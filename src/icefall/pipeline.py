"""End-to-end run: data, selection, held-out scores, audits, oxygen studies, charts."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, data, evaluate, models, oxygen, plots, score, stats


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def death_study(train, valid, test, n_boot):
    trv = pd.concat([train, valid])
    p = evaluate.fit_predict("boosting", trv, test, target="died")
    y = test["died"].to_numpy()
    g = test["expedition_id"].to_numpy()
    base = test["recent_death_rate"].to_numpy()
    auc = stats.cluster_bootstrap(g, lambda i: roc_auc_score(y[i], p[i]), n_boot)
    order = np.argsort(-p)
    tier = np.empty(len(p), dtype=int)
    tier[order] = np.minimum((np.arange(len(p)) * 10) // len(p), 9) + 1
    t = pd.DataFrame({"tier": tier, "died": y, "p": p})
    tiers = t.groupby("tier").agg(climbers=("died", "size"), deaths=("died", "sum"), predicted=("p", "mean"))
    tiers["observed"] = tiers["deaths"] / tiers["climbers"]
    tiers["share_of_deaths"] = tiers["deaths"] / max(tiers["deaths"].sum(), 1)
    res = {
        "deaths_in_test": int(y.sum()),
        "death_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "auc_ci": stats.interval(auc),
        "lookup_auc": float(roc_auc_score(y, base)),
        "top_decile_share_of_deaths": float(tiers.loc[1, "share_of_deaths"]),
        "top_two_deciles_share": float(tiers.loc[[1, 2], "share_of_deaths"].sum()),
        "mean_predicted": float(p.mean()),
    }
    return res, tiers.reset_index()


def oxygen_planning(train, valid, test, n_boot):
    tr, va, te = (d[d["is_8000"] == 1] for d in (train, valid, test))
    p_va = evaluate.fit_predict("boosting", tr, va, target="oxygen_used")
    p_te = evaluate.fit_predict("boosting", pd.concat([tr, va]), te, target="oxygen_used")
    ev = oxygen.expedition_demand(va, p_va)
    et = oxygen.expedition_demand(te, p_te)
    c = config.COSTS
    off = oxygen.safety_offset(ev, c.unused_set, c.missing_set)
    pol = oxygen.policies(et, off)
    demand = et["demand"].to_numpy(dtype=float)
    costs = {k: oxygen.order_cost(v, demand, c.unused_set, c.missing_set) for k, v in pol.items()}
    tab = pd.DataFrame(
        {
            "policy": list(costs),
            "total_cost": [v.sum() for v in costs.values()],
            "sets_ordered": [pol[k].sum() for k in costs],
            "sets_unused": [np.clip(pol[k] - demand, 0, None).sum() for k in costs],
            "sets_short": [np.clip(demand - pol[k], 0, None).sum() for k in costs],
        }
    )
    tab["cost_per_member"] = tab["total_cost"] / et["members"].sum()
    g = et["expedition_id"].to_numpy()
    own, rate, full = costs["model plus safety offset"], costs["peak history rate"], costs["order for every member"]
    d_rate = stats.cluster_bootstrap(g, lambda i: (rate[i] - own[i]).sum(), n_boot)
    d_full = stats.cluster_bootstrap(g, lambda i: (full[i] - own[i]).sum(), n_boot)
    sweep = []
    for ratio in [1, 2, 3, c.missing_set / c.unused_set, 10, 20]:
        un, ms = c.unused_set, c.unused_set * ratio
        o = oxygen.safety_offset(ev, un, ms)
        pp = oxygen.policies(et, o)
        cc = {k: oxygen.order_cost(v, demand, un, ms).sum() for k, v in pp.items()}
        sweep.append(
            {
                "ratio": ratio,
                "offset": o,
                "saving_vs_rate": cc["peak history rate"] - cc["model plus safety offset"],
                "saving_vs_full": cc["order for every member"] - cc["model plus safety offset"],
                "saving_vs_expected": cc["model expected demand"] - cc["model plus safety offset"],
            }
        )
    info = {
        "expeditions": int(len(et)),
        "members": int(et["members"].sum()),
        "sets_used": int(demand.sum()),
        "offset_sets": off,
        "saving_per_member_vs_full": float((full.sum() - own.sum()) / et["members"].sum()),
        "saving_per_member_vs_rate": float((rate.sum() - own.sum()) / et["members"].sum()),
        "saving_vs_rate": float(rate.sum() - own.sum()),
        "saving_vs_rate_ci": stats.interval(d_rate),
        "saving_vs_full": float(full.sum() - own.sum()),
        "saving_vs_full_ci": stats.interval(d_full),
        "expedition_mae_model": float(np.abs(et["expected"] - et["demand"]).mean()),
        "expedition_mae_rate": float(np.abs(et["rate_lookup"] - et["demand"]).mean()),
        "member_auc": float(roc_auc_score(te["oxygen_used"], p_te)),
        "assumptions": {"unused_set_usd": c.unused_set, "missing_set_usd": c.missing_set},
    }
    return info, tab, pd.DataFrame(sweep)


def causal_studies(df, expeditions, n_boot, with_dowhy=True):  # returns (results, per-peak table)
    d = oxygen.causal_frame(df, expeditions)
    res = {"sample": int(len(d)), "expeditions": int(d["expedition_id"].nunique())}
    for treat in ["oxygen_used", "exp_oxygen"]:
        for outcome in ["success", "died"]:
            r = oxygen.effect_with_ci(d, treat, outcome, n_boot=n_boot)
            rr_base = max(r["control_rate"], 1e-6)
            r["risk_ratio_naive"] = r["treated_rate"] / rr_base
            res[f"{treat}__{outcome}"] = r
    s = res["oxygen_used__success"]
    s["e_value_naive_rr"] = oxygen.e_value(s["risk_ratio_naive"])
    # permutation placebo: shuffle the treatment, the adjusted effect should sit near zero
    rng = np.random.default_rng(config.SEED)
    cols = list(oxygen.design(d).columns)
    placebo = []
    for _ in range(10):
        dd = d.copy()
        dd["oxygen_used"] = rng.permutation(dd["oxygen_used"].to_numpy())
        placebo.append(oxygen.estimate_effect(dd, "oxygen_used", "success", cols)["aipw"])
    res["placebo_permuted_aipw_mean"] = float(np.mean(placebo))
    res["placebo_permuted_aipw_sd"] = float(np.std(placebo, ddof=1))
    if with_dowhy:
        res["dowhy"] = oxygen.dowhy_check(d, "oxygen_used", "success")
    arms, by_peak = oxygen.composition(d)
    res["composition"] = arms
    return res, by_peak


def run(fast: bool = False, with_dowhy: bool = True) -> dict:
    t0 = time.time()
    nb = 60 if fast else 400
    config.TABLES.mkdir(parents=True, exist_ok=True)
    config.ASSETS.mkdir(parents=True, exist_ok=True)

    raw = data.load_raw()
    df = data.with_history(data.build_table(raw))
    parts = data.split_by_year(df)
    train, valid, test = parts["train"], parts["valid"], parts["test"]
    trv = pd.concat([train, valid])
    out: dict = {
        "data": {
            "climbers": len(df),
            "expeditions": int(df["expedition_id"].nunique()),
            "peaks": int(df["peak_id"].nunique()),
            "first_year": int(df["year"].min()),
            "last_year": int(df["year"].max()),
            "summit_rate": float(df["success"].mean()),
            "death_rate": float(df["died"].mean()),
            "splits": {
                k: {
                    "rows": len(v),
                    "expeditions": int(v["expedition_id"].nunique()),
                    "summit_rate": float(v["success"].mean()),
                    "years": [int(v["year"].min()), int(v["year"].max())],
                }
                for k, v in parts.items()
            },
        }
    }

    # 1. selection on validation
    chosen, sel, _ = evaluate.select_model(train, valid, n_boot=nb)
    sel.to_csv(config.TABLES / "model_selection.csv", index=False)
    out["selection"] = {"chosen": chosen, "table": sel.round(5).to_dict("records")}

    # 2. held-out scores, shipped model refit on train + validation
    p_te = evaluate.fit_predict(chosen, trv, test)
    base_te = models.peak_lookup_scores(test)
    out["test"] = evaluate.holdout_report(test["success"], p_te, test["expedition_id"], base_te, nb)
    p_train_only = evaluate.fit_predict(chosen, train, test)
    p_va = evaluate.fit_predict(chosen, train, valid)
    out["test"]["train_only_model"] = models.score_all(test["success"], p_train_only)
    out["calibration"] = evaluate.temperature_check(p_va, valid["success"], p_train_only, test["success"])
    out["test"]["mean_predicted"] = float(p_te.mean())
    out["test"]["mean_observed"] = float(test["success"].mean())

    byy = evaluate.by_year(test, p_te, base_te)
    byy.to_csv(config.TABLES / "test_by_year.csv", index=False)
    out["by_year"] = byy.round(4).to_dict("records")
    pt = evaluate.peak_table(test, p_te)
    pt.to_csv(config.TABLES / "test_by_peak.csv", index=False)
    out["by_peak"] = pt.round(4).to_dict("records")

    # 3. evaluation discipline
    split_tab = evaluate.split_study(df, chosen)
    split_tab.to_csv(config.TABLES / "split_study.csv", index=False)
    leak_tab = evaluate.leak_study(trv.copy(), test.copy(), chosen)
    leak_tab.to_csv(config.TABLES / "leak_study.csv", index=False)
    out["split_study"] = split_tab.round(4).to_dict("records")
    out["leak_study"] = leak_tab.round(4).to_dict("records")
    out["dependence"] = evaluate.dependence_study(test, p_te, nb)

    # 4. causal oxygen
    out["causal"], oxy_peaks = causal_studies(df, raw["expeditions"], 20 if fast else 120, with_dowhy and not fast)
    oxy_peaks.to_csv(config.TABLES / "oxygen_by_peak.csv", index=False)

    # 5. oxygen ordering
    info, ptab, sweep = oxygen_planning(train, valid, test, nb)
    ptab.to_csv(config.TABLES / "oxygen_policies.csv", index=False)
    sweep.to_csv(config.TABLES / "oxygen_cost_sweep.csv", index=False)
    out["oxygen_planning"] = {
        **info,
        "policies": ptab.round(2).to_dict("records"),
        "sweep": sweep.round(2).to_dict("records"),
    }

    # 6. death risk
    dres, tiers = death_study(train, valid, test, nb)
    tiers.to_csv(config.TABLES / "death_tiers.csv", index=False)
    out["death"] = dres

    # 7. charts
    plots.era_drift(byy, config.ASSETS / "era_drift.png")
    plots.model_compare(sel, config.ASSETS / "model_selection.png")
    plots.leakage(split_tab, leak_tab, config.ASSETS / "leakage_audit.png")
    plots.reliability(test["success"], p_te, config.ASSETS / "reliability.png")
    c = out["causal"]
    rows = []
    for label, key in [
        ("member used oxygen", "oxygen_used__success"),
        ("expedition took oxygen", "exp_oxygen__success"),
    ]:
        r = c[key]
        rows.append((label, r["naive"], r["aipw"], *r["aipw_ci"]))
    plots.oxygen_effect(rows, config.ASSETS / "oxygen_effect.png")
    plots.oxygen_cost(ptab, sweep, config.ASSETS / "oxygen_cost.png")
    plots.death_lift(tiers, config.ASSETS / "death_tiers.png")

    # 8. ship the app bundle: models refit on every year available
    full_success = models.make_model(chosen, config.SEED).fit(df[models.FEATURES], df["success"])
    d8 = df[df["is_8000"] == 1]
    full_oxy = models.make_model("boosting", config.SEED).fit(d8[models.FEATURES], d8["oxygen_used"])
    full_death = models.make_model("boosting", config.SEED).fit(df[models.FEATURES], df["died"])
    bundle = score.make_bundle(df, raw["peaks"], full_success, full_oxy, full_death)
    score.save_bundle(bundle, config.OUT / "bundle.joblib")

    out["runtime_seconds"] = round(time.time() - t0, 1)
    out["fast"] = fast
    (config.OUT / "metrics.json").write_text(json.dumps(_clean(out), indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the full icefall pipeline")
    ap.add_argument("--fast", action="store_true", help="fewer bootstrap draws, no dowhy (for CI smoke runs)")
    args = ap.parse_args(argv)
    res = run(fast=args.fast)
    print(json.dumps(_clean(res["test"]), indent=2))


if __name__ == "__main__":
    main()
