"""Two oxygen questions: what does using it change, and how many sets to order."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config, stats

CONFOUNDERS_NUM = ["age", "age_missing", "log_team", "log_staff"]
CONFOUNDERS_CAT = ["peak", "era", "sex", "role", "season", "nation"]


def causal_frame(df: pd.DataFrame, expeditions: pd.DataFrame) -> pd.DataFrame:
    """Climbers on 8,000 m peaks since 1980 with a member-level and an expedition-level treatment."""
    d = df[(df["is_8000"] == 1) & (df["year"] >= 1980)].copy()
    ex = expeditions[["expedition_id", "oxygen_used"]].rename(columns={"oxygen_used": "exp_oxygen"})
    ex["exp_oxygen"] = ex["exp_oxygen"].astype(str).str.upper().eq("TRUE").astype(int)
    d = d.merge(ex, on="expedition_id", how="left")
    d["exp_oxygen"] = d["exp_oxygen"].fillna(0).astype(int)
    top_peaks = d["peak_id"].value_counts().head(10).index
    top_nations = d["citizenship"].value_counts().head(12).index
    d["peak"] = d["peak_id"].where(d["peak_id"].isin(top_peaks), "other")
    d["nation"] = d["citizenship"].where(d["citizenship"].isin(top_nations), "other")
    d["era"] = (d["year"] // 5 * 5).astype(str)
    d["log_team"] = np.log1p(d["team_size"])
    d["log_staff"] = np.log1p(d["hired_staff"])
    d["age"] = d["age"].fillna(d["age"].median())
    return d.reset_index(drop=True)


def design(d: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    x = pd.get_dummies(d[CONFOUNDERS_NUM + CONFOUNDERS_CAT], columns=CONFOUNDERS_CAT, drop_first=True, dtype=float)
    x["age"] = (x["age"] - 38) / 10
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0.0)
    return x


def _fit(x, y, c=1.0):
    return LogisticRegression(C=c, max_iter=1000).fit(x, y)


def estimate_effect(d: pd.DataFrame, treat: str, outcome: str, columns: list[str] | None = None) -> dict:
    """Naive gap, g-computation, IPW and AIPW risk differences for one treatment."""
    t = d[treat].to_numpy()
    y = d[outcome].to_numpy()
    x = design(d, columns)
    if t.min() == t.max():
        raise ValueError("treatment has no variation")
    ps_model = _fit(x, t)
    ps = np.clip(ps_model.predict_proba(x)[:, 1], 0.02, 0.98)
    xt = x.assign(__t=t)
    out_model = _fit(xt, y, c=2.0)
    m1 = out_model.predict_proba(x.assign(__t=1))[:, 1]
    m0 = out_model.predict_proba(x.assign(__t=0))[:, 1]
    ipw = (t * y / ps).sum() / (t / ps).sum() - ((1 - t) * y / (1 - ps)).sum() / ((1 - t) / (1 - ps)).sum()
    aipw = np.mean(m1 - m0 + t * (y - m1) / ps - (1 - t) * (y - m0) / (1 - ps))
    return {
        "naive": float(y[t == 1].mean() - y[t == 0].mean()),
        "g_computation": float((m1 - m0).mean()),
        "ipw": float(ipw),
        "aipw": float(aipw),
        "treated_share": float(t.mean()),
        "treated_rate": float(y[t == 1].mean()),
        "control_rate": float(y[t == 0].mean()),
        "n": int(len(d)),
        "propensity_min": float(ps.min()),
        "propensity_max": float(ps.max()),
    }


def effect_with_ci(d: pd.DataFrame, treat: str, outcome: str, n_boot: int = 120) -> dict:
    point = estimate_effect(d, treat, outcome)
    cols = list(design(d).columns)
    g = d["expedition_id"].to_numpy()

    def stat(i):
        sub = d.iloc[i]
        try:
            return estimate_effect(sub, treat, outcome, cols)["aipw"]
        except ValueError:
            return np.nan

    draws = stats.cluster_bootstrap(g, stat, n_boot, seed=config.SEED)
    draws = draws[~np.isnan(draws)]
    point["aipw_ci"] = stats.interval(draws)
    return point


def composition(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Who sits in each arm of the expedition-level comparison, and how each peak splits."""
    d = d.assign(high_end=d["peak_id"].isin(["EVER", "LHOT"]).astype(int))
    arms = d.groupby("exp_oxygen").agg(
        climbers=("success", "size"),
        summit_rate=("success", "mean"),
        mean_height_m=("height_metres", "mean"),
        share_everest_lhotse=("high_end", "mean"),
    )
    by_peak = (
        d.groupby(["peak_id", "exp_oxygen"])
        .agg(climbers=("success", "size"), summit_rate=("success", "mean"))
        .unstack("exp_oxygen")
    )
    by_peak.columns = [f"{a}_{'with' if b == 1 else 'without'}" for a, b in by_peak.columns]
    by_peak = by_peak.reset_index()
    names = d.drop_duplicates("peak_id").set_index("peak_id")["peak_name"]
    by_peak.insert(1, "peak", by_peak["peak_id"].map(names))
    by_peak = by_peak.sort_values("climbers_with", ascending=False).reset_index(drop=True)
    return arms.reset_index().to_dict("records"), by_peak


def e_value(rr: float) -> float:
    rr = max(rr, 1 / rr)
    return float(rr + np.sqrt(rr * (rr - 1)))


def dowhy_check(d: pd.DataFrame, treat: str, outcome: str, sample: int = 8000, seed: int = 7, sims: int = 10) -> dict:
    """Cross-check the effect with dowhy: two estimators, each with two refuters.

    A permuted-treatment placebo should land near zero and a random extra
    confounder should leave the estimate alone. The weighting estimator fails
    the placebo inside dowhy (see the README), so both are reported side by side.
    """
    import logging

    from dowhy import CausalModel

    logging.getLogger("dowhy").setLevel(logging.ERROR)
    sub = d.sample(min(sample, len(d)), random_state=seed)
    x = design(sub)
    data = pd.concat([x.reset_index(drop=True), sub[[treat, outcome]].reset_index(drop=True)], axis=1)
    data[treat] = data[treat].astype(bool)
    model = CausalModel(data=data, treatment=treat, outcome=outcome, common_causes=list(x.columns))
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    out: dict = {"sample": int(len(sub))}
    for tag, method, params in [
        ("regression", "backdoor.linear_regression", None),
        ("weighting", "backdoor.propensity_score_weighting", {"weighting_scheme": "ips_stabilized_weight"}),
    ]:
        kwargs = {"method_params": params} if params else {}
        est = model.estimate_effect(estimand, method_name=method, **kwargs)
        out[f"{tag}_ate"] = float(est.value)
        for name, extra in [
            ("placebo", ("placebo_treatment_refuter", {"placebo_type": "permute", "num_simulations": sims})),
            ("random_cause", ("random_common_cause", {"num_simulations": sims})),
        ]:
            try:
                r = model.refute_estimate(estimand, est, method_name=extra[0], **extra[1])
                out[f"{tag}_{name}"] = float(r.new_effect)
            except Exception as exc:  # refuters are a cross-check, never a blocker
                out[f"{tag}_{name}"] = None
                out[f"{tag}_{name}_error"] = type(exc).__name__
    return out


# ------------------------------------------------------- demand planning ---


def expedition_demand(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    t = df[["expedition_id", "oxygen_used", "recent_oxygen_rate"]].copy()
    t["p"] = p
    g = t.groupby("expedition_id").agg(
        members=("oxygen_used", "size"),
        demand=("oxygen_used", "sum"),
        expected=("p", "sum"),
        rate_lookup=("recent_oxygen_rate", "sum"),
    )
    return g.reset_index()


def order_cost(order: np.ndarray, demand: np.ndarray, unused: float, missing: float) -> np.ndarray:
    return unused * np.clip(order - demand, 0, None) + missing * np.clip(demand - order, 0, None)


def safety_offset(exp_valid: pd.DataFrame, unused: float, missing: float) -> float:
    """Newsvendor offset: the residual quantile where marginal waste meets marginal shortage."""
    q = missing / (missing + unused)
    return float(np.quantile(exp_valid["demand"] - exp_valid["expected"], q))


def policies(exp: pd.DataFrame, offset: float) -> dict[str, np.ndarray]:
    return {
        "order for every member": exp["members"].to_numpy(dtype=float),
        "peak history rate": np.ceil(exp["rate_lookup"].to_numpy()),
        "model expected demand": np.ceil(np.clip(exp["expected"].to_numpy(), 0, exp["members"].to_numpy())),
        "model plus safety offset": np.ceil(np.clip(exp["expected"].to_numpy() + offset, 0, exp["members"].to_numpy())),
        "hindsight (exact demand)": exp["demand"].to_numpy(dtype=float),
    }
