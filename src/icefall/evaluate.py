"""Model selection, held-out scoring and the evaluation-discipline studies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from netcal.scaling import TemperatureScaling
from sklearn.metrics import log_loss, roc_auc_score

from . import config, models, stats

# Order matters: when two models are within one standard error, the earlier one wins.
SIMPLICITY = ["lookup", "logistic", "logistic_loose", "boosting", "boosting_deep"]


def fit_predict(kind: str, train: pd.DataFrame, test: pd.DataFrame, target: str = "success") -> np.ndarray:
    if kind == "lookup":
        return models.peak_lookup_scores(test)
    m = models.make_model(kind, config.SEED).fit(train[models.FEATURES], train[target])
    return m.predict_proba(test[models.FEATURES])[:, 1]


def _row_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def select_model(train: pd.DataFrame, valid: pd.DataFrame, n_boot: int = 300) -> tuple[str, pd.DataFrame, dict]:
    """Pick by validation log loss with the one-standard-error rule."""
    y = valid["success"].to_numpy()
    g = valid["expedition_id"].to_numpy()
    preds = {k: fit_predict(k, train, valid) for k in SIMPLICITY}
    rows = []
    for k, p in preds.items():
        loss = _row_loss(y, p)
        draws = stats.cluster_bootstrap(g, lambda i, loss=loss: loss[i].mean(), n_boot)
        rows.append(
            {
                "model": k,
                "valid_log_loss": loss.mean(),
                "se": draws.std(ddof=1),
                **{f"valid_{a}": b for a, b in models.score_all(y, p).items() if a != "log_loss"},
            }
        )
    tab = pd.DataFrame(rows).set_index("model")
    best = tab["valid_log_loss"].idxmin()
    cutoff = tab.loc[best, "valid_log_loss"] + tab.loc[best, "se"]
    chosen = next(k for k in SIMPLICITY if tab.loc[k, "valid_log_loss"] <= cutoff)
    return chosen, tab.reset_index(), preds


def holdout_report(y, p, groups, baseline, n_boot: int = 400) -> dict:
    """Scores with expedition-clustered intervals and paired gaps to a baseline."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    b = np.clip(np.asarray(baseline), 1e-6, 1 - 1e-6)
    out = models.score_all(y, p)
    auc = stats.cluster_bootstrap(groups, lambda i: roc_auc_score(y[i], p[i]), n_boot)
    ll = stats.cluster_bootstrap(groups, lambda i: log_loss(y[i], p[i], labels=[0, 1]), n_boot)
    gap = stats.cluster_bootstrap(
        groups, lambda i: log_loss(y[i], b[i], labels=[0, 1]) - log_loss(y[i], p[i], labels=[0, 1]), n_boot
    )
    out["auc_ci"] = stats.interval(auc)
    out["log_loss_ci"] = stats.interval(ll)
    out["baseline_log_loss"] = float(log_loss(y, b))
    out["log_loss_gain_vs_baseline"] = float(log_loss(y, b) - log_loss(y, p))
    out["log_loss_gain_ci"] = stats.interval(gap)
    out["baseline_auc"] = float(roc_auc_score(y, b))
    return out


def temperature_check(p_valid, y_valid, p_test, y_test) -> dict:
    """Does a temperature fitted on validation help on test?"""
    ts = TemperatureScaling()
    ts.fit(np.asarray(p_valid), np.asarray(y_valid))
    cal = np.clip(ts.transform(np.asarray(p_test)), 1e-6, 1 - 1e-6)
    return {
        "temperature": float(np.ravel(ts.temperature)[0]),
        "raw": models.score_all(y_test, p_test),
        "calibrated": models.score_all(y_test, cal),
    }


def by_year(df: pd.DataFrame, p: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    t = df[["year", "success"]].copy()
    t["model"] = p
    t["lookup"] = baseline
    out = t.groupby("year").agg(
        n=("success", "size"), observed=("success", "mean"), model=("model", "mean"), lookup=("lookup", "mean")
    )
    return out.reset_index()


def peak_table(df: pd.DataFrame, p: np.ndarray, top: int = 8) -> pd.DataFrame:
    t = df[["peak_id", "peak_name", "success", "recent_success_rate"]].copy()
    t["p"] = p
    rows = []
    for _pid, sub in t.groupby("peak_id"):
        if len(sub) < 150 or sub["success"].nunique() < 2:
            continue
        rows.append(
            {
                "peak": sub["peak_name"].iloc[0],
                "climbers": len(sub),
                "observed": sub["success"].mean(),
                "predicted": sub["p"].mean(),
                "auc_model": roc_auc_score(sub["success"], sub["p"]),
                "auc_lookup": roc_auc_score(sub["success"], sub["recent_success_rate"])
                if sub["recent_success_rate"].nunique() > 1
                else 0.5,
            }
        )
    return pd.DataFrame(rows).sort_values("climbers", ascending=False).head(top).reset_index(drop=True)


# ---------------------------------------------------------------- audits ---


def split_study(df: pd.DataFrame, kind: str, seed: int = 7) -> pd.DataFrame:
    """Same model, three ways to hold data out."""
    rng = np.random.default_rng(seed)
    rows = []

    def run(name, tr, te, note):
        p = fit_predict(kind, tr, te)
        sc = models.score_all(te["success"], p)
        rows.append(
            {
                "split": name,
                "train_rows": len(tr),
                "test_rows": len(te),
                "auc": sc["auc"],
                "log_loss": sc["log_loss"],
                "note": note,
            }
        )

    pre = df[df["year"] <= 2019]
    idx = rng.permutation(len(pre))
    cut = int(0.8 * len(pre))
    run("random rows", pre.iloc[idx[:cut]], pre.iloc[idx[cut:]], "teammates on both sides")
    exps = np.asarray(pre["expedition_id"].unique(), dtype=object)
    rng.shuffle(exps)
    held = set(exps[: int(0.2 * len(exps))])
    mask = pre["expedition_id"].isin(held)
    run("random expeditions", pre[~mask], pre[mask], "whole expeditions held out")
    s = df[df["year"] <= config.VALID_END]
    run(
        "forward in time",
        s[s["year"] <= config.TRAIN_END],
        df[df["year"] > config.VALID_END],
        "train to 2008, test 2014 on",
    )
    return pd.DataFrame(rows)


def leak_study(train: pd.DataFrame, test: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Add one outcome-side column at a time and watch the score."""
    base = fit_predict(kind, train, test)
    rows = [{"scenario": "booking-time features only", "auc": roc_auc_score(test["success"], base)}]

    def with_extra(name, col, fill=-1.0):
        tr, te = train.copy(), test.copy()
        for d in (tr, te):
            d["__x"] = d[col].astype(float).fillna(fill)
        cols = models.FEATURES + ["__x"]
        m = models.make_model(kind, config.SEED, extra_numeric=("__x",)).fit(tr[cols], tr["success"])
        p = m.predict_proba(te[cols])[:, 1]
        rows.append({"scenario": name, "auc": roc_auc_score(te["success"], p)})

    with_extra("+ member used oxygen", "oxygen_used")
    with_extra("+ member highpoint (metres)", "highpoint_metres")
    for d in (train, test):
        d["__exp_ok"] = d["termination_reason"].astype(str).str.startswith("Success").astype(int)
    with_extra("+ expedition ended in success", "__exp_ok")
    return pd.DataFrame(rows)


def dependence_study(test: pd.DataFrame, p: np.ndarray, n_boot: int = 400) -> dict:
    y = test["success"].to_numpy()
    g = test["expedition_id"].to_numpy()
    icc = stats.intraclass_correlation(y, g)
    rng = np.random.default_rng(config.SEED)
    naive = np.array([roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot // 2))])
    clustered = stats.cluster_bootstrap(g, lambda i: roc_auc_score(y[i], p[i]), n_boot)
    return {
        **icc,
        "naive_auc_ci": stats.interval(naive),
        "cluster_auc_ci": stats.interval(clustered),
        "ci_width_ratio": float(
            np.subtract(*stats.interval(clustered)[::-1]) / np.subtract(*stats.interval(naive)[::-1])
        ),
    }
