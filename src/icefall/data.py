"""Load the Himalayan Database extract and build one tidy row per climber."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Columns that describe what happened on the mountain. None of these can be
# known when a climber books, so none of them may reach the booking-time model.
OUTCOME_COLUMNS = ["highpoint_metres", "died", "injured", "oxygen_used"]

ROLE_MAP = {
    "Climber": "climber",
    "Leader": "leader",
    "Deputy Leader": "leader",
    "Co-Leader": "leader",
    "Climbing Leader": "leader",
    "Exp Doctor": "support",
    "Doctor": "support",
}


# Rates used before any earlier year exists. Fixed numbers, so no later outcome leaks backwards.
COLD_START = {"s": 0.25, "d": 0.02, "o": 0.05}


def load_raw() -> dict[str, pd.DataFrame]:
    members = pd.read_csv(config.RAW / "members.csv.gz")
    expeditions = pd.read_csv(config.RAW / "expeditions.csv.gz")
    peaks = pd.read_csv(config.RAW / "peaks.csv")
    return {"members": members, "expeditions": expeditions, "peaks": peaks}


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.upper().eq("TRUE")


def build_table(raw: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per non-hired climber, with booking-time columns plus outcomes.

    Hired staff are removed because their role, pay and incentives differ from
    a paying member's, and their summit rate is double the clients' rate.
    """
    raw = raw or load_raw()
    m = raw["members"].copy()
    e = raw["expeditions"].drop_duplicates("expedition_id", keep="last")  # KANG10101 appears twice
    p = raw["peaks"]

    for col in ["hired", "success", "solo", "oxygen_used", "died", "injured"]:
        m[col] = _as_bool(m[col])

    m = m.drop_duplicates("member_id")
    m = m[~m["hired"] & (m["year"] >= config.MIN_YEAR)].copy()

    exp = e[["expedition_id", "members", "hired_staff", "trekking_agency", "termination_reason"]].rename(
        columns={"members": "team_size"}
    )
    m = m.merge(exp, on="expedition_id", how="left")
    pk = p[["peak_id", "height_metres", "first_ascent_year"]]
    m = m.merge(pk, on="peak_id", how="left")
    m = m[m["height_metres"].notna()].copy()

    m["season"] = m["season"].where(m["season"].isin(["Spring", "Autumn", "Winter", "Summer"]), "Spring")
    m["role"] = m["expedition_role"].map(ROLE_MAP).fillna("other")
    m["age_missing"] = m["age"].isna().astype(int)
    m["has_agency"] = m["trekking_agency"].notna().astype(int)
    m["years_since_first_ascent"] = (m["year"] - m["first_ascent_year"]).where(m["first_ascent_year"].notna(), -1)
    m["years_since_first_ascent"] = m["years_since_first_ascent"].clip(lower=-1)
    m["is_8000"] = (m["height_metres"] >= config.EIGHT_THOUSANDER).astype(int)
    m["solo"] = m["solo"].astype(int)
    for col in ["success", "died", "injured", "oxygen_used"]:
        m[col] = m[col].astype(int)
    m["team_size"] = m["team_size"].fillna(m.groupby("expedition_id")["member_id"].transform("count"))
    m["hired_staff"] = m["hired_staff"].fillna(0)
    return m.reset_index(drop=True)


def split_by_year(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y = df["year"]
    return {
        "train": df[y <= config.TRAIN_END],
        "valid": df[(y > config.TRAIN_END) & (y <= config.VALID_END)],
        "test": df[y > config.VALID_END],
    }


def citizenship_levels(train: pd.Series, top: int = 14) -> list[str]:
    return train.value_counts().head(top).index.tolist()


def prior_rate_features(df: pd.DataFrame, shrink: float = 20.0, window: int = 5, horizon: int = 0) -> pd.DataFrame:
    """Point-in-time peak history: only years strictly before each row's year.

    For every (peak, year) the rate is computed from earlier years alone, so a
    row in 2016 never sees a 2016 outcome from its own peak. Rates are shrunk
    towards the all-peak rate of the same earlier period so a peak with two
    prior climbers does not report a rate of 0 or 1.
    """
    g = (
        df.groupby(["peak_id", "year"])
        .agg(n=("success", "size"), s=("success", "sum"), d=("died", "sum"), o=("oxygen_used", "sum"))
        .reset_index()
    )
    allg = g.groupby("year")[["n", "s", "d", "o"]].sum().sort_index()
    years = np.arange(df["year"].min(), df["year"].max() + 1 + horizon)
    allg = allg.reindex(years, fill_value=0)
    cum_all = allg.cumsum().shift(1, fill_value=0)
    glob = {k: (cum_all[k] / cum_all["n"].replace(0, np.nan)).fillna(COLD_START[k]) for k in ["s", "d", "o"]}

    era = {}
    for k in ["s", "o"]:
        num = allg[k].rolling(window, min_periods=1).sum().shift(1, fill_value=0)
        den = allg["n"].rolling(window, min_periods=1).sum().shift(1, fill_value=0)
        era[k] = (num / den.replace(0, np.nan)).fillna(COLD_START[k])

    rows = []
    for peak, sub in g.groupby("peak_id"):
        sub = sub.drop(columns="peak_id").set_index("year").reindex(years, fill_value=0)
        sub["peak_id"] = peak
        for k in ["n", "s", "d", "o"]:
            sub[f"cum_{k}"] = sub[k].cumsum().shift(1, fill_value=0)
            sub[f"win_{k}"] = sub[k].rolling(window, min_periods=1).sum().shift(1, fill_value=0)
        rows.append(sub.reset_index(names="year"))
    h = pd.concat(rows, ignore_index=True)
    for name, k in [("success", "s"), ("death", "d"), ("oxygen", "o")]:
        gl = h["year"].map(glob[k])
        h[f"prior_{name}_rate"] = (h[f"cum_{k}"] + shrink * gl) / (h["cum_n"] + shrink)
        h[f"recent_{name}_rate"] = (h[f"win_{k}"] + shrink * gl) / (h["win_n"] + shrink)
    h["era_success_rate"] = h["year"].map(era["s"])
    h["era_oxygen_rate"] = h["year"].map(era["o"])
    h["prior_climbers"] = h["cum_n"]
    h["recent_climbers"] = h["win_n"]
    cols = [
        "peak_id",
        "year",
        "prior_success_rate",
        "recent_success_rate",
        "prior_death_rate",
        "recent_death_rate",
        "prior_oxygen_rate",
        "recent_oxygen_rate",
        "prior_climbers",
        "recent_climbers",
        "era_success_rate",
        "era_oxygen_rate",
    ]
    return h[cols]


def with_history(df: pd.DataFrame) -> pd.DataFrame:
    hist = prior_rate_features(df)
    out = df.merge(hist, on=["peak_id", "year"], how="left")
    return out
