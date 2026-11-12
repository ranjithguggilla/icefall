"""Scoring one climber: probability, a plain-language band and the reasons behind it."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import config, data, models

SEASONS = ["Spring", "Autumn", "Winter", "Summer"]
ROLES = ["climber", "leader", "support", "other"]


@dataclass
class Bundle:
    success_model: object
    oxygen_model: object
    death_model: object
    peaks: pd.DataFrame  # one row per peak with its latest point-in-time history
    reference: dict = field(default_factory=dict)
    nations: list[str] = field(default_factory=list)
    next_year: int = 2020


def build_peak_table(df: pd.DataFrame, raw_peaks: pd.DataFrame) -> pd.DataFrame:
    hist = data.prior_rate_features(df, horizon=1)
    last = hist[hist["year"] == df["year"].max() + 1]
    n = df.groupby("peak_id").size().rename("climbers_on_record")
    t = raw_peaks[["peak_id", "peak_name", "height_metres", "first_ascent_year"]].merge(last, on="peak_id", how="inner")
    t = t.merge(n, on="peak_id", how="left")
    return (
        t[t["climbers_on_record"].fillna(0) >= 100].sort_values("height_metres", ascending=False).reset_index(drop=True)
    )


def make_bundle(df, raw_peaks, success_model, oxygen_model, death_model) -> Bundle:
    peaks = build_peak_table(df, raw_peaks)
    ref = {
        "age": float(df["age"].median()),
        "team_size": float(df["team_size"].median()),
        "hired_staff": float(df["hired_staff"].median()),
        "season": df["season"].mode().iloc[0],
        "role": "climber",
        "sex": df["sex"].mode().iloc[0],
        "citizenship": "other",
    }
    nations = df["citizenship"].value_counts().head(25).index.tolist()
    return Bundle(success_model, oxygen_model, death_model, peaks, ref, nations, int(df["year"].max()) + 1)


def save_bundle(b: Bundle, path) -> None:
    joblib.dump(b, path)


def load_bundle(path=None) -> Bundle:
    return joblib.load(path or config.OUT / "bundle.joblib")


def _row(b: Bundle, peak_id: str, **kw) -> pd.DataFrame:
    pk = b.peaks[b.peaks["peak_id"] == peak_id]
    if pk.empty:
        raise KeyError(f"unknown peak id {peak_id!r}")
    pk = pk.iloc[0]
    age = kw.get("age")
    row = {
        "age": np.nan if age is None else float(age),
        "age_missing": int(age is None),
        "height_metres": pk["height_metres"],
        "years_since_first_ascent": max(b.next_year - pk["first_ascent_year"], -1)
        if pd.notna(pk["first_ascent_year"])
        else -1,
        "solo": int(kw.get("solo", 0)),
        "team_size": float(kw.get("team_size", b.reference["team_size"])),
        "hired_staff": float(kw.get("hired_staff", b.reference["hired_staff"])),
        "has_agency": int(kw.get("has_agency", 1)),
        "is_8000": int(pk["height_metres"] >= config.EIGHT_THOUSANDER),
        "season": kw.get("season", b.reference["season"]),
        "sex": kw.get("sex", b.reference["sex"]),
        "role": kw.get("role", "climber"),
        "citizenship": kw.get("citizenship", "other"),
        "peak_id": peak_id,
    }
    for c in [
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
    ]:
        row[c] = pk[c]
    return pd.DataFrame([row])[models.FEATURES]


def probabilities(b: Bundle, frame: pd.DataFrame) -> dict[str, float]:
    return {
        "success": float(b.success_model.predict_proba(frame)[0, 1]),
        "oxygen": float(b.oxygen_model.predict_proba(frame)[0, 1]) if frame["is_8000"].iloc[0] else 0.0,
        "death": float(b.death_model.predict_proba(frame)[0, 1]),
    }


def band(p: float) -> str:
    if p < 0.15:
        return "long shot"
    if p < 0.35:
        return "below even"
    if p < 0.55:
        return "roughly a coin flip"
    if p < 0.75:
        return "better than even"
    return "favourable"


def _swap_peak(b: Bundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Replace every peak-specific column with the median peak's values."""
    f = frame.copy()
    med = b.peaks.median(numeric_only=True)
    for c in [
        "height_metres",
        "prior_success_rate",
        "recent_success_rate",
        "prior_death_rate",
        "recent_death_rate",
        "prior_oxygen_rate",
        "recent_oxygen_rate",
        "prior_climbers",
        "recent_climbers",
    ]:
        f[c] = med[c]
    f["years_since_first_ascent"] = float(b.next_year - med["first_ascent_year"])
    f["is_8000"] = int(med["height_metres"] >= config.EIGHT_THOUSANDER)
    f["peak_id"] = "other"
    return f


def reasons(b: Bundle, frame: pd.DataFrame, top: int = 3) -> list[str]:
    """Swap one input at a time to a typical value and report how far the estimate moves."""
    base = b.success_model.predict_proba(frame)[0, 1]
    tests: dict[str, tuple[str, pd.DataFrame]] = {}
    tests["the peak"] = ("peak", _swap_peak(b, frame))
    for label, col, val in [
        ("the season", "season", b.reference["season"]),
        ("the climber's age", "age", b.reference["age"]),
        ("the team size", "team_size", b.reference["team_size"]),
        ("the hired support", "hired_staff", b.reference["hired_staff"]),
        ("the climber's role", "role", "climber"),
        ("the nationality", "citizenship", "other"),
        ("climbing solo", "solo", 0),
        ("the agency booking", "has_agency", 1),
    ]:
        alt = frame.copy()
        alt[col] = val
        if col == "age":
            alt["age_missing"] = 0
        tests[label] = (col, alt)
    deltas = []
    for label, (_, alt) in tests.items():
        shift = base - b.success_model.predict_proba(alt)[0, 1]
        deltas.append((label, shift))
    deltas.sort(key=lambda t: -abs(t[1]))
    out = []
    for label, shift in deltas[:top]:
        if abs(shift) < 0.005:
            continue
        verb = "adds" if shift > 0 else "takes away"
        out.append(
            f"{label[0].upper() + label[1:]} {verb} about {abs(shift) * 100:.1f} points "
            "compared with a typical climber."
        )
    return out or ["No single input moves the estimate by more than half a point."]


def score_climber(b: Bundle, peak_id: str, **kw) -> dict:
    frame = _row(b, peak_id, **kw)
    pr = probabilities(b, frame)
    pk = b.peaks[b.peaks["peak_id"] == peak_id].iloc[0]
    return {
        "peak": pk["peak_name"],
        "height_m": int(pk["height_metres"]),
        "p_summit": pr["success"],
        "band": band(pr["success"]),
        "p_oxygen": pr["oxygen"],
        "p_death": pr["death"],
        "reasons": reasons(b, frame),
        "peak_recent_rate": float(pk["recent_success_rate"]),
    }
