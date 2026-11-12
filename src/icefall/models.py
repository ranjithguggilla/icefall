"""Booking-time feature set, model constructors and plain metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = [
    "age",
    "age_missing",
    "height_metres",
    "years_since_first_ascent",
    "solo",
    "team_size",
    "hired_staff",
    "has_agency",
    "is_8000",
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
COUNTS = ["prior_climbers", "recent_climbers", "team_size", "hired_staff", "years_since_first_ascent"]
CATEGORICAL = ["season", "sex", "role", "citizenship", "peak_id"]
FEATURES = NUMERIC + CATEGORICAL


class RareLevels(BaseEstimator, TransformerMixin):
    """Keep the most common levels seen in training, call the rest 'other'.

    Living inside the pipeline means the level lists are learned from training
    rows only, in every fold. Citizenship keeps 14 levels and peaks keep 120,
    which also keeps boosting under its 255-category limit.
    """

    def __init__(self, top: tuple = (("citizenship", 14), ("peak_id", 120))):
        self.top = top

    def fit(self, X, y=None):
        self.levels_ = {c: X[c].value_counts().head(n).index.tolist() for c, n in self.top}
        return self

    def transform(self, X):
        X = X.copy()
        for c, keep in self.levels_.items():
            X[c] = X[c].where(X[c].isin(keep), "other").fillna("other")
        return X


def _linear(C: float, numeric: list[str]) -> Pipeline:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer

    plain = [c for c in numeric if c not in COUNTS]
    pre = ColumnTransformer(
        [
            # Age is the only column with real gaps; the missing flag carries the signal.
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), plain),
            (
                "cnt",
                Pipeline(
                    [
                        (
                            "log",
                            FunctionTransformer(
                                lambda a: np.log1p(np.clip(a, 0, None)), feature_names_out="one-to-one"
                            ),
                        ),
                        ("sc", StandardScaler()),
                    ]
                ),
                COUNTS,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=40, sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("levels", RareLevels()), ("pre", pre), ("clf", LogisticRegression(C=C, max_iter=3000))])


def _boosted(learning_rate: float, max_leaf_nodes: int, l2: float, seed: int, numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL,
            ),
        ]
    )
    mask = [False] * len(numeric) + [True] * len(CATEGORICAL)
    return Pipeline(
        [
            ("levels", RareLevels()),
            ("pre", pre),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    categorical_features=mask,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_model(kind: str, seed: int = 7, extra_numeric: tuple[str, ...] = ()) -> Pipeline:
    num = NUMERIC + list(extra_numeric)
    if kind == "logistic":
        return _linear(0.3, num)
    if kind == "logistic_loose":
        return _linear(3.0, num)
    if kind == "boosting":
        return _boosted(0.05, 15, 1.0, seed, num)
    if kind == "boosting_deep":
        return _boosted(0.05, 40, 0.5, seed, num)
    raise ValueError(f"unknown model kind: {kind}")


MODEL_KINDS = ["logistic", "logistic_loose", "boosting", "boosting_deep"]


def peak_lookup_scores(df: pd.DataFrame) -> np.ndarray:
    """Baseline that knows only the peak's own earlier success rate."""
    return df["recent_success_rate"].to_numpy()


def ece(y, p, bins: int = 10) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(total)


def score_all(y, p) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }
