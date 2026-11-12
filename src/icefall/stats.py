"""Resampling helpers that respect the fact climbers share an expedition."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def group_indices(groups) -> list[np.ndarray]:
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    sorted_g = groups[order]
    cuts = np.flatnonzero(sorted_g[1:] != sorted_g[:-1]) + 1
    return np.split(order, cuts)


def cluster_bootstrap(
    groups,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 400,
    seed: int = 7,
) -> np.ndarray:
    """Resample whole expeditions with replacement; ``stat`` gets row indices."""
    rng = np.random.default_rng(seed)
    blocks = group_indices(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        out[b] = stat(np.concatenate([blocks[i] for i in pick]))
    return out


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    return float(lo), float(hi)


def intraclass_correlation(y, groups) -> dict[str, float]:
    """One-way ANOVA estimate of how much of a 0/1 outcome is shared in a group.

    Returns the ICC, the mean group size and the design effect
    ``1 + (m - 1) * icc`` that tells you how many independent rows the data is
    really worth.
    """
    y = np.asarray(y, dtype=float)
    blocks = group_indices(groups)
    n = len(y)
    k = len(blocks)
    sizes = np.array([len(b) for b in blocks])
    means = np.array([y[b].mean() for b in blocks])
    grand = y.mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    msw = float(sum(((y[b] - y[b].mean()) ** 2).sum() for b in blocks) / (n - k))
    m0 = (n - (sizes**2).sum() / n) / (k - 1)
    icc = max((msb - msw) / (msb + (m0 - 1) * msw), 0.0)
    mbar = n / k
    deff = 1 + (mbar - 1) * icc
    return {
        "icc": float(icc),
        "mean_group_size": float(mbar),
        "design_effect": float(deff),
        "effective_n": float(n / deff),
    }
