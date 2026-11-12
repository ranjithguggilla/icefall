import numpy as np

from icefall import stats


def test_cluster_bootstrap_resamples_whole_groups():
    groups = np.repeat(np.arange(50), 4)
    seen = []

    def stat(idx):
        seen.append(len(idx))
        return len(set(groups[idx]))

    draws = stats.cluster_bootstrap(groups, stat, n_boot=30, seed=1)
    assert all(n == 200 for n in seen)
    assert draws.max() <= 50


def test_icc_is_high_when_groups_share_the_outcome():
    groups = np.repeat(np.arange(200), 5)
    y = np.repeat(np.random.default_rng(0).integers(0, 2, 200), 5)
    r = stats.intraclass_correlation(y, groups)
    assert r["icc"] > 0.95
    assert r["design_effect"] > 4


def test_icc_is_near_zero_for_independent_rows():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(400), 5)
    y = rng.integers(0, 2, 2000)
    r = stats.intraclass_correlation(y, groups)
    assert r["icc"] < 0.05
    assert abs(r["design_effect"] - 1) < 0.25


def test_interval_orders_the_ends():
    lo, hi = stats.interval(np.arange(100.0))
    assert lo < hi
