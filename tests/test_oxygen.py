import numpy as np
import pandas as pd
import pytest

from icefall import config, data, oxygen


def test_order_cost_is_asymmetric():
    cost = oxygen.order_cost(np.array([3, 1]), np.array([2, 3]), unused=10, missing=100)
    assert list(cost) == [10, 200]


def test_safety_offset_follows_the_newsvendor_quantile():
    exp = pd.DataFrame({"demand": np.arange(101.0), "expected": np.zeros(101)})
    off = oxygen.safety_offset(exp, unused=1, missing=3)
    assert off == pytest.approx(75.0)


def test_policies_respect_team_size():
    exp = pd.DataFrame({"members": [4, 6], "demand": [2, 5], "expected": [2.2, 9.0], "rate_lookup": [1.2, 3.4]})
    pol = oxygen.policies(exp, offset=1.0)
    assert pol["model plus safety offset"].max() <= 6
    assert list(pol["hindsight (exact demand)"]) == [2, 5]
    assert list(pol["order for every member"]) == [4, 6]


def test_e_value_matches_the_published_formula():
    assert oxygen.e_value(2.0) == pytest.approx(2 + np.sqrt(2))
    assert oxygen.e_value(0.5) == pytest.approx(2 + np.sqrt(2))


def test_causal_frame_and_naive_gap(table):
    raw = data.load_raw()
    d = oxygen.causal_frame(table, raw["expeditions"])
    assert (d["height_metres"] >= config.EIGHT_THOUSANDER).all()
    assert d["year"].min() >= 1980
    r = oxygen.estimate_effect(d.sample(6000, random_state=0), "oxygen_used", "success")
    assert 0.3 < r["naive"] < 0.8
    assert 0.2 < r["aipw"] < 0.8


def test_permuted_treatment_has_no_effect(table):
    raw = data.load_raw()
    d = oxygen.causal_frame(table, raw["expeditions"]).sample(8000, random_state=2)
    d["oxygen_used"] = np.random.default_rng(0).permutation(d["oxygen_used"].to_numpy())
    r = oxygen.estimate_effect(d, "oxygen_used", "success")
    assert abs(r["aipw"]) < 0.03


def test_constant_treatment_is_rejected(table):
    raw = data.load_raw()
    d = oxygen.causal_frame(table, raw["expeditions"]).head(500).copy()
    d["oxygen_used"] = 1
    with pytest.raises(ValueError):
        oxygen.estimate_effect(d, "oxygen_used", "success")


def test_dowhy_cross_check_runs_on_a_small_sample(table):
    raw = data.load_raw()
    d = oxygen.causal_frame(table, raw["expeditions"])
    r = oxygen.dowhy_check(d, "oxygen_used", "success", sample=1500, sims=2)
    assert r["sample"] == 1500
    assert 0.3 < r["regression_ate"] < 0.9
    assert 0.3 < r["weighting_ate"] < 0.9
    assert abs(r["regression_placebo"]) < 0.1


def test_composition_splits_every_climber_once(table):
    raw = data.load_raw()
    d = oxygen.causal_frame(table, raw["expeditions"])
    arms, by_peak = oxygen.composition(d)
    assert sum(a["climbers"] for a in arms) == len(d)
    assert {"peak", "summit_rate_with", "summit_rate_without"} <= set(by_peak.columns)
    assert by_peak["climbers_with"].fillna(0).sum() + by_peak["climbers_without"].fillna(0).sum() == len(d)
