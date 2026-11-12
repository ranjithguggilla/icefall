import numpy as np
import pandas as pd

from icefall import data


def test_table_has_only_paying_members(table):
    assert table["hired"].sum() == 0
    assert table["member_id"].is_unique
    assert table["year"].min() >= 1950
    assert set(table["success"].unique()) == {0, 1}


def test_splits_are_forward_in_time(splits):
    assert splits["train"]["year"].max() < splits["valid"]["year"].min()
    assert splits["valid"]["year"].max() < splits["test"]["year"].min()
    total = sum(len(v) for v in splits.values())
    assert total > 55000


def _toy():
    rows = []
    for year, wins in [(2000, 2), (2001, 0), (2002, 2)]:
        for i in range(4):
            rows.append({"peak_id": "AAAA", "year": year, "success": int(i < wins), "died": 0, "oxygen_used": 0})
    for year in (2000, 2001, 2002):
        rows.append({"peak_id": "BBBB", "year": year, "success": 0, "died": 0, "oxygen_used": 0})
    return pd.DataFrame(rows)


def test_history_never_sees_its_own_year():
    df = _toy()
    base = data.prior_rate_features(df).set_index(["peak_id", "year"])
    flipped = df.copy()
    flipped.loc[(flipped["peak_id"] == "AAAA") & (flipped["year"] == 2001), "success"] = 1
    new = data.prior_rate_features(flipped).set_index(["peak_id", "year"])
    # changing a 2001 outcome must not touch 2000 or 2001 features
    for y in (2000, 2001):
        assert base.loc[("AAAA", y), "prior_success_rate"] == new.loc[("AAAA", y), "prior_success_rate"]
    # but it must show up from 2002 on
    assert new.loc[("AAAA", 2002), "prior_success_rate"] > base.loc[("AAAA", 2002), "prior_success_rate"]


def test_history_shrinks_towards_the_global_rate():
    df = _toy()
    h = data.prior_rate_features(df, shrink=20).set_index(["peak_id", "year"])
    first = h.loc[("AAAA", 2000)]
    assert first["prior_climbers"] == 0
    assert 0 <= first["prior_success_rate"] <= 1
    later = h.loc[("BBBB", 2002), "prior_success_rate"]
    assert later > 0  # two failures do not collapse the rate to exactly zero


def test_horizon_adds_a_scoring_year():
    df = _toy()
    h = data.prior_rate_features(df, horizon=1)
    assert h["year"].max() == df["year"].max() + 1
    assert np.isfinite(h["prior_success_rate"]).all()
