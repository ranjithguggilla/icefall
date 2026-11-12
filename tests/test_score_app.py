import pytest

from icefall import app, data, models, score


@pytest.fixture(scope="module")
def bundle(table):
    sub = table.sample(9000, random_state=3)
    raw = data.load_raw()
    ok = models.make_model("logistic").fit(sub[models.FEATURES], sub["success"])
    d8 = sub[sub["is_8000"] == 1]
    ox = models.make_model("logistic").fit(d8[models.FEATURES], d8["oxygen_used"])
    de = models.make_model("logistic").fit(sub[models.FEATURES], sub["died"])
    return score.make_bundle(table, raw["peaks"], ok, ox, de)


def test_peak_table_has_everest(bundle):
    assert "Everest" in set(bundle.peaks["peak_name"])
    assert bundle.peaks["recent_success_rate"].between(0, 1).all()


def test_score_returns_sane_numbers(bundle):
    pid = bundle.peaks.loc[bundle.peaks["peak_name"] == "Everest", "peak_id"].iloc[0]
    r = score.score_climber(bundle, pid, season="Spring", age=35, sex="M", citizenship="USA")
    assert 0 < r["p_summit"] < 1
    assert 0 < r["p_death"] < 1
    assert r["band"] in {"long shot", "below even", "roughly a coin flip", "better than even", "favourable"}
    assert r["reasons"]


def test_unknown_peak_is_an_error(bundle):
    with pytest.raises(KeyError):
        score.score_climber(bundle, "NOPE")


def test_missing_age_is_handled(bundle):
    pid = bundle.peaks["peak_id"].iloc[0]
    r = score.score_climber(bundle, pid, age=None)
    assert 0 < r["p_summit"] < 1


def test_band_edges():
    assert score.band(0.05) == "long shot"
    assert score.band(0.9) == "favourable"


def test_ui_text_and_app(bundle):
    text = app.predict(bundle, "Everest", "Spring", 40, "M", bundle.nations[0], "climber", 10, 8, False, True)
    assert "Everest" in text and "%" in text
    assert app.predict(bundle, "Not a peak", "Spring", 40, "M", "USA", "climber", 5, 2, False, True).startswith("Pick")
    assert app.build_app(bundle) is not None
