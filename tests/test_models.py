import numpy as np

from icefall import evaluate, models


def test_rare_levels_learn_from_training_only(splits):
    tr = splits["train"]
    r = models.RareLevels().fit(tr)
    out = r.transform(splits["test"])
    assert out["citizenship"].nunique() <= 15
    assert out["peak_id"].nunique() <= 121


def test_ece_is_zero_for_perfect_bins():
    y = np.array([0, 0, 1, 1])
    assert models.ece(y, y.astype(float)) == 0.0
    assert models.ece(y, np.full(4, 0.5)) == 0.0


def test_score_all_keys():
    s = models.score_all(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.4, 0.6]))
    assert set(s) == {"auc", "log_loss", "brier", "ece"}
    assert s["auc"] == 1.0


def test_each_model_fits_and_gives_probabilities(splits):
    tr = splits["train"].sample(4000, random_state=0)
    te = splits["valid"].head(300)
    for kind in ["logistic", "boosting"]:
        p = evaluate.fit_predict(kind, tr, te)
        assert p.shape == (300,)
        assert ((p > 0) & (p < 1)).all()


def test_unknown_model_kind_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        models.make_model("forest")


def test_lookup_baseline_uses_history_column(splits):
    te = splits["test"].head(50)
    assert np.allclose(models.peak_lookup_scores(te), te["recent_success_rate"])


def test_one_se_rule_prefers_the_simpler_model(splits, monkeypatch):
    tr = splits["train"].sample(3000, random_state=1)
    va = splits["valid"].sample(1500, random_state=1)
    monkeypatch.setattr(evaluate, "SIMPLICITY", ["lookup", "logistic", "boosting"])
    chosen, tab, _ = evaluate.select_model(tr, va, n_boot=30)
    best = tab.set_index("model")["valid_log_loss"].idxmin()
    order = ["lookup", "logistic", "boosting"]
    assert order.index(chosen) <= order.index(best)


def test_holdout_report_has_intervals(splits):
    te = splits["test"].head(600)
    p = te["recent_success_rate"].to_numpy()
    rep = evaluate.holdout_report(te["success"], p, te["expedition_id"], np.full(len(te), 0.4), n_boot=30)
    assert rep["log_loss_ci"][0] < rep["log_loss_ci"][1]
    assert "log_loss_gain_vs_baseline" in rep
