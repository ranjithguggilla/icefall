import json

from icefall import config, pipeline


def test_fast_pipeline_writes_every_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUT", tmp_path / "outputs")
    monkeypatch.setattr(config, "TABLES", tmp_path / "outputs" / "tables")
    monkeypatch.setattr(config, "ASSETS", tmp_path / "assets")
    config.OUT.mkdir(parents=True)
    res = pipeline.run(fast=True)
    m = json.loads((config.OUT / "metrics.json").read_text())
    assert m["test"]["auc"] > 0.6
    assert m["test"]["log_loss_gain_vs_baseline"] > 0
    assert (config.OUT / "bundle.joblib").exists()
    for chart in ["era_drift", "leakage_audit", "oxygen_cost", "oxygen_effect", "reliability"]:
        assert (config.ASSETS / f"{chart}.png").stat().st_size > 5000
    assert res["leak_study"][2]["auc"] > 0.95
