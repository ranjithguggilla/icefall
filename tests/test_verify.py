import copy
import json
from pathlib import Path

from icefall import config, verify


def _metrics():
    return json.loads((Path(config.OUT) / "metrics.json").read_text())


def test_identical_metrics_agree():
    m = _metrics()
    assert verify.compare(m, copy.deepcopy(m)) == []


def test_a_moved_number_is_reported():
    m = _metrics()
    other = copy.deepcopy(m)
    other["test"]["auc"] = m["test"]["auc"] - 0.05
    msgs = verify.compare(m, other)
    assert any(x.startswith("test.auc") for x in msgs)


def test_a_changed_model_choice_is_reported():
    m = _metrics()
    other = copy.deepcopy(m)
    other["selection"]["chosen"] = "logistic"
    assert verify.compare(m, other)


def test_cli_exit_codes(tmp_path):
    m = _metrics()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(m))
    b.write_text(json.dumps(m))
    assert verify.main([str(a), str(b)]) == 0
    m["death"]["auc"] = 0.9
    b.write_text(json.dumps(m))
    assert verify.main([str(a), str(b)]) == 1
