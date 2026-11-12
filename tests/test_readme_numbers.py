"""The README quotes numbers from outputs/metrics.json; this keeps the two from drifting apart."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "outputs" / "metrics.json").read_text())
README = (ROOT / "README.md").read_text()


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def usd(x):
    return f"${x:,.0f}"


CHECKS = {
    "climbers": f"{M['data']['climbers']:,}",
    "expeditions": f"{M['data']['expeditions']:,}",
    "test auc": f"{M['test']['auc']:.3f}",
    "test log loss": f"{M['test']['log_loss']:.3f}",
    "baseline log loss": f"{M['test']['baseline_log_loss']:.3f}",
    "log loss gain": f"{M['test']['log_loss_gain_vs_baseline']:.4f}",
    "train summit rate": pct(M["data"]["splits"]["train"]["summit_rate"]),
    "test summit rate": pct(M["data"]["splits"]["test"]["summit_rate"]),
    "icc": f"{M['dependence']['icc']:.3f}",
    "design effect": f"{M['dependence']['design_effect']:.2f}",
    "random rows auc": f"{M['split_study'][0]['auc']:.3f}",
    "forward auc": f"{M['split_study'][2]['auc']:.3f}",
    "highpoint leak": f"{M['leak_study'][2]['auc']:.3f}",
    "member aipw": f"{M['causal']['oxygen_used__success']['aipw'] * 100:+.1f}",
    "expedition aipw": f"{M['causal']['exp_oxygen__success']['aipw'] * 100:+.1f}",
    "e value": f"{M['causal']['oxygen_used__success']['e_value_naive_rr']:.2f}",
    "member oxygen auc": f"{M['oxygen_planning']['member_auc']:.3f}",
    "saving vs full": usd(M["oxygen_planning"]["saving_vs_full"]),
    "saving vs rate": usd(M["oxygen_planning"]["saving_vs_rate"]),
    "death auc": f"{M['death']['auc']:.3f}",
    "model order cost": usd(M["oxygen_planning"]["policies"][3]["total_cost"]),
}


@pytest.mark.parametrize("label", sorted(CHECKS))
def test_readme_quotes_current_metrics(label):
    assert CHECKS[label] in README, f"{label}: {CHECKS[label]} not found in README"


def test_every_embedded_chart_exists():
    import re

    for rel in re.findall(r"\]\((assets/[^)]+\.png)\)", README):
        assert (ROOT / rel).stat().st_size > 5000, rel
