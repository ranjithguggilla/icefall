"""Compare a fresh metrics file with the committed one, within a tolerance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Numbers the README quotes. Dotted paths into metrics.json.
KEYS = [
    "test.auc",
    "test.log_loss",
    "test.log_loss_gain_vs_baseline",
    "dependence.icc",
    "death.auc",
    "causal.oxygen_used__success.aipw",
    "causal.exp_oxygen__success.aipw",
    "oxygen_planning.saving_vs_full",
    "oxygen_planning.saving_vs_rate",
    "oxygen_planning.member_auc",
]


def dig(d: dict, path: str):
    cur = d
    for part in path.split("."):
        cur = cur[part]
    return cur


def compare(committed: dict, fresh: dict, keys=KEYS, rel_tol: float = 0.02, abs_tol: float = 0.005) -> list[str]:
    """Return one message per key that moved more than the tolerance."""
    problems = []
    if committed["selection"]["chosen"] != fresh["selection"]["chosen"]:
        problems.append(f"selected model changed: {committed['selection']['chosen']} -> {fresh['selection']['chosen']}")
    for k in keys:
        a, b = dig(committed, k), dig(fresh, k)
        if abs(a - b) > max(abs_tol, rel_tol * abs(a)):
            problems.append(f"{k}: committed {a} vs fresh {b}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a fresh run against committed metrics")
    ap.add_argument("committed", type=Path)
    ap.add_argument("fresh", type=Path)
    args = ap.parse_args(argv)
    problems = compare(json.loads(args.committed.read_text()), json.loads(args.fresh.read_text()))
    for p in problems:
        print("MISMATCH", p)
    if not problems:
        print(f"all {len(KEYS) + 1} checked values agree")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
