"""Paths, split boundaries and the assumptions behind every dollar figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
ASSETS = ROOT / "assets"

SEED = 7

# Forward-in-time split on the expedition year.
TRAIN_END = 2008
VALID_END = 2013  # test is 2014 onwards (data stops in spring 2019)

EIGHT_THOUSANDER = 8000  # metres, used for the oxygen analyses
MIN_YEAR = 1950


@dataclass(frozen=True)
class Costs:
    """Oxygen logistics assumptions (all per climber set, in USD).

    These are my own planning numbers, not figures from the dataset. The
    report repeats them and sweeps the ratio so the conclusion does not hang
    on one pair of values.
    """

    unused_set: float = 800.0  # bottles hauled, stored or flown back unused
    missing_set: float = 4500.0  # emergency purchase or a climber who cannot go on


COSTS = Costs()

DATA_SOURCE = (
    "https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-09-22 "
    "(The Himalayan Database, Elizabeth Hawley archive, cleaned by Alex Cookson)"
)
