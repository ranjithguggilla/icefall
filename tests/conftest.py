import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def table():
    from icefall import data

    return data.with_history(data.build_table())


@pytest.fixture(scope="session")
def splits(table):
    from icefall import data

    return data.split_by_year(table)
