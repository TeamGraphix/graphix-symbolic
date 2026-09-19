from __future__ import annotations

import pytest
from numpy.random import PCG64, Generator

SEED = 25
DEPTH = 1


@pytest.fixture
def fx_rng() -> Generator:
    return Generator(PCG64(SEED))


@pytest.fixture
def fx_bg() -> PCG64:
    return PCG64(SEED)
