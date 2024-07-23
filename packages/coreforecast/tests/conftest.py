import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)

@pytest.fixture
def indptr(rng):
    lengths = rng.integers(low=200, high=300, size=200)
    return np.append(0, lengths.cumsum()).astype(np.int32)


@pytest.fixture
def data(rng, indptr):
    return rng.normal(size=indptr[-1])
