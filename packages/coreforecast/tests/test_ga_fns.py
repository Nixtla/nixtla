import numpy as np
import pytest
from coreforecast.grouped_array import GroupedArray


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("season_length", [7, 12, 24])
def test_periods(dtype, season_length):
    sizes = np.random.randint(2 * season_length, 100, 500)
    data = np.hstack([np.arange(size, dtype=dtype) % season_length for size in sizes])
    ga = GroupedArray(data, np.append(0, sizes.cumsum()))
    lengths = ga._periods(50)
    unique_lengths = np.unique(lengths)
    assert unique_lengths.size == 1
    assert unique_lengths.item() == season_length
