import numpy as np
import pytest
from coreforecast._lib import rolling as _rolling
from coreforecast._lib.grouped_array import _GroupedArrayFloat32, _GroupedArrayFloat64
from coreforecast.grouped_array import GroupedArray


@pytest.mark.parametrize("stat", ["mean", "std", "min", "max", "quantile"])
def test_every_rolling_statistic_is_fully_registered(stat):
    # the rolling entry points are generated per statistic, so a statistic that
    # is left out of the binder list would only surface as an AttributeError in
    # whoever calls it
    for cls in (_GroupedArrayFloat32, _GroupedArrayFloat64):
        for name in (
            f"_rolling_{stat}",
            f"_rolling_{stat}_update",
            f"_seasonal_rolling_{stat}",
            f"_seasonal_rolling_{stat}_update",
        ):
            assert callable(getattr(cls, name)), name
    assert callable(getattr(_rolling, f"rolling_{stat}"))
    assert callable(getattr(_rolling, f"seasonal_rolling_{stat}"))


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
