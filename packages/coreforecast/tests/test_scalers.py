import numpy as np
import pytest

from coreforecast.grouped_array import GroupedArray
from coreforecast.scalers import (
    LocalMinMaxScaler,
    LocalRobustScalerIqr,
    LocalRobustScalerMad,
    LocalStandardScaler,
)


@pytest.fixture
def ga():
    lengths = np.random.randint(low=100, high=200, size=50_000)
    indptr = np.append(0, lengths.cumsum()).astype(np.int32)
    data = np.random.rand(indptr[-1]).astype(np.float32)
    return GroupedArray(data, indptr)


@pytest.mark.parametrize(
    "scaler",
    [
        LocalMinMaxScaler(),
        LocalRobustScalerIqr(),
        LocalRobustScalerMad(),
        LocalStandardScaler(),
    ],
)
def test_scaler(ga, scaler):
    stats = scaler.fit(ga)
    transformed = scaler.transform(ga)
    transformed_ga = GroupedArray(transformed, ga.indptr)
    restored = scaler.inverse_transform(transformed_ga)
    np.testing.assert_allclose(ga.data, restored, rtol=1e-5, atol=1e-5)
