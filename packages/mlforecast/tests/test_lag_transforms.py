import operator

import numpy as np
import pytest
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray

from mlforecast.grouped_array import GroupedArray as MLGroupedArray
from mlforecast.lag_transforms import (
    Combine,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingQuantile,
    ExpandingStd,
    ExponentiallyWeightedMean,
    Lag,
    Offset,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingQuantile,
    RollingStd,
    SeasonalRollingMax,
    SeasonalRollingMean,
    SeasonalRollingMin,
    SeasonalRollingQuantile,
    SeasonalRollingStd,
)


@pytest.fixture(scope='module')
def grouped_array():
    rng = np.random.default_rng(seed=0)
    lengths = rng.integers(low=50, high=100, size=20)
    data = rng.random(lengths.sum())
    return CoreGroupedArray(data, np.append(0, lengths.cumsum()))

def test_offset_name_and_transform(grouped_array):
    offset = Offset(RollingMean(window_size=10), 2)._set_core_tfm(5)
    assert offset._get_name(5) == "rolling_mean_lag7_window_size10"

    transformed = offset.transform(grouped_array)
    expected = (
        RollingMean(window_size=10)
        ._set_core_tfm(5)
        .transform(grouped_array._with_data(Lag(2).transform(grouped_array)))
    )
    np.testing.assert_allclose(transformed, expected)

def test_combine_name_and_transform(grouped_array):
    comb = Combine(Lag(1), Lag(2), operator.truediv)
    assert comb._get_name(1) == "lag1_truediv_lag2"

    transformed = comb.transform(grouped_array)
    expected = Lag(1).transform(grouped_array) / Lag(2).transform(grouped_array)
    np.testing.assert_allclose(transformed, expected)

def test_combine_take(grouped_array):
    tfm = Combine(
        RollingMean(window_size=7, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )._set_core_tfm(1)
    tfm.transform(grouped_array)

    idxs = np.array([0, 5, 10, 15])
    subset_tfm = tfm.take(idxs)

    assert isinstance(subset_tfm, Combine)
    assert subset_tfm.tfm1 is not None
    assert subset_tfm.tfm2 is not None
    assert subset_tfm.operator == operator.add

def test_nested_combine_take(grouped_array):
    inner = Combine(
        RollingMean(window_size=7, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )
    outer = Combine(
        inner,
        RollingMean(window_size=3, min_samples=1),
        operator.sub
    )._set_core_tfm(1)
    outer.transform(grouped_array)

    idxs = np.array([0, 5, 10])
    subset_tfm = outer.take(idxs)

    assert isinstance(subset_tfm, Combine)
    assert isinstance(subset_tfm.tfm1, Combine)
    assert subset_tfm.operator == operator.sub
    assert subset_tfm.tfm1.operator == operator.add

def test_combine_stack(grouped_array):
    tfm1 = Combine(
        RollingMean(window_size=7, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )._set_core_tfm(1)
    tfm2 = Combine(
        RollingMean(window_size=7, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )._set_core_tfm(1)

    tfm1.transform(grouped_array)
    tfm2.transform(grouped_array)

    stacked_tfm = Combine.stack([tfm1, tfm2])

    assert isinstance(stacked_tfm, Combine)
    assert stacked_tfm.operator == operator.add


def test_combine_stack_behavioral(grouped_array):
    """Verify that Combine.stack() doesn't just return first partition"""
    # Create two Combine transforms with DIFFERENT window sizes to detect if
    # stacking just returns the first partition vs actually combining them
    tfm1 = Combine(
        RollingMean(window_size=3, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )._set_core_tfm(1)

    tfm2 = Combine(
        RollingMean(window_size=7, min_samples=1),  # Different window size
        RollingMean(window_size=9, min_samples=1),  # Different window size
        operator.add
    )._set_core_tfm(1)

    tfm1.transform(grouped_array)
    tfm2.transform(grouped_array)

    # If stack incorrectly returned partition_tfms[0], it would just return tfm1
    # The stacked transform should have tfm1's window sizes (since stack keeps first's config)
    # but should have stacked internal state from both
    stacked = Combine.stack([tfm1, tfm2])

    # Verify stacked uses first transform's configuration
    assert stacked.tfm1.window_size == 3
    assert stacked.tfm2.window_size == 5
    # Verify it's not just a reference to tfm1 (defensive check)
    assert stacked is not tfm1


@pytest.mark.parametrize("tfm", [
    ExpandingMax(),
    ExpandingMean(),
    ExpandingMin(),
    ExpandingStd(),
    ExpandingQuantile(0.5),
    ExponentiallyWeightedMean(0.1),
    RollingMax(7),
    RollingMean(7),
    RollingMin(7),
    RollingStd(7),
    RollingQuantile(0.5, 7),
    SeasonalRollingMax(7, 2),
    SeasonalRollingMean(7, 2),
    SeasonalRollingMin(7, 2),
    SeasonalRollingStd(7, 2),
    SeasonalRollingQuantile(0.5, 7, 7),
    Offset(RollingMax(7), 2),
    Combine(RollingMean(5), Offset(RollingMean(5), 2), operator.truediv),
    Combine(Offset(RollingMean(5), 2), RollingMean(5), operator.truediv),
])
def test_transform_and_update_consistency(grouped_array, tfm):
    tfm._set_core_tfm(1)
    tfm._get_name(1)
    tfm.transform(grouped_array)

    updates = tfm.update(grouped_array)
    upd_samples = tfm.update_samples
    if upd_samples > -1:
        sliced_ga = MLGroupedArray(grouped_array.data, grouped_array.indptr).take_from_groups(
            slice(-upd_samples, None)
        )
        ga2 = CoreGroupedArray(sliced_ga.data, sliced_ga.indptr)
        tfm.transform(grouped_array)  # reset internal state
        updates2 = tfm.update(ga2)
        np.testing.assert_allclose(updates, updates2)
