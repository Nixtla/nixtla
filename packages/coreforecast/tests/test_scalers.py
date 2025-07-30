import math

import numpy as np
import pytest
from coreforecast.differences import diff
from coreforecast.grouped_array import GroupedArray
from coreforecast.scalers import (
    AutoDifferences,
    AutoSeasonalDifferences,
    AutoSeasonalityAndDifferences,
    Difference,
    LocalBoxCoxScaler,
    boxcox,
    boxcox_lambda,
    inv_boxcox,
)
from coreforecast.seasonal import find_season_length

from . import (
    dtypes,
    scaler2core,
    scaler2fns,
    scaler_inverse_transform,
    scaler_transform,
    scalers,
)


@pytest.mark.parametrize("scaler_name", scalers)
@pytest.mark.parametrize("dtype", dtypes)
def test_correctness(data, indptr, scaler_name, dtype):
    # introduce some nans at the starts of groups
    data = data.astype(dtype, copy=True)
    sizes = np.diff(indptr)
    gt10 = np.where(sizes > 10)[0]
    assert gt10.size > 5
    for i in range(5):
        group = gt10[i]
        data[indptr[group] : indptr[group] + 10] = np.nan
    ga = GroupedArray(data, indptr)

    # setup scaler
    scaler = scaler2core[scaler_name]
    stats_fn = scaler2fns[scaler_name]
    scaler.fit(ga)

    # stats
    expected_stats = np.hstack([stats_fn(grp) for grp in ga]).reshape(-1, 2)
    np.testing.assert_allclose(scaler.stats_, expected_stats, atol=1e-6, rtol=1e-6)

    # transform
    transformed = scaler.transform(ga)
    expected_transformed = np.hstack(
        [scaler_transform(grp, scaler.stats_[i]) for i, grp in enumerate(ga)]
    )
    np.testing.assert_allclose(transformed, expected_transformed, atol=1e-6, rtol=1e-6)

    # inverse transform
    transformed_ga = GroupedArray(transformed, ga.indptr)
    restored = scaler.inverse_transform(transformed_ga)
    expected_restored = np.hstack(
        [
            scaler_inverse_transform(grp, scaler.stats_[i])
            for i, grp in enumerate(transformed_ga)
        ]
    )
    np.testing.assert_allclose(restored, expected_restored, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["guerrero", "loglik"])
def test_boxcox_correctness(data, indptr, dtype, method):
    data = data.astype(dtype, copy=True)
    # introduce some nans
    for i in [1, 90, 177]:
        data[indptr[i] : indptr[i] + 19] = np.nan
    if method == "loglik":
        data = np.abs(data)
    ga = GroupedArray(data, indptr)
    sc = LocalBoxCoxScaler(method=method, season_length=10)
    sc.fit(ga)
    transformed = sc.transform(ga)
    restored = sc.inverse_transform(GroupedArray(transformed, ga.indptr))
    atol = 5e-4 if dtype == np.float32 else 1e-8
    np.testing.assert_allclose(ga.data, restored, atol=atol)
    lmbda = boxcox_lambda(ga[0], method=method, season_length=10)
    np.testing.assert_allclose(lmbda, sc.stats_[0, 0])
    first_grp = slice(indptr[0], indptr[1])
    first_tfm = boxcox(ga[0], lmbda)
    first_restored = inv_boxcox(first_tfm, lmbda)
    np.testing.assert_allclose(first_tfm, transformed[first_grp])
    np.testing.assert_allclose(first_restored, restored[first_grp])

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_guerrero_correctness(dtype):
    data = [194, 229, 249, 203, 196, 238, 252, 210, 205, 236]
    expected_lambda = 1.99 # value from R's BoxCox.lambda function (forecast package)
    calculated_lambda = boxcox_lambda(data, method="guerrero", season_length=4)
    np.testing.assert_allclose(calculated_lambda, expected_lambda, atol=0.1)

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_difference_correctness(data, indptr, dtype):
    data = data.astype(dtype)
    ga = GroupedArray(data, indptr)
    d = 2
    sc = Difference(d=d)
    transformed = sc.fit_transform(ga)
    transformed_ga = ga._with_data(transformed)
    for i in range(len(ga)):
        np.testing.assert_allclose(transformed_ga[i], diff(ga[i], d))
    horizon = 4
    preds_data = np.zeros_like(data, shape=horizon * len(ga))
    preds_indptr = np.arange(0, (len(ga) + 1) * horizon, horizon)
    preds_ga = GroupedArray(preds_data, preds_indptr)
    restored = sc.inverse_transform(preds_ga)
    restored_ga = preds_ga._with_data(restored)
    for i in range(len(ga)):
        np.testing.assert_allclose(
            restored_ga[i], np.tile(sc.tails_[i * d : (i + 1) * d], horizon // d)
        )

    # take
    idxs = np.array([1, 4])
    subs = sc.take(idxs)
    np.testing.assert_equal(
        subs.tails_, np.hstack([sc.tails_[d * i : d * (i + 1)] for i in idxs])
    )

    # stack
    combined = sc.stack([sc, sc])
    np.testing.assert_equal(combined.tails_, np.tile(sc.tails_, 2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_differences_correctness(data, indptr, dtype):
    data = data.astype(dtype)
    with_trend = np.empty_like(data)
    expected = data.copy()
    for start, end in zip(indptr[:-1], indptr[1:]):
        with_trend[start:end] = data[start:end].cumsum()
        expected[start] = np.nan
    ga = GroupedArray(with_trend, indptr)
    sc = AutoDifferences(max_diffs=1)
    transformed = sc.fit_transform(ga)
    # we should've got that almost all series require a difference
    assert sc.diffs_.mean() > 0.9
    # the transformation should revert the cumsum, except for those where
    # the test said it was not necessary
    mask = np.repeat(sc.diffs_ == 1, np.diff(indptr))
    expected = np.where(mask, expected, with_trend)
    np.testing.assert_allclose(transformed, expected, atol=1e-5)
    # inverting on zeros should restore the last values
    tails = ga._tail(1)
    horizon = 5
    preds_data = np.zeros_like(data, shape=horizon * len(ga))
    preds_indptr = np.arange(0, (len(ga) + 1) * horizon, horizon)
    preds_ga = GroupedArray(preds_data, preds_indptr)
    np.testing.assert_allclose(
        sc.inverse_transform(preds_ga),
        np.repeat(tails * (sc.diffs_ == 1), horizon),
    )

    # take
    no_diffs_mask = sc.diffs_ == 0
    no_diffs_idxs = np.where(no_diffs_mask)[0]
    assert sc.take(no_diffs_idxs).tails_[0].size == 0
    idxs = np.where(~no_diffs_mask)[0]
    subs = sc.take(idxs)
    np.testing.assert_equal(subs.diffs_, sc.diffs_[idxs])
    np.testing.assert_equal(subs.tails_[0], sc.tails_[0])

    # update method
    ga = GroupedArray(np.arange(10, dtype=dtype), np.array([0, 5, 10]))
    sc = AutoDifferences(1)
    _ = sc.fit_transform(ga)
    # should've applied a diff for both series
    np.testing.assert_equal(sc.diffs_, [1, 1])
    # check the tails
    np.testing.assert_equal(sc.tails_[0], np.array([4, 9], dtype=dtype))
    # update should update the tails and take the difference
    new_ga = GroupedArray(
        np.array([6, 7, 11, 12, 13, 14], dtype=dtype), np.array([0, 2, 6])
    )
    updates = sc.update(new_ga)
    np.testing.assert_equal(updates, np.array([2, 1, 2, 1, 1, 1], dtype=dtype))
    np.testing.assert_equal(sc.tails_[0], np.array([7, 14], dtype=dtype))

    # stack
    combined = sc.stack([sc, sc])
    np.testing.assert_equal(combined.diffs_, np.tile(sc.diffs_, 2))
    np.testing.assert_equal(combined.tails_[0], np.tile(sc.tails_[0], 2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_seasonal_differences_correctness(data, indptr, dtype):
    season_length = 10
    seasonality = np.arange(season_length)

    # this can take a long time, so we'll use fewer series
    indptr = indptr[:100]
    data = data[: indptr[-1]].astype(dtype)

    with_season = np.empty_like(data)
    expected = data.copy()
    keep_as_is = np.random.choice(len(indptr) - 1, size=10, replace=False)
    for i, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        if i in keep_as_is:
            with_season[start:end] = data[start:end]
            continue
        size = end - start
        repeats = math.ceil(size / season_length)
        seasonality = np.tile(np.arange(season_length), repeats)
        with_season[start:end] = seasonality[:size] + data[start:end]
        expected[start : start + season_length] = np.nan
        expected[start + season_length : end] -= data[start : end - season_length]
    ga = GroupedArray(with_season, indptr, num_threads=2)
    sc = AutoSeasonalDifferences(season_length, max_diffs=1)
    transformed = sc.fit_transform(ga)
    # we should've got that almost all series require a seasonal difference
    assert 0.8 < sc.diffs_.mean() < 1.0
    # the transformation should revert the seasonality, except for those where
    # the test said it was not necessary
    mask = np.repeat(sc.diffs_ == 1, np.diff(indptr))
    expected = np.where(mask, expected, with_season)
    np.testing.assert_allclose(transformed, expected, atol=1e-5)
    # inverting on zeros should restore the last season values
    tails = ga._tail(season_length)
    horizon = 20
    preds_data = np.zeros_like(data, shape=horizon * len(ga))
    preds_indptr = np.arange(0, (len(ga) + 1) * horizon, horizon)
    preds_ga = GroupedArray(preds_data, preds_indptr)
    expected2 = preds_data.copy()
    repeats = horizon // season_length
    for i in range(len(ga)):
        if sc.diffs_[i] == 0:
            continue
        grp_tails = tails[i * season_length : (i + 1) * season_length]
        expected2[horizon * i : horizon * (i + 1)] = np.tile(grp_tails, repeats)
    restored = sc.inverse_transform(preds_ga)
    np.testing.assert_allclose(restored, expected2)

    # take
    no_diffs_mask = sc.diffs_ == 0
    no_diffs_idxs = np.where(no_diffs_mask)[0]
    assert sc.take(no_diffs_idxs).tails_[0].size == 0
    idxs = np.where(~no_diffs_mask)[0]
    subs = sc.take(idxs)
    np.testing.assert_equal(subs.diffs_, sc.diffs_[idxs])
    np.testing.assert_equal(subs.tails_[0], sc.tails_[0])

    # update method
    seasonal_x = np.arange(7, dtype=dtype)
    offset = 10
    ga = GroupedArray(seasonal_x[np.arange(28) % 7], np.array([0, 28]))
    sc = AutoSeasonalDifferences(7, 1)
    _ = sc.fit_transform(ga)
    np.testing.assert_equal(sc.diffs_, np.array([1.0], dtype=dtype))
    np.testing.assert_equal(sc.tails_[0], seasonal_x)
    new_ga = GroupedArray(seasonal_x + offset, np.array([0, 7]))
    updates = sc.update(new_ga)
    np.testing.assert_equal(updates, np.repeat(offset, 7))
    np.testing.assert_equal(sc.tails_[0], seasonal_x + offset)

    # stack
    combined = sc.stack([sc, sc])
    np.testing.assert_equal(combined.diffs_, np.tile(sc.diffs_, 2))
    np.testing.assert_equal(combined.tails_[0], np.tile(sc.tails_[0], 2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_seasonality_and_differences_correctness(dtype):
    amplitudes = [3, 5]
    seasonal_periods = [5, 24]
    max_season_length = 24
    max_diffs = 2
    t = 1 + np.arange(500)
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.1, size=t.size).astype(dtype)
    for amplitude, period in zip(amplitudes, seasonal_periods):
        x += amplitude * np.cos(2 * np.pi * t / period)

    period1 = find_season_length(x, max_season_length)
    y = diff(x, period1)
    period2 = find_season_length(y, max_season_length)
    z = diff(y, period2)
    period3 = find_season_length(z, max_season_length)
    assert period3 == 0
    assert sorted([period1, period2]) == seasonal_periods

    sc = AutoSeasonalityAndDifferences(
        max_season_length=max_season_length, max_diffs=max_diffs, n_seasons=None
    )
    ga = GroupedArray(np.hstack([x, x]), np.array([0, x.size, 2 * x.size]))
    diffed = sc.fit_transform(ga)
    np.testing.assert_allclose(diffed, np.hstack([z, z]))
    min_period = min(seasonal_periods)
    zeros_ga = GroupedArray(
        np.zeros(2 * min_period, dtype=dtype), np.array([0, min_period, 2 * min_period])
    )
    inv_transformed = sc.inverse_transform(zeros_ga)
    inv_ga = zeros_ga._with_data(inv_transformed)
    actual = np.hstack([inv_ga[0][:min_period], inv_ga[1][:min_period]])
    expected = x[-period1:][:min_period] + y[-period2:][:min_period]
    np.testing.assert_allclose(actual, np.hstack([expected, expected]))

    # take
    idxs = np.array([1])
    subs = sc.take(idxs)
    for i in range(max_diffs):
        np.testing.assert_equal(subs.diffs_[i], sc.diffs_[i][idxs])
        np.testing.assert_equal(subs.tails_[i], sc.tails_[i][subs.diffs_[i][0] :])

    # update
    new_n = 50
    new_data = np.arange(new_n, dtype=dtype)
    first_diff = sc.diffs_[0][0]
    second_diff = sc.diffs_[1][0]
    t1 = diff(np.append(sc.tails_[0][:first_diff], new_data), first_diff)[-new_n:]
    t2 = diff(np.append(sc.tails_[1][:second_diff], t1), second_diff)[-new_n:]
    expected_updates = np.hstack([t2, t2])
    new_ga = GroupedArray(np.tile(new_data, 2), np.array([0, new_n, 2 * new_n]))
    updates = sc.update(new_ga)
    np.testing.assert_equal(sc.tails_[0], np.tile(new_data[-min_period:], 2))
    np.testing.assert_allclose(updates, expected_updates)

    # stack
    combined = sc.stack([sc, sc])
    np.testing.assert_equal(combined.diffs_[0], np.tile(sc.diffs_[0], 2))
    np.testing.assert_equal(combined.tails_[0], np.tile(sc.tails_[0], 2))
