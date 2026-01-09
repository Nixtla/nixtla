import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from utilsforecast.processing import counts_by_id

from mlforecast import MLForecast
from mlforecast.grouped_array import GroupedArray
from mlforecast.target_transforms import (
    AutoDifferences,
    AutoSeasonalDifferences,
    AutoSeasonalityAndDifferences,
    Differences,
    GlobalSklearnTransformer,
    LocalBoxCox,
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
)
from mlforecast.target_transforms import Differences as ExportedDifferences
from mlforecast.utils import generate_daily_series


@pytest.fixture
def setup_series():
    series = generate_daily_series(10, min_length=50, max_length=100)
    diffs = Differences([1, 2, 5])
    id_counts = counts_by_id(series, 'unique_id')
    indptr = np.append(0, id_counts['counts'].cumsum())
    ga = GroupedArray(series['y'].values, indptr)
    return series, diffs, id_counts, indptr, ga


# differences are applied correctly
def test_differences_applied(setup_series):
    series, diffs, _, _, ga = setup_series
    transformed = diffs.fit_transform(ga)
    assert diffs.fitted_ == []
    expected = series.copy()
    for d in diffs.differences:
        expected['y'] -= expected.groupby('unique_id', observed=True)['y'].shift(d)
    np.testing.assert_allclose(transformed.data, expected['y'].values)

# fitted differences are restored correctly
def test_differences_restored(setup_series):
    _, diffs, _, _, ga = setup_series
    diffs.store_fitted = True
    transformed = diffs.fit_transform(ga)
    keep_mask = ~np.isnan(transformed.data)
    restored = diffs.inverse_transform_fitted(transformed)
    np.testing.assert_allclose(ga.data[keep_mask], restored.data[keep_mask])

# test transform
def test_transforms(setup_series):
    _, diffs, _, _, ga = setup_series

    diffs.store_fitted = True
    transformed = diffs.fit_transform(ga)

    new_ga = GroupedArray(np.random.rand(10), np.arange(11))
    prev_orig = [diffs.scalers_[i].tails_[::d].copy() for i, d in enumerate(diffs.differences)]
    expected = new_ga.data - np.add.reduce(prev_orig)
    updates = diffs.update(new_ga)
    np.testing.assert_allclose(expected, updates.data)
    np.testing.assert_allclose(diffs.scalers_[0].tails_, new_ga.data)
    np.testing.assert_allclose(diffs.scalers_[1].tails_[1::2], new_ga.data - prev_orig[0])
    np.testing.assert_allclose(diffs.scalers_[2].tails_[4::5], new_ga.data - np.add.reduce(prev_orig[:2]))


# variable sizes
def test_variable_sizes(setup_series):
    _, _, _, _, ga = setup_series
    diff1 = Differences([1])
    ga = GroupedArray(np.arange(10), np.array([0, 3, 10]))
    diff1.fit_transform(ga)
    new_ga = GroupedArray(np.arange(4), np.array([0, 1, 4]))
    updates = diff1.update(new_ga)
    np.testing.assert_allclose(updates.data, np.array([0 - 2, 1 - 9, 2 - 1, 3 - 2]))
    np.testing.assert_allclose(diff1.scalers_[0].tails_, np.array([0, 3]))

# short series
def test_short_series(setup_series):
    series, diffs, _, _, ga = setup_series
    ga = GroupedArray(np.arange(20), np.array([0, 2, 20]))
    with pytest.raises(Exception) as exec_info:
        diffs.fit_transform(ga)
    assert "[0]" in str(exec_info.value)

# stack
def test_stack(setup_series):
    series, _, _, indptr, ga = setup_series
    diffs = Differences([1, 2, 5])
    ga = GroupedArray(series['y'].values, indptr)
    diffs.fit_transform(ga)
    stacked = Differences.stack([diffs, diffs])
    for i in range(len(diffs.differences)):
        np.testing.assert_allclose(
            stacked.scalers_[i].tails_,
            np.tile(diffs.scalers_[i].tails_, 2)
        )

def test_autodifferences():
    sc = AutoDifferences(1)
    ga = GroupedArray(np.arange(10), np.array([0, 10]))
    transformed = sc.fit_transform(ga)
    np.testing.assert_allclose(transformed.data, np.append(np.nan, np.ones(9)))
    np.testing.assert_equal(sc.scaler_.diffs_, np.array([1.0], dtype=np.float32))
    inv = sc.inverse_transform(ga).data
    np.testing.assert_allclose(9 + np.arange(10).cumsum(), inv)
    upd = sc.update(GroupedArray(np.array([10]), np.array([0, 1])))
    np.testing.assert_equal(np.array([1.0]), upd.data)
    np.testing.assert_equal(sc.scaler_.tails_[0], np.array([10]))

    # stack
    stacked = AutoDifferences.stack([sc, sc])
    np.testing.assert_allclose(
        stacked.scaler_.diffs_,
        np.tile(sc.scaler_.diffs_, 2),
    )
    np.testing.assert_allclose(
        stacked.scaler_.tails_,
        np.tile(sc.scaler_.tails_, 2),
    )

def test_autoseasonaldifferences():
    sc = AutoSeasonalDifferences(season_length=5, max_diffs=1)
    ga = GroupedArray(np.arange(5)[np.arange(10) % 5], np.array([0, 10]))
    transformed = sc.fit_transform(ga)
    sc.inverse_transform(ga)
    sc.update(ga)

    # stack
    stacked = AutoDifferences.stack([sc, sc])
    np.testing.assert_allclose(
        stacked.scaler_.diffs_,
        np.tile(sc.scaler_.diffs_, 2),
    )
    np.testing.assert_allclose(
        stacked.scaler_.tails_,
        np.tile(sc.scaler_.tails_, 2),
    )

def test_autoseasonality_and_differences():
    sc = AutoSeasonalityAndDifferences(max_season_length=5, max_diffs=1)
    ga = GroupedArray(np.arange(5)[np.arange(10) % 5], np.array([0, 10]))
    transformed = sc.fit_transform(ga)
    sc.inverse_transform(ga)
    sc.update(ga)

    # stack
    stacked = AutoDifferences.stack([sc, sc])
    np.testing.assert_allclose(
        stacked.scaler_.diffs_,
        np.tile(sc.scaler_.diffs_, 2),
    )
    np.testing.assert_allclose(
        stacked.scaler_.tails_,
        np.tile(sc.scaler_.tails_, 2),
    )


@pytest.mark.parametrize("scaler", [
    LocalStandardScaler(),
    LocalMinMaxScaler(),
    LocalRobustScaler(scale='iqr'),
    LocalRobustScaler(scale='mad'),
    LocalBoxCox(),
])
def test_scaler(scaler, setup_series):
    series, diffs, id_counts, indptr, ga = setup_series
    id_counts = counts_by_id(series, 'unique_id')
    indptr = np.append(0, id_counts['counts'].cumsum())
    ga = GroupedArray(series['y'].values, indptr)
    transformed = scaler.fit_transform(ga)
    np.testing.assert_allclose(
        scaler.inverse_transform(transformed).data,
        ga.data,
    )
    transformed2 = scaler.update(ga)
    np.testing.assert_allclose(transformed.data, transformed2.data)

    idxs = [0, 7]
    subset = ga.take(idxs)
    transformed_subset = transformed.take(idxs)
    subsc = scaler.take(idxs)
    np.testing.assert_allclose(
        subsc.inverse_transform(transformed_subset).data,
        subset.data,
    )

    stacked = scaler.stack([scaler, scaler])
    stacked_stats = stacked.scaler_.stats_
    np.testing.assert_allclose(
        stacked_stats,
        np.tile(scaler.scaler_.stats_, (2, 1)),
    )


@pytest.fixture
def mlforecast_setup():
    sk_boxcox = PowerTransformer(method='box-cox', standardize=False)
    boxcox_global = GlobalSklearnTransformer(sk_boxcox)
    single_difference = ExportedDifferences([1])
    models = [LinearRegression(), HistGradientBoostingRegressor()]
    lags = [1, 2]
    target_transforms = [boxcox_global, single_difference]

    return {
        'sk_boxcox': sk_boxcox,
        'boxcox_global': boxcox_global,
        'single_difference': single_difference,
        'models': models,
        'lags': lags,
        'target_transforms': target_transforms
    }


def test_mlforecast_differences(mlforecast_setup):
    setup = mlforecast_setup
    series = generate_daily_series(10)
    fcst = MLForecast(
        models=setup['models'],
        freq='D',
        lags=setup['lags'],
        target_transforms=setup['target_transforms']
    )
    prep = fcst.preprocess(series, dropna=False)
    expected = (
        pd.Series(
            setup['sk_boxcox'].fit_transform(series[['y']])[:, 0],
            index=series['unique_id']
        ).groupby('unique_id', observed=True)
        .diff()
        .dropna()
        .values
    )
    np.testing.assert_allclose(prep['y'].values, expected)
    preds = fcst.fit(series).predict(5)
    # Ensure predictions are generated successfully
    assert preds is not None
    assert len(preds) > 0


def test_mlforecast_polars(mlforecast_setup):
    setup = mlforecast_setup
    series_pl = generate_daily_series(10, engine='polars')
    fcst_pl = MLForecast(
        models=setup['models'],
        freq='1d',
        lags=setup['lags'],
        target_transforms=setup['target_transforms']
    )

    # Also test pandas version for comparison
    series_pd = generate_daily_series(10)
    fcst_pd = MLForecast(
        models=setup['models'],
        freq='D',
        lags=setup['lags'],
        target_transforms=setup['target_transforms']
    )

    prep_pl = fcst_pl.preprocess(series_pl, dropna=False)
    prep_pd = fcst_pd.preprocess(series_pd, dropna=False)

    pd.testing.assert_frame_equal(prep_pd.reset_index(drop=True), prep_pl.to_pandas())

    pl_preds = fcst_pl.fit(series_pl).predict(5)
    pd_preds = fcst_pd.fit(series_pd).predict(5)

    pd.testing.assert_frame_equal(pd_preds, pl_preds.to_pandas())


def test_autoseasonality_and_differences_validation():
    """Test validation logic for AutoSeasonalityAndDifferences safety guard."""

    # Test 1: ValueError when series has insufficient data for STL after differencing
    # With max_diffs=12, we need at least 12 + 4 = 16 observations per series
    # Create a series with only 15 observations (too short)
    sc = AutoSeasonalityAndDifferences(max_season_length=12, max_diffs=12, n_seasons=3)
    ga_short = GroupedArray(np.arange(15, dtype=float), np.array([0, 15]))

    with pytest.raises(ValueError) as exc_info:
        sc.fit_transform(ga_short)
    error_msg = str(exc_info.value)
    assert "Insufficient data" in error_msg
    assert "requires at least 16 observations" in error_msg
    assert "[15]" in error_msg  # Shows the actual length

    # Test 2: Success when series has sufficient data
    # Create a series with 20 observations (sufficient for max_diffs=12)
    ga_long = GroupedArray(np.arange(20, dtype=float), np.array([0, 20]))
    transformed = sc.fit_transform(ga_long)
    assert transformed.data is not None

    # Test 3: Multiple series - some too short
    # Series 1: 20 obs (ok), Series 2: 15 obs (too short), Series 3: 25 obs (ok)
    data = np.concatenate([np.arange(20, dtype=float), np.arange(15, dtype=float), np.arange(25, dtype=float)])
    indptr = np.array([0, 20, 35, 60])
    ga_mixed = GroupedArray(data, indptr)

    with pytest.raises(ValueError) as exc_info:
        sc.fit_transform(ga_mixed)
    error_msg = str(exc_info.value)
    assert "Insufficient data in 1 series" in error_msg
    assert "[15]" in error_msg

    # Test 4: No validation error with longer series even if shorter than max_season_length * n_seasons
    # The safeguard only checks series_length >= max_diffs + 4, not the n_seasons constraint
    sc_lenient = AutoSeasonalityAndDifferences(max_season_length=12, max_diffs=1, n_seasons=10)
    ga_medium = GroupedArray(np.arange(20, dtype=float), np.array([0, 20]))
    # Should not raise even though 20 < 12 * 10 = 120, because 20 >= 1 + 4 = 5
    transformed = sc_lenient.fit_transform(ga_medium)
    assert transformed.data is not None

    # Test 5: Edge case - exactly the minimum required length
    sc_exact = AutoSeasonalityAndDifferences(max_season_length=10, max_diffs=1, n_seasons=2)
    ga_exact = GroupedArray(np.arange(5, dtype=float), np.array([0, 5]))  # Exactly max_diffs + 4 = 5
    transformed = sc_exact.fit_transform(ga_exact)
    assert transformed.data is not None

    # One observation short should fail
    ga_one_short = GroupedArray(np.arange(4, dtype=float), np.array([0, 4]))  # 4 < 5
    with pytest.raises(ValueError) as exc_info:
        sc_exact.fit_transform(ga_one_short)
    assert "requires at least 5 observations" in str(exc_info.value)
