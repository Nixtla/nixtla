import random
import tempfile
import warnings
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pytest
import utilsforecast.processing as ufp
import xgboost as xgb
from sklearn import set_config
from sklearn.linear_model import LinearRegression
from utilsforecast.feature_engineering import fourier, time_features
from utilsforecast.processing import match_if_categorical

from mlforecast import MLForecast
from mlforecast.forecast import _get_conformal_method
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExponentiallyWeightedMean,
    RollingMean,
)
from mlforecast.lgb_cv import LightGBMCV
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.utils import (
    PredictionIntervals,
    generate_daily_series,
    generate_prices_for_series,
    generate_series,
)

set_config(display='text')
warnings.simplefilter('ignore', UserWarning)


def test_conformal_method():
    with pytest.raises(ValueError):
        _get_conformal_method('my_method')

@pytest.fixture
def setup_forecast_data():
    df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
    ids = df['unique_id'].unique()
    random.seed(0)
    sample_ids = random.choices(ids, k=4)
    sample_df = df[df['unique_id'].isin(sample_ids)]
    horizon = 48
    valid = sample_df.groupby('unique_id').tail(horizon)
    train = sample_df.drop(valid.index)
    return df, train, valid

@pytest.fixture
def fcst():
    """Main forecast object for testing."""
    # train, valid = setup_forecast_data
    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq=1,
        lags=[24 * (i+1) for i in range(7)],
        lag_transforms={
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        num_threads=1,
        target_transforms=[Differences([24])],
    )
    return fcst

@pytest.fixture
def fcst2():
    """Secondary forecast object with additional transforms for testing."""
    # train, valid = setup_forecast_data
    fcst2 = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq=1,
        lags=[24 * (i+1) for i in range(7)],
        lag_transforms={
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        num_threads=1,
        target_transforms=[Differences([24]), LocalStandardScaler()],
    )
    return fcst2

@pytest.fixture
def fitted_fcst(fcst, setup_forecast_data):
    """Pre-fitted forecast object for tests that need fitted models."""
    df, train, _ = setup_forecast_data
    fcst.fit(train, fitted=True)
    return fcst

@pytest.fixture
def predictions(fitted_fcst):
    """Predictions from the fitted forecast object."""
    # train, valid = setup_forecast_data
    horizon = 48
    return fitted_fcst.predict(horizon)


def test_missing_future(fcst, setup_forecast_data):
    df, train, _ = setup_forecast_data
    train2 = train.copy()
    train2['weight'] = np.random.default_rng(seed=0).random(train2.shape[0])
    fcst.fit(train2, weight_col='weight', as_numpy=True).predict(5)
    fcst.cross_validation(train2, n_windows=2, h=5, weight_col='weight', as_numpy=True)
    fcst.fit(train, fitted=True);
    expected_future = fcst.make_future_dataframe(h=1)


    missing_future = fcst.get_missing_future(h=1, X_df=expected_future.head(2))
    pd.testing.assert_frame_equal(
        missing_future,
        expected_future.tail(2).reset_index(drop=True)
    )


# check that the fitted target from the fitted_values matches the original target after applying the transformation
def test_fitted_target(fcst2, setup_forecast_data):
    df, train, valid = setup_forecast_data
    fcst2.fit(train, fitted=True)
    fitted_vals = fcst2.forecast_fitted_values()
    train_restored = train.merge(
        fitted_vals.drop(columns='LGBMRegressor'),
        on=['unique_id', 'ds'],
        suffixes=('_expected', '_actual')
    )
    np.testing.assert_allclose(
        train_restored['y_expected'].values,
        train_restored['y_actual'].values,
    )

    # check fitted + max_horizon
    max_horizon = 7
    fcst2.fit(train, fitted=True, max_horizon=max_horizon)
    max_horizon_fitted_values = fcst2.forecast_fitted_values()
    # h is 1 to max_horizon
    np.testing.assert_equal(
        np.sort(max_horizon_fitted_values['h'].unique()),
        np.arange(1, max_horizon + 1),
    )
    # predictions for the first horizon are equal to the recursive
    pd.testing.assert_frame_equal(
        fitted_vals.reset_index(drop=True),
        max_horizon_fitted_values[max_horizon_fitted_values['h'] == 1].drop(columns='h'),
    )
    # restored values match
    xx = max_horizon_fitted_values[lambda x: x['unique_id'].eq('H413')].pivot_table(
        index=['unique_id', 'ds'], columns='h', values='y'
    ).loc['H413']
    first_ds = xx.index.min()
    last_ds = xx.index.max()
    for h in range(1, max_horizon):
        np.testing.assert_allclose(
            xx.loc[first_ds + h :, 1].values,
            xx.loc[: last_ds - h, h + 1].values,
        )


def test_new_df_argument(fitted_fcst, setup_forecast_data, predictions):
    """Test that predictions with new_df argument work correctly."""
    df, train, _ = setup_forecast_data
    horizon = 48
    pd.testing.assert_frame_equal(
        fitted_fcst.predict(horizon, new_df=train),
        predictions
    )
@pytest.fixture
def fcst_with_intervals(fcst, setup_forecast_data):
    """Forecast object fitted with prediction intervals."""
    _, train, _ = setup_forecast_data
    fcst.fit(
        train,
        prediction_intervals=PredictionIntervals(n_windows=3, h=48)
    )
    return fcst

@pytest.fixture
def predictions_w_intervals(fcst_with_intervals):
    """Predictions with prediction intervals."""
    return fcst_with_intervals.predict(48, level=[50, 80, 95])

def test_prediction_intervals_lower_horizon(fcst_with_intervals):
    """Test we can forecast horizon lower than h with prediction intervals."""
    # Test different horizons
    preds_h1 = fcst_with_intervals.predict(1, level=[50, 80, 95])
    preds_h30 = fcst_with_intervals.predict(30, level=[50, 80, 95])

    # Test monotonicity of intervals for h=1
    monotonic_count = preds_h1.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(preds_h1)

    # Test monotonicity of intervals for h=30
    monotonic_count = preds_h30.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(preds_h30)

def test_prediction_intervals_error_conditions(fcst_with_intervals):
    """Test error conditions for prediction intervals."""
    # Should fail when predicting beyond fitted horizon
    with pytest.raises(Exception):  # Replace with specific exception if known
        fcst_with_intervals.predict(49, level=[68])

def test_recover_point_forecasts(predictions, predictions_w_intervals):
    """Test we can recover point forecasts from interval predictions."""
    pd.testing.assert_frame_equal(
        predictions,
        predictions_w_intervals[predictions.columns]
    )

def test_recover_mean_forecasts_level_zero(predictions, fcst_with_intervals):
    """Test we can recover mean forecasts with level 0."""
    level_zero_preds = fcst_with_intervals.predict(48, level=[0])
    np.testing.assert_allclose(
        predictions['LGBMRegressor'].values,
        level_zero_preds['LGBMRegressor-lo-0'].values,
    )

def test_prediction_intervals_monotonicity(predictions_w_intervals):
    """Test monotonicity of prediction intervals."""
    monotonic_count = predictions_w_intervals.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(predictions_w_intervals)


def test_indexed_data_datetime_ds():
    # test indexed data, datetime ds
    fcst_test = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq='D',
        lags=[1],
        num_threads=1,
    )
    df_test = generate_daily_series(1)
    fcst_test.fit(
        df_test,
        # prediction_intervals=PredictionIntervals()
    )
    pred_test = fcst_test.predict(12)
    pred_int_test = fcst_test.predict(12, level=[80, 90])
    # test same structure
    assert pred_test.values.all() == pred_int_test[pred_test.columns].values.all()
    # test monotonicity of intervals
    assert pred_int_test.filter(regex='lo|hi').apply(
            lambda x: x.is_monotonic_increasing,
            axis=1
        ).sum() == len(pred_int_test)


def namer(tfm, lag, *args): # noqa
    # transforms namer
    return f'hello_from_{tfm.__class__.__name__.lower()}'

def test_transform_name(setup_forecast_data, fcst, predictions):
    """Test custom lag transform namer and manual model fitting."""
    _, train, _ = setup_forecast_data

    # Test custom transform namer
    fcst2 = MLForecast(
        models=LinearRegression(),
        freq=1,
        lag_transforms={1: [ExpandingMean()]},
        lag_transforms_namer=namer,
    )
    prep = fcst2.preprocess(train)
    assert 'hello_from_expandingmean' in prep

    # Test manual model fitting produces same results as regular fitting
    fcst.fit(train, fitted=True)  # Regular fitting

    # Manual fitting
    prep_df = fcst.preprocess(train)
    X, y = prep_df.drop(columns=['unique_id', 'ds', 'y']), prep_df['y']
    fcst.fit_models(X, y)

    horizon = 48
    predictions2 = fcst.predict(horizon)
    pd.testing.assert_frame_equal(predictions, predictions2)

# test intervals multioutput
def test_intervals_multioutput(setup_forecast_data, fcst):
    _, train, _ = setup_forecast_data
    max_horizon = 24
    _ = fcst.fit(
        train,
        max_horizon=max_horizon,
        prediction_intervals=PredictionIntervals(h=max_horizon)
    )
    individual_preds = fcst.predict(max_horizon)
    individual_preds_intervals = fcst.predict(max_horizon, level=[90, 80])
    # test monotonicity of intervals
    monotonic_intervals = individual_preds_intervals.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    )
    assert monotonic_intervals.sum() == len(individual_preds_intervals)
    # test we can recover point forecasts with intervals
    pd.testing.assert_frame_equal(
        individual_preds,
        individual_preds_intervals[individual_preds.columns]
    )

# check for bad max_horizon & models_ states before predict
def test_bad_max_horizon(setup_forecast_data):
    df, *_ = setup_forecast_data
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=[12],
    )
    fcst.fit(df, max_horizon=2)
    fcst.preprocess(df, max_horizon=None)
    with pytest.raises(ValueError) as exec:
        fcst.predict(1)
    assert 'Found one model per horizon' in str(exec.value)

    fcst.fit(df, max_horizon=None)
    fcst.preprocess(df, max_horizon=2)
    with pytest.raises(ValueError) as exec:
        fcst.predict(1)
    assert 'Found a single model for all horizons' in str(exec.value)


# test fitted
def test_fitted_max_horizon(setup_forecast_data):
    df, train, valid = setup_forecast_data
    fcst2 = MLForecast(
        models=lgb.LGBMRegressor(n_estimators=5, random_state=0, verbosity=-1),
        freq=1,
        lags=[24],
        num_threads=1,
    )
    _ = fcst2.cross_validation(
        train,
        n_windows=2,
        h=2,
        fitted=True,
        refit=False,
    )
    fitted_cv_results = fcst2.cross_validation_fitted_values()
    train_with_cv_fitted_values = train.merge(fitted_cv_results, on=['unique_id', 'ds'], suffixes=('_expected', ''))
    np.testing.assert_allclose(
        train_with_cv_fitted_values['y_expected'].values,
        train_with_cv_fitted_values['y'].values,
    )

    # test with max_horizon
    _ = fcst2.cross_validation(
        train,
        n_windows=2,
        h=2,
        fitted=True,
        refit=False,
        max_horizon=2,
    )
    max_horizon_fitted_cv_results = fcst2.cross_validation_fitted_values()
    pd.testing.assert_frame_equal(
        fitted_cv_results[lambda df: df['fold'].eq(0)],
        (
            max_horizon_fitted_cv_results
            [lambda df: df['fold'].eq(0) & df['h'].eq(1)]
            .drop(columns='h')
            .reset_index(drop=True)
        ),
    )

def test_refit(setup_forecast_data):
    _, train, _ = setup_forecast_data
    fcst = MLForecast(
        models=LinearRegression(),
        freq=1,
        lags=[1, 24],
    )
    horizon = 48

    for refit, expected_models in zip([True, False, 2], [4, 1, 2]):
        fcst.cross_validation(
            train,
            n_windows=4,
            h=horizon,
            refit=refit,
        )
        assert len(fcst.cv_models_) == expected_models

def test_cv_no_refit(setup_forecast_data):
    _, train, _ = setup_forecast_data

    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq=1,
        lags=[24 * (i+1) for i in range(7)],
        lag_transforms={
            1: [RollingMean(window_size=24)],
            24: [RollingMean(window_size=24)],
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        num_threads=1,
        target_transforms=[Differences([24])],
    )
    horizon = 48

    cv_results_no_refit = fcst.cross_validation(
        train,
        n_windows=2,
        h=horizon,
        step_size=horizon,
        refit=False
    )

    cv_results = fcst.cross_validation(
        train,
        n_windows=2,
        h=horizon,
        step_size=horizon,
        fitted=True,
    )
    pd.testing.assert_frame_equal(cv_results_no_refit.drop(columns='LGBMRegressor'), cv_results.drop(columns='LGBMRegressor'))
    # test the first window has the same forecasts
    first_cutoff = cv_results['cutoff'].iloc[0] # noqa
    pd.testing.assert_frame_equal(cv_results_no_refit.query('cutoff == @first_cutoff'), cv_results.query('cutoff == @first_cutoff'))

    # test next windows have different forecasts
    no_refit_other_windows = cv_results_no_refit.query('cutoff != @first_cutoff')
    refit_other_windows = cv_results.query('cutoff != @first_cutoff')
    # The forecasts should be different for the second window when refit=True vs refit=False
    assert not no_refit_other_windows['LGBMRegressor'].equals(refit_other_windows['LGBMRegressor'])

def test_cv_input_size(setup_forecast_data, fcst):
    _, train, _ = setup_forecast_data
    horizon = 48
    # cv with input_size
    input_size = 300
    _ = fcst.cross_validation(
        train,
        n_windows=2,
        h=horizon,
        step_size=horizon,
        input_size=input_size,
        keep_last_n=input_size,
    )
    series_lengths = np.diff(fcst.ts.ga.indptr)
    unique_lengths = np.unique(series_lengths)
    assert unique_lengths.size == 1
    assert unique_lengths[0] == input_size


# one model per horizon
def test_model_per_horizon(setup_forecast_data, fcst):
    _, train, _ = setup_forecast_data
    horizon = 48

    cv_results2 = fcst.cross_validation(
        train,
        n_windows=2,
        h=horizon,
        step_size=horizon,
        max_horizon=horizon,
    )

    cv_results = fcst.cross_validation(
        train,
        n_windows=2,
        h=horizon,
        step_size=horizon,
        fitted=True,
    )

    # the first entry per id and window is the same
    pd.testing.assert_frame_equal(
        cv_results.groupby(['unique_id', 'cutoff']).head(1),
        cv_results2.groupby(['unique_id', 'cutoff']).head(1)
    )
    # the rest is different
    assert not cv_results.equals(cv_results2)


# one model per horizon with prediction intervals
def test_model_per_horizon_with_prediction_intervals(setup_forecast_data, fcst):
    _, train, _ = setup_forecast_data
    horizon = 48

    cv_results2_intervals = fcst.cross_validation(
        train,
        n_windows=2,
        h=horizon,
        step_size=horizon,
        max_horizon=horizon,
        prediction_intervals=PredictionIntervals(n_windows=2, h=horizon),
        level=[80, 90]
    )

    cv_results_intervals = fcst.cross_validation(
        train,
        n_windows=2,
        h=horizon,
        step_size=horizon,
        prediction_intervals=PredictionIntervals(h=horizon),
        level=[80, 90]
    )
    # the first entry per id and window is the same
    pd.testing.assert_frame_equal(
        cv_results_intervals.groupby(['unique_id', 'cutoff']).head(1),
        cv_results2_intervals.groupby(['unique_id', 'cutoff']).head(1)
    )
    # the rest is different
    assert not cv_results_intervals.equals(cv_results2_intervals)

def test_wrong_frequency_error():
    # wrong frequency raises error
    df_wrong_freq = pd.DataFrame({'ds': pd.to_datetime(['2020-01-02', '2020-02-02', '2020-03-02', '2020-04-02'])})
    df_wrong_freq['unique_id'] = 'id1'
    df_wrong_freq['y'] = 1
    fcst_wrong_freq = MLForecast(
        models=[LinearRegression()],
        freq='MS',
        lags=[1],
    )
    with pytest.raises(ValueError) as exec:
        fcst_wrong_freq.cross_validation(df_wrong_freq, n_windows=1, h=1)
    assert 'Cross validation result produced less results than expected' in str(exec.value)


def test_best_iter(setup_forecast_data):
    _, train, _ = setup_forecast_data
    horizon = 48

    cv = LightGBMCV(
        freq=1,
        lags=[24 * (i+1) for i in range(7)],
        lag_transforms={
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        num_threads=1,
        target_transforms=[Differences([24])]
    )
    _ = cv.fit(
        train,
        n_windows=2,
        h=horizon,
        params={'verbosity': -1},
    )

    fcst = MLForecast.from_cv(cv)
    assert cv.best_iteration_ == fcst.models['LGBMRegressor'].n_estimators


def test_preds_day():
    series = generate_daily_series(100, equal_ends=True, n_static_features=2, static_as_categorical=False)
    non_std_series = series.copy()
    non_std_series['ds'] = non_std_series.groupby('unique_id', observed=True).cumcount()
    non_std_series = non_std_series.rename(columns={'unique_id': 'some_id', 'ds': 'time', 'y': 'value'})
    models = [
        lgb.LGBMRegressor(n_jobs=1, random_state=0, verbosity=-1),
        xgb.XGBRegressor(n_jobs=1, random_state=0),
    ]
    flow_params = dict(
        models=models,
        lags=[7],
        lag_transforms={
            1: [ExpandingMean()],
            7: [RollingMean(window_size=14)]
        },
        num_threads=2,
    )
    fcst = MLForecast(freq=1, **flow_params)
    non_std_preds = fcst.fit(non_std_series, id_col='some_id', time_col='time', target_col='value').predict(7)
    non_std_preds = non_std_preds.rename(columns={'some_id': 'unique_id'})
    fcst = MLForecast(freq='D', **flow_params)
    preds = fcst.fit(series).predict(7)
    pd.testing.assert_frame_equal(preds.drop(columns='ds'), non_std_preds.drop(columns='time'))

@pytest.fixture
def non_std_series():
    series = generate_daily_series(100, equal_ends=True, n_static_features=2, static_as_categorical=False)
    non_std_series = series.copy()
    non_std_series['ds'] = non_std_series.groupby('unique_id', observed=True).cumcount()
    non_std_series = non_std_series.rename(columns={'unique_id': 'some_id', 'ds': 'time', 'y': 'value'})
    return non_std_series

def assert_cross_validation(data, add_exogenous=False):
    n_windows = 2
    h = 14
    fcst = MLForecast(lgb.LGBMRegressor(verbosity=-1), freq=1, lags=[7, 14])
    if add_exogenous:
        data = data.assign(ex1 = lambda x: np.arange(0, len(x)))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        backtest_results = fcst.cross_validation(
            df=data,
            n_windows=n_windows,
            h=h,
            id_col='some_id',
            time_col='time',
            target_col='value',
            static_features=['some_id', 'static_0', 'static_1'],
        )
    renamer = {'some_id': 'unique_id', 'time': 'ds', 'value': 'y'}
    backtest_results = backtest_results.rename(columns=renamer)
    renamed = data.rename(columns=renamer)
    manual_results = []
    for cutoff, train, valid in ufp.backtest_splits(renamed, n_windows, h, 'unique_id', 'ds', 1):
        fcst.fit(train, static_features=['unique_id', 'static_0', 'static_1'])
        if add_exogenous:
            X_df = valid.drop(columns=['y', 'static_0', 'static_1']).reset_index()
        else:
            X_df = None
        pred = fcst.predict(h, X_df=X_df)
        res = valid[['unique_id', 'ds', 'y']].copy()
        res = res.merge(cutoff, on='unique_id')
        res = res[['unique_id', 'ds', 'cutoff', 'y']].copy()
        manual_results.append(res.merge(pred, on=['unique_id', 'ds'], how='left'))
    manual_results = pd.concat(manual_results)
    pd.testing.assert_frame_equal(backtest_results, manual_results.reset_index(drop=True))

def test_cross_validation(non_std_series):
    assert_cross_validation(non_std_series)
    assert_cross_validation(non_std_series, add_exogenous=True)

# test short series in cv
def test_short_series_in_cv():
    series = generate_daily_series(
        n_series=100, min_length=20, max_length=51, equal_ends=True,
    )
    horizon = 10
    n_windows = 4
    fcst = MLForecast(models=[LinearRegression()], freq='D', lags=[1])
    cv_res = fcst.cross_validation(series, h=horizon, n_windows=n_windows)
    series_per_cutoff = cv_res.groupby('cutoff')['unique_id'].nunique()
    series_sizes = series['unique_id'].value_counts().sort_index()
    for i in range(4):
        assert series_per_cutoff.iloc[i] == series_sizes.gt((n_windows - i) * horizon).sum()




@pytest.fixture
def polars_pandas_test_data():
    """Generate test data for polars/pandas compatibility tests."""
    horizon = 2
    series_pl = generate_daily_series(
        10, n_static_features=2, static_as_categorical=False, equal_ends=True, engine='polars'
    )
    series_pd = generate_daily_series(
        10, n_static_features=2, static_as_categorical=False, equal_ends=True, engine='pandas'
    )
    series_pd = series_pd.rename(columns={'static_0': 'product_id'})
    prices_pd = generate_prices_for_series(series_pd, horizon)
    prices_pd['weight'] = np.random.rand(prices_pd.shape[0])
    prices_pd['unique_id'] = prices_pd['unique_id'].astype(series_pd['unique_id'].dtype)
    series_pd = series_pd.merge(prices_pd, on=['unique_id', 'ds'])

    prices_pl = pl.from_pandas(prices_pd)
    uids_series, uids_prices = match_if_categorical(series_pl['unique_id'], prices_pl['unique_id'])
    series_pl = series_pl.with_columns(uids_series)
    prices_pl = prices_pl.with_columns(uids_prices)
    series_pl = series_pl.rename({'static_0': 'product_id'}).join(prices_pl, on=['unique_id', 'ds'])
    permutation = np.random.choice(series_pl.shape[0], series_pl.shape[0], replace=False)
    series_pl = series_pl[permutation]
    series_pd = series_pd.iloc[permutation]

    cfg = dict(
        models=[LinearRegression(), lgb.LGBMRegressor(verbosity=-1)],
        freq='1d',
        lags=[1, 2],
        lag_transforms={
            1: [ExpandingMean()],
            2: [ExpandingMean()],
        },
        date_features=['day', 'month', 'week', 'year'],
        target_transforms=[Differences([1, 2]), LocalStandardScaler()],
    )
    fit_kwargs = dict(
        fitted=True,
        prediction_intervals=PredictionIntervals(h=horizon),
        static_features=['product_id', 'static_1'],
        weight_col='weight',
    )
    predict_kwargs = dict(
        h=2,
        level=[80, 95],
    )
    cv_kwargs = dict(
        n_windows=2,
        h=horizon,
        fitted=True,
        static_features=['product_id', 'static_1'],
        weight_col='weight',
    )

    return {
        'series_pl': series_pl,
        'series_pd': series_pd,
        'prices_pl': prices_pl,
        'prices_pd': prices_pd,
        'cfg': cfg,
        'fit_kwargs': fit_kwargs,
        'predict_kwargs': predict_kwargs,
        'cv_kwargs': cv_kwargs,
        'horizon': horizon,
    }

@pytest.mark.parametrize("max_horizon,as_numpy", product([None, 2], [True, False]))
def test_polars_pandas_compatibility(polars_pandas_test_data, max_horizon, as_numpy):
    """Test that polars and pandas produce identical results."""
    data = polars_pandas_test_data

    # Skip as_numpy=False for now due to LightGBM/Polars compatibility issues
    if not as_numpy:
        pytest.skip("LightGBM with polars DataFrames not fully supported yet")

    fcst_pl = MLForecast(**data['cfg'])
    fcst_pl.fit(data['series_pl'], max_horizon=max_horizon, as_numpy=as_numpy, **data['fit_kwargs'])
    fitted_pl = fcst_pl.forecast_fitted_values()
    preds_pl = fcst_pl.predict(X_df=data['prices_pl'], **data['predict_kwargs'])
    preds_pl_subset = fcst_pl.predict(X_df=data['prices_pl'], ids=fcst_pl.ts.uids[[0, 6]], **data['predict_kwargs'])
    cv_pl = fcst_pl.cross_validation(data['series_pl'], as_numpy=as_numpy, **data['cv_kwargs'])
    cv_fitted_pl = fcst_pl.cross_validation_fitted_values()

    fcst_pd = MLForecast(**data['cfg'])
    fcst_pd.fit(data['series_pd'], max_horizon=max_horizon, as_numpy=as_numpy, **data['fit_kwargs'])
    fitted_pd = fcst_pd.forecast_fitted_values()
    preds_pd = fcst_pd.predict(X_df=data['prices_pd'], **data['predict_kwargs'])
    preds_pd_subset = fcst_pd.predict(X_df=data['prices_pd'], ids=fcst_pd.ts.uids[[0, 6]], **data['predict_kwargs'])
    assert preds_pd_subset['unique_id'].unique().tolist() == ['id_0', 'id_6']
    cv_pd = fcst_pd.cross_validation(data['series_pd'], as_numpy=as_numpy, **data['cv_kwargs'])
    cv_fitted_pd = fcst_pd.cross_validation_fitted_values()

    if max_horizon is not None:
        fitted_pl = fitted_pl.with_columns(pl.col('h').cast(pl.Int64))
        for h in range(max_horizon):
            fitted_h = fitted_pl.filter(pl.col('h').eq(h + 1))
            series_offset = (
                data['series_pl']
                .sort('unique_id', 'ds')
                .with_columns(pl.col('y').shift(-h).over('unique_id'))
            )
            series_filtered = (
                fitted_h
                [['unique_id', 'ds']]
                .join(series_offset, on=['unique_id', 'ds'])
                .sort(['unique_id', 'ds'])
            )
            np.testing.assert_allclose(
                series_filtered['y'],
                fitted_h['y']
            )
    else:
        series_filtered = (
            fitted_pl
            [['unique_id', 'ds']]
            .join(data['series_pl'], on=['unique_id', 'ds'])
            .sort(['unique_id', 'ds'])
        )
        np.testing.assert_allclose(
            series_filtered['y'],
            fitted_pl['y']
        )

    pd.testing.assert_frame_equal(fitted_pl.to_pandas(), fitted_pd)
    pd.testing.assert_frame_equal(preds_pl.to_pandas(), preds_pd)
    pd.testing.assert_frame_equal(preds_pl_subset.to_pandas(), preds_pd_subset)
    pd.testing.assert_frame_equal(cv_pl.to_pandas(), cv_pd)
    pd.testing.assert_frame_equal(
        cv_fitted_pl.with_columns(pl.col('fold').cast(pl.Int64)).to_pandas(),
        cv_fitted_pd,
    )
# test transforms are inverted correctly when series were dropped
def test_transforms_inverted_when_series_dropped():
    """Test that transforms are inverted correctly when series were dropped."""
    series = generate_daily_series(10, min_length=5, max_length=20)
    fcst = MLForecast(
        models=LinearRegression(),
        freq='D',
        lags=[10],
        target_transforms=[Differences([1]), LocalStandardScaler()],
    )
    fcst.fit(series, fitted=True)
    assert fcst.ts._dropped_series.size > 0
    fitted_vals = fcst.forecast_fitted_values()
    full = fitted_vals.merge(series, on=['unique_id', 'ds'], suffixes=('_fitted', '_orig'))
    np.testing.assert_allclose(
        full['y_fitted'].values,
        full['y_orig'].values,
    )

# save & load
def test_save_load():
    """Test saving and loading MLForecast objects."""
    series = generate_daily_series(10)
    fcst = MLForecast(
        models=LinearRegression(),
        freq='D',
        lags=[10],
        target_transforms=[Differences([1]), LocalStandardScaler()],
    )
    fcst.fit(series)
    preds = fcst.predict(10)
    with tempfile.TemporaryDirectory() as tmpdir:
        savedir = Path(tmpdir) / 'fcst'
        savedir.mkdir()
        fcst.save(savedir)
        fcst2 = MLForecast.load(savedir)
    preds2 = fcst2.predict(10)
    pd.testing.assert_frame_equal(preds, preds2)

# direct approach requires only one timestamp and produces same results for two models
def test_direct_approach_single_timestamp():
    """Test direct approach works with partial X_df (backward compatibility)."""
    series = generate_daily_series(5)
    h = 5
    freq = 'D'
    train, future = time_features(series, freq=freq, features=['day'], h=h)
    models = [LinearRegression(), lgb.LGBMRegressor(n_estimators=5)]

    # Test 1: Full features work correctly
    fcst1 = MLForecast(models=models, freq=freq, date_features=['dayofweek'])
    fcst1.fit(train, max_horizon=h, static_features=[])
    preds1 = fcst1.predict(h=h, X_df=future)  # full timestamps

    # Test 2: Partial features work (backward compatibility)
    fcst2 = MLForecast(models=models[::-1], freq=freq, date_features=['dayofweek'])
    fcst2.fit(train, max_horizon=h, static_features=[])
    X_df_one = future.groupby('unique_id', observed=True).head(1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        preds2 = fcst2.predict(h=h, X_df=X_df_one)  # single timestamp (reused)
        # Verify warning was raised
        assert len(w) == 1
        assert 'missing horizon steps' in str(w[0].message)

    # Shape validation
    assert preds1.shape[0] == len(series['unique_id'].unique()) * h
    assert preds2.shape[0] == len(series['unique_id'].unique()) * h

    # Sanity checks for partial-feature predictions (preds2)
    model_cols = [col for col in preds2.columns if col not in ['unique_id', 'ds']]
    for col in model_cols:
        # 1. No NaN/Inf values
        assert not preds2[col].isna().any(), f"{col} contains NaN values"
        assert not np.isinf(preds2[col]).any(), f"{col} contains Inf values"

        # 2. Values within reasonable bounds (based on training data statistics)
        lower_bound = train['y'].min()
        upper_bound = train['y'].max()
        assert preds2[col].min() >= lower_bound, \
            f"{col} has values below training data bound: {preds2[col].min()} < {lower_bound}"
        assert preds2[col].max() <= upper_bound, \
            f"{col} has values above training data bound: {preds2[col].max()} > {upper_bound}"

    # First period predictions should be identical (same features used)
    first_preds1 = preds1.groupby('unique_id', observed=True).head(1)
    first_preds2 = preds2.groupby('unique_id', observed=True).head(1)
    pd.testing.assert_frame_equal(first_preds1, first_preds2[first_preds1.columns])

    # Full predictions should be highly correlated
    for col in model_cols:
        correlation = np.corrcoef(preds1[col], preds2[col])[0, 1]
        assert correlation > 0.95, \
            f"{col} predictions have low correlation: {correlation:.3f}"


def test_direct_forecasting_exogenous_alignment():
    """Test that direct forecasting correctly aligns exogenous features to each horizon.

    This test reproduces the bug reported in issue #496 where direct forecasting
    only uses exogenous features from horizon 1 for all horizons. The test creates
    two versions of exogenous data:
    1. test_X_df: Correct Fourier features for all horizons
    2. test_X_df_zeros: Fourier features zeroed out for horizon > 1

    Before fix: Predictions are identical (bug - only horizon 1 features used)
    After fix: Predictions differ (correct - each horizon uses its aligned features)
    """
    H = 3
    df = generate_series(3, freq="W", min_length=104, n_static_features=1)
    df, _ = fourier(df, freq="W", season_length=52, k=1, h=H)

    # Train/test split
    test = df.groupby("unique_id").tail(H)
    train = df.drop(test.index)
    test_X_df = test.drop(columns=["static_0", "y"])

    # Create version with zeros for horizon > 1
    test_X_df_zeros = test_X_df.copy()
    test_X_df_zeros["is_first_ds"] = test_X_df_zeros.groupby("unique_id")["ds"].transform(
        lambda x: x == x.min()
    )
    test_X_df_zeros.loc[~test_X_df_zeros["is_first_ds"], ["sin1_52", "cos1_52"]] = 0
    test_X_df_zeros = test_X_df_zeros.drop(columns="is_first_ds")

    # Fit and predict
    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq="W",
        lags=[1],
        date_features=["month"],
    )
    fcst.fit(train, static_features=["static_0"], max_horizon=H)

    preds_correct = fcst.predict(h=H, X_df=test_X_df)
    preds_zeros = fcst.predict(h=H, X_df=test_X_df_zeros)

    # After fix: predictions should be DIFFERENT
    # (before fix they are identical, proving the bug)
    assert not np.allclose(
        preds_correct["LGBMRegressor"].values,
        preds_zeros["LGBMRegressor"].values
    )


def test_direct_forecasting_perfect_exogenous_fit():
    """Test that direct forecasting correctly uses exogenous features for each horizon.

    This test validates that when a model is trained with a perfect linear relationship
    between the target and an exogenous feature (y = X), the predictions at each horizon
    correctly use the aligned exogenous features from X_df. If properly implemented,
    predictions should equal the exogenous feature values.
    """
    H = 3

    # Create simple training data where y = X (perfect linear relationship)
    df = pd.DataFrame({
        "ds": np.arange(7),
        "X": np.arange(7),
        "unique_id": "ex",
        "y": np.arange(7),
    })

    # Create test dataframe with exogenous features for prediction
    test_df = pd.DataFrame({
        "ds": np.arange(7, 10),
        "X": np.arange(7, 10),
        "unique_id": "ex",
    })

    # Fit with max_horizon to enable direct forecasting
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
    )
    fcst.fit(df, static_features=[], max_horizon=H)

    # Predict using exogenous features
    individual_preds = fcst.predict(h=H, X_df=test_df)

    # Validation: predictions should equal the exogenous feature values
    # This proves each horizon model uses the correct aligned features
    np.testing.assert_allclose(
        individual_preds['LinearRegression'].values,
        test_df['X'].values,
        rtol=1e-10,
        err_msg="Direct forecasting predictions do not match expected exogenous feature values"
    )
