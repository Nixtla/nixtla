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
    default_direct_fitted_values = fcst2.forecast_fitted_values()
    assert default_direct_fitted_values["h"].eq(1).all()
    max_horizon_fitted_values = pd.concat(
        [
            fcst2.forecast_fitted_values(h=horizon_idx)
            for horizon_idx in range(1, max_horizon + 1)
        ],
        ignore_index=True,
    )
    # h is 1 to max_horizon
    np.testing.assert_equal(
        np.sort(max_horizon_fitted_values['h'].unique()),
        np.arange(1, max_horizon + 1),
    )
    # predictions for the first horizon are equal to the recursive
    pd.testing.assert_frame_equal(
        fitted_vals.reset_index(drop=True),
        max_horizon_fitted_values[max_horizon_fitted_values['h'] == 1].reset_index(drop=True),
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


def test_recursive_forecast_fitted_values_on_demand_h():
    df = generate_daily_series(2, min_length=50, max_length=50)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
    )
    fcst.fit(df, fitted=True, static_features=[])

    fitted_h1 = fcst.forecast_fitted_values()
    fitted_h3 = fcst.forecast_fitted_values(h=3)

    assert "h" in fitted_h1.columns
    assert "h" in fitted_h3.columns
    assert fitted_h1["h"].eq(1).all()
    assert fitted_h3["h"].eq(3).all()
    assert fitted_h3.shape[0] < fitted_h1.shape[0]

    assert np.isfinite(fitted_h1["LinearRegression"].to_numpy()).all()
    assert np.isfinite(fitted_h3["LinearRegression"].to_numpy()).all()
    overlap = fitted_h1[["unique_id", "ds", "LinearRegression"]].merge(
        fitted_h3[["unique_id", "ds", "LinearRegression"]],
        on=["unique_id", "ds"],
        suffixes=("_h1", "_h3"),
    )
    assert len(overlap) > 0
    assert not np.allclose(overlap["LinearRegression_h1"], overlap["LinearRegression_h3"])


def test_recursive_forecast_fitted_values_on_demand_h_matches_slow_baseline():
    h = 3
    df = generate_daily_series(2, min_length=40, max_length=40)
    df["x"] = np.linspace(0.0, 1.0, len(df))
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
        lag_transforms={1: [RollingMean(3)]},
    )
    fcst.fit(df, fitted=True, static_features=[])

    fast = fcst.forecast_fitted_values(h=h).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    train_pd = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    one_step_pd = fcst.fcst_fitted_values_[["unique_id", "ds"]].copy()
    valid_one_step_times = {
        uid: set(group["ds"].tolist())
        for uid, group in one_step_pd.groupby("unique_id", observed=True)
    }
    static = [c for c in fcst.ts.static_features_.columns if c != "unique_id"]
    exclude = {"unique_id", "ds", "y", *static}
    if fcst.ts.weight_col is not None:
        exclude.add(fcst.ts.weight_col)
    dynamic = [c for c in train_pd.columns if c not in exclude]

    rows = []
    original_ts = fcst.ts
    for uid, group in train_pd.groupby("unique_id", observed=True):
        group = group.sort_values("ds").reset_index(drop=True)
        valid_uid_times = valid_one_step_times.get(uid, set())
        for target_idx in range(h - 1, len(group)):
            origin_idx = target_idx - h
            if origin_idx < 0:
                continue
            hist = group.iloc[: origin_idx + 1]
            future = group.iloc[origin_idx + 1 : origin_idx + h + 1]
            if future.shape[0] < h:
                continue
            if future.iloc[0]["ds"] not in valid_uid_times:
                continue
            X_df = future[["unique_id", "ds", *dynamic]] if dynamic else None
            try:
                preds = fcst.predict(h=h, new_df=hist, X_df=X_df)
            finally:
                fcst.ts = original_ts
            pred_last = preds.iloc[-1]
            rows.append(
                {
                    "unique_id": group.iloc[target_idx]["unique_id"],
                    "ds": group.iloc[target_idx]["ds"],
                    "y": group.iloc[target_idx]["y"],
                    "h": h,
                    "LinearRegression": pred_last["LinearRegression"],
                }
            )
    expected = pd.DataFrame(rows).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(fast, expected)


def test_recursive_forecast_fitted_values_on_demand_uses_cached_train_snapshot():
    h = 3
    df = generate_daily_series(2, min_length=40, max_length=40)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
    )
    fcst.fit(df, fitted=True, static_features=[])
    before = fcst.forecast_fitted_values(h=h).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Mutate caller-side dataframe after fit; cached train snapshot should be unaffected.
    df.loc[:, "y"] = df["y"] + 1000
    after = fcst.forecast_fitted_values(h=h).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(before, after)


def test_recursive_forecast_fitted_values_on_demand_h_rejects_global_group_tfms():
    df = generate_daily_series(2, min_length=50, max_length=50)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1],
        lag_transforms={1: [RollingMean(2, global_=True)]},
    )
    fcst.fit(df, fitted=True, static_features=[])

    with pytest.raises(ValueError, match="not supported.*global or grouped lag transforms"):
        fcst.forecast_fitted_values(h=2)


def test_recursive_forecast_fitted_values_on_demand_h_with_static_features():
    df = generate_daily_series(2, min_length=50, max_length=50, n_static_features=1, static_as_categorical=False)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
    )
    fcst.fit(df, fitted=True, static_features=['static_0'])

    # Should not raise a KeyError due to missing static columns in hist
    fitted_h3 = fcst.forecast_fitted_values(h=3)
    assert "h" in fitted_h3.columns
    assert fitted_h3["h"].eq(3).all()
    assert np.isfinite(fitted_h3["LinearRegression"].to_numpy()).all()


def test_recursive_forecast_fitted_values_on_demand_h_with_train_df_override():
    h = 3
    df = generate_daily_series(2, min_length=50, max_length=50)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
    )
    fcst.fit(df, fitted=True, static_features=[], cache_train_df=False)

    assert not hasattr(fcst, "_fitted_train_df_")
    fitted_h3 = fcst.forecast_fitted_values(h=h, train_df=df)
    assert "h" in fitted_h3.columns
    assert fitted_h3["h"].eq(h).all()
    assert np.isfinite(fitted_h3["LinearRegression"].to_numpy()).all()


def test_recursive_forecast_fitted_values_on_demand_h_requires_train_df_without_cache():
    df = generate_daily_series(2, min_length=50, max_length=50)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
    )
    fcst.fit(df, fitted=True, static_features=[], cache_train_df=False)

    with pytest.raises(ValueError, match="Pass `train_df` to `forecast_fitted_values`"):
        fcst.forecast_fitted_values(h=3)


def test_direct_forecast_fitted_values_h_filter():
    df = generate_daily_series(2, min_length=40, max_length=40)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
    )
    fcst.fit(df, fitted=True, max_horizon=3, static_features=[])

    fitted_default = fcst.forecast_fitted_values()
    fitted_h2 = fcst.forecast_fitted_values(h=2)

    assert fitted_default["h"].eq(1).all()
    assert fitted_h2["h"].eq(2).all()
    assert fitted_h2.shape[0] > 0
    with pytest.raises(ValueError, match="No fitted values found for h=4"):
        fcst.forecast_fitted_values(h=4)


def test_direct_forecast_fitted_values_requires_h_metadata():
    df = generate_daily_series(2, min_length=40, max_length=40)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
    )
    fcst.fit(df, fitted=True, max_horizon=3, static_features=[])
    fcst.fcst_fitted_values_ = fcst.fcst_fitted_values_.drop(columns=["h"])

    with pytest.raises(ValueError, match="missing horizon information"):
        fcst.forecast_fitted_values(h=2)


def test_forecast_fitted_values_positional_level_compat():
    df = generate_daily_series(2, min_length=40, max_length=40)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
    )
    fcst.fit(df, fitted=True, static_features=[])

    positional = fcst.forecast_fitted_values([80])
    keyword = fcst.forecast_fitted_values(level=[80])

    pd.testing.assert_frame_equal(positional, keyword)


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


@pytest.mark.parametrize("refit", [True, False])
def test_cv_weight_col(refit):
    """Test that cross_validation works with weight_col and weights are used.

    When refit=False, this is a regression test for issue #497 where the
    second CV window would fail because weight_col was not passed to new_ts._fit().
    Also verifies that weights actually affect predictions.
    """
    from sklearn.linear_model import Ridge

    series = generate_daily_series(2, min_length=100, max_length=100)

    fcst = MLForecast(
        models={'lr': Ridge()},
        freq='D',
        lags=[1],
        date_features=['dayofweek'],
    )

    # Run with uniform weights
    series['weight'] = 1.0
    result_uniform = fcst.cross_validation(
        series,
        weight_col='weight',
        static_features=[],
        n_windows=2,
        h=7,
        refit=refit,
    )

    # Run with skewed weights (higher weight on recent samples)
    series['weight'] = np.arange(len(series), dtype=float)
    result_skewed = fcst.cross_validation(
        series,
        weight_col='weight',
        static_features=[],
        n_windows=2,
        h=7,
        refit=refit,
    )

    assert result_uniform.shape[0] == 2 * 2 * 7  # 2 series * 2 windows * 7 horizon
    assert result_skewed.shape[0] == 2 * 2 * 7
    # Predictions should differ when weights change, proving weights are used
    assert not np.allclose(result_uniform['lr'].values, result_skewed['lr'].values)


@pytest.mark.parametrize("max_horizon", [None, 2])
def test_cv_refit_false_with_grouped_expanding_mean(max_horizon, grouped_expanding_mean_df):
    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D",
        lags=[1],
        lag_transforms={1: [ExpandingMean(groupby=["cat_code"])]},
    )

    cv = fcst.cross_validation(
        grouped_expanding_mean_df,
        n_windows=2,
        h=2,
        static_features=["cat_code"],
        refit=False,
        max_horizon=max_horizon,
    )

    assert not cv.empty
    assert "LinearRegression" in cv.columns
    assert cv["LinearRegression"].notna().all()


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
    with pytest.raises(ValueError, match="freq"):
        fcst_wrong_freq.cross_validation(df_wrong_freq, n_windows=1, h=1)


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

    base_cfg = dict(
        models=[LinearRegression(), lgb.LGBMRegressor(verbosity=-1)],
        lags=[1, 2],
        lag_transforms={
            1: [ExpandingMean()],
            2: [ExpandingMean()],
        },
        date_features=['day', 'month', 'week', 'year'],
        target_transforms=[Differences([1, 2]), LocalStandardScaler()],
    )
    cfg_pl = dict(**base_cfg, freq='1d')
    cfg_pd = dict(**base_cfg, freq='D')
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
        'cfg_pl': cfg_pl,
        'cfg_pd': cfg_pd,
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

    fcst_pl = MLForecast(**data['cfg_pl'])
    fcst_pl.fit(data['series_pl'], max_horizon=max_horizon, as_numpy=as_numpy, **data['fit_kwargs'])
    fitted_pl = fcst_pl.forecast_fitted_values()
    if max_horizon is not None:
        fitted_pl = pl.concat(
            [fcst_pl.forecast_fitted_values(h=hidx) for hidx in range(1, max_horizon + 1)]
        )
    preds_pl = fcst_pl.predict(X_df=data['prices_pl'], **data['predict_kwargs'])
    preds_pl_subset = fcst_pl.predict(X_df=data['prices_pl'], ids=fcst_pl.ts.uids[[0, 6]], **data['predict_kwargs'])
    cv_pl = fcst_pl.cross_validation(data['series_pl'], as_numpy=as_numpy, **data['cv_kwargs'])
    cv_fitted_pl = fcst_pl.cross_validation_fitted_values()

    fcst_pd = MLForecast(**data['cfg_pd'])
    fcst_pd.fit(data['series_pd'], max_horizon=max_horizon, as_numpy=as_numpy, **data['fit_kwargs'])
    fitted_pd = fcst_pd.forecast_fitted_values()
    if max_horizon is not None:
        fitted_pd = pd.concat(
            [fcst_pd.forecast_fitted_values(h=hidx) for hidx in range(1, max_horizon + 1)],
            ignore_index=True,
        )
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


@pytest.mark.parametrize("horizons", [
    [1, 3],
    [1, 3, 5, 7],
    [1, 7],
    [2, 4, 6],
    [1, 2, 3],  # contiguous horizons
])
def test_direct_forecasting_perfect_exogenous_fit_with_horizons(horizons):
    """Test that sparse horizons correctly use exogenous features for each trained horizon.

    This test validates that when using the `horizons` parameter:
    1. Predictions are only made for the specified horizons
    2. Each horizon's prediction correctly uses its aligned exogenous features
    3. The output timestamps (ds) are correct for the sparse horizons

    With y = X (perfect linear relationship), predictions should equal the
    exogenous feature values at the corresponding horizon timestamps.
    """
    max_h = max(horizons)
    last_train_ds = 10

    # Create training data where y = X (perfect linear relationship)
    df = pd.DataFrame({
        "ds": np.arange(last_train_ds + 1),
        "X": np.arange(last_train_ds + 1),
        "unique_id": "ex",
        "y": np.arange(last_train_ds + 1),
    })

    # Create test dataframe with exogenous features for all possible prediction timestamps
    test_df = pd.DataFrame({
        "ds": np.arange(last_train_ds + 1, last_train_ds + 1 + max_h),
        "X": np.arange(last_train_ds + 1, last_train_ds + 1 + max_h),
        "unique_id": "ex",
    })

    # Fit with sparse horizons
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
    )
    fcst.fit(df, static_features=[], horizons=horizons)

    # Verify correct number of models trained
    assert len(fcst.models_['LinearRegression']) == len(horizons)
    # Models should be stored with 0-indexed keys
    expected_keys = {h - 1 for h in horizons}
    assert set(fcst.models_['LinearRegression'].keys()) == expected_keys

    # Predict using exogenous features
    preds = fcst.predict(h=max_h, X_df=test_df)

    # Verify correct number of predictions (one per horizon)
    assert preds.shape[0] == len(horizons)

    # Verify timestamps are correct
    expected_ds = [last_train_ds + h for h in horizons]
    np.testing.assert_array_equal(
        preds['ds'].values,
        expected_ds,
        err_msg=f"Timestamps mismatch for horizons={horizons}"
    )

    # Verify predictions equal the exogenous feature values at the correct timestamps
    # For y = X, prediction at ds=t should equal X[t] = t
    expected_values = np.array(expected_ds, dtype=float)
    np.testing.assert_allclose(
        preds['LinearRegression'].values,
        expected_values,
        rtol=1e-10,
        err_msg=f"Predictions do not match expected values for horizons={horizons}"
    )


def test_direct_forecasting_perfect_exogenous_fit_horizons_vs_max_horizon():
    """Test that horizons=[1,2,3] produces identical results to max_horizon=3.

    This validates that the new horizons parameter, when given contiguous horizons,
    produces the same predictions as the equivalent max_horizon parameter.
    """
    last_train_ds = 10
    H = 5

    # Create training data where y = X
    df = pd.DataFrame({
        "ds": np.arange(last_train_ds + 1),
        "X": np.arange(last_train_ds + 1),
        "unique_id": "ex",
        "y": np.arange(last_train_ds + 1),
    })

    # Create test dataframe
    test_df = pd.DataFrame({
        "ds": np.arange(last_train_ds + 1, last_train_ds + 1 + H),
        "X": np.arange(last_train_ds + 1, last_train_ds + 1 + H),
        "unique_id": "ex",
    })

    # Fit with max_horizon
    fcst_max = MLForecast(models=[LinearRegression()], freq=1)
    fcst_max.fit(df, static_features=[], max_horizon=H)
    preds_max = fcst_max.predict(h=H, X_df=test_df)

    # Fit with equivalent horizons
    fcst_horizons = MLForecast(models=[LinearRegression()], freq=1)
    fcst_horizons.fit(df, static_features=[], horizons=list(range(1, H + 1)))
    preds_horizons = fcst_horizons.predict(h=H, X_df=test_df)

    # Results should be identical
    pd.testing.assert_frame_equal(preds_max, preds_horizons)


def test_direct_forecasting_perfect_exogenous_fit_multiple_series():
    """Test sparse horizons with multiple time series.

    Validates that sparse horizons work correctly when forecasting multiple series,
    with correct timestamps and predictions for each series.
    """
    horizons = [1, 3, 5]
    max_h = max(horizons)
    n_series = 5

    # Generate series structure using generate_daily_series
    df = generate_daily_series(n_series, min_length=20, max_length=20, equal_ends=True)

    # Add exogenous feature X and set y = X for perfect linear relationship
    # Use cumcount within each series to create unique X values
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    df['X'] = df.groupby('unique_id', observed=True).cumcount().astype(float)
    df['y'] = df['X']

    # Fit with sparse horizons
    fcst = MLForecast(models=[LinearRegression()], freq='D')
    fcst.fit(df, static_features=[], horizons=horizons)

    # Create future dataframe using make_future_dataframe
    future_df = fcst.make_future_dataframe(h=max_h)

    # Add exogenous feature X to future dataframe
    # X continues from where training left off (20, 21, 22, ...)
    last_train_idx = df.groupby('unique_id', observed=True)['X'].transform('max')
    future_df = future_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    future_df['X'] = future_df.groupby('unique_id', observed=True).cumcount() + 20.0

    preds = fcst.predict(h=max_h, X_df=future_df)

    # Should have len(horizons) predictions per series
    assert preds.shape[0] == n_series * len(horizons)

    # Get last dates from training data
    last_dates = df.groupby('unique_id', observed=True)['ds'].max()

    # Verify each series has correct predictions
    for uid in df['unique_id'].unique():
        series_preds = preds[preds['unique_id'] == uid].sort_values('ds')
        assert series_preds.shape[0] == len(horizons)

        # Verify timestamps are correct for sparse horizons
        expected_dates = [last_dates[uid] + pd.Timedelta(days=h) for h in horizons]
        assert series_preds['ds'].tolist() == expected_dates

        # Verify predictions match expected X values (y = X)
        # X values at horizons [1, 3, 5] are [20, 22, 24] (0-indexed from end of training)
        expected_values = np.array([20 + h - 1 for h in horizons], dtype=float)
        np.testing.assert_allclose(
            series_preds['LinearRegression'].values,
            expected_values,
            rtol=1e-10
        )


# Tests for horizons parameter
def test_horizons_basic():
    """Test that horizons=[7, 14] creates exactly 2 models."""
    df = generate_daily_series(10, min_length=50, max_length=100)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    fcst.fit(df, horizons=[7, 14])

    # Verify only 2 models trained per model name
    assert len(fcst.models_['LinearRegression']) == 2
    # Models are stored as dict keyed by 0-indexed horizon
    assert set(fcst.models_['LinearRegression'].keys()) == {6, 13}


def test_horizons_mutual_exclusivity():
    """Test error when both max_horizon and horizons provided."""
    df = generate_daily_series(10, min_length=50, max_length=100)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    with pytest.raises(ValueError, match="Cannot specify both"):
        fcst.fit(df, max_horizon=5, horizons=[7, 14])


def test_horizons_validation_empty():
    """Test error for empty horizons list."""
    df = generate_daily_series(10, min_length=50, max_length=100)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        fcst.fit(df, horizons=[])


def test_horizons_validation_non_positive():
    """Test error for non-positive horizons."""
    df = generate_daily_series(10, min_length=50, max_length=100)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    with pytest.raises(ValueError, match="positive integers"):
        fcst.fit(df, horizons=[0, 7, 14])

    with pytest.raises(ValueError, match="positive integers"):
        fcst.fit(df, horizons=[-1, 7])


def test_horizons_validation_non_integer():
    """Test error for non-integer horizons."""
    df = generate_daily_series(10, min_length=50, max_length=100)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    with pytest.raises(ValueError, match="positive integers"):
        fcst.fit(df, horizons=[7.5, 14])


def test_horizons_duplicate_handling():
    """Test that horizons=[7, 7, 14] deduplicates to [7, 14]."""
    df = generate_daily_series(10, min_length=50, max_length=100)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    fcst.fit(df, horizons=[7, 7, 14, 14])

    # Should have exactly 2 models after deduplication
    assert len(fcst.models_['LinearRegression']) == 2
    assert set(fcst.models_['LinearRegression'].keys()) == {6, 13}


def test_horizons_prediction():
    """Test that predict(h=14) with horizons=[7, 14] works correctly."""
    df = generate_daily_series(10, min_length=50, max_length=100)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    fcst.fit(df, horizons=[7, 14])

    # Predict for horizon 14
    preds = fcst.predict(h=14)

    # Should only have predictions for the trained horizons (7 and 14)
    # Each series gets 2 predictions
    n_series = df['unique_id'].nunique()
    assert preds.shape[0] == n_series * 2

    # Check that dates are correct (at h=7 and h=14 from last date)
    last_dates = df.groupby('unique_id', observed=True)['ds'].max()
    for uid in df['unique_id'].unique():
        uid_preds = preds[preds['unique_id'] == uid]
        expected_dates = [
            last_dates[uid] + pd.Timedelta(days=7),
            last_dates[uid] + pd.Timedelta(days=14),
        ]
        assert uid_preds['ds'].tolist() == expected_dates


def test_horizons_prediction_partial():
    """Test that predict(h=10) with horizons=[7, 14] only returns h=7."""
    df = generate_daily_series(10, min_length=50, max_length=100)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    fcst.fit(df, horizons=[7, 14])

    # Predict for horizon 10 (only h=7 is available)
    preds = fcst.predict(h=10)

    # Should only have predictions for h=7
    n_series = df['unique_id'].nunique()
    assert preds.shape[0] == n_series * 1


def test_horizons_cross_validation():
    """Test that cross_validation works with sparse horizons parameter.

    Note: With sparse horizons, predictions are only made for trained horizons,
    so the CV result will have fewer rows than the validation set. The join
    filters to only matching timestamps.
    """
    df = generate_daily_series(10, min_length=100, max_length=150)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )

    # Use max_horizon=14 which is equivalent to horizons=[1,2,...,14]
    # This should work the same as before
    cv_results = fcst.cross_validation(
        df,
        n_windows=2,
        h=14,
        max_horizon=14,
    )

    # Should have predictions for all 14 horizons per series per window
    assert cv_results.shape[0] > 0
    assert 'LinearRegression' in cv_results.columns


def test_horizons_backward_compatibility():
    """Test that max_horizon=5 works identically to before."""
    df = generate_daily_series(10, min_length=50, max_length=100)

    # Fit with max_horizon
    fcst1 = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    fcst1.fit(df, max_horizon=5)
    preds1 = fcst1.predict(h=5)

    # Models should be stored as dict with keys 0, 1, 2, 3, 4
    assert isinstance(fcst1.models_['LinearRegression'], dict)
    assert set(fcst1.models_['LinearRegression'].keys()) == {0, 1, 2, 3, 4}

    # Predictions should have 5 rows per series
    n_series = df['unique_id'].nunique()
    assert preds1.shape[0] == n_series * 5


def test_horizons_with_exogenous():
    """Test horizons with exogenous features."""
    df = generate_daily_series(5, min_length=50, max_length=60)
    df['exog'] = np.random.randn(len(df))

    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7],
    )
    fcst.fit(df, horizons=[7, 14], static_features=[])

    # Create future exogenous
    future_df = fcst.make_future_dataframe(h=14)
    future_df['exog'] = np.random.randn(len(future_df))

    preds = fcst.predict(h=14, X_df=future_df)
    assert preds.shape[0] > 0


@pytest.fixture(params=["pandas", "polars"])
def horizon_features_data(request):
    """Horizon-specific exogenous train/future data in pandas or polars format.

    Builds the dataframe in pandas (NaN assignment is straightforward), then
    converts to polars when needed so both formats share identical values.
    """
    H = 3
    n = 50
    df_pd = generate_daily_series(1, min_length=n, max_length=n)
    df_pd["bookings_horizon_1"] = df_pd["y"] * 0.9
    df_pd["bookings_horizon_2"] = df_pd["y"] * 0.7
    df_pd["bookings_horizon_3"] = df_pd["y"] * 0.5
    # Introduce NaNs that horizon routing must handle: h1 missing for last 2
    # rows, h2 missing for the last row.
    df_pd.iloc[-2:, df_pd.columns.get_loc("bookings_horizon_1")] = None
    df_pd.iloc[-1, df_pd.columns.get_loc("bookings_horizon_2")] = None

    exog_cols = ["unique_id", "ds", "bookings_horizon_1", "bookings_horizon_2", "bookings_horizon_3"]
    horizon_features = {
        1: ["bookings_horizon_1"],
        2: ["bookings_horizon_2"],
        3: ["bookings_horizon_3"],
    }

    if request.param == "pandas":
        train = df_pd.iloc[:-H]
        future = df_pd.iloc[-H:][exog_cols]
        freq = "D"
    else:
        df_pl = pl.from_pandas(df_pd)  # pandas NaN → polars null automatically
        train = df_pl.head(n - H)
        future = df_pl.tail(H).select(exog_cols)
        freq = "1d"

    return {
        "train": train,
        "future": future,
        "H": H,
        "freq": freq,
        "horizon_features": horizon_features,
    }


def _to_pandas(df):
    """Normalise a pandas or polars DataFrame to pandas for assertions."""
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def test_horizon_feature_templates_route_horizon_specific_exog(horizon_features_data):
    """Ensure direct models can route horizon-scoped exogenous features via templates."""
    data = horizon_features_data
    H, freq = data["H"], data["freq"]
    train, future = data["train"], data["future"]

    baseline = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    baseline.fit(train, static_features=[], max_horizon=H)
    with pytest.raises(ValueError, match="Input X contains NaN"):
        baseline.predict(h=H, X_df=future)

    templated = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    templated.fit(
        train,
        static_features=[],
        max_horizon=H,
        horizon_feature_templates=["bookings_horizon_{h}"],
    )
    assert templated.horizon_features_ == data["horizon_features"]
    preds = _to_pandas(templated.predict(h=H, X_df=future))
    assert preds.shape[0] == H
    assert preds["LinearRegression"].notna().all()


def test_horizon_features_route_horizon_specific_exog(horizon_features_data):
    """Ensure explicit horizon feature mappings are used directly and persist."""
    data = horizon_features_data
    H, freq = data["H"], data["freq"]
    train, future = data["train"], data["future"]

    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    fcst.fit(
        train,
        static_features=[],
        max_horizon=H,
        horizon_features=data["horizon_features"],
    )
    assert fcst.horizon_features_ == data["horizon_features"]
    preds = _to_pandas(fcst.predict(h=H, X_df=future))
    assert preds.shape[0] == H
    assert preds["LinearRegression"].notna().all()

    with tempfile.TemporaryDirectory() as tmpdir:
        savedir = Path(tmpdir) / "fcst"
        savedir.mkdir()
        fcst.save(savedir)
        loaded = MLForecast.load(savedir)
    assert loaded.horizon_features_ == fcst.horizon_features_
    preds_loaded = _to_pandas(loaded.predict(h=H, X_df=future))
    pd.testing.assert_frame_equal(preds, preds_loaded)


@pytest.mark.parametrize("df_engine", ["pandas", "polars"])
def test_horizon_feature_templates_require_direct_mode(df_engine):
    """Templates are only valid for direct forecasting setups."""
    df = generate_daily_series(2, min_length=30, max_length=30, engine=df_engine)
    freq = "1d" if df_engine == "polars" else "D"
    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    with pytest.raises(ValueError, match="only supported when using `max_horizon` or `horizons`"):
        fcst.fit(df, static_features=[], horizon_feature_templates=["bookings_horizon_{h}"])


@pytest.mark.parametrize("df_engine", ["pandas", "polars"])
def test_horizon_features_require_direct_mode(df_engine):
    """Explicit horizon feature mappings are only valid for direct forecasting setups."""
    df = generate_daily_series(2, min_length=30, max_length=30, engine=df_engine)
    freq = "1d" if df_engine == "polars" else "D"
    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    with pytest.raises(ValueError, match="only supported when using `max_horizon` or `horizons`"):
        fcst.fit(df, static_features=[], horizon_features={1: ["bookings_horizon_1"]})


@pytest.mark.parametrize("df_engine", ["pandas", "polars"])
def test_horizon_features_with_weights_and_fitted_values(df_engine):
    """Direct fitted values should work when weights and horizon-specific inputs are both used."""
    H = 3
    n = 50
    df_pd = generate_daily_series(1, min_length=n, max_length=n)
    df_pd["weight"] = np.linspace(1.0, 2.0, n)
    df_pd["bookings_horizon_1"] = df_pd["y"] * 0.9
    df_pd["bookings_horizon_2"] = df_pd["y"] * 0.7
    df_pd["bookings_horizon_3"] = df_pd["y"] * 0.5

    if df_engine == "polars":
        train = pl.from_pandas(df_pd).head(n - H)
        freq = "1d"
    else:
        train = df_pd.iloc[:-H]
        freq = "D"

    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    fcst.fit(
        train,
        static_features=[],
        max_horizon=H,
        horizon_features={
            1: ["bookings_horizon_1"],
            2: ["bookings_horizon_2"],
            3: ["bookings_horizon_3"],
        },
        weight_col="weight",
        fitted=True,
    )
    fitted = fcst.forecast_fitted_values()
    assert fitted.shape[0] > 0
    assert "LinearRegression" in _to_pandas(fitted).columns


@pytest.mark.parametrize("df_engine", ["pandas", "polars"])
def test_horizon_features_transfer_learning_predict(df_engine):
    """Transfer-learning predictions should preserve horizon-specific feature routing."""
    H = 3
    n = 80
    df_pd = generate_daily_series(1, min_length=n, max_length=n)
    df_pd["bookings_horizon_1"] = df_pd["y"] * 0.9
    df_pd["bookings_horizon_2"] = df_pd["y"] * 0.7
    df_pd["bookings_horizon_3"] = df_pd["y"] * 0.5
    df_pd.iloc[-2:, df_pd.columns.get_loc("bookings_horizon_1")] = None
    df_pd.iloc[-1, df_pd.columns.get_loc("bookings_horizon_2")] = None

    exog_cols = ["unique_id", "ds", "bookings_horizon_1", "bookings_horizon_2", "bookings_horizon_3"]

    if df_engine == "polars":
        df = pl.from_pandas(df_pd)
        base_train = df.head(n - 2 * H)
        transfer_df = df.slice(n - 2 * H, H)
        future = df.tail(H).select(exog_cols)
        freq = "1d"
    else:
        df = df_pd
        base_train = df.iloc[: -2 * H]
        transfer_df = df.iloc[-2 * H : -H]
        future = df.iloc[-H:][exog_cols]
        freq = "D"

    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    fcst.fit(
        base_train,
        static_features=[],
        max_horizon=H,
        horizon_features={
            1: ["bookings_horizon_1"],
            2: ["bookings_horizon_2"],
            3: ["bookings_horizon_3"],
        },
    )
    preds = _to_pandas(fcst.predict(h=H, new_df=transfer_df, X_df=future))
    assert preds.shape[0] == H
    assert preds["LinearRegression"].notna().all()


def test_horizons_with_target_transforms():
    """Test that sparse horizons work correctly with target transforms.

    This is a regression test for the bug where inverse_transform failed
    because indptr was computed using the requested horizon instead of
    the actual number of predictions (output_horizon).
    """
    from mlforecast.target_transforms import Differences

    df = generate_daily_series(5, min_length=50, max_length=60)

    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7, 14],
        target_transforms=[Differences([7])],
    )
    fcst.fit(df, horizons=[7, 14])

    # This should not raise an error about indptr/data size mismatch
    preds = fcst.predict(h=14)

    # Should have 2 predictions per series (h=7 and h=14)
    n_series = df['unique_id'].nunique()
    assert preds.shape[0] == n_series * 2


def test_horizons_cross_validation_with_target_transforms():
    """Test cross_validation with sparse horizons and target transforms.

    This is a regression test for the bug where cross_validation failed
    with sparse horizons because:
    1. inverse_transform used wrong indptr size
    2. result row count check didn't account for sparse output
    """
    from mlforecast.target_transforms import Differences

    df = generate_daily_series(5, min_length=100, max_length=120)

    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7, 14],
        target_transforms=[Differences([7])],
    )

    # This should not raise any errors
    cv_results = fcst.cross_validation(
        df,
        n_windows=2,
        h=14,
        horizons=[7, 14],
    )

    # Should have 2 predictions per series per window
    n_series = df['unique_id'].nunique()
    n_windows = 2
    n_horizons = 2  # [7, 14]
    assert cv_results.shape[0] == n_series * n_windows * n_horizons


def test_polars_catboost_cross_validation():
    catboost = pytest.importorskip("catboost")
    df = generate_daily_series(
        3,
        min_length=80,
        max_length=80,
        equal_ends=True,
        engine="polars",
    )
    fcst = MLForecast(
        models=[catboost.CatBoostRegressor(iterations=10, verbose=0)],
        freq="1d",
        lags=[1, 2, 7],
        date_features=["weekday"],
    )

    cv_results = fcst.cross_validation(
        df=df,
        h=7,
        n_windows=2,
        step_size=7,
        refit=False,
    )

    n_series = df["unique_id"].n_unique()
    assert cv_results.shape[0] == n_series * 2 * 7


def test_polars_catboost_horizons_cross_validation():
    catboost = pytest.importorskip("catboost")
    df = generate_daily_series(
        3,
        min_length=100,
        max_length=100,
        equal_ends=True,
        engine="polars",
    )
    fcst = MLForecast(
        models=[catboost.CatBoostRegressor(iterations=10, verbose=0)],
        freq="1d",
        lags=[1, 7, 14],
    )

    cv_results = fcst.cross_validation(
        df=df,
        h=14,
        horizons=[7, 14],
        n_windows=2,
        refit=False,
    )

    n_series = df["unique_id"].n_unique()
    assert cv_results.shape[0] == n_series * 2 * 2


def test_horizons_cross_validation_polars_refit_false_as_numpy():
    df = generate_daily_series(
        4,
        min_length=100,
        max_length=100,
        equal_ends=True,
        engine="polars",
    )
    fcst = MLForecast(
        models=[LinearRegression()],
        freq="1d",
        lags=[1, 7, 14],
    )

    cv_results = fcst.cross_validation(
        df=df,
        n_windows=2,
        h=14,
        horizons=[7, 14],
        refit=False,
        as_numpy=True,
    )

    n_series = df["unique_id"].n_unique()
    assert cv_results.shape[0] == n_series * 2 * 2
    assert "LinearRegression" in cv_results.columns


def test_fit_with_validate_data_clean():
    """Test that fit with validate_data=True works on clean data without warnings."""
    # Use equal_ends=True to ensure all series have the same length
    df = generate_daily_series(3, min_length=50, max_length=50, equal_ends=True)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7]
    )

    # Should not produce any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        fcst.fit(df, validate_data=True)

    # Model should be fitted
    assert hasattr(fcst, 'models_')
    assert len(fcst.models_) > 0


def test_fit_with_validate_data_duplicates():
    """Test that fit with validate_data=True raises ValueError on duplicates."""
    df = generate_daily_series(3, min_length=50, max_length=100)

    # Add duplicate rows
    duplicate_rows = df.head(5).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7]
    )

    # Duplicates are a hard error - raises ValueError and stops execution
    with pytest.raises(ValueError, match="duplicate"):
        fcst.fit(df, validate_data=True)


def test_fit_with_validate_data_missing_dates():
    """Test that fit with validate_data=True warns about missing dates but continues."""
    # Create data with gaps
    df = pd.DataFrame({
        'unique_id': ['A'] * 10 + ['B'] * 8,
        'ds': pd.date_range('2020-01-01', periods=10, freq='D').tolist() +
             [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'),
              pd.Timestamp('2020-01-04'), pd.Timestamp('2020-01-05'),
              pd.Timestamp('2020-01-06'), pd.Timestamp('2020-01-08'),
              pd.Timestamp('2020-01-09'), pd.Timestamp('2020-01-10')],
        'y': list(range(18))
    })

    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1]
    )

    # Should produce error about missing dates
    with pytest.raises(ValueError, match="missing"):
        fcst.fit(df, validate_data=True)


def test_fit_validate_data_default_true():
    """Test that validate_data defaults to True - raises on duplicates by default."""
    # Create data with duplicate rows
    df = generate_daily_series(3, min_length=50, max_length=100)
    duplicate_rows = df.head(5).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7]
    )

    # Should raise by default (validate_data=True)
    with pytest.raises(ValueError, match="duplicate"):
        fcst.fit(df)

    # Should succeed when validation is explicitly disabled
    fcst2 = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 7]
    )
    fcst2.fit(df, validate_data=False)
    assert hasattr(fcst2, 'models_')


def test_preprocess_with_validate_data():
    """Test that preprocess method honors validate_data parameter."""
    df = pd.DataFrame({
        'unique_id': ['A'] * 10,
        'ds': pd.date_range('2020-01-01', periods=10, freq='D'),
        'y': list(range(10))
    })

    # Add duplicate rows
    duplicate_rows = df.head(2).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1]
    )

    # Should raise ValueError when validate_data=True and duplicates are present
    with pytest.raises(ValueError, match="duplicate"):
        fcst.preprocess(df, validate_data=True)

    # Should not produce warning when validate_data=False (default)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fcst.preprocess(df, validate_data=False)


def test_cross_validation_with_validate_data():
    """Test that cross_validation method honors validate_data parameter."""
    # Create data with gaps
    df = pd.DataFrame({
        'unique_id': ['A'] * 50 + ['B'] * 48,
        'ds': pd.date_range('2020-01-01', periods=50, freq='D').tolist() +
             [pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
              for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                        39, 40, 41, 42, 43, 44, 45, 46, 47, 49]],  # Missing day 48
        'y': list(range(98))
    })

    fcst = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1]
    )

    # Should produce warning when validate_data=True
    with pytest.raises(ValueError, match="missing"):
        cv_results = fcst.cross_validation(
            df,
            n_windows=2,
            h=5,
            validate_data=True
        )

    # Should not produce warning when validate_data=False
    fcst2 = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cv_results2 = fcst2.cross_validation(
            df,
            n_windows=2,
            h=5,
            validate_data=False
        )
def test_mlforecast_num_threads_minus_one():
    """Test that MLForecast correctly handles num_threads=-1."""
    import pandas as pd
    from joblib import cpu_count
    from sklearn.linear_model import LinearRegression

    from mlforecast import MLForecast
    from mlforecast.utils import generate_daily_series

    series = generate_daily_series(5, min_length=50, max_length=100, equal_ends=True)

    mlf_multi = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 2, 3],
        num_threads=-1,
    )
    assert mlf_multi.ts.num_threads == cpu_count()
    assert mlf_multi.ts.num_threads >= 1

    # Verify it works in fit and predict
    mlf_multi.fit(series)
    preds_multi = mlf_multi.predict(h=7)
    assert preds_multi is not None
    assert len(preds_multi) > 0

    # Compare with num_threads=1
    mlf_single = MLForecast(
        models=[LinearRegression()],
        freq='D',
        lags=[1, 2, 3],
        num_threads=1,
    )
    mlf_single.fit(series)
    preds_single = mlf_single.predict(h=7)

    pd.testing.assert_frame_equal(preds_multi, preds_single)


def test_transfer_learning_updates_uids():
    """Test that predict with new_df updates self.ts to reflect the new dataset."""
    train_a = generate_daily_series(3, min_length=50, max_length=50)
    train_b = generate_daily_series(
        2, min_length=50, max_length=50, n_static_features=0
    )
    # Make sure train_b has different uids
    train_b["unique_id"] = train_b["unique_id"].cat.rename_categories(
        {c: f"new_{c}" for c in train_b["unique_id"].cat.categories}
    )

    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D",
        lags=[1, 2, 3],
    )
    fcst.fit(train_a)
    original_uids = set(fcst.ts.uids)
    assert len(original_uids) == 3

    fcst.predict(h=5, new_df=train_b)
    new_uids = set(fcst.ts.uids)
    expected_uids = set(train_b["unique_id"].unique())
    assert new_uids == expected_uids
    assert new_uids != original_uids


def test_transfer_learning_failed_predict_keeps_original_state():
    """Test that transfer learning doesn't mutate state if prediction fails."""
    train_a = generate_daily_series(3, min_length=50, max_length=50, n_static_features=0)
    train_b = generate_daily_series(2, min_length=50, max_length=50, n_static_features=0)
    train_b["unique_id"] = train_b["unique_id"].cat.rename_categories(
        {c: f"new_{c}" for c in train_b["unique_id"].cat.categories}
    )

    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D",
        lags=[1, 2, 3],
    )
    fcst.fit(train_a)
    original_uids = set(fcst.ts.uids)

    with pytest.raises(ValueError, match="weren't seen during training"):
        fcst.predict(h=5, new_df=train_b, ids=["id_0"])

    assert set(fcst.ts.uids) == original_uids


def test_transfer_learning_then_intervals_raises_clear_error():
    """Test that intervals after transfer learning fail with a clear message."""
    train_a = generate_daily_series(3, min_length=50, max_length=50, n_static_features=0)
    train_b = generate_daily_series(2, min_length=50, max_length=50, n_static_features=0)
    train_b["unique_id"] = train_b["unique_id"].cat.rename_categories(
        {c: f"new_{c}" for c in train_b["unique_id"].cat.categories}
    )

    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D",
        lags=[1, 2, 3],
    )
    fcst.fit(train_a, prediction_intervals=PredictionIntervals(n_windows=2, h=5))
    fcst.predict(h=5, new_df=train_b)

    with pytest.raises(ValueError, match="calibrated on a different set of series"):
        fcst.predict(h=5, level=[90])


def test_transfer_learning_then_intervals_raises_with_same_series_count():
    """Intervals should fail if ids differ, even when the number of series matches."""
    train_a = generate_daily_series(2, min_length=50, max_length=50, n_static_features=0)
    train_b = generate_daily_series(2, min_length=50, max_length=50, n_static_features=0)
    train_b["unique_id"] = train_b["unique_id"].cat.rename_categories(
        {c: f"new_{c}" for c in train_b["unique_id"].cat.categories}
    )

    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D",
        lags=[1, 2, 3],
    )
    fcst.fit(train_a, prediction_intervals=PredictionIntervals(n_windows=2, h=5))
    fcst.predict(h=5, new_df=train_b)

    with pytest.raises(ValueError, match="calibrated on a different set of series"):
        fcst.predict(h=5, level=[90])


def test_transfer_learning_subset_predict_keeps_full_transform_state():
    """Subset forecasts after transfer learning shouldn't persist subset-mutated transforms."""
    train_a = generate_daily_series(2, min_length=60, max_length=60, n_static_features=0)
    train_b = generate_daily_series(2, min_length=60, max_length=60, n_static_features=0)
    train_b["unique_id"] = train_b["unique_id"].cat.rename_categories(
        {c: f"new_{c}" for c in train_b["unique_id"].cat.categories}
    )

    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D",
        lags=[1],
        lag_transforms={1: [ExpandingMean()]},
    )
    fcst.fit(train_a)
    expected_uids = set(train_b["unique_id"].unique())
    subset_id = [next(iter(expected_uids))]

    # Persist transfer state through a subset forecast first.
    fcst.predict(h=3, new_df=train_b, ids=subset_id)
    # A full forecast should still work for all transfer-learning ids.
    preds = fcst.predict(h=3)
    assert set(preds["unique_id"].unique()) == expected_uids


def test_transfer_learning_with_sparse_horizons():
    """Regression test: _horizons must be propagated to new_ts during transfer learning.

    Previously, predict(new_df=...) created a fresh TimeSeries object without copying
    _horizons, causing a KeyError when sparse horizons (e.g. [3, 10]) were used.
    """
    train_a = generate_daily_series(3, min_length=50, max_length=50, n_static_features=0)
    train_b = generate_daily_series(3, min_length=50, max_length=50, n_static_features=0)
    train_b["unique_id"] = train_b["unique_id"].cat.rename_categories(
        {c: f"new_{c}" for c in train_b["unique_id"].cat.categories}
    )

    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D",
        lags=[1, 2, 3],
    )
    fcst.fit(train_a, horizons=[3, 10])

    preds = fcst.predict(h=10, new_df=train_b)

    n_series = train_b["unique_id"].nunique()
    # Only horizons 3 and 10 should be returned
    assert preds.shape[0] == n_series * 2
    assert set(preds["unique_id"].unique()) == set(train_b["unique_id"].unique())


def test_cross_validation_with_horizon_features(horizon_features_data):
    """cross_validation() should work correctly with horizon_features."""
    data = horizon_features_data
    H, freq = data["H"], data["freq"]
    train = data["train"]
    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    cv = fcst.cross_validation(
        train,
        n_windows=2,
        h=H,
        static_features=[],
        max_horizon=H,
        horizon_features=data["horizon_features"],
    )
    cv_pd = _to_pandas(cv)
    assert cv_pd.shape[0] > 0
    assert "LinearRegression" in cv_pd.columns
    assert cv_pd["LinearRegression"].notna().all()


def test_horizon_features_all_horizon_specific():
    """Fit and predict should work when every exog column is horizon-specific (no common_exog)."""
    H = 2
    n = 40
    df = generate_daily_series(1, min_length=n, max_length=n)
    df["feat_h1"] = df["y"] * 0.8
    df["feat_h2"] = df["y"] * 0.6
    exog_cols = ["unique_id", "ds", "feat_h1", "feat_h2"]
    train = df.iloc[:-H]
    future = df.iloc[-H:][exog_cols]

    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    fcst.fit(
        train,
        static_features=[],
        max_horizon=H,
        horizon_features={1: ["feat_h1"], 2: ["feat_h2"]},
    )
    from mlforecast.core import TimeSeries  # noqa: F401 — verifying internal state
    exog_cols_all = fcst.ts._get_dynamic_exog_cols(fcst.ts.features_order_)
    common, _ = fcst.ts._split_horizon_exog_cols(exog_cols_all, fcst.horizon_features_)
    assert common == []
    preds = fcst.predict(h=H, X_df=future)
    assert _to_pandas(preds).shape[0] == H
    assert _to_pandas(preds)["LinearRegression"].notna().all()


@pytest.mark.parametrize("df_engine", ["pandas", "polars"])
def test_horizon_feature_templates_no_matches(df_engine):
    """A template that matches no columns should raise ValueError."""
    H = 2
    df = generate_daily_series(1, min_length=30, max_length=30, engine=df_engine)
    if df_engine == "polars":
        df = df.with_columns(pl.lit(1.0).alias("some_feature"))
        freq = "1d"
    else:
        df = df.copy()
        df["some_feature"] = 1.0
        freq = "D"
    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    with pytest.raises(ValueError, match="No dynamic exogenous columns matched"):
        fcst.fit(
            df,
            static_features=[],
            max_horizon=H,
            horizon_feature_templates=["nonexistent_{h}"],
        )


def test_horizon_features_save_load_templates(horizon_features_data):
    """Save/load round-trip should preserve horizon_features_ when fit via templates."""
    data = horizon_features_data
    H, freq = data["H"], data["freq"]
    train, future = data["train"], data["future"]

    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    fcst.fit(
        train,
        static_features=[],
        max_horizon=H,
        horizon_feature_templates=["bookings_horizon_{h}"],
    )
    preds = _to_pandas(fcst.predict(h=H, X_df=future))

    with tempfile.TemporaryDirectory() as tmpdir:
        savedir = Path(tmpdir) / "fcst"
        savedir.mkdir()
        fcst.save(savedir)
        loaded = MLForecast.load(savedir)

    assert loaded.horizon_features_ == fcst.horizon_features_
    preds_loaded = _to_pandas(loaded.predict(h=H, X_df=future))
    pd.testing.assert_frame_equal(preds, preds_loaded)

    # Loaded model should still validate missing horizon features
    if isinstance(future, pl.DataFrame):
        future_missing = future.drop("bookings_horizon_2")
    else:
        future_missing = future.drop(columns=["bookings_horizon_2"])
    with pytest.raises(ValueError, match="missing the following exogenous features"):
        loaded.predict(h=H, X_df=future_missing)


@pytest.mark.parametrize("df_engine", ["pandas", "polars"])
def test_horizon_feature_templates_conflicting_patterns(df_engine):
    """A column matching two templates at different horizon numbers should raise ValueError."""
    H = 2
    # "feat_1_h2" matches "feat_{h}_h2" -> h=1 AND "feat_1_h{h}" -> h=2 (conflicting)
    col_name = "feat_1_h2"
    df = generate_daily_series(1, min_length=30, max_length=30, engine=df_engine)
    if df_engine == "polars":
        df = df.with_columns(pl.lit(1.0).alias(col_name))
        freq = "1d"
    else:
        df = df.copy()
        df[col_name] = 1.0
        freq = "D"
    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    with pytest.raises(ValueError, match="matches multiple horizon templates with conflicting horizons"):
        fcst.fit(
            df,
            static_features=[],
            max_horizon=H,
            horizon_feature_templates=["feat_{h}_h2", "feat_1_h{h}"],
        )


def test_predict_x_df_missing_horizon_features(horizon_features_data):
    """predict() should raise a clear error when X_df is missing a horizon-specific feature."""
    data = horizon_features_data
    H, freq = data["H"], data["freq"]
    train, future = data["train"], data["future"]

    fcst = MLForecast(models=[LinearRegression()], freq=freq, lags=[1])
    fcst.fit(
        train,
        static_features=[],
        max_horizon=H,
        horizon_features=data["horizon_features"],
    )
    if isinstance(future, pl.DataFrame):
        future_missing = future.drop("bookings_horizon_2")
    else:
        future_missing = future.drop(columns=["bookings_horizon_2"])

    with pytest.raises(ValueError, match="missing the following exogenous features"):
        fcst.predict(h=H, X_df=future_missing)
