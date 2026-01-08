import time

import optuna
import pandas as pd
import polars as pl
import pytest
from datasetsforecast.m4 import M4, M4Info
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from mlforecast.auto import (
    AutoLightGBM,
    AutoMLForecast,
    AutoModel,
    AutoRidge,
    PredictionIntervals,
    ridge_space,
)

from .conftest import assert_raises_with_message

optuna.logging.set_verbosity(optuna.logging.ERROR)


@pytest.fixture(scope="module")
def weekly_data():
    group = "Weekly"
    M4.async_download("data", group=group)
    df, *_ = M4.load(directory="data", group=group)
    df["ds"] = df["ds"].astype("int")
    horizon = M4Info[group].horizon
    valid = df.groupby("unique_id").tail(horizon).copy()
    train = df.drop(valid.index).reset_index(drop=True)
    train["unique_id"] = train["unique_id"].astype("category")
    valid["unique_id"] = valid["unique_id"].astype(train["unique_id"].dtype)
    return train, valid, M4Info[group]


def test_automlforecast_pipeline(weekly_data):
    train, valid, info = weekly_data
    h = info.horizon
    season_length = info.seasonality

    ridge_pipeline = make_pipeline(
        ColumnTransformer(
            [("encoder", OneHotEncoder(), ["unique_id"])],
            remainder="passthrough",
        ),
        Ridge(),
    )
    auto_ridge = AutoModel(
        ridge_pipeline,
        lambda trial: {f"ridge__{k}": v for k, v in ridge_space(trial).items()},
    )

    auto_mlf = AutoMLForecast(
        freq=1,
        season_length=season_length,
        models={"lgb": AutoLightGBM(), "ridge": auto_ridge},
        fit_config=lambda trial: {"static_features": ["unique_id"]},
        num_threads=2,
    )

    auto_mlf.fit(
        df=train,
        n_windows=2,
        h=h,
        num_samples=2,
        optimize_kwargs={"timeout": 60},
        fitted=True,
        prediction_intervals=PredictionIntervals(n_windows=2, h=h),
    )
    preds = auto_mlf.predict(h, level=[80])
    assert not preds.empty
    fitted_vals = auto_mlf.forecast_fitted_values(level=[95])
    assert not fitted_vals.empty
    
def test_automlforecast_weight_col(weekly_data):

    def custom_loss(val_df, train_df, weight_col):
        error = (val_df['y'] - val_df['model']).abs()
        indices = val_df.index
        weights = train.loc[indices, weight_col].values
        weighted_sum = sum(e * w for e, w in zip(error, weights))
        total_weight = sum(weights)

        weighted_error = weighted_sum / total_weight if total_weight != 0 else float('inf')
        return weighted_error

    train, valid, info = weekly_data
    h = info.horizon
    season_length = info.seasonality

    train['weights'] = 1

    auto_ridge = AutoModel(
        Ridge(),
        lambda trial: {
            'alpha': trial.suggest_float('alpha', 0.0, 1.0)  
        }
    )

    auto_mlf = AutoMLForecast(
        freq=1,
        season_length=season_length,
        models={"ridge": auto_ridge},  
        fit_config=lambda trial: {'static_features': []}
    )

    auto_mlf.fit(
        df=train,
        n_windows=2,
        h=h,
        num_samples=2,  
        loss=custom_loss, 
        weight_col='weights',  
        fitted=True
    )

    preds = auto_mlf.predict(h)
    assert not preds.empty
    fitted_vals = auto_mlf.forecast_fitted_values(level=[95])
    assert not fitted_vals.empty


def test_automlforecast_errors_and_warnings():
    assert_raises_with_message(
        lambda: AutoMLForecast(
            freq=1,
            season_length=None,
            init_config=None,
            models=[AutoLightGBM()],
        ),
        "`season_length` is required",
    )
    with pytest.warns(Warning, match="`season_length` is not used when `init_config` is provided."):
        AutoMLForecast(
            freq=1,
            season_length=1,
            init_config=lambda trial: {},  # noqa: ARG005
            models=[AutoLightGBM()],
        )


def test_polars_input_compatibility(weekly_data):
    train, _, info = weekly_data
    h = info.horizon
    season_length = info.seasonality
    train_pl = pl.from_pandas(train.astype({"unique_id": "str"}))

    auto_mlf = AutoMLForecast(
        freq=1,
        season_length=season_length,
        models={"ridge": AutoRidge()},
        num_threads=2,
    )

    auto_mlf.fit(
        df=train_pl,
        n_windows=2,
        h=h,
        num_samples=2,
        optimize_kwargs={"timeout": 60},
        fitted=True,
        prediction_intervals=PredictionIntervals(n_windows=2, h=h),
    )

    preds = auto_mlf.predict(h, level=[80])
    assert not preds.is_empty()
    auto_mlf.forecast_fitted_values(level=[95])


def test_step_size_impact(weekly_data):
    train, _, info = weekly_data
    h = info.horizon
    season_length = info.seasonality
    train_pl = pl.from_pandas(train.astype({"unique_id": "str"}))

    base = AutoMLForecast(
        freq=1,
        season_length=season_length,
        models={"ridge": AutoRidge()},
        num_threads=2,
    )
    base.fit(
        df=train_pl,
        n_windows=2,
        h=h,
        step_size=h,
        num_samples=2,
        optimize_kwargs={"timeout": 60},
        fitted=True,
        prediction_intervals=PredictionIntervals(n_windows=2, h=h),
    )
    base2 = AutoMLForecast(
        freq=1,
        season_length=season_length,
        models={"ridge": AutoRidge()},
        num_threads=2,
    )
    base2.fit(
        df=train_pl,
        n_windows=2,
        h=h,
        step_size=1,
        num_samples=2,
        optimize_kwargs={"timeout": 60},
        fitted=True,
        prediction_intervals=PredictionIntervals(n_windows=2, h=h),
    )
    val_h = base.results_["ridge"].best_trial.value
    val_1 = base2.results_["ridge"].best_trial.value
    assert abs(val_h / val_1 - 1) > 0.02


def test_nonstandard_column_names(weekly_data):
    train, _, info = weekly_data
    h = info.horizon
    season_length = info.seasonality

    fit_kwargs = dict(
        n_windows=2,
        h=h,
        step_size=1,
        num_samples=2,
        optimize_kwargs={"timeout": 60},
    )
    model = AutoMLForecast(
        freq=1, season_length=season_length, models={"ridge": AutoRidge()}
    )
    preds = model.fit(train, **fit_kwargs).predict(5)

    train2 = train.rename(columns={"unique_id": "id", "ds": "time", "y": "target"})
    preds2 = model.fit(
        df=train2,
        id_col="id",
        time_col="time",
        target_col="target",
        **fit_kwargs,
    ).predict(5)

    pd.testing.assert_frame_equal(
        preds,
        preds2.rename(columns={"id": "unique_id", "time": "ds"}),
    )


def test_input_size_speedup(weekly_data):
    train, _, info = weekly_data
    h = info.horizon
    season_length = info.seasonality
    model = AutoMLForecast(
        freq=1, season_length=season_length, models={"ridge": AutoRidge()}
    )
    fit_kwargs = dict(
        n_windows=3,
        h=h,
        num_samples=5,
        optimize_kwargs={"timeout": 60},
    )

    start = time.perf_counter()
    model.fit(df=train, **fit_kwargs)
    time_no_limit = time.perf_counter() - start

    start = time.perf_counter()
    model.fit(df=train, input_size=50, **fit_kwargs)
    time_with_limit = time.perf_counter() - start

    assert time_with_limit < time_no_limit
    

def test_reuse_cv_splits_same_predictions(weekly_data):
    train, valid, info = weekly_data
    h = info.horizon
    n_windows = 2
    num_samples = 5 
    
    def ridge_config(trial):  
        return {"alpha": 1.0, "fit_intercept": True, "solver": "svd"}
    def fit_config(trial):  
        return {"dropna": True}
    def init_config(trial):  
        return {
            "lags": [1, 2, 3],
        }
    ridge_auto = AutoModel(model=Ridge(), config=ridge_config)

    common_kwargs = dict(
        models={"ridge": ridge_auto},
        freq=1,
        fit_config=fit_config,
        init_config=init_config,
    )

    automl_a = AutoMLForecast(**common_kwargs, reuse_cv_splits=False).fit(
        train,
        n_windows=n_windows,
        h=h,
        num_samples=num_samples,
    )

    automl_b = AutoMLForecast(**common_kwargs, reuse_cv_splits=True).fit(
        train,
        n_windows=n_windows,
        h=h,
        num_samples=num_samples,
    )

    preds_a = automl_a.predict(h=h)
    preds_b = automl_b.predict(h=h)

    assert preds_a.columns.tolist() == preds_b.columns.tolist()
    assert (preds_a["ridge"].to_numpy() == preds_b["ridge"].to_numpy()).all()