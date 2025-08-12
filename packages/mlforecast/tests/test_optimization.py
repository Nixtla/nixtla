import lightgbm as lgb
import optuna
import pandas as pd
import pytest
from datasetsforecast.m4 import M4, M4Evaluation, M4Info
from utilsforecast.losses import smape

from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.optimization import mlforecast_objective
from mlforecast.target_transforms import Differences, LocalBoxCox, LocalStandardScaler


@pytest.fixture(scope="module")
def weekly_data():
    group = "Weekly"
    import asyncio
    asyncio.run(M4.async_download("data", group=group))
    df, *_ = M4.load(directory="data", group=group)
    df["ds"] = df["ds"].astype("int")
    horizon = M4Info[group].horizon
    valid = df.groupby("unique_id").tail(horizon).copy()
    train = df.drop(valid.index)
    train["unique_id"] = train["unique_id"].astype("category")
    valid["unique_id"] = valid["unique_id"].astype(train["unique_id"].dtype)
    return train, valid, horizon


def config_fn(trial):
    candidate_lags = [
        [1],
        [13],
        [1, 13],
        range(1, 33),
    ]
    lag_idx = trial.suggest_categorical("lag_idx", range(len(candidate_lags)))
    candidate_lag_tfms = [
        {1: [RollingMean(window_size=13)]},
        {
            1: [RollingMean(window_size=13)],
            13: [RollingMean(window_size=13)],
        },
        {
            13: [RollingMean(window_size=13)],
        },
        {
            4: [ExpandingMean(), RollingMean(window_size=4)],
            8: [ExpandingMean(), RollingMean(window_size=4)],
        },
    ]
    lag_tfms_idx = trial.suggest_categorical("lag_tfms_idx", range(len(candidate_lag_tfms)))
    candidate_targ_tfms = [
        [Differences([1])],
        [LocalBoxCox()],
        [LocalStandardScaler()],
        [LocalBoxCox(), Differences([1])],
        [LocalBoxCox(), LocalStandardScaler()],
        [LocalBoxCox(), Differences([1]), LocalStandardScaler()],
    ]
    targ_tfms_idx = trial.suggest_categorical("targ_tfms_idx", range(len(candidate_targ_tfms)))
    return {
        "model_params": {
            "learning_rate": 0.05,
            "objective": "l1",
            "bagging_freq": 1,
            "num_threads": 2,
            "verbose": -1,
            "force_col_wise": True,
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 1024, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 10, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.75, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.75, 1.0),
        },
        "mlf_init_params": {
            "lags": candidate_lags[lag_idx],
            "lag_transforms": candidate_lag_tfms[lag_tfms_idx],
            "target_transforms": candidate_targ_tfms[targ_tfms_idx],
        },
        "mlf_fit_params": {
            "static_features": ["unique_id"],
        },
    }


def loss(df: pd.DataFrame, train_df: pd.DataFrame) -> float:  # noqa: ARG001
    return smape(df, models=["model"])["model"].mean()


def test_optuna_optimization_and_evaluation(weekly_data):
    train, valid, horizon = weekly_data
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective = mlforecast_objective(
        df=train,
        config_fn=config_fn,
        loss=loss,
        model=lgb.LGBMRegressor(),
        freq=1,
        n_windows=2,
        h=horizon,
    )
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=2)
    best_cfg = study.best_trial.user_attrs["config"]

    final_model = MLForecast(
        models=[lgb.LGBMRegressor(**best_cfg["model_params"])],
        freq=1,
        **best_cfg["mlf_init_params"],
    )
    final_model.fit(train, **best_cfg["mlf_fit_params"])
    preds = final_model.predict(horizon)

    # Get unique series count and reshape predictions accordingly
    n_series = train["unique_id"].nunique()
    pred_values = preds["LGBMRegressor"].values.reshape(n_series, horizon)

    results = M4Evaluation.evaluate("data", "Weekly", pred_values)
    assert isinstance(results, pd.DataFrame) and "SMAPE" in results.columns
