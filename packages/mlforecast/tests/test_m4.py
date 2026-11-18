import lightgbm as lgb
import pandas as pd
import pytest
from datasetsforecast.m4 import M4, M4Evaluation, M4Info
from sklearn.linear_model import ElasticNet

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExponentiallyWeightedMean,
    RollingMean,
)
from mlforecast.target_transforms import Differences

configs = {
    "Hourly": {
        "lgb_params": {
            "n_estimators": 200,
            "bagging_freq": 1,
            "learning_rate": 0.05,
            "num_leaves": 2500,
            "lambda_l1": 0.03,
            "lambda_l2": 0.5,
            "bagging_fraction": 0.9,
            "feature_fraction": 0.8,
        },
        "mlf_params": {
            "target_transforms": [Differences([24])],
            "lags": [24 * i for i in range(1, 15)],
            "lag_transforms": {
                24: [
                    ExponentiallyWeightedMean(alpha=0.3),
                    RollingMean(7 * 24),
                    RollingMean(7 * 48),
                ],
                48: [
                    ExponentiallyWeightedMean(alpha=0.3),
                    RollingMean(7 * 24),
                    RollingMean(7 * 48),
                ],
            },
        },
        "metrics": {
            "lgb": {
                "SMAPE": 10.206856,
                "MASE": 0.861700,
                "OWA": 0.457511,
            },
            "enet": {
                "SMAPE": 26.721835,
                "MASE": 22.954763,
                "OWA": 5.518959,
            },
        },
    },
    "Daily": {
        "lgb_params": {
            "n_estimators": 30,
            "num_leaves": 128,
        },
        "mlf_params": {
            "target_transforms": [Differences([1])],
            "lags": [i + 1 for i in range(14)],
            "lag_transforms": {
                7: [RollingMean(7)],
                14: [RollingMean(7)],
            },
        },
        "metrics": {
            "lgb": {
                "SMAPE": 2.984652,
                "MASE": 3.205519,
                "OWA": 0.978931,
            },
            "enet": {
                "SMAPE": 2.989489,
                "MASE": 3.221004,
                "OWA": 0.982087,
            },
        },
    },
    "Weekly": {
        "lgb_params": {
            "n_estimators": 100,
            "objective": "l1",
            "num_leaves": 256,
        },
        "mlf_params": {
            "target_transforms": [Differences([1])],
            "lags": [i + 1 for i in range(32)],
            "lag_transforms": {
                4: [ExpandingMean(), RollingMean(4)],
                8: [ExpandingMean(), RollingMean(4)],
            },
        },
        "metrics": {
            "lgb": {
                "SMAPE": 8.238175,
                "MASE": 2.222099,
                "OWA": 0.849666,
            },
            "enet": {
                "SMAPE": 9.794393,
                "MASE": 3.270274,
                "OWA": 1.123305,
            },
        },
    },
    "Yearly": {
        "lgb_params": {
            "n_estimators": 100,
            "objective": "l1",
            "num_leaves": 256,
        },
        "mlf_params": {
            "target_transforms": [Differences([1])],
            "lags": [i + 1 for i in range(6)],
            "lag_transforms": {
                1: [ExpandingMean()],
                6: [ExpandingMean()],
            },
        },
        "metrics": {
            "lgb": {
                "SMAPE": 13.281131,
                "MASE": 3.018999,
                "OWA": 0.786155,
            },
            "enet": {
                "SMAPE": 15.363430,
                "MASE": 3.953421,
                "OWA": 0.967420,
            },
        },
    },
}


def train_valid_split(group):
    df, *_ = M4.load(directory="data", group=group)
    df["ds"] = df["ds"].astype("int")
    horizon = M4Info[group].horizon
    valid = df.groupby("unique_id").tail(horizon)
    train = df.drop(valid.index)
    return train, valid, horizon


@pytest.mark.parametrize("group", configs.keys())
def test_performance(group):
    cfg = configs[group]
    train, _, horizon = train_valid_split(group)
    fcst = MLForecast(
        models={
            "lgb": lgb.LGBMRegressor(
                random_state=0, n_jobs=1, verbosity=-1, **cfg["lgb_params"]
            ),
            "enet": ElasticNet(),
        },
        freq=1,
        **cfg["mlf_params"],
        num_threads=2,
    )
    fcst.fit(train)
    preds = fcst.predict(horizon)
    for model, expected in cfg["metrics"].items():
        model_preds = preds[model].values.reshape(-1, horizon)
        model_eval = M4Evaluation.evaluate("data", group, model_preds).loc[group]
        pd.testing.assert_series_equal(model_eval, pd.Series(expected, name=group))
