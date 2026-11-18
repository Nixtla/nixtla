import os
from time import time
from typing import List, Tuple

os.environ["NIXTLA_NUMBA_RELEASE_GIL"] = "1"
os.environ["NIXTLA_NUMBA_CACHE"] = "1"

import fire
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoCES,
    DynamicOptimizedTheta,
    SeasonalNaive,
)

from src.utils import ExperimentHandler


def run_seasonal_naive(
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
    seasonality: int,
    level: List[int],
) -> Tuple[pd.DataFrame, float, str]:
    os.environ["NIXTLA_ID_AS_COL"] = "true"
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=seasonality)],
        freq=freq,
        n_jobs=-1,
    )
    model = sf
    init_time = time()
    fcsts_df = model.forecast(df=train_df, h=horizon, level=level)
    total_time = time() - init_time
    return fcsts_df, total_time, "SeasonalNaive"


def ensemble_forecasts(
    fcsts_df: pd.DataFrame,
    quantiles: List[float],
    name_models: List[str],
    model_name: str,
) -> pd.DataFrame:
    fcsts_df[model_name] = fcsts_df[name_models].mean(axis=1).values  # type: ignore
    # compute quantiles based on the mean of the forecasts
    sigma_models = []
    for model in name_models:
        fcsts_df[f"sigma_{model}"] = fcsts_df[f"{model}-hi-68.27"] - fcsts_df[model]
        sigma_models.append(f"sigma_{model}")
    fcsts_df[f"std_{model_name}"] = (
        fcsts_df[sigma_models].pow(2).sum(axis=1).div(len(sigma_models) ** 2).pow(0.5)
    )
    z = norm.ppf(quantiles)
    q_cols = []
    for q, zq in zip(quantiles, z):
        q_col = f"{model_name}-q-{q}"
        fcsts_df[q_col] = fcsts_df[model_name] + zq * fcsts_df[f"std_{model_name}"]
        q_cols.append(q_col)
    fcsts_df = fcsts_df[["unique_id", "ds"] + [model_name] + q_cols]
    return fcsts_df


def run_statistical_ensemble(
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
    seasonality: int,
    quantiles: List[float],
) -> Tuple[pd.DataFrame, float, str]:
    os.environ["NIXTLA_ID_AS_COL"] = "true"
    models = [
        AutoARIMA(season_length=seasonality),
        AutoETS(season_length=seasonality),
        AutoCES(season_length=seasonality),
        DynamicOptimizedTheta(season_length=seasonality),
    ]
    init_time = time()
    series_per_core = 15
    n_series = train_df["unique_id"].nunique()
    n_jobs = min(n_series // series_per_core, os.cpu_count())
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=n_jobs,
    )
    fcsts_df = sf.forecast(df=train_df, h=horizon, level=[68.27])
    name_models = [repr(model) for model in models]
    model_name = "StatisticalEnsemble"
    fcsts_df = ensemble_forecasts(
        fcsts_df,
        quantiles,
        name_models,
        model_name,
    )
    total_time = time() - init_time
    return fcsts_df, total_time, model_name


def main(dataset: str):
    exp = ExperimentHandler(dataset)
    # seasonal naive benchmark
    fcst_df, total_time, model_name = run_seasonal_naive(
        train_df=exp.train_df,
        horizon=exp.horizon,
        freq=exp.freq,
        seasonality=exp.seasonality,
        level=exp.level,
    )
    fcst_df = exp.fcst_from_level_to_quantiles(fcst_df, model_name)
    exp.save_results(fcst_df, total_time, model_name)
    # statistical ensemble
    fcst_df, total_time, model_name = run_statistical_ensemble(
        train_df=exp.train_df,
        horizon=exp.horizon,
        freq=exp.freq,
        seasonality=exp.seasonality,
        quantiles=exp.quantiles,
    )
    exp.save_results(fcst_df, total_time, model_name)


if __name__ == "__main__":
    from statsforecast.utils import AirPassengers as ap

    AutoARIMA(season_length=12).forecast(ap.astype(np.float32), h=12)
    fire.Fire(main)
