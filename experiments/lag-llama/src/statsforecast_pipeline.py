import os
from time import time
from typing import List, Tuple

import fire
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from src.utils import ExperimentHandler


def run_statsforecast(
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
    seasonality: int,
    level: List[int],
) -> Tuple[pd.DataFrame, float, str]:
    os.environ["NIXTLA_ID_AS_COL"] = "true"
    models = [SeasonalNaive(season_length=seasonality)]
    init_time = time()
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=-1,
    )
    fcsts_df = sf.forecast(df=train_df, h=horizon, level=level)
    total_time = time() - init_time
    model_name = repr(models[0])
    return fcsts_df, total_time, model_name


def main(dataset: str):
    exp = ExperimentHandler(dataset)
    fcst_df, total_time, model_name = run_statsforecast(
        train_df=exp.train_df,
        horizon=exp.horizon,
        freq=exp.freq,
        seasonality=exp.seasonality,
        level=exp.level,
    )
    fcst_df = exp._fcst_from_level_to_quantiles(fcst_df, model_name)
    exp._save_results(fcst_df, total_time, model_name)


if __name__ == "__main__":
    fire.Fire(main)
