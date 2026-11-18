import os
from pathlib import Path
from time import time

import fire
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from src.utils.data_handler import ExperimentDataset, ForecastDataset


def sn_forecast(dataset_path: str, results_dir: str = "./results"):
    os.environ["NIXTLA_ID_AS_COL"] = "true"
    dataset = ExperimentDataset.from_parquet(parquet_path=dataset_path)
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=dataset.seasonality)],
        freq=dataset.pandas_frequency,
    )
    start = time()
    forecast_df = sf.forecast(
        df=dataset.Y_df_train,
        h=dataset.horizon,
    )
    end = time()
    total_time = end - start
    forecast_dataset = ForecastDataset(forecast_df=forecast_df, total_time=total_time)
    experiment_name = dataset_path.split("/")[-1].split(".")[0]
    results_path = Path(results_dir) / "statsforecast_sn" / experiment_name
    forecast_dataset.save_to_dir(results_path)


if __name__ == "__main__":
    fire.Fire(sn_forecast)
