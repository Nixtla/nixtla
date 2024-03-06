from pathlib import Path
from time import time

import fire
from dotenv import load_dotenv
from nixtlats import TimeGPT

from src.utils.data_handler import ExperimentDataset, ForecastDataset

load_dotenv()


def timegpt_forecast(dataset_path: str, results_dir: str = "./results"):
    dataset = ExperimentDataset.from_parquet(parquet_path=dataset_path)
    timegpt = TimeGPT(max_retries=1)
    start = time()
    forecast_df = timegpt.forecast(
        df=dataset.Y_df_train,
        h=dataset.horizon,
        freq=dataset.pandas_frequency,
        model="timegpt-1-long-horizon",
    )
    end = time()
    total_time = end - start
    forecast_dataset = ForecastDataset(
        forecast_df=forecast_df,
        total_time=total_time,
    )
    experiment_name = dataset_path.split("/")[-1].split(".")[0]
    results_path = Path(results_dir) / "nixtla_timegpt" / experiment_name
    forecast_dataset.save_to_dir(results_path)


if __name__ == "__main__":
    fire.Fire(timegpt_forecast)
