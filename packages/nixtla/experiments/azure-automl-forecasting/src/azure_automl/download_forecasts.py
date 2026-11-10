import logging
from pathlib import Path

import fire

from .automl_handler import AzureAutoML
from .forecasting import AzureAutoMLJobs
from src.utils.data_handler import ForecastDataset

logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)


def download_forecasts(dir: str = "./results"):
    azure_automl = AzureAutoML.from_environment()
    azure_automl_experiments = AzureAutoMLJobs()
    results_path = Path(dir) / "azure_automl"

    jobs_df = azure_automl_experiments.get_jobs_df()
    jobs_df = jobs_df.sort_values("created_at", ascending=False).drop_duplicates(
        "experiment_name"
    )

    for _, row in jobs_df.iterrows():
        experiment_name = row.experiment_name
        job_name = row.job_name
        main_logger.info(
            f"Downloading forecasts for experiment {experiment_name} and job {job_name}"
        )
        try:
            forecast_df = azure_automl.get_forecast_df(job_name)
            total_time = azure_automl.get_job_total_time(job_name)
        except Exception:
            main_logger.info(
                f"Failed to download forecasts for experiment {experiment_name} and job {job_name}"
            )
            continue
        if forecast_df is None:
            main_logger.info(
                f"Failed to download forecasts for experiment {experiment_name} and job {job_name}"
                "probably because the job is not finished yet or failed"
            )
            continue
        fcst_dataset = ForecastDataset(forecast_df=forecast_df, total_time=total_time)
        experiment_name = row.experiment_name
        fcst_dataset.save_to_dir(results_path / experiment_name)
        main_logger.info(
            f"Saved forecasts for experiment {experiment_name} and job {job_name}"
        )


if __name__ == "__main__":
    fire.Fire(download_forecasts)
