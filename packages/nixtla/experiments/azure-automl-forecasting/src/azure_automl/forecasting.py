from pathlib import Path

import fire
import pandas as pd
from azure.ai.ml.entities import AmlCompute

from .automl_handler import AzureAutoML
from src.utils.data_handler import ExperimentDataset


class AzureAutoMLJobs:
    """
    This class stores and updates the Azure AutoML Experiments,
    to keep track of the pipeline jobs.
    We need this to later downlaod the forecasts.
    """

    file_name = "forecasting_jobs.csv"

    def __init__(self, dir: str = "./azure_automl_results"):
        self.dir = dir
        self.jobs_path = Path(self.dir) / self.file_name
        self.setup()

    def setup(self):
        self.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.jobs_path.exists():
            pd.DataFrame(columns=["created_at", "experiment_name", "job_name"]).to_csv(
                self.jobs_path,
                index=False,
            )

    def get_jobs_df(self) -> pd.DataFrame:
        return pd.read_csv(self.jobs_path)

    def save_job(self, job_name: str, experiment_name: str):
        jobs_df = self.get_jobs_df()
        new_row = pd.DataFrame(
            {
                "created_at": [pd.Timestamp.now()],
                "experiment_name": [experiment_name],
                "job_name": [job_name],
            }
        )
        jobs_df = pd.concat([jobs_df, new_row])
        jobs_df.to_csv(self.jobs_path, index=False)


def start_forecasting_job(
    dataset_path: str,
    begin_create_or_update_aml_compute: bool = False,
):
    experiment_name = dataset_path.split("/")[-1].split(".")[0]
    dataset = ExperimentDataset.from_parquet(parquet_path=dataset_path)
    azure_automl = AzureAutoML.from_environment()
    azure_automl_jobs = AzureAutoMLJobs()

    aml_compute = AmlCompute(
        name="azure-automl-fcst-cluster-nixtla",
        min_instances=11,
        max_instances=11,
        size="STANDARD_DS5_V2",
    )

    job_name = azure_automl.forecast(
        df=dataset.Y_df_train,
        df_test=dataset.Y_df_test,
        aml_compute=aml_compute,
        h=dataset.horizon,
        freq=dataset.pandas_frequency,
        n_cross_validations=2,
        experiment_name=experiment_name,
        begin_create_or_update_aml_compute=begin_create_or_update_aml_compute,
        max_nodes=11,
        max_concurrency_per_node=8,
    )

    azure_automl_jobs.save_job(job_name, experiment_name)


if __name__ == "__main__":
    fire.Fire(start_forecasting_job)
