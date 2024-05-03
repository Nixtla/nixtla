import json
import logging
import os
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from azure.ai.ml import Input
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AmlCompute, Job
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)

loggers = logging.Logger.manager.loggerDict
for logger_name in loggers:
    if logger_name.startswith("azure"):
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False


def str_to_datetime(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str)


def df_to_parquet_azureml_input(df: pd.DataFrame, dir: str) -> Input:
    series_path = Path(dir) / "series.parquet"
    df.to_parquet(series_path, index=False)
    table_data_input = Input(type=AssetTypes.URI_FOLDER, path=dir)
    return table_data_input


def config_to_yaml_azureml_input(config: dict, dir: str) -> Input:
    config_path = Path(dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    config = Input(type="uri_file", path=str(config_path))
    return config


class AzureAutoML:
    """
    Before using this class, you need to login to Azure.
    Use the following command to login:
    $ az login
    """

    def __init__(
        self,
        subscription_id: str,
        resource_group_name: str,
        workspace_name: str,
    ):
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.workspace_name = workspace_name

    @classmethod
    def from_environment(cls) -> "AzureAutoML":
        return cls(
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
        )

    def get_ml_client(self, registry_name: str | None = None) -> MLClient:
        kwargs = {}
        if not registry_name:
            kwargs["workspace_name"] = self.workspace_name
        else:
            kwargs["registry_name"] = registry_name
        credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
        ml_client = MLClient(
            credential=credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group_name,
            **kwargs,
        )
        return ml_client

    def get_train_and_inference_components(self) -> tuple:
        ml_client_reqistry = self.get_ml_client("azureml")
        train_component = ml_client_reqistry.components.get(
            name="automl_many_models_training",
            label="latest",
        )
        inference_component = ml_client_reqistry.components.get(
            name="automl_many_models_inference",
            label="latest",
        )
        return train_component, inference_component

    def forecast(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame,
        aml_compute: AmlCompute,
        h: int,
        freq: str,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        primary_metric: str = "normalized_root_mean_squared_error",
        n_cross_validations: str | int = "auto",
        experiment_name: str | None = None,
        begin_create_or_update_aml_compute: bool = False,
        max_trials: int = 25,
        enable_early_stopping: bool = True,
        max_nodes: int = 1,
        max_concurrency_per_node: int = 1,
        forecast_mode: str = "rolling",
        retrain_failed_model: bool = False,
    ) -> str:
        if experiment_name is None:
            random_id = np.random.randint(10000, 99999)
            experiment_name = f"automl-forecasting-job-{random_id}"
        ml_client = self.get_ml_client()
        train_component, inference_component = self.get_train_and_inference_components()
        automl_config_dict = dict(
            task="forecasting",
            forecast_horizon=h,
            forecast_step=h,
            frequency=freq,
            time_series_id_column_names=id_col,
            partition_column_names=[id_col],
            time_column_name=time_col,
            label_column_name=target_col,
            primary_metric=primary_metric,
            n_cross_validations=n_cross_validations,
            max_trials=max_trials,
            enable_early_stopping=enable_early_stopping,
            track_child_runs=False,
            allow_multi_partitions=False,
            #            allowed_training_algorithms=["Naive"],
        )

        @pipeline(description="pipeline for automl forecasting")
        def forecasting_pipeline(
            training_data: Input,
            test_data: Input,
            automl_config: Input,
            compute_name: str,
        ):
            # training node
            training_node = train_component(
                raw_data=training_data,
                automl_config=automl_config,
                max_concurrency_per_node=max_concurrency_per_node,
                max_nodes=max_nodes,
                retrain_failed_model=retrain_failed_model,
                compute_name=compute_name,
            )
            # inference node
            inference_node = inference_component(
                raw_data=test_data,
                max_nodes=max_nodes,
                max_concurrency_per_node=max_concurrency_per_node,
                optional_train_metadata=training_node.outputs.run_output,
                forecast_mode=forecast_mode,
                forecast_step=h,
                compute_name=compute_name,
            )
            return {"forecast_output": inference_node.outputs.raw_predictions}

        if begin_create_or_update_aml_compute:
            main_logger.info("Begin create or update aml compute")
            ml_client.compute.begin_create_or_update(aml_compute).result()

        cwd = Path.cwd()
        with TemporaryDirectory(dir=cwd) as tmp_dir, TemporaryDirectory(
            dir=cwd
        ) as tmp_dir_test, TemporaryDirectory(dir=cwd) as tmp_dir_config:
            main_logger.info("Transforming datasets to parquet")
            table_data_input = df_to_parquet_azureml_input(df, dir=tmp_dir)
            table_data_input_test = df_to_parquet_azureml_input(
                df_test,
                dir=tmp_dir_test,
            )
            automl_config = config_to_yaml_azureml_input(
                automl_config_dict,
                dir=tmp_dir_config,
            )
            pipeline_job = forecasting_pipeline(
                training_data=table_data_input,
                test_data=table_data_input_test,
                automl_config=automl_config,
                compute_name=aml_compute.name,
            )
            pipeline_job.settings.default_compute = aml_compute.name
            main_logger.info("Begin submitting pipeline job")
            returned_pipeline_job = ml_client.jobs.create_or_update(
                pipeline_job,
                experiment_name=experiment_name,
            )
        return returned_pipeline_job.name

    def get_job(self, job_name: str) -> Job:
        ml_client = self.get_ml_client()
        job = ml_client.jobs.get(job_name)
        return job

    def get_job_status(self, job_name: str) -> str | None:
        job = self.get_job(job_name)
        return job.status

    def get_job_total_time(self, job_name: str) -> float | None:
        job = self.get_job(job_name)
        if job.status == "NotStarted":
            main_logger.info(f"Job {job_name} is not started yet")
            return None
        stages_key = "azureml.pipelines.stages"
        if stages_key not in job.properties:
            main_logger.info(f"Job {job_name} has no stages yet")
            return None
        stages = json.loads(job.properties[stages_key])
        execution_info = stages["Execution"]
        status = execution_info["Status"]
        if status == "Failed":
            raise Exception(f"Job {job_name} failed")
        start_time = str_to_datetime(execution_info["StartTime"])
        if "EndTime" not in execution_info:
            total_time = pd.Timestamp.now(tz=start_time.tz) - start_time
            main_logger.info(
                f"Job has status {status}, total time so far: {total_time.total_seconds()}"
            )
        end_time = str_to_datetime(execution_info["EndTime"])
        total_time = end_time - start_time
        return total_time.total_seconds()

    def get_forecast_df(self, job_name: str) -> pd.DataFrame | None:
        job_status = self.get_job_status(job_name)
        if job_status != "Completed":
            main_logger.info(f"Job {job_name} is not completed yet")
            return None
        ml_client = self.get_ml_client()
        cwd = Path.cwd()
        with TemporaryDirectory(dir=cwd) as tmp_dir:
            ml_client.jobs.download(
                job_name,
                download_path=tmp_dir,
                output_name="forecast_output",
            )
            output_path = Path(tmp_dir) / "named-outputs" / "forecast_output"
            forecast_df = pd.read_parquet(output_path)
        return forecast_df
