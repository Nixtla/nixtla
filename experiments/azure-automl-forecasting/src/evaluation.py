import logging
from pathlib import Path
from typing import List
from unicodedata import numeric

import fire
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.utils.data_handler import ExperimentDataset, ForecastDataset

logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)


def print_df_rich(df: pd.DataFrame):
    console = Console()
    table = Table()
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].apply(lambda x: f"{x:.3f}")
    for col in df.columns:
        table.add_column(col)
    for row in df.itertuples(index=False):
        table.add_row(*row)
    console.print(table)


METHODS = {
    "azure_automl": "automl_prediction",
    "nixtla_timegpt": "TimeGPT",
    "statsforecast_sn": "SeasonalNaive",
}


def get_model_name(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"Invalid method: {method}")
    return METHODS[method]


def evaluate_experiments(
    datasets_paths: str,
    methods_to_evaluate: List[str] = list(METHODS.keys()),
    results_dir: str = "./results",
):
    datasets_paths_ = datasets_paths.split(",")
    eval_datasets_df: pd.DataFrame | None = None
    for dataset_path in datasets_paths_:
        experiment_name = dataset_path.split("/")[-1].split(".")[0]
        eval_method_df: pd.DataFrame | None = None
        dataset: None | ExperimentDataset = None
        for method in methods_to_evaluate:
            results_experiment_dir = Path(results_dir) / method / experiment_name
            if ForecastDataset.is_forecast_ready(results_experiment_dir):
                main_logger.info(
                    f"Evaluating experiment {experiment_name} and method {method}"
                )
                forecast_dataset = ForecastDataset.from_dir(results_experiment_dir)
                if dataset is None:
                    dataset = ExperimentDataset.from_parquet(parquet_path=dataset_path)
                eval_df = dataset.evaluate_forecast_df(
                    forecast_df=forecast_dataset.forecast_df,
                    model=get_model_name(method),
                    total_time=forecast_dataset.total_time,
                )
                if eval_method_df is None:
                    eval_method_df = eval_df
                else:
                    eval_method_df = pd.concat(
                        [eval_method_df, eval_df],
                        axis=1,
                    )  # type: ignore
            else:
                main_logger.info(
                    f"Skipping evaluation for experiment {experiment_name} and method {method}"
                    " because the forecasts are not ready yet"
                )
        if eval_method_df is not None:
            eval_method_df.reset_index(inplace=True)
            eval_method_df.insert(0, "dataset", experiment_name)
            if eval_datasets_df is None:
                eval_datasets_df = eval_method_df
            else:
                eval_datasets_df = pd.concat(
                    [eval_datasets_df, eval_method_df],
                    ignore_index=True,
                )  # type: ignore
    if eval_datasets_df is not None:
        azure_renamer = {"automl_prediction": "AzureAutoML"}
        if "azure_automl" in methods_to_evaluate:
            eval_datasets_df = eval_datasets_df.rename(columns=azure_renamer)
        eval_datasets_df.to_csv(Path(results_dir) / "eval_datasets.csv", index=False)
        eval_datasets_df["metric"] = (
            eval_datasets_df["metric"].str.upper().str.replace("TOTAL_", "")
        )
        # scale by SeasonalNaive
        if "SeasonalNaive" in eval_datasets_df.columns:
            time_mask = eval_datasets_df["metric"] == "TIME"
            for model in eval_datasets_df.columns.drop(["dataset", "metric"]):
                if model == "SeasonalNaive":
                    continue
                eval_datasets_df.loc[~time_mask, model] = (
                    eval_datasets_df.loc[~time_mask, model]
                    / eval_datasets_df.loc[~time_mask, "SeasonalNaive"]
                )
            eval_datasets_df = eval_datasets_df.drop(columns=["SeasonalNaive"])

        def pivot_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
            return df.pivot(
                index="dataset",
                columns="metric",
                values=col,
            )

        result_list = []
        models = []
        for method in methods_to_evaluate:
            if method == "statsforecast_sn":
                continue
            if method == "azure_automl":
                col = "AzureAutoML"
            else:
                col = get_model_name(method)
            pivotted_df = pivot_df(eval_datasets_df, col)
            result_list.append(pivotted_df)
            models.append(col)
        result = pd.concat(result_list, axis=1, keys=models)
        result = result.swaplevel(axis=1).sort_index(axis=1)
        flattened_columns = ["_".join(col) for col in result.columns.values]
        result.columns = flattened_columns
        result = result.reset_index()
        print_df_rich(result)


if __name__ == "__main__":
    fire.Fire(evaluate_experiments)
