import os
import pandas as pd
from metaflow import FlowSpec, Parameter, batch, catch, step, timeout
from metaflow.metaflow_config import from_conf

dataset_names = [
    "australian_electricity_demand",
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "dominick",
    "ercot",
    "ett_small_15min",
    "ett_small_1h",
    "exchange_rate",
    "fred_md",
    "hospital",
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    "m3_monthly",
    "m3_other",
    "m3_quarterly",
    "m3_yearly",
    "m4_quarterly",
    "m4_yearly",
    "m5",
    "nn5_daily_without_missing",
    "nn5_weekly",
    "tourism_monthly",
    "tourism_quarterly",
    "tourism_yearly",
    "traffic",
    "weather",
]

model_paths = {
    "chronos_mini": "amazon/chronos-t5-mini",
    "chronos_small": "amazon/chronos-t5-small",
    "chronos_base": "amazon/chronos-t5-base",
    "chronos_large": "amazon/chronos-t5-large",
}

batch_sizes = {
    "chronos_mini": 64,
    "chronos_small": 32,
    "chronos_base": 16,
    "chronos_large": 16,
}

# Configuration for g5.2xlarge instance used to run Chronos
BATCH_NUM_CPUS = int(os.environ.get("BATCH_NUM_CPUS", 8))
BATCH_NUM_GPUS = int(os.environ.get("BATCH_NUM_GPUS", 1))
BATCH_MEMORY_MB = int(os.environ.get("BATCH_MEMORY_MB", 31_000))
# For c5a.24xlarge set BATCH_NUM_CPUS=96 BATCH_NUM_GPUS=0 BATCH_MEMORY_MB=190000


class ForecastEvaluation(FlowSpec):
    model = Parameter(
        "model",
        default="SeasonalNaive",
        help="Model to evaluate. One of 'SeasonalNaive', 'StatisticalEnsemble', 'chronos_mini', 'chronos_small', 'chronos_base', 'chronos_large'",
    )

    @step
    def start(self):
        assert self.model in model_paths or self.model in [
            "StatisticalEnsemble",
            "SeasonalNaive",
        ]
        self.dataset_names = dataset_names
        self.next(self.save_dataset_name, foreach="dataset_names")

    @step
    def save_dataset_name(self):
        # Save dataset name as a variable in case evaluate_dataset crashes / times out
        self.dataset = self.input
        self.next(self.evaluate_dataset)

    @catch(var="error_message")
    @timeout(seconds=24 * 60 * 60)
    # Uncomment lines below to run experiments in parallel using AWS Batch
    # @batch(
    #     cpu=BATCH_NUM_CPUS,
    #     gpu=BATCH_NUM_GPUS,
    #     memory=BATCH_MEMORY_MB,
    #     image=f"{from_conf('BATCH_CONTAINER_IMAGE')}:nixtla-eval",
    # )
    @step
    def evaluate_dataset(self):
        import numpy as np
        import torch

        from eval_utils.amazon_chronos.pipeline import run_amazon_chronos
        from eval_utils.statsforecast_pipeline import (
            run_seasonal_naive,
            run_statistical_ensemble,
        )
        from eval_utils.utils import ExperimentHandler

        torch.manual_seed(123)
        np.random.seed(123)

        dataset = self.dataset
        print(f"Evaluating model {self.model} on dataset {dataset}")
        exp = ExperimentHandler(dataset)
        experiment_kwargs = dict(
            train_df=exp.train_df,
            horizon=exp.horizon,
            freq=exp.freq,
        )

        if self.model == "SeasonalNaive":
            run_experiment = run_seasonal_naive
            experiment_kwargs["level"] = exp.level
            experiment_kwargs["seasonality"] = exp.seasonality
        elif self.model == "StatisticalEnsemble":
            run_experiment = run_statistical_ensemble
            experiment_kwargs["quantiles"] = exp.quantiles
            experiment_kwargs["seasonality"] = exp.seasonality
        else:
            run_experiment = run_amazon_chronos
            experiment_kwargs["model_name"] = model_paths[self.model]
            experiment_kwargs["quantiles"] = exp.quantiles
            experiment_kwargs["batch_size"] = batch_sizes[self.model]

        fcst_df, total_time, model_name = run_experiment(**experiment_kwargs)
        if self.model == "SeasonalNaive":
            fcst_df = exp.fcst_from_level_to_quantiles(fcst_df, model_name)
        time_df = pd.DataFrame({"time": [total_time], "model": model_name})
        self.results = exp.evaluate_from_predictions(
            models=[model_name], fcsts_df=fcst_df, times_df=time_df
        )
        print(self.results)

        self.next(self.join)

    @step
    def join(self, inputs):
        results_list = []
        for ip in inputs:
            if ip.error_message is not None:
                results_list.append(
                    pd.DataFrame(
                        [
                            {
                                "dataset": ip.dataset,
                                "model": ip.model,
                                "error_message": ip.error_message,
                            }
                        ]
                    )
                )
            else:
                results_list.append(ip.results)
        self.results_full = pd.concat(results_list)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ForecastEvaluation()
