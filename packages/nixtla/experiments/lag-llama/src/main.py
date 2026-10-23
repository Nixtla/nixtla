import logging
import subprocess

import pandas as pd

from src.utils import ExperimentHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

not_included_datasets = [
    "m1_yearly",
    "m1_quarterly",
    "m1_monthly",
    "m3_yearly",
    "m3_quarterly",
    "m3_monthly",
    "m3_other",
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "tourism_yearly",
    "tourism_quarterly",
    "tourism_monthly",
]

test_paper_datasets = [
    "pedestrian_counts",
    "weather",
]

datasets = {
    "not_included": not_included_datasets,
    "test_set": test_paper_datasets,
}


def evaluate():
    eval_df = []
    prefix_process = ["python", "-m"]

    for name_group, groups in datasets.items():
        for dataset in groups:
            logger.info(f"Evaluating {dataset}...")
            suffix_process = ["--dataset", dataset]
            process = (
                lambda middle_process: prefix_process + middle_process + suffix_process
            )
            # running statsforecast and lagllama in separated
            # processes because gluonts sets multiprocessing context
            # see: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/__init__.py
            logger.info("Running SeasonalNaive")
            subprocess.run(process(["src.statsforecast_pipeline"]))
            logger.info("Running LagLLama")
            subprocess.run(process(["src.lag_llama_pipeline"]))
            logger.info("Running dataset evaluation")
            exp = ExperimentHandler(dataset)
            eval_dataset_df = exp.evaluate_models(["LagLlama", "SeasonalNaive"])
            eval_dataset_df.insert(0, "paper", name_group)
            eval_df.append(eval_dataset_df)
    eval_df = pd.concat(eval_df).reset_index(drop=True)
    exp.save_dataframe(eval_df, "complete-results.csv")


if __name__ == "__main__":
    evaluate()
