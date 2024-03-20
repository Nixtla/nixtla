import logging
import subprocess

import fire
import pandas as pd

from src.utils import ExperimentHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

datasets = [
    "m1_yearly",
    "m1_quarterly",
    "m1_monthly",
    "m3_yearly",
    "m3_quarterly",
    "m3_monthly",
    "m3_other",
    "tourism_yearly",
    "tourism_quarterly",
    "tourism_monthly",
    "m4_yearly",
    "m4_quarterly",
]

amazon_chronos_models = [
    "amazon/chronos-t5-large",
    "amazon/chronos-t5-tiny",
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
    "amazon/chronos-t5-base",
]


def main(mode: str):
    prefix_process = ["python", "-m"]

    eval_df = None
    for dataset in datasets:
        logger.info(f"Evaluating {dataset}...")
        if mode in ["fcst_statsforecast", "fcst_chronos"]:
            suffix_process = ["--dataset", dataset]

            def process(middle_process):
                return prefix_process + middle_process + suffix_process

            if mode == "fcst_statsforecast":
                logger.info("Running StatisticalEnsemble")
                subprocess.run(process(["src.statsforecast_pipeline"]))
            elif mode == "fcst_chronos":
                for model in amazon_chronos_models:
                    logger.info(f"Running Amazon Chronos {model}")
                    chronos_process = process(["src.amazon_chronos.pipeline"])
                    chronos_process.extend(["--model_name", model])
                    subprocess.run(chronos_process)
        elif mode == "evaluation":
            if eval_df is None:
                eval_df = []
            logger.info("Running dataset evaluation")
            exp = ExperimentHandler(dataset)
            try:
                eval_dataset_df = exp.evaluate_models(
                    amazon_chronos_models + ["StatisticalEnsemble", "SeasonalNaive"]
                )
                print(eval_dataset_df)
                eval_df.append(eval_dataset_df)
            except Exception as e:
                logger.error(e)
    if eval_df is not None:
        eval_df = pd.concat(eval_df).reset_index(drop=True)
        exp.save_dataframe(eval_df, "complete-results.csv")


if __name__ == "__main__":
    fire.Fire(main)
