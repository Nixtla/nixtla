import sys
from time import time

import fire
from dotenv import load_dotenv
from nixtla import NixtlaClient

from src.tools import ExperimentHandler

load_dotenv()


def evaluate_experiment(file: str):
    exp_handler = ExperimentHandler(file=file, method="timegpt")
    model_name = "TimeGPT"
    print(model_name)
    # timegpt does not need the full history to
    # make zero shot predictions
    Y_df, freq, pandas_freq, h, seasonality = exp_handler.read_data(
        max_insample_length=300
    )
    size_df = sys.getsizeof(Y_df) / (1024 * 1024)
    max_partition_size_mb = 20
    num_partitions = int(size_df / max_partition_size_mb) + 1
    timegpt = NixtlaClient(
        base_url="https://timegpt-endpoint.eastus.inference.ml.azure.com/",
        max_retries=1,
    )
    start = time()
    Y_hat_df = timegpt.cross_validation(
        df=Y_df,
        h=h,
        n_windows=1,
        freq=pandas_freq,
        num_partitions=num_partitions,
    )
    total_time = time() - start
    print(total_time)
    # evaluation
    eval_df, total_time_df = exp_handler.evaluate_model(
        Y_hat_df=Y_hat_df,
        model_name=model_name,
        total_time=total_time,
    )
    exp_handler.save_results(
        freq=freq,
        eval_df=eval_df,
        total_time_df=total_time_df,
    )


if __name__ == "__main__":
    fire.Fire(evaluate_experiment)
