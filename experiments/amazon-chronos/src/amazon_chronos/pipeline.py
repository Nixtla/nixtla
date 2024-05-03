import os
from time import time
from typing import List, Tuple

import fire
import pandas as pd


from ..utils import ExperimentHandler
from .forecaster import AmazonChronos


def run_amazon_chronos(
    train_df: pd.DataFrame,
    model_name: str,
    horizon: int,
    freq: str,
    quantiles: List[float],
) -> Tuple[pd.DataFrame, float, str]:
    ac = AmazonChronos(model_name)
    init_time = time()
    fcsts_df = ac.forecast(
        df=train_df,
        h=horizon,
        freq=freq,
        batch_size=8,
        quantiles=quantiles,
        # parameters as in https://github.com/amazon-science/chronos-forecasting/blob/73be25042f5f587823d46106d372ba133152fb00/README.md?plain=1#L62-L65
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    total_time = time() - init_time
    return fcsts_df, total_time, model_name


def main(dataset: str, model_name: str):
    exp = ExperimentHandler(dataset)
    fcst_df, total_time, model_name = run_amazon_chronos(
        train_df=exp.train_df,
        model_name=model_name,
        horizon=exp.horizon,
        freq=exp.freq,
        quantiles=exp.quantiles,
    )
    exp.save_results(fcst_df, total_time, model_name)


if __name__ == "__main__":
    fire.Fire(main)
