from time import time

import fire
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, ZeroModel

from src.tools import ExperimentHandler


def evaluate_experiment(file: str):
    exp_handler = ExperimentHandler(file=file, method="statsforecast")
    Y_df, freq, pandas_freq, h, seasonality = exp_handler.read_data()
    models = [
        SeasonalNaive(season_length=seasonality),
        ZeroModel(),
    ]
    # even though statsforecast can handle multiple models, we only use one
    # at a time to calculate time for each
    eval_df = []
    total_time_df = []
    for model in models:
        model_name = repr(model)
        print(model_name)
        sf = StatsForecast(
            models=[model],
            freq=pandas_freq,
            n_jobs=-1,
        )
        start = time()
        Y_hat_df_model = sf.cross_validation(
            df=Y_df,
            h=h,
            n_windows=1,
        ).reset_index()
        total_time = time() - start
        print(total_time)
        # evaluation
        eval_df_model, total_time_df_model = exp_handler.evaluate_model(
            Y_hat_df=Y_hat_df_model,
            model_name=model_name,
            total_time=total_time,
        )
        eval_df.append(eval_df_model.set_index(["metric", "unique_id"]))
        total_time_df.append(total_time_df_model)
    eval_df = pd.concat(eval_df, axis=1).reset_index()
    total_time_df = pd.concat(total_time_df)
    exp_handler.save_results(
        freq=freq,
        eval_df=eval_df,
        total_time_df=total_time_df,
        df=Y_df,
    )


if __name__ == "__main__":
    fire.Fire(evaluate_experiment)
