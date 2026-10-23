from time import time
from typing import Iterable, List, Tuple

import fire
import pandas as pd
import torch
from gluonts.dataset import Dataset
from gluonts.model.forecast import Forecast
from gluonts.torch.model.predictor import PyTorchPredictor
from tqdm import tqdm

from lag_llama.gluon.estimator import LagLlamaEstimator
from src.utils import ExperimentHandler


def get_lag_llama_predictor(
    prediction_length: int, models_dir: str
) -> PyTorchPredictor:
    model_path = f"{models_dir}/lag-llama.ckpt"
    map_location = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if map_location == "cpu":
        raise ValueError("cpu is not supported in lagllama (there is a bug)")
    ckpt = torch.load(model_path, map_location=map_location)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    # this context length is reported in the paper
    context_length = 32
    estimator = LagLlamaEstimator(
        ckpt_path=model_path,
        prediction_length=prediction_length,
        context_length=context_length,
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
    )
    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor


def gluonts_instance_fcst_to_df(
    fcst: Forecast,
    quantiles: List[float],
    model_name: str,
) -> pd.DataFrame:
    point_forecast = fcst.mean
    h = len(point_forecast)
    dates = pd.date_range(
        fcst.start_date.to_timestamp(),
        freq=fcst.freq,
        periods=h,
    )
    fcst_df = pd.DataFrame(
        {
            "ds": dates,
            "unique_id": fcst.item_id,
            model_name: point_forecast,
        }
    )
    for q in quantiles:
        fcst_df[f"{model_name}-q-{q}"] = fcst.quantile(q)
    return fcst_df


def gluonts_fcsts_to_df(
    fcsts: Iterable[Forecast],
    quantiles: List[float],
    model_name: str,
) -> pd.DataFrame:
    df = []
    for fcst in tqdm(fcsts):
        fcst_df = gluonts_instance_fcst_to_df(fcst, quantiles, model_name)
        df.append(fcst_df)
    return pd.concat(df).reset_index(drop=True)


def run_lag_llama(
    gluonts_dataset: Dataset,
    horizon: int,
    quantiles: List[float],
    models_dir: str,
) -> Tuple[pd.DataFrame, float, str]:
    init_time = time()
    predictor = get_lag_llama_predictor(horizon, models_dir)
    fcsts = predictor.predict(gluonts_dataset, num_samples=100)
    model_name = "LagLlama"
    fcsts_df = gluonts_fcsts_to_df(
        fcsts,
        quantiles=quantiles,
        model_name=model_name,
    )
    total_time = time() - init_time
    return fcsts_df, total_time, model_name


def main(dataset: str):
    exp = ExperimentHandler(dataset)
    fcst_df, total_time, model_name = run_lag_llama(
        gluonts_dataset=exp.gluonts_train_dataset,
        horizon=exp.horizon,
        quantiles=exp.quantiles,
        models_dir=exp.models_dir,
    )
    exp._save_results(fcst_df, total_time, model_name)


if __name__ == "__main__":
    fire.Fire(main)
