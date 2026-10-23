from time import time
from typing import Iterable, List, Tuple

import fire
import pandas as pd
import torch
from gluonts.dataset import Dataset
from gluonts.model.forecast import Forecast
from gluonts.torch.model.predictor import PyTorchPredictor
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from uni2ts.model.moirai import MoiraiForecast

from src.utils import ExperimentHandler


def get_morai_predictor(
    model_size: str,
    prediction_length: int,
    target_dim: int,
    batch_size: int,
) -> PyTorchPredictor:
    model = MoiraiForecast.load_from_checkpoint(
        checkpoint_path=hf_hub_download(
            repo_id=f"Salesforce/moirai-1.0-R-{model_size}",
            filename="model.ckpt",
        ),
        prediction_length=prediction_length,
        context_length=200,
        patch_size="auto",
        num_samples=100,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    predictor = model.create_predictor(batch_size)

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


def run_moirai(
    gluonts_dataset: Dataset,
    model_size: str,
    horizon: int,
    target_dim: int,
    batch_size: int,
    quantiles: List[float],
) -> Tuple[pd.DataFrame, float, str]:
    init_time = time()
    predictor = get_morai_predictor(model_size, horizon, target_dim, batch_size)
    fcsts = predictor.predict(gluonts_dataset)
    model_name = "SalesforceMoirai"
    fcsts_df = gluonts_fcsts_to_df(
        fcsts,
        quantiles=quantiles,
        model_name=model_name,
    )
    total_time = time() - init_time
    return fcsts_df, total_time, model_name


def main(dataset: str):
    exp = ExperimentHandler(dataset)
    fcst_df, total_time, model_name = run_moirai(
        gluonts_dataset=exp.gluonts_train_dataset,
        model_size="large",
        horizon=exp.horizon,
        target_dim=1,
        batch_size=32,
        quantiles=exp.quantiles,
    )
    exp.save_results(fcst_df, total_time, model_name)


if __name__ == "__main__":
    fire.Fire(main)
