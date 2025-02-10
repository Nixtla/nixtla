from tabpfn_client import init
from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import generate_test_X

from tabpfn_time_series import FeatureTransformer, DefaultFeatures
from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

from time import time
from pathlib import Path
CURRENT_PATH = Path(__file__).parent

import numpy as np
import pandas as pd
#%%
predictor = TabPFNTimeSeriesPredictor(
    tabpfn_mode=TabPFNMode.LOCAL,
)

selected_features = [
    DefaultFeatures.add_running_index,
    DefaultFeatures.add_calendar_features,
]
#%%
def read_and_prepare_data(file_path: str, value_name: str = "y"):
    """Reads data in wide format, and returns it in long format with columns `unique_id`, `ds`, `y`"""
    df = pd.read_csv(file_path)
    uid_cols = ["Client", "Warehouse", "Product"]
    df["unique_id"] = df[uid_cols].astype(str).agg("-".join, axis=1)
    df = df.drop(uid_cols, axis=1)
    df = df.melt(id_vars=["unique_id"], var_name="ds", value_name=value_name)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    return df
#%%
def get_train_data():
    """Reads all train data and returns it in long format with columns `unique_id`, `ds`, `y`"""
    train_list = [CURRENT_PATH.joinpath(f"data/phase_{i}_sales.csv") for i in [0, 1]]
    train_list = [read_and_prepare_data(train_file) for train_file in train_list]
    train_df = pd.concat(train_list).reset_index(drop=True)
    train_df = train_df.sort_values(by=["unique_id", "ds"])

    def remove_leading_zeros(group):
        first_non_zero_index = group["y"].ne(0).idxmax()
        return group.loc[first_non_zero_index:]

    train_df = (
        train_df.groupby("unique_id").apply(remove_leading_zeros).reset_index(drop=True)
    )
    train_df = train_df.rename(columns={
        'unique_id': 'item_id',
        'ds': 'timestamp',
        'y': 'target'
    })
    train_df = train_df.set_index(['item_id', 'timestamp'])
    train_tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
    return train_tsdf
#%%
def get_competition_forecasts() -> pd.DataFrame:
    """Reads all competition forecasts and returns it in long format with columns `unique_id`, `ds`, `y`"""
    fcst_df: pd.DataFrame | None = None
    for place in ["1st", "2nd", "3rd", "4th", "5th"]:
        fcst_df_place = read_and_prepare_data(
            CURRENT_PATH.joinpath(f"data/solution_{place}_place.csv"), place
        )
        if fcst_df is None:
            fcst_df = fcst_df_place
        else:
            fcst_df = fcst_df.merge(
                fcst_df_place,
                on=["unique_id", "ds"],
                how="left",
            )
    return fcst_df
#%%
def vn1_competition_evaluation(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Computes competition evaluation scores"""
    actual = read_and_prepare_data(CURRENT_PATH.joinpath("data/phase_2_sales.csv"))
    res = actual[["unique_id", "ds", "y"]].merge(
        forecasts, on=["unique_id", "ds"], how="left"
    )
    ids_forecasts = forecasts["unique_id"].unique()
    ids_res = res["unique_id"].unique()
    assert set(ids_forecasts) == set(ids_res), "Some unique_ids are missing"
    scores = {}
    for model in [col for col in forecasts.columns if col not in ["unique_id", "ds"]]:
        abs_err = np.nansum(np.abs(res[model] - res["y"]))
        err = np.nansum(res[model] - res["y"])
        score = abs_err + abs(err)
        score = score / res["y"].sum()
        scores[model] = round(score, 4)
    score_df = pd.DataFrame(list(scores.items()), columns=["model", "score"])
    score_df = score_df.sort_values(by="score")
    return score_df
#%%
def main():
    """Complete pipeline"""
    h=13
    tabpfn_results = CURRENT_PATH.joinpath("tabpfn_preds.csv")
    if not tabpfn_results.exists():
        train_df = get_train_data()
        test_df = generate_test_X(train_df, h)
        train_df, test_df = FeatureTransformer.add_features(
            train_df, test_df, selected_features
        )
        init = time()
        fcst_df = predictor.predict(train_df, test_df)
        print(f"TabPFN time: {time() - init}")
        fcst_df = fcst_df.reset_index()
        fcst_df = fcst_df[['item_id', 'timestamp', 'target']]
        fcst_df = fcst_df.rename(columns={
            'item_id': 'unique_id',
            'timestamp': 'ds',
            'target': 'tabpfn'
        })
        fcst_df.to_csv(CURRENT_PATH.joinpath("tabpfn_preds.csv"), index=False, header=True)
    else:
        fcst_df = pd.read_csv(tabpfn_results, parse_dates=['ds'])
        fcst_df.rename(columns={'y': 'tabpfn'}, inplace=True)
    fcst_df_comp = get_competition_forecasts()
    fcst_df = fcst_df.merge(fcst_df_comp, on=["unique_id", "ds"], how="left")
    eval_df = vn1_competition_evaluation(fcst_df)
    print(eval_df)
#%%
if __name__ == '__main__':
    main()