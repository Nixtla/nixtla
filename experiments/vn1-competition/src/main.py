from time import time


import numpy as np
import pandas as pd
from dotenv import load_dotenv
from nixtla import NixtlaClient

load_dotenv()


def read_and_prepare_data(file_path: str, value_name: str = "y") -> pd.DataFrame:
    """Reads data in wide format, and returns it in long format with columns `unique_id`, `ds`, `y`"""
    df = pd.read_csv(file_path)
    uid_cols = ["Client", "Warehouse", "Product"]
    df["unique_id"] = df[uid_cols].astype(str).agg("-".join, axis=1)
    df = df.drop(uid_cols, axis=1)
    df = df.melt(id_vars=["unique_id"], var_name="ds", value_name=value_name)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    return df


def get_train_data() -> pd.DataFrame:
    """Reads all train data and returns it in long format with columns `unique_id`, `ds`, `y`"""
    train_list = [read_and_prepare_data(f"./data/phase_{i}_sales.csv") for i in [0, 1]]
    train_df = pd.concat(train_list).reset_index(drop=True)
    train_df = train_df.sort_values(by=["unique_id", "ds"])

    def remove_leading_zeros(group):
        first_non_zero_index = group["y"].ne(0).idxmax()
        return group.loc[first_non_zero_index:]

    train_df = (
        train_df.groupby("unique_id").apply(remove_leading_zeros).reset_index(drop=True)
    )
    return train_df


def get_competition_forecasts() -> pd.DataFrame:
    """Reads all competition forecasts and returns it in long format with columns `unique_id`, `ds`, `y`"""
    fcst_df: pd.DataFrame | None = None
    for place in ["1st", "2nd", "3rd", "4th", "5th"]:
        fcst_df_place = read_and_prepare_data(
            f"./data/solution_{place}_place.csv", place
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


def vn1_competition_evaluation(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Computes competition evaluation scores"""
    actual = read_and_prepare_data("./data/phase_2_sales.csv")
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


def main():
    """Complete pipeline"""
    train_df = get_train_data()
    client = NixtlaClient()
    init = time()
    fcst_df = client.forecast(train_df, h=13, model="timegpt-1-long-horizon")
    print(f"TimeGPT time: {time() - init}")
    fcst_df_comp = get_competition_forecasts()
    fcst_df = fcst_df.merge(fcst_df_comp, on=["unique_id", "ds"], how="left")
    eval_df = vn1_competition_evaluation(fcst_df)
    print(eval_df)


if __name__ == "__main__":
    main()
