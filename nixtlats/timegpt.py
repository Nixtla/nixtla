# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/timegpt.ipynb.

# %% auto 0
__all__ = ['main_logger', 'httpx_logger']

# %% ../nbs/timegpt.ipynb 5
import logging
import inspect
import json
import requests
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .client import Nixtla, SingleSeriesForecast

logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.ERROR)

# %% ../nbs/timegpt.ipynb 7
date_features_by_freq = {
    # Daily frequencies
    "B": ["year", "month", "day", "weekday"],
    "C": ["year", "month", "day", "weekday"],
    "D": ["year", "month", "day", "weekday"],
    # Weekly
    "W": ["year", "week", "weekday"],
    # Monthly
    "M": ["year", "month"],
    "SM": ["year", "month", "day"],
    "BM": ["year", "month"],
    "CBM": ["year", "month"],
    "MS": ["year", "month"],
    "SMS": ["year", "month", "day"],
    "BMS": ["year", "month"],
    "CBMS": ["year", "month"],
    # Quarterly
    "Q": ["year", "quarter"],
    "BQ": ["year", "quarter"],
    "QS": ["year", "quarter"],
    "BQS": ["year", "quarter"],
    # Yearly
    "A": ["year"],
    "Y": ["year"],
    "BA": ["year"],
    "BY": ["year"],
    "AS": ["year"],
    "YS": ["year"],
    "BAS": ["year"],
    "BYS": ["year"],
    # Hourly
    "BH": ["year", "month", "day", "hour", "weekday"],
    "H": ["year", "month", "day", "hour"],
    # Minutely
    "T": ["year", "month", "day", "hour", "minute"],
    "min": ["year", "month", "day", "hour", "minute"],
    # Secondly
    "S": ["year", "month", "day", "hour", "minute", "second"],
    # Milliseconds
    "L": ["year", "month", "day", "hour", "minute", "second", "millisecond"],
    "ms": ["year", "month", "day", "hour", "minute", "second", "millisecond"],
    # Microseconds
    "U": ["year", "month", "day", "hour", "minute", "second", "microsecond"],
    "us": ["year", "month", "day", "hour", "minute", "second", "microsecond"],
    # Nanoseconds
    "N": [],
}

# %% ../nbs/timegpt.ipynb 8
class TimeGPT:
    """
    A class used to interact with the TimeGPT API.
    """

    def __init__(self, token: str):
        """
        Constructs all the necessary attributes for the TimeGPT object.

        Parameters
        ----------
        token : str
            The authorization token to interact with the TimeGPT API.
        """
        self.client = Nixtla(base_url="https://dashboard.nixtla.io/api", token=token)
        self.weights_x: pd.DataFrame = None

    @property
    def request_headers(self):
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.client._client_wrapper._token}",
        }
        return headers

    def _parse_response(self, response) -> Dict:
        """Parses responde."""
        response.raise_for_status()
        try:
            resp = response.json()
        except Exception as e:
            raise Exception(response)
        return resp

    def validate_token(self, log: bool = True) -> bool:
        """Returns True if your token is valid."""
        validation = self.client.validate_token()
        valid = False
        if "message" in validation:
            if validation["message"] == "success":
                valid = True
        if "support" in validation and log:
            main_logger.info(f'Happy Forecasting! :), {validation["support"]}')
        return valid

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        X_df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ):
        renamer = {
            id_col: "unique_id",
            time_col: "ds",
            target_col: "y",
        }
        main_logger.info("Validating inputs...")
        df = df.rename(columns=renamer)
        if df.dtypes.ds != "object":
            df["ds"] = df["ds"].astype(str)
        drop_uid = False
        if "unique_id" not in df.columns:
            # Insert unique_id column
            df = df.assign(unique_id="ts_0")
            drop_uid = True
        if X_df is not None:
            X_df = X_df.rename(columns=renamer)
            if "unique_id" not in X_df.columns:
                X_df = X_df.assign(unique_id="ts_0")
            if X_df.dtypes.ds != "object":
                X_df["ds"] = X_df["ds"].astype(str)
        return df, X_df, drop_uid

    def _validate_outputs(
        self,
        fcst_df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        drop_uid: bool,
    ):
        renamer = {
            "unique_id": id_col,
            "ds": time_col,
            "target_col": target_col,
        }
        if drop_uid:
            fcst_df = fcst_df.drop(columns="unique_id")
        fcst_df = fcst_df.rename(columns=renamer)
        return fcst_df

    def _infer_freq(self, df: pd.DataFrame):
        unique_id = df.iloc[0]["unique_id"]
        df_id = df.query("unique_id == @unique_id")
        freq = pd.infer_freq(df_id["ds"])
        if freq is None:
            raise Exception(
                "Could not infer frequency of ds column. This could be due to "
                "inconsistent intervals. Please check your data for missing, "
                "duplicated or irregular timestamps"
            )
        return freq

    def _resample_dataframe(
        self,
        df: pd.DataFrame,
        freq: str,
    ):
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        resampled_df = df.set_index("ds").groupby("unique_id").resample(freq).bfill()
        resampled_df = resampled_df.drop(columns="unique_id").reset_index()
        resampled_df["ds"] = resampled_df["ds"].astype(str)
        return resampled_df

    def _compute_date_feature(self, dates, feature):
        if callable(feature):
            feat_name = feature.__name__
            feat_vals = feature(dates)
        else:
            feat_name = feature
            if feature in ("week", "weekofyear"):
                dates = dates.isocalendar()
            feat_vals = getattr(dates, feature)
        vals = np.asarray(feat_vals)
        return feat_name, vals

    def _make_future_dataframe(
        self, df: pd.DataFrame, h: int, freq: str, reconvert: bool = True
    ):
        last_dates = df.groupby("unique_id")["ds"].max()

        def _future_date_range(last_date):
            future_dates = pd.date_range(last_date, freq=freq, periods=h + 1)[-h:]
            return future_dates

        future_df = last_dates.apply(_future_date_range).reset_index()
        future_df = future_df.explode("ds").reset_index(drop=True)
        if reconvert and df.dtypes["ds"] == "object":
            # avoid date 000
            future_df["ds"] = future_df["ds"].astype(str)
        return future_df

    def _add_date_features(
        self,
        df: pd.DataFrame,
        X_df: Optional[pd.DataFrame],
        h: int,
        freq: str,
        date_features: List[str],
        date_features_to_one_hot: Optional[List[str]],
    ):
        # df contains exogenous variables
        # X_df are the future values of the exogenous variables
        # construct dates
        train_dates = df["ds"].unique().tolist()
        # if we dont have future exogenos variables
        # we need to compute the future dates
        if X_df is None:
            X_df = self._make_future_dataframe(df=df, h=h, freq=freq)
        future_dates = X_df["ds"].unique().tolist()
        dates = pd.DatetimeIndex(train_dates + future_dates)
        date_features_df = pd.DataFrame({"ds": dates})
        for feature in date_features:
            feat_name, feat_vals = self._compute_date_feature(dates, feature)
            date_features_df[feat_name] = feat_vals
        if df.dtypes["ds"] == "object":
            date_features_df["ds"] = date_features_df["ds"].astype(str)
        if date_features_to_one_hot is not None:
            date_features_df = pd.get_dummies(
                date_features_df,
                columns=date_features_to_one_hot,
                dtype=int,
            )
        # remove duplicated columns if any
        date_features_df = date_features_df.drop(
            columns=[
                col
                for col in date_features_df.columns
                if col in df.columns and col not in ["unique_id", "ds"]
            ]
        )
        # add date features to df
        df = df.merge(date_features_df, on="ds", how="left")
        # add date features to X_df
        X_df = X_df.merge(date_features_df, on="ds", how="left")
        return df, X_df

    def _preprocess_dataframes(
        self,
        df: pd.DataFrame,
        h: int,
        X_df: Optional[pd.DataFrame],
        freq: str,
    ):
        """Returns Y_df and X_df dataframes in the structure expected by the endpoints."""
        y_cols = ["unique_id", "ds", "y"]
        Y_df = df[y_cols]
        if Y_df["y"].isna().any():
            raise Exception("Your target variable contains NA, please check")
        # Azul: efficient this code
        # and think about returning dates that are not in the training set
        Y_df = self._resample_dataframe(Y_df, freq)
        x_cols = []
        if X_df is not None:
            x_cols = X_df.drop(columns=["unique_id", "ds"]).columns.to_list()
            if not all(col in df.columns for col in x_cols):
                raise Exception(
                    "You must include the exogenous variables in the `df` object, "
                    f'exogenous variables {",".join(x_cols)}'
                )
            if len(X_df) != df["unique_id"].nunique() * h:
                raise Exception(
                    f"You have to pass the {h} future values of your "
                    "exogenous variables for each time series"
                )
            X_df_history = df[["unique_id", "ds"] + x_cols]
            X_df = pd.concat([X_df_history, X_df])
            if X_df[x_cols].isna().any().any():
                raise Exception(
                    "Some of your exogenous variables contain NA, please check"
                )
            X_df = X_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            X_df = self._resample_dataframe(X_df, freq)
        return Y_df, X_df, x_cols

    def _get_to_dict_args(self):
        to_dict_args = {"orient": "split"}
        if "index" in inspect.signature(pd.DataFrame.to_dict).parameters:
            to_dict_args["index"] = False
        return to_dict_args

    def _transform_dataframes(self, Y_df: pd.DataFrame, X_df: pd.DataFrame):
        # contruction of y and x for the payload
        to_dict_args = self._get_to_dict_args()
        y = Y_df.to_dict(**to_dict_args)
        x = X_df.to_dict(**to_dict_args) if X_df is not None else None
        return y, x

    def _get_model_params(self, freq: str):
        model_params = self.client.timegpt_model_params(
            request=SingleSeriesForecast(freq=freq)
        )
        model_params = model_params["data"]["detail"]
        input_size, model_horizon = model_params["input_size"], model_params["horizon"]
        return input_size, model_horizon

    def _validate_input_size(
        self,
        Y_df: pd.DataFrame,
        input_size: int,
        model_horizon: int,
        require_history: bool,
    ):
        if require_history:
            min_history = Y_df.groupby("unique_id").size().min()
            if min_history < input_size + model_horizon:
                raise Exception(
                    "Your time series data is too short "
                    "Please be sure that your unique time series contain "
                    f"at least {input_size + model_horizon} observations"
                )
        return True

    def _hit_multi_series_endpoint(
        self,
        Y_df: pd.DataFrame,
        X_df: pd.DataFrame,
        x_cols: List[str],
        h: int,
        freq: str,
        finetune_steps: int,
        clean_ex_first: bool,
        level: Optional[List[Union[int, float]]],
        input_size: int,
        model_horizon: int,
    ):
        # restrict input if
        # - we dont want to finetune
        # - we dont have exogenous regegressors
        # - and we dont want to produce pred intervals
        restrict_input = finetune_steps == 0 and X_df is None and level is not None
        if restrict_input:
            # add sufficient info to compute
            # conformal interval
            new_input_size = 3 * input_size + max(model_horizon, h)
            Y_df = Y_df.groupby("unique_id").tail(new_input_size)
            if X_df is not None:
                X_df = X_df.groupby("unique_id").tail(
                    new_input_size + h
                )  # history plus exogenous
        self._validate_input_size(
            Y_df=Y_df,
            input_size=input_size,
            model_horizon=model_horizon,
            require_history=finetune_steps > 0 or level is not None,
        )
        y, x = self._transform_dataframes(Y_df, X_df)
        response_timegpt = self.client.timegpt_multi_series(
            y=y,
            x=x,
            fh=h,
            freq=freq,
            level=level,
            finetune_steps=finetune_steps,
            clean_ex_first=clean_ex_first,
        )
        if "weights_x" in response_timegpt["data"]:
            self.weights_x = pd.DataFrame(
                {
                    "features": x_cols,
                    "weights": response_timegpt["data"]["weights_x"],
                }
            )
        return pd.DataFrame(**response_timegpt["data"]["forecast"])

    def _hit_multi_series_historic_endpoint(
        self,
        Y_df: pd.DataFrame,
        freq: str,
        level: Optional[List[Union[int, float]]],
        input_size: int,
        model_horizon: int,
    ):
        self._validate_input_size(
            Y_df=Y_df,
            input_size=input_size,
            model_horizon=model_horizon,
            require_history=True,
        )
        y, x = self._transform_dataframes(Y_df, None)
        response_timegpt = self.client.timegpt_multi_series_historic(
            freq=freq, level=level, y=y
        )
        return pd.DataFrame(**response_timegpt["data"]["forecast"])

    def _multi_series(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        X_df: Optional[pd.DataFrame],
        level: Optional[List[Union[int, float]]],
        finetune_steps: int,
        clean_ex_first: bool,
        add_history: bool,
        date_features: Union[bool, List[str]],
        date_features_to_one_hot: Union[bool, List[str]],
    ):
        if freq is None:
            freq = self._infer_freq(df)
        # add date features logic
        if isinstance(date_features, bool):
            if date_features:
                date_features = date_features_by_freq.get(freq)
                if date_features is None:
                    warnings.warn(
                        f"Non default date features for {freq} "
                        "please pass a list of date features"
                    )
            else:
                date_features = None

        if date_features is not None:
            if isinstance(date_features_to_one_hot, bool):
                if date_features_to_one_hot:
                    date_features_to_one_hot = date_features
                else:
                    date_features_to_one_hot = None
            df, X_df = self._add_date_features(
                df=df,
                X_df=X_df,
                h=h,
                freq=freq,
                date_features=date_features,
                date_features_to_one_hot=date_features_to_one_hot,
            )
        main_logger.info("Preprocessing dataframes...")
        Y_df, X_df, x_cols = self._preprocess_dataframes(
            df=df,
            h=h,
            X_df=X_df,
            freq=freq,
        )
        input_size, model_horizon = self._get_model_params(freq)
        main_logger.info("Calling Forecast Endpoint...")
        fcst_df = self._hit_multi_series_endpoint(
            Y_df=Y_df,
            X_df=X_df,
            h=h,
            freq=freq,
            clean_ex_first=clean_ex_first,
            finetune_steps=finetune_steps,
            x_cols=x_cols,
            level=level,
            input_size=input_size,
            model_horizon=model_horizon,
        )
        if add_history:
            main_logger.info("Calling Historical Forecast Endpoint...")
            fitted_df = self._hit_multi_series_historic_endpoint(
                Y_df=Y_df,
                freq=freq,
                level=level,
                input_size=input_size,
                model_horizon=model_horizon,
            )
            fitted_df = fitted_df.drop(columns="y")
            fcst_df = pd.concat([fitted_df, fcst_df]).sort_values(["unique_id", "ds"])
        return fcst_df

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: Optional[str] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        X_df: Optional[pd.DataFrame] = None,
        level: Optional[List[Union[int, float]]] = None,
        finetune_steps: int = 0,
        clean_ex_first: bool = True,
        validate_token: bool = False,
        add_history: bool = False,
        date_features: Union[bool, List[str]] = False,
        date_features_to_one_hot: Union[bool, List[str]] = True,
    ):
        """Forecast your time series using TimeGPT.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame on which the function will operate. Expected to contain at least the following columns:
            - time_col:
                Column name in `df` that contains the time indices of the time series. This is typically a datetime
                column with regular intervals, e.g., hourly, daily, monthly data points.
            - target_col:
                Column name in `df` that contains the target variable of the time series, i.e., the variable we
                wish to predict or analyze.
            Additionally, you can pass multiple time series (stacked in the dataframe) considering an additional column:
            - id_col:
                Column name in `df` that identifies unique time series. Each unique value in this column
                corresponds to a unique time series.
        h : int
            Forecast horizon.
        freq : str
            Frequency of the data. By default, the freq will be inferred automatically.
            See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        X_df : pandas.DataFrame, optional (default=None)
            DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous.
        level : List[float], optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.
        finetune_steps : int (default=0)
            Number of steps used to finetune TimeGPT in the
            new data.
        clean_ex_first : bool (default=True)
            Clean exogenous signal before making forecasts
            using TimeGPT.
        validate_token : bool (default=False)
            If True, validates token before
            sending requests.
        add_history : bool (default=False)
            Return fitted values of the model.
        date_features : bool or list of str or callable, optional (default=False)
            Features computed from the dates.
            Can be pandas date attributes or functions that will take the dates as input.
            If True automatically adds most used date features for the
            frequency of `df`.
        date_features_to_one_hot : bool or list of str (default=True)
            Apply one-hot encoding to these date features.
            If `date_features=True`, then all date features are
            one-hot encoded by default.

        Returns
        -------
        fcsts_df : pandas.DataFrame
            DataFrame with TimeGPT forecasts for point predictions and probabilistic
            predictions (if level is not None).
        """
        if not self.validate_token(log=False):
            raise Exception("Token not valid, please email ops@nixtla.io")

        df, X_df, drop_uid = self._validate_inputs(
            df=df,
            X_df=X_df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        fcst_df = self._multi_series(
            df=df,
            h=h,
            freq=freq,
            X_df=X_df,
            level=level,
            finetune_steps=finetune_steps,
            clean_ex_first=clean_ex_first,
            add_history=add_history,
            date_features=date_features,
            date_features_to_one_hot=date_features_to_one_hot,
        )
        fcst_df = self._validate_outputs(
            fcst_df=fcst_df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            drop_uid=drop_uid,
        )
        return fcst_df
