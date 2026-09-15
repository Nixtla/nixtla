"""The `client.jobs` namespace: submitting work that runs server-side.

Every method here mirrors the blocking `NixtlaClient` method of the same name,
but returns a `Job` handle instead of a result. Nothing here holds state of its
own -- payload construction, HTTP and polling all stay on the client, reached
through `self._client`.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from utilsforecast.compat import DataFrame, DFType

from . import _async_transport
from .async_job import Job
from .nixtla_client import _ensure_local_dataframe, _validate_simulate_args
from ._types import (
    _ANOMALY_DETECTION_ENDPOINT,
    _ExplainMethod,
    _ExtraParamDataType,
    _FinetuneDepth,
    _Freq,
    _Loss,
    _Model,
    _NonNegativeInt,
    _PositiveInt,
    _ThresholdMethod,
    extra_param_checker,
)
from .steps import build_request as _build_step_request

if TYPE_CHECKING:
    from .nixtla_client import NixtlaClient


class Jobs:
    """Submit long-running work to run server-side, reached as `client.jobs`.

    Each method submits a job and returns immediately with a `Job` handle
    rather than blocking for the result: call `job.wait()` to poll until it
    finishes and get the result, `job.refresh()` to check on it, or
    `job.cancel()` to ask the server to stop it.

    Not constructed directly -- `NixtlaClient` builds it on first access to
    `client.jobs`.
    """

    def __init__(self, client: "NixtlaClient"):
        self._client = client

    def forecast(
        self,
        df: DFType,
        h: _PositiveInt,
        freq: Optional[_Freq] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        X_df: Optional[DFType] = None,
        level: Optional[list[Union[int, float]]] = None,
        quantiles: Optional[list[float]] = None,
        finetune_steps: _NonNegativeInt = 0,
        finetune_depth: _FinetuneDepth = 1,
        finetune_loss: _Loss = "default",
        finetuned_model_id: Optional[str] = None,
        clean_ex_first: bool = True,
        hist_exog_list: Optional[list[str]] = None,
        categorical_exog_list: Optional[list[str]] = None,
        validate_api_key: bool = False,
        date_features: Union[bool, list[Union[str, Callable]]] = False,
        date_features_to_one_hot: Union[bool, list[str]] = False,
        model: _Model = "timegpt-2.1",
        feature_contributions: bool = False,
        model_parameters: _ExtraParamDataType = None,
        multivariate: bool = False,
        job_timeout_seconds: Optional[int] = None,
    ) -> Job:
        """Submit a forecast job to run asynchronously.

        Unlike `forecast()`, this does not block until the job finishes. It
        submits the job and immediately returns a `Job` handle; call
        `job.wait()` to poll until it completes and get the resulting
        DataFrame, or `job.cancel()` to request that the server stop it.

        Not supported in this version: `num_partitions` (distributed/threaded
        fan-out) and `add_history`. Use `forecast()` for those.

        Args:
            df (pandas or polars DataFrame): The DataFrame on which the
                function will operate. Expected to contain at least the
                following columns:
                - time_col:
                    Column name in `df` that contains the time indices of
                    the time series. This is typically a datetime column
                    with regular intervals, e.g., hourly, daily, monthly
                    data points.
                - target_col:
                    Column name in `df` that contains the target variable of
                    the time series, i.e., the variable we wish to predict
                    or analyze.
                Additionally, you can pass multiple time series (stacked in
                    the dataframe) considering an additional column:
                - id_col:
                    Column name in `df` that identifies unique time series.
                    Each unique value in this column corresponds to a unique
                    time series.
            h (int): Forecast horizon.
            freq (str, int or pandas offset, optional): Frequency of the
                timestamps. If `None`, it will be inferred automatically.
                See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
                Defaults to None.
            id_col (str): Column that identifies each series. Defaults to
                'unique_id'.
            time_col (str): Column that identifies each timestep, its values
                can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            X_df (pandas or polars DataFrame, optional):
                DataFrame with [`unique_id`, `ds`] columns and `df`'s future
                exogenous. Defaults to None.
            level (list[float], optional): Confidence levels between 0 and 100
                for prediction intervals. Defaults to None.
            quantiles (list[float], optional): Quantiles to forecast, list
                between (0, 1). `level` and `quantiles` should not be
                used simultaneously. The output dataframe will have
                the quantile columns formatted as TimeGPT-q-(100 * q) for each
                q. 100 * q represents percentiles but we choose this notation
                to avoid having dots in column names. Defaults to None.
            finetune_steps (int): Number of steps used to finetune learning
                TimeGPT in the new data. Defaults to 0.
            finetune_depth (int): The depth of the finetuning. Uses a scale
                from 1 to 5, where 1 means little finetuning, and 5 means that
                the entire model is finetuned. Defaults to 1.
            finetune_loss (str): Loss function to use for finetuning. Options
                are: `default`, `mae`, `mse`, `rmse`, `mape`, and `smape`.
                Defaults to 'default'.
            finetuned_model_id (str, optional): ID of previously fine-tuned model
                to use. Defaults to None.
            clean_ex_first (bool): Clean exogenous signal before making
                forecasts using TimeGPT. Defaults to True.
            hist_exog_list (list[str], optional): Column names of the
                historical exogenous features. Defaults to None.
            categorical_exog_list (list[str], optional): Column names of
                categorical exogenous features (can be strings or numbers).
                Future categoricals must be provided via `X_df`; historical-only
                categoricals must appear in `df` and be listed in
                `hist_exog_list`. Defaults to None.
            validate_api_key (bool):
                If True, validates api_key before sending requests. Defaults
                to False.
            date_features (bool or list[str] or callable, optional): Features
                computed from the dates. Can be pandas date attributes
                or functions that will take the dates as input. If True
                automatically adds most used date features for the
                frequency of `df`. Defaults to False.
            date_features_to_one_hot (bool or list[str]): Apply one-hot
                encoding to these date features. If
                `date_features=True`, then all date features are
                one-hot encoded by default. Defaults to False.
            model (str): Model to use as a string. Options are: `timegpt-1`,
                and `timegpt-1-long-horizon`,`timegpt-2`, `timegpt-2-mini`,
                `timegpt-2-pro`, `timegpt-2.1`. We recommend using
                `timegpt-1-long-horizon` for forecasting if you want to
                predict more than one seasonal period given the frequency of
                your data. Defaults to 'timegpt-2.1'.
            feature_contributions (bool): Compute SHAP values.
                Gives access to computed SHAP values to explain the impact
                of features on the final predictions. Defaults to False.
            model_parameters (dict): The dictionary settings that determine
                the behavior of the model. Default is None
            multivariate (bool): If True, enables multivariate predictions.
                Defaults to False. Note: multivariate predictions are only
                supported for a select set of TimeGPT models.
            job_timeout_seconds (int, optional): Maximum seconds the server
                allows this job to run before terminating it server-side.
                This is separate from `poll_timeout` in `Job.wait()`, which
                only controls how long the client polls locally. Capped by a
                server-side per-task maximum; requesting a higher value
                raises `ApiError` (422) when submitting. Defaults to the
                server's default for this task type if not specified.

        Returns:
            Job: Handle to the submitted job. `job.wait()` returns a pandas
                or polars DataFrame with TimeGPT forecasts.
        """
        extra_param_checker.validate_python(model_parameters)
        _ensure_local_dataframe(
            df, method_name="jobs.forecast()", sync_method_name="forecast()"
        )
        payload, _, _, _, parse_result = self._client._prepare_forecast(
            df=df,
            h=h,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            X_df=X_df,
            level=level,
            quantiles=quantiles,
            finetune_steps=finetune_steps,
            finetune_depth=finetune_depth,
            finetune_loss=finetune_loss,
            finetuned_model_id=finetuned_model_id,
            clean_ex_first=clean_ex_first,
            hist_exog_list=hist_exog_list,
            categorical_exog_list=categorical_exog_list,
            validate_api_key=validate_api_key,
            add_history=False,
            date_features=date_features,
            date_features_to_one_hot=date_features_to_one_hot,
            model=model,
            feature_contributions=feature_contributions,
            model_parameters=model_parameters,
            multivariate=multivariate,
        )
        return _async_transport.submit_and_wrap_job(
            self._client,
            "v2/forecast",
            payload,
            job_timeout_seconds,
            parse_result,
            task="forecast",
        )

    def cross_validation(
        self,
        df: DFType,
        h: _PositiveInt,
        freq: Optional[_Freq] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        level: Optional[list[Union[int, float]]] = None,
        quantiles: Optional[list[float]] = None,
        validate_api_key: bool = False,
        n_windows: _PositiveInt = 1,
        step_size: Optional[_PositiveInt] = None,
        finetune_steps: _NonNegativeInt = 0,
        finetune_depth: _FinetuneDepth = 1,
        finetune_loss: _Loss = "default",
        finetuned_model_id: Optional[str] = None,
        refit: bool = True,
        clean_ex_first: bool = True,
        hist_exog_list: Optional[list[str]] = None,
        date_features: Union[bool, list[str]] = False,
        date_features_to_one_hot: Union[bool, list[str]] = False,
        model: _Model = "timegpt-2.1",
        model_parameters: _ExtraParamDataType = None,
        multivariate: bool = False,
        categorical_exog_list: Optional[list[str]] = None,
        job_timeout_seconds: Optional[int] = None,
    ) -> Job:
        """Submit a cross-validation job to run asynchronously.

        Unlike `cross_validation()`, this does not block until the job
        finishes. It submits the job and immediately returns a `Job` handle;
        call `job.wait()` to poll until it completes and get the resulting
        DataFrame, or `job.cancel()` to request that the server stop it.

        Not supported in this version: `num_partitions` (distributed/threaded
        fan-out). Use `cross_validation()` for that.

        Args:
            df (pandas or polars DataFrame): The DataFrame on which the
                function will operate. Expected to contain at least the
                following columns:
                - time_col:
                    Column name in `df` that contains the time indices of the
                    time series. This is typically a datetime column with
                    regular intervals, e.g., hourly, daily, monthly data points.
                - target_col:
                    Column name in `df` that contains the target variable of the
                    time series, i.e., the variable we wish to predict or analyze.
                Additionally, you can pass multiple time series (stacked in the
                dataframe) considering an additional column:
                - id_col:
                    Column name in `df` that identifies unique time series.
                    Each unique value in this column corresponds to a unique
                    time series.
            h (int): Forecast horizon.
            freq (str, int or pandas offset, optional): Frequency of the
                timestamps. If `None`, it will be inferred automatically.
                See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
                Defaults to None.
            id_col (str): Column that identifies each series. Defaults to
                'unique_id'.
            time_col (str): Column that identifies each timestep, its values
                can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            level (list[float], optional): Confidence levels between 0 and 100
                for prediction intervals. Defaults to None.
            quantiles (list[float], optional): Quantiles to forecast, list
                between (0, 1). `level` and `quantiles` should not be
                used simultaneously. The output dataframe will have
                the quantile columns formatted as TimeGPT-q-(100 * q) for each
                q. 100 * q represents percentiles but we choose this notation
                to avoid having dots in column names. Defaults to None.
            validate_api_key (bool): If True, validates api_key before sending
                requests. Defaults to False.
            n_windows (int): Number of windows to evaluate. Defaults to 1.
            step_size (int, optional): Step size between each cross validation
                window. If None it will be equal to `h`. Defaults to None.
            finetune_steps (int): Number of steps used to finetune learning
                TimeGPT in the new data. Defaults to 0.
            finetune_depth (int): The depth of the finetuning. Uses a scale
                from 1 to 5, where 1 means little finetuning, and 5 means that
                the entire model is finetuned. Defaults to 1.
            finetune_loss (str): Loss function to use for finetuning. Options
                are: `default`, `mae`, `mse`, `rmse`, `mape`, and `smape`.
                Defaults to 'default'.
            finetuned_model_id (str, optional): ID of previously fine-tuned
                model to use. Defaults to None.
            refit (bool):
                Fine-tune the model in each window. If `False`, only
                fine-tunes on the first window. Only used if `finetune_steps`
                > 0. Defaults to True.
            clean_ex_first (bool):
                Clean exogenous signal before making forecasts using TimeGPT.
                Defaults to True.
            hist_exog_list (list[str], optional):
                Column names of the historical exogenous features. Defaults
                to None.
            date_features (bool or list[str] or callable, optional): Features
                computed from the dates. Can be pandas date attributes
                or functions that will take the dates as input. If True
                automatically adds most used date features for the
                frequency of `df`. Defaults to False.
            date_features_to_one_hot (bool or list[str]): Apply one-hot
                encoding to these date features. If
                `date_features=True`, then all date features are
                one-hot encoded by default. Defaults to False.
            model (str): Model to use as a string. Options are: `timegpt-1`,
                and `timegpt-1-long-horizon`, `timegpt-2`, `timegpt-2-mini`,
                `timegpt-2-pro`, `timegpt-2.1`. We recommend using
                `timegpt-1-long-horizon` for forecasting if you want to
                predict more than one seasonal period given the frequency of
                your data. Defaults to 'timegpt-2.1'.
            model_parameters (dict): The dictionary settings that determine
                the behavior of the model. Default is None.
            multivariate (bool): If True, enables multivariate predictions.
                Defaults to False. Note: multivariate predictions are only
                supported for a select set of TimeGPT models.
            categorical_exog_list (list[str], optional): Column names of
                categorical exogenous features in (can be strings or
                numbers). Defaults to None.
            job_timeout_seconds (int, optional): Maximum seconds the server
                allows this job to run before terminating it server-side.
                This is separate from `poll_timeout` in `Job.wait()`, which
                only controls how long the client polls locally. Capped by a
                server-side per-task maximum; requesting a higher value
                raises `ApiError` (422) when submitting. Defaults to the
                server's default for this task type if not specified.

        Returns:
            Job: Handle to the submitted job. `job.wait()` returns a pandas
                or polars DataFrame with cross validation forecasts.
        """
        extra_param_checker.validate_python(model_parameters)
        _ensure_local_dataframe(
            df,
            method_name="jobs.cross_validation()",
            sync_method_name="cross_validation()",
        )
        payload, parse_result = self._client._prepare_cross_validation(
            df=df,
            h=h,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
            quantiles=quantiles,
            validate_api_key=validate_api_key,
            n_windows=n_windows,
            step_size=step_size,
            finetune_steps=finetune_steps,
            finetune_depth=finetune_depth,
            finetune_loss=finetune_loss,
            finetuned_model_id=finetuned_model_id,
            refit=refit,
            clean_ex_first=clean_ex_first,
            hist_exog_list=hist_exog_list,
            date_features=date_features,
            date_features_to_one_hot=date_features_to_one_hot,
            model=model,
            model_parameters=model_parameters,
            multivariate=multivariate,
            categorical_exog_list=categorical_exog_list,
        )
        return _async_transport.submit_and_wrap_job(
            self._client,
            "v2/cross_validation",
            payload,
            job_timeout_seconds,
            parse_result,
            task="cross_validation",
        )

    def finetune(
        self,
        df: DataFrame,
        freq: Optional[_Freq] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        finetune_steps: _NonNegativeInt = 10,
        finetune_depth: _FinetuneDepth = 1,
        finetune_loss: _Loss = "default",
        output_model_id: Optional[str] = None,
        finetuned_model_id: Optional[str] = None,
        model: _Model = "timegpt-2.1",
        job_timeout_seconds: Optional[int] = None,
    ) -> Job:
        """Submit a fine-tuning job to run asynchronously.

        Unlike `finetune()`, this does not block until the job finishes. It
        submits the job and immediately returns a `Job` handle; call
        `job.wait()` to poll until it completes and get the fine-tuned model
        id, or `job.cancel()` to request that the server stop it.

        Args:
            df (pandas or polars DataFrame): The DataFrame on which the
                function will operate. Expected to contain at least the
                following columns:
                - time_col:
                    Column name in `df` that contains the time indices of
                    the time series. This is typically a datetime column with
                    regular intervals, e.g., hourly, daily, monthly data
                    points.
                - target_col:
                    Column name in `df` that contains the target variable of
                    the time series, i.e., the variable we wish to predict
                    or analyze.
                Additionally, you can pass multiple time series (stacked in
                the dataframe) considering an additional column:
                - id_col:
                    Column name in `df` that identifies unique time series.
                    Each unique value in this column corresponds to a unique
                    time series.
            freq (str, int, pandas offset, optional): Frequency of the
                timestamps.  If `None`, it will be inferred automatically.
                See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
                Defaults to None.
            id_col (str): Column that identifies each series. Defaults to
                'unique_id'.
            time_col (str): Column that identifies each timestep, its values
                can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            finetune_steps (int): Number of steps used to finetune learning
                TimeGPT in the new data. Defaults to 10.
            finetune_depth (int): The depth of the finetuning. Uses a scale
                from 1 to 5, where 1 means little finetuning, and 5 means that
                the entire model is finetuned. Defaults to 1.
            finetune_loss (str): Loss function to use for finetuning. Options
                are: `default`, `mae`, `mse`, `rmse`, `mape`, and `smape`.
                Defaults to 'default'.
            output_model_id (str, optional): ID to assign to the fine-tuned model.
                If `None`, an UUID is used. Defaults to None.
            finetuned_model_id (str, optional): ID of previously fine-tuned
                model to use as base. Defaults to None.
            model (str):
                Model to use as a string. Options are: `timegpt-1`, and
                `timegpt-1-long-horizon`, `timegpt-2`, `timegpt-2-mini`, `timegpt-2-pro`,
                `timegpt-2.1`. We recommend using
                `timegpt-1-long-horizon` for forecasting if you want to
                predict more than one seasonal period given the frequency
                of your data. Defaults to 'timegpt-2.1'.
            job_timeout_seconds (int, optional): Maximum seconds the server
                allows this job to run before terminating it server-side.
                This is separate from `poll_timeout` in `Job.wait()`, which
                only controls how long the client polls locally. Capped by a
                server-side per-task maximum; requesting a higher value
                raises `ApiError` (422) when submitting. Defaults to the
                server's default for this task type if not specified.

        Returns:
            Job: Handle to the submitted job. `job.wait()` returns the
                fine-tuned model id (str).
        """
        payload = self._client._prepare_finetune_payload(
            df=df,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            finetune_steps=finetune_steps,
            finetune_depth=finetune_depth,
            finetune_loss=finetune_loss,
            output_model_id=output_model_id,
            finetuned_model_id=finetuned_model_id,
            model=model,
        )
        return _async_transport.submit_and_wrap_job(
            self._client,
            "v2/finetune",
            payload,
            job_timeout_seconds,
            lambda resp: resp["finetuned_model_id"],
            task="finetune",
        )

    def detect_anomalies(
        self,
        df: DFType,
        h: _PositiveInt,
        detection_size: _PositiveInt,
        threshold_method: _ThresholdMethod = "univariate",
        freq: Optional[_Freq] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        level: Union[int, float] = 99,
        clean_ex_first: bool = True,
        step_size: Optional[_PositiveInt] = None,
        finetune_steps: _NonNegativeInt = 0,
        finetune_depth: _FinetuneDepth = 1,
        finetune_loss: _Loss = "default",
        finetuned_model_id: Optional[str] = None,
        hist_exog_list: Optional[list[str]] = None,
        date_features: Union[bool, list[str]] = False,
        date_features_to_one_hot: Union[bool, list[str]] = False,
        model: _Model = "timegpt-2.1",
        model_parameters: _ExtraParamDataType = None,
        refit: bool = False,
        multivariate: bool = False,
        job_timeout_seconds: Optional[int] = None,
    ) -> Job:
        """Submit an online anomaly detection job to run asynchronously.

        Unlike `detect_anomalies_online()`, this does not block until the job
        finishes. It submits the job and immediately returns a `Job` handle;
        call `job.wait()` to poll until it completes and get the resulting
        DataFrame, or `job.cancel()` to request that the server stop it.

        Not supported in this version: `num_partitions` (distributed/threaded
        fan-out). Use `detect_anomalies_online()` for that.

        Args:
            df (pandas or polars DataFrame):
                The DataFrame on which the function will operate. Expected
                to contain at least the following columns:
                - time_col:
                    Column name in `df` that contains the time indices of the
                    time series. This is typically a datetime column with
                    regular intervals, e.g., hourly, daily, monthly data
                    points.
                - target_col:
                    Column name in `df` that contains the target variable of
                    the time series, i.e., the variable we wish to predict or
                    analyze.
                - id_col:
                    Column name in `df` that identifies unique time series.
                    Each unique value in this column corresponds to a unique
                    time series.

            h (int): Forecast horizon.
            detection_size (int): The length of the sequence where anomalies
                will be detected starting from the end of the dataset.
            threshold_method (str, optional): The method used to calculate the
                intervals for anomaly detection. Use `univariate` to flag
                anomalies independently for each series in the dataset.
                Use `multivariate` to have a global threshold across all series
                in the dataset. For this method, all series must have the same
                length. Defaults to 'univariate'.
            freq (str, optional): Frequency of the data. By default, the freq
                will be inferred automatically. See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
            id_col (str, optional): Column that identifies each series.
                Defaults to 'unique_id'
            time_col (str, optional): Column that identifies each timestep,
                its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str, optional): Column that contains the target.
                Defaults to 'y'.
            level (float, optional):
                Confidence level between 0 and 100 for detecting the anomalies.
                Defaults to 99.
            clean_ex_first (bool, optional): Clean exogenous signal before
                making forecasts using TimeGPT. Defaults to True.
            step_size (int, optional): Step size between each cross validation
                window. If None it will be equal to `h`. Defaults to None.
            finetune_steps (int): Number of steps used to finetune TimeGPT in
                the new data. Defaults to 0.
            finetune_depth (int): The depth of the finetuning. Uses a scale
                from 1 to 5, where 1 means little finetuning, and 5 means that
                the entire model is finetuned. Defaults to 1.
            finetune_loss (str): Loss function to use for finetuning.
                Options are: `default`, `mae`, `mse`, `rmse`, `mape`, and
                `smape`. Defaults to 'default'.
            finetuned_model_id (str, optional): ID of previously fine-tuned model
                to use. Defaults to None.
            hist_exog_list (list[str], optional): Column names of the historical
                exogenous features. Defaults to None.
            date_features (bool or list[str] or callable, optional): Features
                computed from the dates. Can be pandas date attributes
                or functions that will take the dates as input. If True
                automatically adds most used date features for the
                frequency of `df`. Defaults to False.
            date_features_to_one_hot (bool or list[str]): Apply one-hot
                encoding to these date features. If
                `date_features=True`, then all date features are
                one-hot encoded by default. Defaults to False.
            model (str, optional): Model to use as a string. Options are:
                `timegpt-1`, and `timegpt-1-long-horizon`, `timegpt-2`,
                `timegpt-2-mini`, `timegpt-2-pro`, `timegpt-2.1`.
                We recommend using
                `timegpt-1-long-horizon` for forecasting if you want to
                predict more than one seasonal period given the frequency of
                your data. Defaults to 'timegpt-2.1'.
            model_parameters (dict): The dictionary settings that determine
                the behavior of the model. Default is None.
            refit (bool, optional): Fine-tune the model in each window. If
                False, only fine-tunes on the first window. Only used if
                finetune_steps > 0. Defaults to False.
            multivariate (bool): If True, enables multivariate predictions.
                Defaults to False. Note: multivariate predictions are only
                supported for a select set of TimeGPT models. This variable
                is different from the `threshold_method` parameter. The latter
                controls the method used for anomaly detection (univariate vs
                multivariate) whereas `multivariate` determines how the model
                creates the predictions.
            job_timeout_seconds (int, optional): Maximum seconds the server
                allows this job to run before terminating it server-side.
                This is separate from `poll_timeout` in `Job.wait()`, which
                only controls how long the client polls locally. Capped by a
                server-side per-task maximum; requesting a higher value
                raises `ApiError` (422) when submitting. Defaults to the
                server's default for this task type if not specified.

        Returns:
            Job: Handle to the submitted job. `job.wait()` returns a pandas
                or polars DataFrame with anomalies flagged by TimeGPT.
        """
        extra_param_checker.validate_python(model_parameters)
        _ensure_local_dataframe(
            df,
            method_name="jobs.detect_anomalies()",
            sync_method_name="detect_anomalies_online()",
        )
        payload, parse_result = self._client._prepare_anomaly_detection(
            df=df,
            h=h,
            detection_size=detection_size,
            threshold_method=threshold_method,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
            clean_ex_first=clean_ex_first,
            step_size=step_size,
            finetune_steps=finetune_steps,
            finetune_depth=finetune_depth,
            finetune_loss=finetune_loss,
            finetuned_model_id=finetuned_model_id,
            hist_exog_list=hist_exog_list,
            date_features=date_features,
            date_features_to_one_hot=date_features_to_one_hot,
            model=model,
            model_parameters=model_parameters,
            refit=refit,
            multivariate=multivariate,
        )
        return _async_transport.submit_and_wrap_job(
            self._client,
            _ANOMALY_DETECTION_ENDPOINT,
            payload,
            job_timeout_seconds,
            parse_result,
            task="anomaly_detection",
        )

    def simulate(
        self,
        df: DataFrame,
        h: _PositiveInt,
        freq: Optional[_Freq] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        X_df: Optional[DataFrame] = None,
        n_paths: _PositiveInt = 100,
        quantiles: Optional[list[float]] = None,
        seed: Optional[int] = None,
        finetuned_model_id: Optional[str] = None,
        clean_ex_first: bool = True,
        hist_exog_list: Optional[list[str]] = None,
        categorical_exog_list: Optional[list[str]] = None,
        validate_api_key: bool = False,
        date_features: Union[bool, list[Union[str, Callable]]] = False,
        date_features_to_one_hot: Union[bool, list[str]] = False,
        model: _Model = "timegpt-2.1",
        multivariate: bool = False,
        job_timeout_seconds: Optional[int] = None,
    ) -> Job:
        """Submit a simulation job to run asynchronously.

        Unlike `simulate()`, this does not block until the job finishes. It
        submits the job and immediately returns a `Job` handle; call
        `job.wait()` to poll until it completes and get the resulting
        DataFrame, or `job.cancel()` to request that the server stop it.

        Not supported in this version: `num_partitions` (concurrent fan-out).
        Use `simulate()` for that.

        Args:
            df (pandas or polars DataFrame): Historical time series data.
                It must contain the time and target columns and may contain an
                ID column and exogenous feature columns.
            h (int): Number of future timesteps in every sample path.
            freq (str, int or pandas offset, optional): Frequency of the
                timestamps. If `None`, it is inferred from `df` (pandas only);
                pass it explicitly for polars.
            id_col (str): Column that identifies each series. Defaults to
                `"unique_id"`.
            time_col (str): Column that identifies each timestep. Defaults to
                `"ds"`.
            target_col (str): Column that contains the target. Defaults to
                `"y"`.
            X_df (pandas or polars DataFrame, optional): Future exogenous
                values with ID and time columns.
            n_paths (int): Number of paths generated for each series. Must be
                between 1 and 10,000. Defaults to 100.
            quantiles (list[float], optional): Strictly increasing marginal
                quantiles inside `(0, 1)`. Between 2 and 200 values may be
                provided. They refine the marginal distribution the paths are
                drawn from and add no columns to the result.
            seed (int, optional): Random seed. Reusing a seed with the same
                inputs produces the same paths. Must be between `-2**63` and
                `2**64 - 1`.
            finetuned_model_id (str, optional): ID of a previously fine-tuned
                model.
            clean_ex_first (bool): Clean exogenous signals before inference.
                Defaults to True.
            hist_exog_list (list[str], optional): Historical-only exogenous
                feature names.
            categorical_exog_list (list[str], optional): Categorical
                exogenous feature names.
            validate_api_key (bool): Validate the API key before the request.
                Defaults to False.
            date_features (bool or list, optional): Date-derived exogenous
                features to add.
            date_features_to_one_hot (bool or list[str]): Date features to
                one-hot encode.
            model (str): Model used to generate the marginal forecasts.
                Defaults to `"timegpt-2.1"`.
            multivariate (bool): Request coherent paths across series. The
                returned `coupled` column reports whether cross-series
                coupling was applied. Defaults to False.
            job_timeout_seconds (int, optional): Maximum seconds the server
                allows this job to run before terminating it server-side.
                This is separate from `poll_timeout` in `Job.wait()`, which
                only controls how long the client polls locally. Capped by a
                server-side per-task maximum; requesting a higher value
                raises `ApiError` (422) when submitting. Defaults to the
                server's default for this task type if not specified.

        Returns:
            Job: Handle to the submitted job. `job.wait()` returns a pandas or
                polars DataFrame of long-format sample paths with ID, time,
                `sample_id`, `TimeGPT`, and `coupled` columns.
        """
        h, n_paths, seed, _ = _validate_simulate_args(
            h, n_paths, seed, None, multivariate
        )
        payload, parse_result = self._client._prepare_simulate(
            df=df,
            h=h,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            X_df=X_df,
            n_paths=n_paths,
            quantiles=quantiles,
            seed=seed,
            finetuned_model_id=finetuned_model_id,
            clean_ex_first=clean_ex_first,
            hist_exog_list=hist_exog_list,
            categorical_exog_list=categorical_exog_list,
            validate_api_key=validate_api_key,
            date_features=date_features,
            date_features_to_one_hot=date_features_to_one_hot,
            model=model,
            multivariate=multivariate,
            method_name="jobs.simulate()",
        )
        return _async_transport.submit_and_wrap_job(
            self._client,
            "v2/simulate",
            payload,
            job_timeout_seconds,
            parse_result,
            task="simulate",
        )

    def explain(
        self,
        df: DataFrame,
        method: _ExplainMethod = "granger",
        features: Optional[list[str]] = None,
        freq: Optional[_Freq] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        categorical_exog_list: Optional[list[str]] = None,
        validate_api_key: bool = False,
        job_timeout_seconds: Optional[int] = None,
    ) -> Job:
        """Submit an explanation job to run asynchronously.

        Unlike `explain()`, this does not block until the job finishes. It
        submits the job and immediately returns a `Job` handle; call
        `job.wait()` to poll until it completes and get the resulting
        DataFrame, or `job.cancel()` to request that the server stop it.

        The returned weights describe lagged predictive relationships in the
        supplied data. They do not establish that changing a feature will cause
        the target to change.

        Args:
            df (pandas or polars DataFrame): Historical time series containing
                the target and candidate feature columns.
            method (str): `"granger"` for linear lagged relationships or
                `"transfer_entropy"` for potentially nonlinear relationships.
                Defaults to `"granger"`.
            features (list[str], optional): Features to analyze. By default,
                every column other than the ID, time, and target columns is
                used.
            freq (str, int or pandas offset, optional): Frequency of the
                timestamps, used to verify that every series is complete and
                regularly spaced. If `None`, it is inferred from `df` (pandas
                only); pass it explicitly for polars.
            id_col (str): Column that identifies each series. Defaults to
                `"unique_id"`.
            time_col (str): Column that identifies each timestep. Defaults to
                `"ds"`.
            target_col (str): Column that contains the target. Defaults to
                `"y"`.
            categorical_exog_list (list[str], optional): Feature names that
                should be treated as categorical.
            validate_api_key (bool): Validate the API key before the request.
                Defaults to False.
            job_timeout_seconds (int, optional): Maximum seconds the server
                allows this job to run before terminating it server-side.
                This is separate from `poll_timeout` in `Job.wait()`, which
                only controls how long the client polls locally. Capped by a
                server-side per-task maximum; requesting a higher value
                raises `ApiError` (422) when submitting. Defaults to the
                server's default for this task type if not specified.

        Returns:
            Job: Handle to the submitted job. `job.wait()` returns a pandas or
                polars DataFrame with one row per feature and `feature`,
                `weight`, and `method` columns.
        """
        payload, parse_result = self._client._prepare_explain(
            df=df,
            method=method,
            features=features,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            categorical_exog_list=categorical_exog_list,
            validate_api_key=validate_api_key,
            method_name="jobs.explain()",
        )
        return _async_transport.submit_and_wrap_job(
            self._client,
            "v2/explain",
            payload,
            job_timeout_seconds,
            parse_result,
            task="explain",
        )

    def execute_step(
        self,
        func_name: str,
        params: dict[str, Any],
        data: Optional[dict[str, Any]] = None,
        job_timeout_seconds: Optional[int] = None,
    ) -> Job:
        """Submit a single TSMP step to run asynchronously.

        `execute_step` runs one TSMP top-level API call server-side. Each
        call is independent: no state carries over from one to the next, so
        every request is self-contained and carries its own data.
        This does not block; it submits the job and immediately returns a
        `Job` handle. Call `job.wait()` to poll until it completes and get a
        `StepResult`, or `job.cancel()` to request that the server stop it.

        Tables are referenced from `params` with `nixtla.ref(key)`, naming a
        key of `data`. Because a step's output tables can be passed straight
        back in as the next step's `data`, calls chain without any file or
        byte handling::

            from nixtla import NixtlaClient, ref

            nixtla_client = NixtlaClient()

            step1 = nixtla_client.jobs.execute_step(
                "make_forecast_input",
                {"data": ref("panel"), "freq": "D"},
                data={"panel": df},
            ).wait()

            step2 = nixtla_client.jobs.execute_step(
                "forecast",
                {"resource": ref("result"), "models": ["timegpt-1"], "h": 7},
                data=step1.data,
            ).wait()

        Chaining relies on arrow schema metadata that a pandas round-trip
        discards, so pass `step.data` between steps rather than
        `step.to_pandas()`.

        There is no `model` argument: a step names the models it runs inside
        `params`.

        Not reachable over this transport: `optimize_model`, which requires a
        `tune.Space` that has no JSON encoding. A pandas index is never sent
        as data; a named one is folded into a column, anything else dropped.

        Args:
            func_name (str): TSMP top-level API to run, e.g. `'forecast'`,
                `'make_forecast_input'`, `'cross_validate'`, `'preprocess'`,
                `'select_by_sql'`.
            params (dict): Arguments for that API. Tables are referenced by
                `ref(key)` envelopes naming a key of `data`; everything else
                is passed through as-is and must be JSON serializable.
            data (dict, optional): Tables the params reference, as pyarrow
                Tables or eager pandas/polars DataFrames, keyed by the name
                used in the `ref` envelopes. Keys must be bare names.
                Defaults to None.
            job_timeout_seconds (int, optional): Maximum seconds the server
                allows this job to run before terminating it server-side.
                This is separate from `poll_timeout` in `Job.wait()`, which
                only controls how long the client polls locally. Capped by a
                server-side per-task maximum; requesting a higher value
                raises `ApiError` (422) when submitting. Defaults to the
                server's default for this task type if not specified.

        Raises:
            TypeError: If a `data` value is not a pyarrow Table or an eager
                pandas/polars DataFrame.
            ValueError: If a `ref` names a table that `data` does not supply,
                a `ref` envelope nests another `ref` (the server would ignore
                it), a `data` key is not a bare name, `func_name` is empty or
                over 128 characters, `job_timeout_seconds` is not positive, or
                the request is over one of the server's budgets (metadata
                header size or nesting, table count, body size). All are
                raised locally, before anything is uploaded.

        Returns:
            Job: Handle to the submitted job. `job.wait()` returns a `StepResult` with `.data`
                (result tables as pyarrow Tables) and `.metadata` (the server's `func_name`,
                `result` envelope and output `profile`).
        """
        metadata, body = _build_step_request(
            func_name=func_name,
            params=params,
            data=data,
            job_timeout_seconds=job_timeout_seconds,
        )
        return _async_transport.submit_and_wrap_binary_job(
            self._client, "v2/execute_step", metadata, body, task="execute_step"
        )
