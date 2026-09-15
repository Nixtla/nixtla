__all__ = [
    "ApiError",
    "JobCancelledError",
    "JobError",
    "JobTimeoutError",
    "Job",
    "JobStatus",
    "NixtlaClient",
    "StepResult",
    "ref",
]

import datetime
import functools
from http import HTTPStatus
from importlib.metadata import PackageNotFoundError, version
import logging
import math
import os
from collections.abc import Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from functools import partial
from threading import Event
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    overload,
)

import httpx
import numpy as np
import orjson
import pandas as pd
import utilsforecast.processing as ufp
import zstandard as zstd
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
)
from utilsforecast.compat import DataFrame, DFType, pl_DataFrame
from utilsforecast.validation import ensure_time_dtype, validate_format

from . import _audit, _payloads
from .jobs import Jobs, _transport
from ._http import (
    _is_retriable_error,
    _parse_retry_after,
    ApiError,
    logger,
)
from ._preprocessing import (
    _coerce_coupled_flag,
    _extract_categorical_exog,
    _forecast_payload_to_in_sample,
    _get_in_sample_horizon_and_windows,
    _maybe_drop_id,
    _maybe_infer_freq,
    _parse_in_sample_output,
    _partition_series,
    _preprocess,
    _series_starts,
    _standardize_freq,
    _validate_freq_regularity,
    _validate_simulate_args,
)
from ._types import (
    _ANOMALY_DETECTION_ENDPOINT,
    _ExplainMethod,
    _ExtraParamDataType,
    _FeatureContributionsType,
    _FinetuneDepth,
    _Freq,
    _FreqType,
    _Loss,
    _MAX_CONCURRENT_ASYNC_JOBS,
    _MAX_SEED,
    _MIN_SEED,
    _Model,
    _NonNegativeInt,
    _ONLINE_ANOMALY_DETECTION_ENDPOINT,
    _PositiveInt,
    _ThresholdMethod,
    AnyDFType,
    DistributedDFType,
    extra_param_checker,
    FinetunedModel,
)
from .jobs._job import (
    JobCancelledError,
    JobError,
    JobTimeoutError,
    Job,
    JobStatus,
    _DEFAULT_POLL_INTERVAL,
    _DEFAULT_POLL_TIMEOUT,
    _validate_poll_settings,
)
from .steps import (
    StepResult,
    ref,
)

if TYPE_CHECKING:
    try:
        from fugue import AnyDataFrame
    except ModuleNotFoundError:
        pass
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        pass
    try:
        import plotly
    except ModuleNotFoundError:
        pass
    try:
        import triad
    except ModuleNotFoundError:
        pass

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)


def _resolve_nixtla_client_version() -> Optional[str]:
    try:
        return version("nixtla")
    except PackageNotFoundError:
        return None


def _retry_strategy(max_retries: int, retry_interval: int, max_wait_time: int):
    def after_retry(retry_state: RetryCallState) -> None:
        error = retry_state.outcome.exception()
        logger.error(f"Attempt {retry_state.attempt_number} failed with error: {error}")

    return retry(
        retry=retry_if_exception(_is_retriable_error),
        wait=wait_fixed(retry_interval),
        after=after_retry,
        stop=stop_after_attempt(max_retries) | stop_after_delay(max_wait_time),
        reraise=True,
    )


class NixtlaClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = 60,
        max_retries: int = 6,
        retry_interval: int = 10,
        max_wait_time: int = 6 * 60,
    ):
        """
        Client to interact with the Nixtla API.


        Args:
            api_key (str, optional): The authorization API key to interact
                with the Nixtla API. If not provided, will use the
                NIXTLA_API_KEY environment variable.
            base_url (str, optional): Custom base URL.
                If not provided, will use the NIXTLA_BASE_URL environment
                variable.
            timeout (int, optional): Request timeout in seconds.
                Set to `None` to disable it. Defaults to 60.
            max_retries (int, optional): The maximum number of attempts to
                make when calling the API before giving up. It defines how
                many times the client will retry the API call if it fails.
                Default value is 6, indicating the client will attempt the
                API call up to 6 times in total. Job submissions (including
                `execute_step`) retry only connection failures and HTTP 429;
                read timeouts and gateway errors are not retried because the
                server may already have accepted the job.
            retry_interval (int, optional): The interval in seconds between
                consecutive retry attempts. This is the waiting period before
                the client tries to call the API again after a failed attempt.
                Default value is 10 seconds, meaning the client waits for
                10 seconds between retries. Defaults to 10.
            max_wait_time (int, optional): The maximum total time in seconds
                that the client will spend on all retry attempts before
                giving up. This sets an upper limit on the cumulative
                waiting time for all retry attempts. If this time is
                exceeded, the client will stop retrying and raise an
                exception. Job-submission retry sleeps, including
                `Retry-After`, are capped to the remaining budget; no new
                attempt starts once that budget expires. An in-flight request
                is governed by `timeout`. Default value is 360 seconds.
                The client throws a ReadTimeout error
                after 60 seconds of inactivity. If you want to catch these
                errors, use max_wait_time >> 60. Defaults to 360.

        Note:
            How long the client waits for a server-side asynchronous job is
            set per call, not on the client: `poll_interval` and
            `poll_timeout` on `simulate()`, `explain()` and `Job.wait()`.
        """
        if api_key is None:
            api_key = os.environ["NIXTLA_API_KEY"]
        if base_url is None:
            base_url = os.getenv("NIXTLA_BASE_URL") or "https://api.nixtla.io"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        client_version = _resolve_nixtla_client_version()
        if client_version is not None:
            headers["nixtla-client-version"] = client_version
        self._client_kwargs = {
            "base_url": base_url,
            "headers": headers,
            "timeout": timeout,
        }
        self._retry_settings = {
            "max_retries": max_retries,
            "retry_interval": retry_interval,
            "max_wait_time": max_wait_time,
        }
        self._retry_strategy = _retry_strategy(**self._retry_settings)
        self._model_params: dict[tuple[str, str], tuple[int, int]] = {}
        self._is_azure = "ai.azure" in base_url

    @functools.cached_property
    def jobs(self) -> Jobs:
        """Namespace for work that runs server-side: `client.jobs.forecast(...)`.

        Each method there mirrors the blocking method of the same name but
        returns a `Job` handle instead of a result. See `Jobs`.
        """
        # Built lazily to keep the back-reference out of `__dict__` for clients
        # that Fugue pickles out to Dask/Ray/Spark workers and that never touch
        # `.jobs`.
        return Jobs(self)

    def _encode_payload(
        self,
        payload: dict[str, Any],
        multithreaded_compress: bool,
        *,
        task: str,
    ) -> tuple[bytes, dict[str, str]]:
        """Serialize `payload` and return the request body and headers.

        `task` is the logical operation (`"forecast"`, `"simulate"`, ...) and
        only drives the guidance given when the body exceeds the size limit.
        """

        def ensure_contiguous_if_array(x):
            if not isinstance(x, np.ndarray):
                return x
            if np.issubdtype(x.dtype, np.floating):
                x = np.nan_to_num(
                    np.ascontiguousarray(x, dtype=np.float32),
                    nan=np.nan,
                    posinf=np.finfo(np.float32).max,
                    neginf=np.finfo(np.float32).min,
                    copy=False,
                )
            else:
                x = np.ascontiguousarray(x)
            return x

        def ensure_contiguous_arrays(d: dict[str, Any]) -> None:
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    d[k] = ensure_contiguous_if_array(v)
                elif isinstance(v, list):
                    d[k] = [ensure_contiguous_if_array(x) for x in v]
                elif isinstance(v, dict):
                    ensure_contiguous_arrays(v)

        ensure_contiguous_arrays(payload)
        content = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
        content_size_mb = len(content) / 2**20
        if content_size_mb > 200:
            if task == "explain":
                raise ValueError(
                    f"The payload is too large ({content_size_mb:.0f}MB, limit "
                    "200MB). `explain` cannot be partitioned because the weights "
                    "are pooled across all series. Reduce the number of series, "
                    "the length of the history, or the number of features."
                )
            if task == "simulate" and payload.get("multivariate"):
                raise ValueError(
                    f"The payload is too large ({content_size_mb:.0f}MB, limit "
                    "200MB). `multivariate=True` cannot be partitioned because "
                    "cross-series coupling is computed across all series in a "
                    "single request. Reduce the number of series or the length "
                    "of the history, or set `multivariate=False` to allow "
                    "partitioning."
                )
            raise ValueError(
                f"The payload is too large. Set num_partitions={math.ceil(content_size_mb / 200)}"
            )
        headers = {}
        if "model" in payload:
            headers["nixtla-model"] = payload["model"]
        if content_size_mb > 1:
            threads = -1 if multithreaded_compress else 0
            content = zstd.ZstdCompressor(level=1, threads=threads).compress(content)
            headers["content-encoding"] = "zstd"
        return content, headers

    @staticmethod
    def _parse_json_response(
        resp: httpx.Response,
        expected_status: Union[int, tuple[int, ...]] = HTTPStatus.OK,
    ) -> Any:
        try:
            resp_body = orjson.loads(resp.content)
        except orjson.JSONDecodeError:
            raise ApiError(
                status_code=resp.status_code,
                body=f"Could not parse JSON: {resp.content}",
                retry_after=_parse_retry_after(resp.headers),
            )
        expected = (
            (expected_status,) if isinstance(expected_status, int) else expected_status
        )
        if resp.status_code not in expected:
            raise ApiError(
                status_code=resp.status_code,
                body=resp_body,
                retry_after=_parse_retry_after(resp.headers),
            )
        return resp_body

    def _make_request(
        self,
        client: httpx.Client,
        endpoint: str,
        payload: dict[str, Any],
        multithreaded_compress: bool,
    ) -> dict[str, Any]:
        content, headers = self._encode_payload(
            payload,
            multithreaded_compress,
            task=_transport._task_name(endpoint),
        )
        resp = client.post(url=endpoint, content=content, headers=headers)
        # async job submissions ({endpoint}/async) respond with 202 ACCEPTED
        resp_body = self._parse_json_response(
            resp, expected_status=(HTTPStatus.OK, HTTPStatus.ACCEPTED)
        )
        if isinstance(resp_body, dict) and "data" in resp_body:
            resp_body = resp_body["data"]
        return resp_body

    def _make_request_with_retries(
        self,
        client: httpx.Client,
        endpoint: str,
        payload: dict[str, Any],
        multithreaded_compress: bool = True,
    ) -> dict[str, Any]:
        return self._retry_strategy(self._make_request)(
            client=client,
            endpoint=endpoint,
            payload=payload,
            multithreaded_compress=multithreaded_compress,
        )

    def _get_request(
        self,
        client: httpx.Client,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {"params": params}
        if timeout is not None:
            # Bound the request by the job deadline without increasing any
            # shorter connection/read/write/pool timeout configured by the caller.
            request_kwargs["timeout"] = httpx.Timeout(
                **{
                    phase: min(limit, timeout) if limit is not None else timeout
                    for phase, limit in client.timeout.as_dict().items()
                }
            )
        resp = client.get(endpoint, **request_kwargs)
        return self._parse_json_response(resp)

    def _collect_concurrent_results(
        self,
        requests: Sequence[Callable[[], dict[str, Any]]],
        max_workers: int,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cancellation_event: Optional[Event] = None,
    ) -> list[dict[str, Any]]:
        from tqdm.auto import tqdm

        results: list[dict[str, Any]] = [{} for _ in requests]
        errors: list[BaseException] = []

        def run(request: Callable[[], dict[str, Any]]) -> dict[str, Any]:
            if cancellation_event is not None and cancellation_event.is_set():
                raise CancelledError
            try:
                return request()
            except BaseException as exc:
                if not isinstance(exc, CancelledError):
                    errors.append(exc)
                if cancellation_event is not None:
                    # Signal before this worker can pick up another request.
                    cancellation_event.set()
                raise

        executor = ThreadPoolExecutor(max_workers)
        future2pos = {}
        try:
            future2pos = {
                executor.submit(run, request): i for i, request in enumerate(requests)
            }
            for future in tqdm(as_completed(future2pos), total=len(future2pos)):
                pos = future2pos[future]
                res = future.result()
                results[pos] = transform(res) if transform is not None else res
        except BaseException as exc:
            if cancellation_event is not None:
                cancellation_event.set()
            for future in future2pos:
                future.cancel()
            if isinstance(exc, CancelledError) and errors:
                # A sibling's cancellation may finish before the failed
                # future is collected. Preserve the original failure.
                raise errors[0]
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        return results

    def _dispatch_partitioned_requests(
        self,
        client: httpx.Client,
        endpoint: str,
        payloads: list[dict[str, Any]],
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        *,
        is_async_job: bool = False,
        poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
        poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
        job_timeout_seconds: Optional[int] = None,
        task: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        # Each async worker completes its job before submitting another. On
        # failure, stop queued submissions and cancel the other running jobs.
        cancellation_event = Event() if is_async_job else None
        requests = []
        for payload in payloads:
            if is_async_job:
                request = partial(
                    partial(_transport.run_async_job, self),
                    client,
                    endpoint,
                    payload,
                    poll_interval=poll_interval,
                    poll_timeout=poll_timeout,
                    multithreaded_compress=False,
                    job_timeout_seconds=job_timeout_seconds,
                    task=task,
                    cancellation_event=cancellation_event,
                )
            else:
                request = partial(
                    self._make_request_with_retries,
                    client=client,
                    endpoint=endpoint,
                    payload=payload,
                    multithreaded_compress=False,
                )
            requests.append(request)
        max_workers = _MAX_CONCURRENT_ASYNC_JOBS if is_async_job else 10
        return self._collect_concurrent_results(
            requests,
            max_workers=min(max_workers, len(payloads)),
            transform=transform,
            cancellation_event=cancellation_event,
        )

    def _make_partitioned_requests(
        self,
        client: httpx.Client,
        endpoint: str,
        payloads: list[dict[str, Any]],
        _is_async_job: bool = False,
        _poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
        _poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
        _job_timeout_seconds: Optional[int] = None,
    ) -> dict[str, Any]:
        results = self._dispatch_partitioned_requests(
            client,
            endpoint,
            payloads,
            is_async_job=_is_async_job,
            poll_interval=_poll_interval,
            poll_timeout=_poll_timeout,
            job_timeout_seconds=_job_timeout_seconds,
        )
        resp = {"mean": np.hstack([res["mean"] for res in results])}
        first_res = results[0]
        for k in ("sizes", "anomaly"):
            if k in first_res:
                resp[k] = np.hstack([res[k] for res in results])
        if "idxs" in first_res:
            part_rows = [sum(p["series"]["sizes"]) for p in payloads]
            offsets = np.cumsum([0, *part_rows[:-1]])
            resp["idxs"] = np.hstack(
                [
                    np.array(res["idxs"], dtype=np.int64) + offset
                    for res, offset in zip(results, offsets)
                ]
            )
        if "anomaly_score" in first_res:
            resp["anomaly_score"] = np.hstack([res["anomaly_score"] for res in results])
        if first_res["intervals"] is None:
            resp["intervals"] = None
        else:
            resp["intervals"] = {}
            for k in first_res["intervals"].keys():
                resp["intervals"][k] = np.hstack(
                    [res["intervals"][k] for res in results]
                )
        if "weights_x" not in first_res or first_res["weights_x"] is None:
            resp["weights_x"] = None
        else:
            resp["weights_x"] = [res["weights_x"] for res in results]
        if (
            "feature_contributions" not in first_res
            or first_res["feature_contributions"] is None
        ):
            resp["feature_contributions"] = None
        else:
            resp["feature_contributions"] = np.vstack(
                [np.stack(res["feature_contributions"], axis=1) for res in results]
            ).T
        return resp

    def _make_partitioned_simulate_requests(
        self,
        client: httpx.Client,
        payloads: list[dict[str, Any]],
        n_paths: int,
        h: int,
        job_timeout_seconds: Optional[int],
        poll_interval: Optional[float],
        poll_timeout: Optional[float],
    ) -> dict[str, Any]:
        def _samples_to_array(res: dict[str, Any]) -> dict[str, Any]:
            samples = res.get("samples")
            if samples is not None:
                try:
                    res["samples"] = np.asarray(samples, dtype=np.float64)
                except (TypeError, ValueError):
                    pass  # the merge loop below raises the descriptive error
            return res

        results = self._dispatch_partitioned_requests(
            client,
            "v2/simulate",
            payloads,
            transform=_samples_to_array,
            is_async_job=True,
            poll_interval=poll_interval,
            poll_timeout=poll_timeout,
            job_timeout_seconds=job_timeout_seconds,
            task="simulate",
        )
        blocks = []
        sizes = []
        for payload, res in zip(payloads, results):
            n_series = len(payload["series"]["sizes"])
            if res.get("n_paths") != n_paths or res.get("h") != h:
                raise RuntimeError(
                    "Simulation response metadata does not match the request."
                )
            try:
                part = np.asarray(res.pop("samples", None), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Simulation response contains non-numeric samples."
                ) from exc
            if part.size != n_paths * n_series * h:
                raise RuntimeError(
                    f"Simulation response contains {part.size:,} values; "
                    f"expected {n_paths * n_series * h:,}."
                )
            # samples are [sample_id][series][h]; join partitions within each
            # sample so the series stay in request order.
            blocks.append(part.reshape(n_paths, n_series * h))
            try:
                sizes.append(np.asarray(res.get("sizes"), dtype=np.int64))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Simulation response contains non-numeric sizes."
                ) from exc
        if any(_coerce_coupled_flag(res.get("coupled")) for res in results):
            raise RuntimeError(
                "Simulation response reported coupled paths for a partitioned request."
            )
        return {
            "samples": np.concatenate(blocks, axis=1).reshape(-1),
            "sizes": np.hstack(sizes),
            "n_paths": n_paths,
            "h": h,
            "coupled": False,
        }

    def _maybe_override_model(self, model: _Model) -> _Model:
        if self._is_azure and model != "azureai":
            logger.warning("Azure endpoint detected, setting `model` to 'azureai'.")
            model = "azureai"
        return model

    def _make_client(self, **kwargs: Any) -> httpx.Client:
        return httpx.Client(**kwargs)

    def _get_model_params(self, model: _Model, freq: str) -> tuple[int, int]:
        key = (model, freq)
        if key not in self._model_params:
            logger.info("Querying model metadata...")
            payload = {"model": model, "freq": freq}
            with self._make_client(**self._client_kwargs) as client:
                if self._is_azure:
                    resp_body = self._make_request_with_retries(
                        client, "model_params", payload
                    )
                else:
                    resp_body = self._retry_strategy(self._get_request)(
                        client, "/model_params", payload
                    )
            params = resp_body["detail"]
            self._model_params[key] = (params["input_size"], params["horizon"])
        return self._model_params[key]

    def _maybe_assign_weights(
        self,
        weights: Optional[Union[list[float], list[list[float]]]],
        df: DataFrame,
        x_cols: list[str],
    ) -> None:
        if weights is None:
            return
        if isinstance(weights[0], list):
            self.weights_x = [
                type(df)({"features": x_cols, "weights": w}) for w in weights
            ]
        else:
            self.weights_x = type(df)({"features": x_cols, "weights": weights})

    def _maybe_assign_feature_contributions(
        self,
        expected_contributions: bool,
        resp: dict[str, Any],
        x_cols: list[str],
        out_df: DataFrame,
        insample_feat_contributions: Optional[list[list[float]]],
    ) -> None:
        if not expected_contributions:
            return
        if "feature_contributions" not in resp:
            if self._is_azure:
                logger.warning("feature_contributions aren't implemented in Azure yet.")
                return
            else:
                raise RuntimeError(
                    "feature_contributions expected in response but not found"
                )
        feature_contributions = resp["feature_contributions"]
        if feature_contributions is None:
            return
        shap_cols = x_cols + ["base_value"]
        shap_df = type(out_df)(dict(zip(shap_cols, feature_contributions)))
        if insample_feat_contributions is not None:
            insample_shap_df = type(out_df)(
                dict(zip(shap_cols, insample_feat_contributions))
            )
            shap_df = ufp.vertical_concat([insample_shap_df, shap_df])
        self.feature_contributions = ufp.horizontal_concat([out_df, shap_df])

    def _run_validations(
        self,
        df: DFType,
        X_df: Optional[DFType],
        id_col: str,
        time_col: str,
        target_col: str,
        validate_api_key: bool,
        freq: Optional[_FreqType],
    ) -> tuple[DFType, Optional[DFType], bool, _FreqType]:
        if validate_api_key and not self.validate_api_key(log=False):
            raise Exception("API Key not valid, please email support@nixtla.io")
        drop_id = id_col not in df.columns
        if drop_id:
            df = ufp.copy_if_pandas(df, deep=False)
            df = ufp.assign_columns(df, id_col, 0)
            if X_df is not None:
                X_df = ufp.copy_if_pandas(X_df, deep=False)
                X_df = ufp.assign_columns(X_df, id_col, 0)
        if (
            isinstance(df, pd.DataFrame)
            and time_col not in df
            and pd.api.types.is_datetime64_any_dtype(df.index)
        ):
            df = df.rename_axis(time_col).reset_index()
        df = ensure_time_dtype(df, time_col=time_col)
        validate_format(df=df, id_col=id_col, time_col=time_col, target_col=target_col)
        inferred_freq = _maybe_infer_freq(
            df, freq=freq, id_col=id_col, time_col=time_col
        )
        _validate_freq_regularity(
            df=df, freq=inferred_freq, id_col=id_col, time_col=time_col
        )
        return df, X_df, drop_id, inferred_freq

    def validate_api_key(self, log: bool = True) -> bool:
        """Check API key status.

        Args:
            log (bool): Show the endpoint's response. Defaults to True.

        Returns:
            bool: Whether API key is valid.
        """
        if self._is_azure:
            raise NotImplementedError(
                "validate_api_key is not implemented for Azure deployments, "
                "you can try using the forecasting methods directly."
            )
        with self._make_client(**self._client_kwargs) as client:
            resp = client.get("/validate_api_key")
            body = resp.json()
        if log:
            logger.info(body["detail"])
        return resp.status_code == 200

    def usage(self) -> dict[str, dict[str, int]]:
        """Query consumed requests and limits

        Returns:
            dict: Consumed requests and limits by minute and month.
        """
        if self._is_azure:
            raise NotImplementedError("usage is not implemented for Azure deployments")
        with self._make_client(**self._client_kwargs) as client:
            return self._get_request(client, "/usage")

    def list_models(self) -> list[str]:
        """List the models available to your API key.

        Returns:
            list of str: Model names, sorted. Each can be passed as the `model`
                argument of `forecast`, `cross_validation`, `detect_anomalies`
                and the other forecasting methods; a model absent from this
                list is one those methods would reject.
        """
        with self._make_client(**self._client_kwargs) as client:
            resp_body = self._retry_strategy(self._get_request)(client, "/v2/models")
        return sorted(m["name"] for m in resp_body["models"])

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
    ) -> str:
        """Fine-tune TimeGPT to your series.

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

        Returns:
            str: ID of the fine-tuned model

        """
        payload = _payloads.prepare_finetune_payload(
            self,
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

        with self._make_client(**self._client_kwargs) as client:
            resp = self._make_request_with_retries(client, "v2/finetune", payload)
        return resp["finetuned_model_id"]

    @overload
    def finetuned_models(self, as_df: Literal[False]) -> list[FinetunedModel]: ...

    @overload
    def finetuned_models(self, as_df: Literal[True]) -> pd.DataFrame: ...

    def finetuned_models(
        self,
        as_df: bool = False,
    ) -> Union[list[FinetunedModel], pd.DataFrame]:
        """List fine-tuned models

        Args:
            as_df (bool): Return the fine-tuned models as a pandas dataframe.

        Returns:
            List of FinetunedModel: List of available fine-tuned models.
        """
        with self._make_client(**self._client_kwargs) as client:
            resp_body = self._get_request(client, "/v2/finetuned_models")
        models = [FinetunedModel(**m) for m in resp_body["finetuned_models"]]
        if as_df:
            models = pd.DataFrame([m.model_dump() for m in models])
        return models

    def finetuned_model(self, finetuned_model_id: str) -> FinetunedModel:
        """Get fine-tuned model metadata

        Args:
            finetuned_model_id (str): ID of the fine-tuned model to get
                metadata from.

        Returns:
            FinetunedModel: Fine-tuned model metadata.
        """
        with self._make_client(**self._client_kwargs) as client:
            resp_body = self._get_request(
                client, f"/v2/finetuned_models/{finetuned_model_id}"
            )
        return FinetunedModel(**resp_body)

    def delete_finetuned_model(self, finetuned_model_id: str) -> bool:
        """Delete a previously fine-tuned model

        Args:
            finetuned_model_id (str): ID of the fine-tuned model to be deleted.

        Returns:
            bool: Whether delete was successful.
        """
        with self._make_client(**self._client_kwargs) as client:
            resp = client.delete(
                f"/v2/finetuned_models/{finetuned_model_id}",
                headers={"accept-encoding": "identity"},
            )
        return resp.status_code == 204

    def _distributed_forecast(
        self,
        df: DistributedDFType,
        h: _PositiveInt,
        freq: Optional[_Freq],
        id_col: str,
        time_col: str,
        target_col: str,
        X_df: Optional[DistributedDFType],
        level: Optional[list[Union[int, float]]],
        quantiles: Optional[list[float]],
        finetune_steps: _NonNegativeInt,
        finetune_depth: _FinetuneDepth,
        finetune_loss: _Loss,
        finetuned_model_id: Optional[str],
        clean_ex_first: bool,
        hist_exog_list: Optional[list[str]],
        categorical_exog_list: Optional[list[str]],
        validate_api_key: bool,
        add_history: bool,
        date_features: Union[bool, list[Union[str, Callable]]],
        date_features_to_one_hot: Union[bool, list[str]],
        model: _Model,
        num_partitions: Optional[int],
        feature_contributions: bool,
        model_parameters: _ExtraParamDataType,
        multivariate: bool,
        feature_contributions_type: _FeatureContributionsType,
        _job_timeout_seconds: Optional[int] = None,
        # Internal-only params used for the num_partitions/distributed async
        # fan-out; not part of the public API. NOTE: when _is_async_job=True,
        # each Fugue partition submits and polls its own async job
        # independently on whichever worker executes it, so if poll_timeout
        # exceeds the underlying compute engine's own task/worker timeout, the
        # worker task can be killed by the compute framework before the async
        # job completes, independent of poll_timeout.
        *,
        _is_async_job: bool = False,
        _poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
        _poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
    ) -> DistributedDFType:
        import fugue.api as fa

        schema, partition_config = _distributed_setup(
            df=df,
            method="forecast",
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
            quantiles=quantiles,
            num_partitions=num_partitions,
        )
        if X_df is not None:

            def format_df(df: pd.DataFrame) -> pd.DataFrame:
                return df.assign(_in_sample=True)

            def format_X_df(
                X_df: pd.DataFrame,
                target_col: str,
                df_cols: list[str],
            ) -> pd.DataFrame:
                return X_df.assign(**{"_in_sample": False, target_col: 0.0})[df_cols]

            df = fa.transform(df, format_df, schema="*,_in_sample:bool")
            X_df = fa.transform(
                X_df,
                format_X_df,
                schema=fa.get_schema(df),
                params={"target_col": target_col, "df_cols": fa.get_column_names(df)},
            )
            df = fa.union(df, X_df)
        result_df = fa.transform(
            df,
            using=_forecast_wrapper,
            schema=schema,
            params=dict(
                client=self,
                h=h,
                freq=freq,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
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
                add_history=add_history,
                date_features=date_features,
                date_features_to_one_hot=date_features_to_one_hot,
                model=model,
                num_partitions=None,
                feature_contributions=feature_contributions,
                model_parameters=model_parameters,
                multivariate=multivariate,
                feature_contributions_type=feature_contributions_type,
                _job_timeout_seconds=_job_timeout_seconds,
                _is_async_job=_is_async_job,
                _poll_interval=_poll_interval,
                _poll_timeout=_poll_timeout,
            ),
            partition=partition_config,
            as_fugue=True,
        )
        return fa.get_native_as_df(result_df)

    def forecast(
        self,
        df: AnyDFType,
        h: _PositiveInt,
        freq: Optional[_Freq] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        X_df: Optional[AnyDFType] = None,
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
        add_history: bool = False,
        date_features: Union[bool, list[Union[str, Callable]]] = False,
        date_features_to_one_hot: Union[bool, list[str]] = False,
        model: _Model = "timegpt-2.1",
        num_partitions: Optional[_PositiveInt] = None,
        feature_contributions: bool = False,
        model_parameters: _ExtraParamDataType = None,
        multivariate: bool = False,
        feature_contributions_type: _FeatureContributionsType = "shapley",
        # Internal-only params used by the num_partitions/distributed async fan-out.
        *,
        _is_async_job: bool = False,
        _poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
        _poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
        # Per-job server-side time limit, applied to each job this call submits. Only valid with
        # _is_async_job.
        _job_timeout_seconds: Optional[int] = None,
    ) -> AnyDFType:
        """Forecast your time series using TimeGPT.

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
            add_history (bool): Return fitted values of the model. Defaults
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
            num_partitions (int):
                Number of partitions to use. If None, the number of partitions
                will be equal to the available parallel resources in
                distributed environments. Defaults to None.
            feature_contributions (bool): Compute feature contributions and
                store them in `self.feature_contributions`. Defaults to False.
            model_parameters (dict): The dictionary settings that determine
                the behavior of the model. Default is None
            multivariate (bool): If True, enables multivariate predictions.
                Defaults to False. Note: multivariate predictions are only
                supported for a select set of TimeGPT models.
            feature_contributions_type (str): Explanation used for feature
                contributions. One of `"shapley"`, `"intervention"`,
                `"granger"`, or `"transfer_entropy"`. Defaults to `"shapley"`.

        Returns:
            pandas, polars, dask or spark DataFrame or ray Dataset:
                DataFrame with TimeGPT forecasts for point predictions and
                probabilistic predictions (if level is not None).
        """
        extra_param_checker.validate_python(model_parameters)
        _transport._validate_job_timeout_seconds(_job_timeout_seconds)
        if _job_timeout_seconds is not None and not _is_async_job:
            raise ValueError(
                "_job_timeout_seconds requires _is_async_job; a synchronous request "
                "creates no job for it to bound."
            )

        if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
            return self._distributed_forecast(
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
                add_history=add_history,
                date_features=date_features,
                date_features_to_one_hot=date_features_to_one_hot,
                model=model,
                num_partitions=num_partitions,
                feature_contributions=feature_contributions,
                feature_contributions_type=feature_contributions_type,
                model_parameters=model_parameters,
                multivariate=multivariate,
                _job_timeout_seconds=_job_timeout_seconds,
                _is_async_job=_is_async_job,
                _poll_interval=_poll_interval,
                _poll_timeout=_poll_timeout,
            )
        payload, sizes, model_horizon, model_input_size, parse_result = (
            _payloads.prepare_forecast(
                self,
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
                add_history=add_history,
                date_features=date_features,
                date_features_to_one_hot=date_features_to_one_hot,
                model=model,
                feature_contributions=feature_contributions,
                model_parameters=model_parameters,
                multivariate=multivariate,
                feature_contributions_type=feature_contributions_type,
            )
        )

        with self._make_client(**self._client_kwargs) as client:
            insample_feat_contributions = None
            in_sample_resp = None
            if num_partitions is None:
                if _is_async_job:
                    resp = _transport.run_async_job(
                        self,
                        client,
                        "v2/forecast",
                        payload,
                        _poll_interval,
                        _poll_timeout,
                        job_timeout_seconds=_job_timeout_seconds,
                    )
                else:
                    resp = self._make_request_with_retries(
                        client, "v2/forecast", payload
                    )
                if add_history:
                    insample_h, n_windows = _get_in_sample_horizon_and_windows(
                        sizes=sizes,
                        model_horizon=model_horizon,
                        model_input_size=model_input_size,
                        clean_ex_first=clean_ex_first,
                        level=level,
                    )
                    in_sample_payload = _forecast_payload_to_in_sample(
                        payload, insample_h, n_windows
                    )
                    logger.info("Calling Historical Forecast Endpoint...")
                    if _is_async_job:
                        in_sample_resp = _transport.run_async_job(
                            self,
                            client,
                            "v2/cross_validation",
                            in_sample_payload,
                            _poll_interval,
                            _poll_timeout,
                            job_timeout_seconds=_job_timeout_seconds,
                        )
                    else:
                        in_sample_resp = self._make_request_with_retries(
                            client, "v2/cross_validation", in_sample_payload
                        )
                    insample_feat_contributions = in_sample_resp.get(
                        "feature_contributions", None
                    )
            else:
                payloads = _partition_series(payload, num_partitions, h)
                resp = self._make_partitioned_requests(
                    client,
                    "v2/forecast",
                    payloads,
                    _is_async_job=_is_async_job,
                    _poll_interval=_poll_interval,
                    _poll_timeout=_poll_timeout,
                    _job_timeout_seconds=_job_timeout_seconds,
                )
                if add_history:
                    insample_h, n_windows = _get_in_sample_horizon_and_windows(
                        sizes=sizes,
                        model_horizon=model_horizon,
                        model_input_size=model_input_size,
                        clean_ex_first=clean_ex_first,
                        level=level,
                    )
                    in_sample_payloads = [
                        _forecast_payload_to_in_sample(p, insample_h, n_windows)
                        for p in payloads
                    ]
                    logger.info("Calling Historical Forecast Endpoint...")
                    in_sample_resp = self._make_partitioned_requests(
                        client,
                        "v2/cross_validation",
                        in_sample_payloads,
                        _is_async_job=_is_async_job,
                        _poll_interval=_poll_interval,
                        _poll_timeout=_poll_timeout,
                        _job_timeout_seconds=_job_timeout_seconds,
                    )
                    insample_feat_contributions = in_sample_resp.get(
                        "feature_contributions", None
                    )

        return parse_result(resp, in_sample_resp, insample_feat_contributions)

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
        num_partitions: Optional[_PositiveInt] = None,
        job_timeout_seconds: Optional[int] = None,
        poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
        poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
    ) -> DataFrame:
        """Generate temporally correlated forecast sample paths.

        The request runs as an asynchronous job on the server: it is submitted,
        then polled until it finishes, so this call blocks until the paths are
        available. Use `submit_simulate_job()` to get a `Job` handle back
        immediately instead.

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
                drawn from and add no columns to the result. A wide grid also
                counts towards a second size limit:
                `n_series * h * (n_paths + len(quantiles))` may not exceed
                10,000,000.
            seed (int, optional): Random seed. Reusing a seed with the same
                inputs produces the same paths. Must be between `-2**63` and
                `2**64 - 1`. The seed drives both the coupled and the per-series
                shuffle, so repeating a request with the same seed and a
                different `multivariate` setting reorders the first series by
                ID identically in both, and returns the very same paths for it
                when its marginal forecast is unchanged too. Vary the seed
                when comparing coupled against uncoupled paths.
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
            num_partitions (int, optional): Split the series across this many
                concurrent requests, which keeps large jobs under the request
                size limit. Cannot be combined with `multivariate=True`, since
                coupling is computed across the series in a single request.
                Each partition is sent a distinct seed derived from `seed`, so
                partitions never share their random draws and the call stays
                reproducible, but a partitioned call returns different paths
                than an unpartitioned one for the same `seed`. Defaults to
                None (a single request). At most five partition jobs run
                concurrently. If a partition fails, the client cancels sibling
                jobs and stops queued submissions. Not supported by
                `submit_simulate_job()`.
            job_timeout_seconds (int, optional): Maximum seconds the server
                allows this job to run before terminating it server-side (each
                partition when `num_partitions` is set). This is separate from
                `poll_timeout`, which only controls how long the client polls
                locally. Capped by a server-side per-task maximum; requesting a
                higher value raises `ApiError` (422) when submitting. Defaults
                to the server's default for this task type if not specified.
            poll_interval (float, optional): Seconds to wait between job-status
                polls, held fixed. Must be finite and non-negative. Defaults to
                `None`, which polls on an adaptive cadence instead: the first
                check comes after half a second and the interval doubles up to
                one check every 15 seconds.
            poll_timeout (float, optional): Maximum seconds to wait for the
                job to reach a terminal state before raising
                `JobTimeoutError`, measured from a successful submission
                and including the time the job spends queued on the server.
                Submission retries are excluded, and each partition gets its
                own timeout. When it elapses the client requests the job's
                cancellation. Set to `None` to wait until the server reports a
                terminal status. Defaults to 3600.

        Returns:
            pandas or polars DataFrame: Long-format sample paths with ID, time,
                `sample_id`, `TimeGPT`, and `coupled` columns. It contains
                `n_series * n_paths * h` rows. The ID column is omitted if `df`
                did not contain one.

        Raises:
            ValueError: Invalid arguments, missing or duplicate timestamps,
                or timestamps that do not match the provided frequency.
            ApiError: An HTTP request failed, including an unsupported deployment.
            JobError: The job failed or returned an invalid job response.
            JobCancelledError: The job was cancelled on the server.
            JobTimeoutError: The job did not finish within `poll_timeout`
                seconds; cancellation was requested.
        """
        _transport._validate_job_timeout_seconds(job_timeout_seconds)
        _validate_poll_settings(poll_interval, poll_timeout, allow_unbounded=True)
        h, n_paths, seed, num_partitions = _validate_simulate_args(
            h, n_paths, seed, num_partitions, multivariate
        )
        payload, parse_result = _payloads.prepare_simulate(
            self,
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
            method_name="simulate",
        )

        logger.info("Calling Simulate Endpoint...")
        if num_partitions is None:
            job = _transport.submit_and_wrap_job(
                self,
                "v2/simulate",
                payload,
                job_timeout_seconds,
                parse_result,
                task="simulate",
            )
            # Abandoning the wait leaves the job burning server-side compute,
            # so cancel it on the way out.
            with job:
                return job.wait(poll_interval, poll_timeout)

        payloads = _partition_series(payload, num_partitions, h)
        if seed is not None:
            seed_span = _MAX_SEED - _MIN_SEED + 1
            for i, part in enumerate(payloads[1:], start=1):
                part["seed"] = (seed - _MIN_SEED + i) % seed_span + _MIN_SEED
        with self._make_client(**self._client_kwargs) as client:
            resp = self._make_partitioned_simulate_requests(
                client,
                payloads,
                n_paths=n_paths,
                h=h,
                job_timeout_seconds=job_timeout_seconds,
                poll_interval=poll_interval,
                poll_timeout=poll_timeout,
            )
        return parse_result(resp)

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
        poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
        poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
    ) -> DataFrame:
        """Compute model-independent historical feature importance weights.

        The returned weights describe lagged predictive relationships in the
        supplied data. They do not establish that changing a feature will cause
        the target to change.

        The request runs as an asynchronous job on the server: it is submitted,
        then polled until it finishes, so this call blocks until the weights
        are available. Use `submit_explain_job()` to get a `Job` handle back
        immediately instead.

        Args:
            df (pandas or polars DataFrame): Historical time series containing
                the target and candidate feature columns.
            method (str): `"granger"` for linear lagged relationships or
                `"transfer_entropy"` for potentially nonlinear relationships.
                Defaults to `"granger"`.
            features (list[str], optional): Features to analyze. By default,
                every column other than the ID, time, and target columns is
                used. Missing feature values are allowed: rows with a missing
                value in a lagged column are excluded from that feature's
                weight estimation, which reduces the effective sample.
            freq (str, int or pandas offset, optional): Frequency of the
                timestamps, used to verify that every series is complete and
                regularly spaced. Both methods are lag-based, so gaps or
                duplicate timestamps distort the weights. If `None`, it is
                inferred from `df` (pandas only); pass it explicitly for polars.
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
                allows this job to run before terminating it server-side. This
                is separate from `poll_timeout`, which only controls how long
                the client polls locally. Capped by a server-side per-task
                maximum; requesting a higher value raises `ApiError` (422) when
                submitting. Defaults to the server's default for this task type
                if not specified.
            poll_interval (float, optional): Seconds to wait between job-status
                polls, held fixed. Must be finite and non-negative. Defaults to
                `None`, which polls on an adaptive cadence instead: the first
                check comes after half a second and the interval doubles up to
                one check every 15 seconds.
            poll_timeout (float, optional): Maximum seconds to wait for the
                job to reach a terminal state before raising
                `JobTimeoutError`, measured from a successful submission
                and including the time the job spends queued on the server.
                Submission retries are excluded. When it elapses the client
                requests the job's cancellation. Set to `None` to wait until
                the server reports a terminal status. Defaults to 3600.

        Returns:
            pandas or polars DataFrame: One row per feature with `feature`,
                `weight`, and `method` columns.

        Raises:
            ValueError: Invalid arguments, missing or duplicate timestamps,
                or timestamps that do not match the provided frequency.
            ApiError: An HTTP request failed, including an unsupported deployment.
            JobError: The job failed or returned an invalid job response.
            JobCancelledError: The job was cancelled on the server.
            JobTimeoutError: The job did not finish within `poll_timeout`
                seconds; cancellation was requested.
        """
        _transport._validate_job_timeout_seconds(job_timeout_seconds)
        _validate_poll_settings(poll_interval, poll_timeout, allow_unbounded=True)
        payload, parse_result = _payloads.prepare_explain(
            self,
            df=df,
            method=method,
            features=features,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            categorical_exog_list=categorical_exog_list,
            validate_api_key=validate_api_key,
            method_name="explain",
        )

        logger.info("Calling Explain Endpoint...")
        job = _transport.submit_and_wrap_job(
            self,
            "v2/explain",
            payload,
            job_timeout_seconds,
            parse_result,
            task="explain",
        )
        # Abandoning the wait leaves the job burning server-side compute, so
        # cancel it on the way out.
        with job:
            return job.wait(poll_interval, poll_timeout)

    def _distributed_detect_anomalies(
        self,
        df: DistributedDFType,
        freq: Optional[_Freq],
        id_col: str,
        time_col: str,
        target_col: str,
        level: Union[int, float],
        finetuned_model_id: Optional[str],
        clean_ex_first: bool,
        validate_api_key: bool,
        date_features: Union[bool, list[str]],
        date_features_to_one_hot: Union[bool, list[str]],
        model: _Model,
        num_partitions: Optional[int],
        multivariate: bool,
        categorical_exog_list: Optional[list[str]] = None,
    ) -> DistributedDFType:
        import fugue.api as fa

        schema, partition_config = _distributed_setup(
            df=df,
            method="detect_anomalies",
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
            quantiles=None,
            num_partitions=num_partitions,
        )
        result_df = fa.transform(
            df,
            using=_detect_anomalies_wrapper,
            schema=schema,
            params=dict(
                client=self,
                freq=freq,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                level=level,
                finetuned_model_id=finetuned_model_id,
                clean_ex_first=clean_ex_first,
                validate_api_key=validate_api_key,
                date_features=date_features,
                date_features_to_one_hot=date_features_to_one_hot,
                model=model,
                num_partitions=None,
                multivariate=multivariate,
                categorical_exog_list=categorical_exog_list,
            ),
            partition=partition_config,
            as_fugue=True,
        )
        return fa.get_native_as_df(result_df)

    def detect_anomalies(
        self,
        df: AnyDFType,
        freq: Optional[_Freq] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        level: Union[int, float] = 99,
        finetuned_model_id: Optional[str] = None,
        clean_ex_first: bool = True,
        validate_api_key: bool = False,
        date_features: Union[bool, list[str]] = False,
        date_features_to_one_hot: Union[bool, list[str]] = False,
        model: _Model = "timegpt-2.1",
        num_partitions: Optional[_PositiveInt] = None,
        multivariate: bool = False,
        categorical_exog_list: Optional[list[str]] = None,
    ) -> AnyDFType:
        """Detect anomalies in your time series using TimeGPT.

        Args:
            df (pandas or polars DataFrame): The DataFrame on which the
                function will operate. Expected to contain at least the
                following columns:
                - time_col:
                    Column name in `df` that contains the time indices of the
                    time series. This is typically a datetime column with
                    regular intervals, e.g., hourly, daily, monthly data points.
                - target_col:
                    Column name in `df` that contains the target variable of
                    the time series, i.e., the variable we wish to predict
                    or analyze.
                Additionally, you can pass multiple time series (stacked in
                the dataframe) considering an additional column:
                - id_col:
                    Column name in `df` that identifies unique time series.
                    Each unique value in this column corresponds to a unique time series.
            freq (str, int, pandas offset, optional): Frequency of the
                timestamps.  If `None`, it will be inferred automatically.
                See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
                Defaults to None.
            id_col (str): Column that identifies each series. Defaults to
                'unique_id'.
            time_col (str): Column that identifies each timestep, its values
                can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            level (float): Confidence level between 0 and 100 for detecting
                the anomalies. Defaults to 99.
            finetuned_model_id (str, optional): ID of previously fine-tuned
                model to use. Defaults to None.
            clean_ex_first (bool): Clean exogenous signal before making
                forecasts using TimeGPT. Defaults to True.
            validate_api_key (bool):
                If True, validates api_key before sending requests. Defaults
                to False.
            date_features (bool or list[str] or callable, optional): Features
                computed from the dates. Can be pandas date attributes or
                functions that will take the dates as input. If True
                automatically adds most used date features for the frequency
                of `df`. Defaults to False.
            date_features_to_one_hot (bool or list[str]): Apply one-hot
                encoding to these date features. If
                `date_features=True`, then all date features are
                one-hot encoded by default. Defaults to False.
            model (str): str (default='timegpt-2.1')
                Model to use as a string. Options are: `timegpt-1`, and
                `timegpt-1-long-horizon`, `timegpt-2`, `timegpt-2-mini`,
                `timegpt-2-pro`, `timegpt-2.1`. We recommend using
                `timegpt-1-long-horizon` for forecasting if you want to predict
                more than one seasonal period given the frequency of your data.
                Defaults to 'timegpt-2.1'.
            num_partitions (int): Number of partitions to use. If None, the
                number of partitions will be equal to the available parallel
                resources in distributed environments. Defaults to None.
            multivariate (bool): If True, enables multivariate predictions.
                Defaults to False. Note: multivariate predictions are only
                supported for a select set of TimeGPT models.
            categorical_exog_list (list[str], optional): Column names of
                categorical exogenous features in `df` (can be strings or
                numbers). Defaults to None.

        Returns:
            pandas, polars, dask or spark DataFrame or ray Dataset:
                DataFrame with anomalies flagged by TimeGPT.
        """
        if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
            return self._distributed_detect_anomalies(
                df=df,
                freq=freq,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                level=level,
                finetuned_model_id=finetuned_model_id,
                clean_ex_first=clean_ex_first,
                validate_api_key=validate_api_key,
                date_features=date_features,
                date_features_to_one_hot=date_features_to_one_hot,
                model=model,
                num_partitions=num_partitions,
                multivariate=multivariate,
                categorical_exog_list=categorical_exog_list,
            )
        self.__dict__.pop("weights_x", None)
        model = self._maybe_override_model(model)
        logger.info("Validating inputs...")
        df, _, drop_id, freq = self._run_validations(
            df=df,
            X_df=None,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            validate_api_key=validate_api_key,
            freq=freq,
        )

        df, _, df_cat_vals, _, cat_cols, _ = _extract_categorical_exog(
            df=df,
            categorical_exog_list=categorical_exog_list,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

        logger.info("Preprocessing dataframes...")
        processed, _, x_cols, _ = _preprocess(
            df=df,
            X_df=None,
            h=0,
            freq=freq,
            date_features=date_features,
            date_features_to_one_hot=date_features_to_one_hot,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        standard_freq = _standardize_freq(freq, processed)
        model_input_size, model_horizon = self._get_model_params(model, standard_freq)

        # Sort categorical arrays to match _preprocess row ordering.
        sorted_cat: list[list] = []
        if cat_cols:
            for c in cat_cols:
                vals = df_cat_vals[c]
                if processed.sort_idxs is not None:
                    vals = vals[processed.sort_idxs]
                sorted_cat.append(vals.tolist())

        X: Optional[list[Any]] = None
        categorical_exog_payload: Optional[list[int]] = None
        if processed.data.shape[1] > 1 or sorted_cat:
            X_num = list(processed.data[:, 1:].T) if processed.data.shape[1] > 1 else []
            X = X_num + sorted_cat
            cat_indices = list(range(len(x_cols), len(x_cols) + len(cat_cols)))
            if cat_indices:
                categorical_exog_payload = cat_indices
            if x_cols:
                logger.info(f"Using the following exogenous features: {x_cols}")
            if cat_cols:
                logger.info(f"Using categorical exogenous features: {cat_cols}")

        series_payload: dict[str, Any] = {
            "y": processed.data[:, 0],
            "sizes": np.diff(processed.indptr),
            "X": X,
        }
        start_datetime = _series_starts(df, processed, time_col)
        if start_datetime is not None:
            series_payload["start_datetime"] = start_datetime
        if categorical_exog_payload is not None:
            series_payload["categorical_exog"] = categorical_exog_payload

        logger.info("Calling Anomaly Detector Endpoint...")
        payload = {
            "series": series_payload,
            "model": model,
            "freq": standard_freq,
            "finetuned_model_id": finetuned_model_id,
            "clean_ex_first": clean_ex_first,
            "level": level,
            "multivariate": multivariate,
        }
        with self._make_client(**self._client_kwargs) as client:
            if num_partitions is None:
                resp = self._make_request_with_retries(
                    client, _ANOMALY_DETECTION_ENDPOINT, payload
                )
            else:
                payloads = _partition_series(payload, num_partitions, h=0)
                resp = self._make_partitioned_requests(
                    client, _ANOMALY_DETECTION_ENDPOINT, payloads
                )

        # assemble result
        out = _parse_in_sample_output(
            in_sample_output=resp,
            df=df,
            processed=processed,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        out = ufp.assign_columns(out, "anomaly", resp["anomaly"])
        out = _maybe_drop_id(df=out, id_col=id_col, drop=drop_id)
        weights_x_cols = x_cols + cat_cols
        self._maybe_assign_weights(
            weights=resp["weights_x"], df=df, x_cols=weights_x_cols
        )
        return out

    def _distributed_detect_anomalies_online(
        self,
        df: DistributedDFType,
        h: _PositiveInt,
        detection_size: _PositiveInt,
        threshold_method: _ThresholdMethod,
        freq: Optional[_Freq],
        id_col: str,
        time_col: str,
        target_col: str,
        level: Union[int, float],
        clean_ex_first: bool,
        step_size: Optional[_PositiveInt],
        finetune_steps: _NonNegativeInt,
        finetune_depth: _FinetuneDepth,
        finetune_loss: _Loss,
        finetuned_model_id: Optional[str],
        hist_exog_list: Optional[list[str]],
        date_features: Union[bool, list[str]],
        date_features_to_one_hot: Union[bool, list[str]],
        model: _Model,
        model_parameters: _ExtraParamDataType,
        refit: bool,
        num_partitions: Optional[int],
        multivariate: bool,
    ) -> DistributedDFType:
        import fugue.api as fa

        schema, partition_config = _distributed_setup(
            df=df,
            method="detect_anomalies_online",
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
            quantiles=None,
            num_partitions=num_partitions,
        )
        result_df = fa.transform(
            df,
            using=_detect_anomalies_online_wrapper,
            schema=schema,
            params=dict(
                client=self,
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
                finetune_loss=finetune_loss,
                finetune_depth=finetune_depth,
                finetuned_model_id=finetuned_model_id,
                hist_exog_list=hist_exog_list,
                date_features=date_features,
                date_features_to_one_hot=date_features_to_one_hot,
                model=model,
                model_parameters=model_parameters,
                refit=refit,
                num_partitions=None,
                multivariate=multivariate,
            ),
            partition=partition_config,
            as_fugue=True,
        )
        return fa.get_native_as_df(result_df)

    def detect_anomalies_online(
        self,
        df: AnyDFType,
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
        num_partitions: Optional[_PositiveInt] = None,
        multivariate: bool = False,
    ) -> AnyDFType:
        """
        Online anomaly detection in your time series using TimeGPT.

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
            num_partitions (int):
                Number of partitions to use. If None, the number of partitions
                will be equal to the available parallel resources in
                distributed environments. Defaults to None.
            multivariate (bool): If True, enables multivariate predictions.
                Defaults to False. Note: multivariate predictions are only
                supported for a select set of TimeGPT models. This variable
                is different from the `threshold_method` parameter. The latter
                controls the method used for anomaly detection (univariate vs
                multivariate) whereas `multivariate` determines how the model
                creates the predictions.

        Returns:
            pandas, polars, dask or spark DataFrame or ray Dataset:
                DataFrame with anomalies flagged by TimeGPT.
        """
        extra_param_checker.validate_python(model_parameters)
        if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
            return self._distributed_detect_anomalies_online(
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
                num_partitions=num_partitions,
                multivariate=multivariate,
            )
        if (
            threshold_method == "multivariate"
            and num_partitions is not None
            and num_partitions > 1
        ):
            raise ValueError(
                "Cannot use more than 1 partition for multivariate anomaly detection. "
                "Either set threshold_method to univariate "
                "or set num_partitions to None."
            )
        payload, parse_result = _payloads.prepare_anomaly_detection(
            self,
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
        logger.info("Calling Online Anomaly Detector Endpoint...")
        with self._make_client(**self._client_kwargs) as client:
            if num_partitions is None:
                resp = self._make_request_with_retries(
                    client, _ONLINE_ANOMALY_DETECTION_ENDPOINT, payload
                )
            else:
                payloads = _partition_series(payload, num_partitions, h=0)
                resp = self._make_partitioned_requests(
                    client, _ONLINE_ANOMALY_DETECTION_ENDPOINT, payloads
                )
        return parse_result(resp)

    def _distributed_cross_validation(
        self,
        df: DistributedDFType,
        h: _PositiveInt,
        freq: Optional[_Freq],
        id_col: str,
        time_col: str,
        target_col: str,
        level: Optional[list[Union[int, float]]],
        quantiles: Optional[list[float]],
        validate_api_key: bool,
        n_windows: _PositiveInt,
        step_size: Optional[_PositiveInt],
        finetune_steps: _NonNegativeInt,
        finetune_depth: _FinetuneDepth,
        finetune_loss: _Loss,
        finetuned_model_id: Optional[str],
        refit: bool,
        clean_ex_first: bool,
        hist_exog_list: Optional[list[str]],
        date_features: Union[bool, Sequence[Union[str, Callable]]],
        date_features_to_one_hot: Union[bool, list[str]],
        model: _Model,
        num_partitions: Optional[int],
        model_parameters: _ExtraParamDataType,
        multivariate: bool,
        categorical_exog_list: Optional[list[str]] = None,
        _job_timeout_seconds: Optional[int] = None,
        # Internal-only params used for the num_partitions/distributed async
        # fan-out; not part of the public API. NOTE: when _is_async_job=True,
        # each Fugue partition submits and polls its own async job
        # independently on whichever worker executes it, so if poll_timeout
        # exceeds the underlying compute engine's own task/worker timeout, the
        # worker task can be killed by the compute framework before the async
        # job completes, independent of poll_timeout.
        *,
        _is_async_job: bool = False,
        _poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
        _poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
    ) -> DistributedDFType:
        import fugue.api as fa

        schema, partition_config = _distributed_setup(
            df=df,
            method="cross_validation",
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
            quantiles=quantiles,
            num_partitions=num_partitions,
        )
        result_df = fa.transform(
            df,
            using=_cross_validation_wrapper,
            schema=schema,
            params=dict(
                client=self,
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
                num_partitions=None,
                model_parameters=model_parameters,
                multivariate=multivariate,
                categorical_exog_list=categorical_exog_list,
                _job_timeout_seconds=_job_timeout_seconds,
                _is_async_job=_is_async_job,
                _poll_interval=_poll_interval,
                _poll_timeout=_poll_timeout,
            ),
            partition=partition_config,
            as_fugue=True,
        )
        return fa.get_native_as_df(result_df)

    def cross_validation(
        self,
        df: AnyDFType,
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
        num_partitions: Optional[_PositiveInt] = None,
        model_parameters: _ExtraParamDataType = None,
        multivariate: bool = False,
        categorical_exog_list: Optional[list[str]] = None,
        # Internal-only params used by the num_partitions/distributed async fan-out.
        *,
        _is_async_job: bool = False,
        _poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
        _poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
        # Per-job server-side time limit, applied to each job this call submits. Only valid with
        # _is_async_job.
        _job_timeout_seconds: Optional[int] = None,
    ) -> AnyDFType:
        """Perform cross validation in your time series using TimeGPT.

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
            num_partitions (int):
                Number of partitions to use. If None, the number of partitions
                will be equal to the available parallel resources in
                distributed environments. Defaults to None.
            model_parameters (dict): The dictionary settings that determine
                the behavior of the model. Default is None.
            multivariate (bool): If True, enables multivariate predictions.
                Defaults to False. Note: multivariate predictions are only
                supported for a select set of TimeGPT models.
            categorical_exog_list (list[str], optional): Column names of
                categorical exogenous features in (can be strings or
                numbers). Defaults to None.

        Returns:
            pandas, polars, dask or spark DataFrame or ray Dataset:
                DataFrame with cross validation forecasts.
        """
        extra_param_checker.validate_python(model_parameters)
        _transport._validate_job_timeout_seconds(_job_timeout_seconds)
        if _job_timeout_seconds is not None and not _is_async_job:
            raise ValueError(
                "_job_timeout_seconds requires _is_async_job; a synchronous request "
                "creates no job for it to bound."
            )
        if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
            return self._distributed_cross_validation(
                df=df,
                h=h,
                freq=freq,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                level=level,
                quantiles=quantiles,
                n_windows=n_windows,
                step_size=step_size,
                validate_api_key=validate_api_key,
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
                num_partitions=num_partitions,
                model_parameters=model_parameters,
                multivariate=multivariate,
                categorical_exog_list=categorical_exog_list,
                _job_timeout_seconds=_job_timeout_seconds,
                _is_async_job=_is_async_job,
                _poll_interval=_poll_interval,
                _poll_timeout=_poll_timeout,
            )
        payload, parse_result = _payloads.prepare_cross_validation(
            self,
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
        with self._make_client(**self._client_kwargs) as client:
            if num_partitions is None:
                if _is_async_job:
                    resp = _transport.run_async_job(
                        self,
                        client,
                        "v2/cross_validation",
                        payload,
                        _poll_interval,
                        _poll_timeout,
                        job_timeout_seconds=_job_timeout_seconds,
                    )
                else:
                    resp = self._make_request_with_retries(
                        client, "v2/cross_validation", payload
                    )
            else:
                payloads = _partition_series(payload, num_partitions, h=0)
                resp = self._make_partitioned_requests(
                    client,
                    "v2/cross_validation",
                    payloads,
                    _is_async_job=_is_async_job,
                    _poll_interval=_poll_interval,
                    _poll_timeout=_poll_timeout,
                    _job_timeout_seconds=_job_timeout_seconds,
                )

        return parse_result(resp)

    def plot(
        self,
        df: Optional[DataFrame] = None,
        forecasts_df: Optional[DataFrame] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        unique_ids: Union[Optional[list[str]], np.ndarray] = None,
        plot_random: bool = True,
        max_ids: int = 8,
        models: Optional[list[str]] = None,
        level: Optional[list[Union[int, float]]] = None,
        max_insample_length: Optional[int] = None,
        plot_anomalies: bool = False,
        engine: Literal["matplotlib", "plotly", "plotly-resampler"] = "matplotlib",
        resampler_kwargs: Optional[dict] = None,
        ax: Optional[
            Union["plt.Axes", np.ndarray, "plotly.graph_objects.Figure"]
        ] = None,
    ):
        """Plot forecasts and insample values.

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
            forecasts_df (pandas or polars DataFrame, optional): DataFrame with
                columns [`unique_id`, `ds`] and models. Defaults to None.
            id_col (str): Column that identifies each series. Defaults to
                'unique_id'.
            time_col (str): Column that identifies each timestep, its values
                can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            unique_ids (list[str], optional): Time Series to plot. If None,
                time series are selected randomly. Defaults to None.
            plot_random (bool):
                Select time series to plot randomly. Defaults to True.
            max_ids (int):
                Maximum number of ids to plot. Defaults to 8.
            models (list[str], optional): List of models to plot. Defaults to
                None.
            level (list[float], optional): List of prediction intervals to
                plot if paseed. Defaults to None.
            max_insample_length (int, optional): Max number of train/insample
                observations to be plotted. Defaults to None.
            plot_anomalies (bool): Plot anomalies for each prediction interval.
                Defaults to False.
            engine (str): Library used to plot. 'matplotlib', 'plotly' or
                'plotly-resampler'. Defaults to 'matplotlib'.
            resampler_kwargs (dict): Kwargs to be passed to plotly-resampler
                constructor. For further custumization ("show_dash") call
                the method, store the plotting object and add the extra
                arguments to its `show_dash` method.
            ax (matplotlib axes, array of matplotlib axes or plotly Figure,
                optional): Object where plots will be added. Defaults to None.
        """
        try:
            from utilsforecast.plotting import plot_series
        except ModuleNotFoundError:
            raise Exception(
                "You have to install additional dependencies to use this method, "
                'please install them using `pip install "nixtla[plotting]"`'
            )
        if df is not None and id_col not in df.columns:
            df = ufp.copy_if_pandas(df, deep=False)
            df = ufp.assign_columns(df, id_col, "ts_0")
        df = ensure_time_dtype(df, time_col=time_col)
        if forecasts_df is not None:
            if id_col not in forecasts_df.columns:
                forecasts_df = ufp.copy_if_pandas(forecasts_df, deep=False)
                forecasts_df = ufp.assign_columns(forecasts_df, id_col, "ts_0")
            forecasts_df = ensure_time_dtype(forecasts_df, time_col=time_col)
            if "anomaly" in forecasts_df.columns:
                # special case to plot outputs
                # from detect_anomalies
                df = None
                forecasts_df = ufp.drop_columns(forecasts_df, "anomaly")
                cols = [
                    c.replace("TimeGPT-lo-", "")
                    for c in forecasts_df.columns
                    if "TimeGPT-lo-" in c
                ]
                level = [float(c) if "." in c else int(c) for c in cols]
                plot_anomalies = True
                models = ["TimeGPT"]
        return plot_series(
            df=df,
            forecasts_df=forecasts_df,
            ids=unique_ids,
            plot_random=plot_random,
            max_ids=max_ids,
            models=models,
            level=level,
            max_insample_length=max_insample_length,
            plot_anomalies=plot_anomalies,
            engine=engine,
            resampler_kwargs=resampler_kwargs,
            palette="tab20b",
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            ax=ax,
        )

    @staticmethod
    def audit_data(
        df: AnyDFType,
        freq: _Freq,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        start: Union[str, int, datetime.date, datetime.datetime] = "per_serie",
        end: Union[str, int, datetime.date, datetime.datetime] = "global",
    ) -> tuple[bool, dict[str, DataFrame], dict[str, DataFrame]]:
        """Audit data quality.

        Args:
            df (pandas or polars DataFrame): The dataframe to be audited.
            freq (str, int or pandas offset): Frequency of the timestamps.
                Must be specified. See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
            id_col (str): Column that identifies each series. Defaults to
                'unique_id'.
            time_col (str): Column that identifies each timestep, its values
                can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            start (Union[str, int, datetime.date, datetime.datetime], optional):
                Initial timestamp for the series.
                    * 'per_serie' uses each series first timestamp
                    * 'global' uses the first timestamp seen in the data
                    * Can also be a specific timestamp or integer,
                    e.g. '2000-01-01', 2000 or datetime(2000, 1, 1)
                , by default "per_serie"
            end (Union[str, int, datetime.date, datetime.datetime], optional):
                Final timestamp for the series.
                    * 'per_serie' uses each series last timestamp
                    * 'global' uses the last timestamp seen in the data
                    * Can also be a specific timestamp or integer,
                    e.g. '2000-01-01', 2000 or datetime(2000, 1, 1)
                , by default "global"

        Returns:
            tuple[bool, dict[str, DataFrame], dict[str, DataFrame]]:
                Tuple containing:
                - bool: True if all tests pass, False otherwise
                - dict: Dictionary mapping test IDs to error DataFrames for failed
                        tests or None if the test could not be performed.
                - dict: Dictionary mapping test IDs to error DataFrames for
                        case-specific tests.

                Test IDs:
                - D001: Test for duplicate rows
                - D002: Test for missing dates
                - F001: Test for presence of categorical feature columns
                - V001: Test for negative values
                - V002: Test for leading zeros
        """
        return _audit.audit_data(df, freq, id_col, time_col, target_col, start, end)

    def clean_data(
        self,
        df: AnyDFType,
        fail_dict: dict[str, DataFrame],
        case_specific_dict: dict[str, DataFrame],
        freq: _Freq,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        clean_case_specific: bool = False,
        agg_dict: Optional[dict[str, Union[str, Callable]]] = None,
    ) -> tuple[AnyDFType, bool, dict[str, DataFrame], dict[str, DataFrame]]:
        """Clean the data. This should be run after running `audit_data`.

        Args:
            df (AnyDFType): The dataframe to be cleaned
            fail_dict (dict[str, DataFrame]): The failure dictionary from the
                audit_data method.
            case_specific_dict (dict[str, DataFrame]): The case specific
                dictionary from the audit_data method.
            freq (str, int or pandas offset): Frequency of the timestamps.
                Must be specified. See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
            id_col (str): Column that identifies each series. Defaults to
                'unique_id'.
            time_col (str): Column that identifies each timestep, its values
                can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            clean_case_specific (bool, optional): If True, clean case
                specific issues. Defaults to False.
            agg_dict (Optional[dict[str, Union[str, Callable]]], optional):
                The aggregation methods to use when there are duplicate rows (D001),
                by default None

        Returns:
            tuple[AnyDFType, bool, dict[str, DataFrame], dict[str, DataFrame]]:
                Tuple containing:
                - AnyDFType: The cleaned dataframe
                - The three outputs from audit_data that are run at the end of cleansing.

        Raises:
            ValueError: Any exceptions during the cleaning process.
        """
        return _audit.clean_data(
            df,
            fail_dict,
            case_specific_dict,
            freq,
            id_col,
            time_col,
            target_col,
            clean_case_specific,
            agg_dict,
        )


def _forecast_wrapper(
    df: pd.DataFrame,
    client: NixtlaClient,
    h: _PositiveInt,
    freq: Optional[_Freq],
    id_col: str,
    time_col: str,
    target_col: str,
    level: Optional[list[Union[int, float]]],
    quantiles: Optional[list[float]],
    finetune_steps: _NonNegativeInt,
    finetune_depth: _FinetuneDepth,
    finetune_loss: _Loss,
    finetuned_model_id: Optional[str],
    clean_ex_first: bool,
    hist_exog_list: Optional[list[str]],
    categorical_exog_list: Optional[list[str]],
    validate_api_key: bool,
    add_history: bool,
    date_features: Union[bool, list[Union[str, Callable]]],
    date_features_to_one_hot: Union[bool, list[str]],
    model: _Model,
    num_partitions: Optional[_PositiveInt],
    feature_contributions: bool,
    model_parameters: _ExtraParamDataType,
    multivariate: bool,
    feature_contributions_type: _FeatureContributionsType,
    _is_async_job: bool = False,
    _poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
    _poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
    _job_timeout_seconds: Optional[int] = None,
) -> pd.DataFrame:
    if "_in_sample" in df:
        in_sample_mask = df["_in_sample"]
        X_df = df.loc[~in_sample_mask].drop(columns=["_in_sample", target_col])
        df = df.loc[in_sample_mask].drop(columns="_in_sample")
    else:
        X_df = None
    return client.forecast(
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
        add_history=add_history,
        date_features=date_features,
        date_features_to_one_hot=date_features_to_one_hot,
        model=model,
        num_partitions=num_partitions,
        feature_contributions=feature_contributions,
        feature_contributions_type=feature_contributions_type,
        model_parameters=model_parameters,
        multivariate=multivariate,
        _is_async_job=_is_async_job,
        _poll_interval=_poll_interval,
        _poll_timeout=_poll_timeout,
        _job_timeout_seconds=_job_timeout_seconds,
    )


def _detect_anomalies_wrapper(
    df: pd.DataFrame,
    client: NixtlaClient,
    freq: Optional[_Freq],
    id_col: str,
    time_col: str,
    target_col: str,
    level: Union[int, float],
    finetuned_model_id: Optional[str],
    clean_ex_first: bool,
    validate_api_key: bool,
    date_features: Union[bool, list[str]],
    date_features_to_one_hot: Union[bool, list[str]],
    model: _Model,
    num_partitions: Optional[_PositiveInt],
    multivariate: bool,
    categorical_exog_list: Optional[list[str]] = None,
) -> pd.DataFrame:
    return client.detect_anomalies(
        df=df,
        freq=freq,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        level=level,
        finetuned_model_id=finetuned_model_id,
        clean_ex_first=clean_ex_first,
        validate_api_key=validate_api_key,
        date_features=date_features,
        date_features_to_one_hot=date_features_to_one_hot,
        model=model,
        num_partitions=num_partitions,
        multivariate=multivariate,
        categorical_exog_list=categorical_exog_list,
    )


def _detect_anomalies_online_wrapper(
    df: pd.DataFrame,
    client: NixtlaClient,
    h: _PositiveInt,
    detection_size: _PositiveInt,
    threshold_method: _ThresholdMethod,
    freq: Optional[_Freq],
    id_col: str,
    time_col: str,
    target_col: str,
    level: Union[int, float],
    clean_ex_first: bool,
    step_size: _PositiveInt,
    finetune_steps: _NonNegativeInt,
    finetune_depth: _FinetuneDepth,
    finetune_loss: _Loss,
    finetuned_model_id: Optional[str],
    hist_exog_list: Optional[list[str]],
    date_features: Union[bool, list[str]],
    date_features_to_one_hot: Union[bool, list[str]],
    model: _Model,
    model_parameters: _ExtraParamDataType,
    refit: bool,
    num_partitions: Optional[_PositiveInt],
    multivariate: bool,
) -> pd.DataFrame:
    return client.detect_anomalies_online(
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
        num_partitions=num_partitions,
        multivariate=multivariate,
    )


def _cross_validation_wrapper(
    df: pd.DataFrame,
    client: NixtlaClient,
    h: _PositiveInt,
    freq: Optional[_Freq],
    id_col: str,
    time_col: str,
    target_col: str,
    level: Optional[list[Union[int, float]]],
    quantiles: Optional[list[float]],
    validate_api_key: bool,
    n_windows: _PositiveInt,
    step_size: Optional[_PositiveInt],
    finetune_steps: _NonNegativeInt,
    finetune_depth: _FinetuneDepth,
    finetune_loss: _Loss,
    finetuned_model_id: Optional[str],
    refit: bool,
    clean_ex_first: bool,
    hist_exog_list: Optional[list[str]],
    date_features: Union[bool, list[str]],
    date_features_to_one_hot: Union[bool, list[str]],
    model: _Model,
    num_partitions: Optional[_PositiveInt],
    model_parameters: _ExtraParamDataType,
    multivariate: bool,
    categorical_exog_list: Optional[list[str]] = None,
    _is_async_job: bool = False,
    _poll_interval: Optional[float] = _DEFAULT_POLL_INTERVAL,
    _poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
    _job_timeout_seconds: Optional[int] = None,
) -> pd.DataFrame:
    return client.cross_validation(
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
        num_partitions=num_partitions,
        model_parameters=model_parameters,
        multivariate=multivariate,
        categorical_exog_list=categorical_exog_list,
        _is_async_job=_is_async_job,
        _poll_interval=_poll_interval,
        _poll_timeout=_poll_timeout,
        _job_timeout_seconds=_job_timeout_seconds,
    )


def _get_schema(
    df: "AnyDataFrame",
    method: str,
    id_col: str,
    time_col: str,
    target_col: str,
    level: Optional[Union[int, float, list[Union[int, float]]]],
    quantiles: Optional[list[float]],
) -> "triad.Schema":
    import fugue.api as fa

    base_cols = [id_col, time_col]
    if method != "forecast":
        base_cols.append(target_col)
    schema = fa.get_schema(df).extract(base_cols).copy()
    schema.append("TimeGPT:double")
    if method == "detect_anomalies":
        schema.append("anomaly:bool")
    if method == "detect_anomalies_online":
        schema.append("anomaly:bool")
        schema.append("anomaly_score:double")
    elif method == "cross_validation":
        schema.append(("cutoff", schema[time_col].type))
    if level is not None and quantiles is not None:
        raise ValueError("You should provide `level` or `quantiles` but not both.")
    if level is not None:
        if not isinstance(level, list):
            level = [level]
        level = sorted(level)
        schema.append(",".join(f"TimeGPT-lo-{lv}:double" for lv in reversed(level)))
        schema.append(",".join(f"TimeGPT-hi-{lv}:double" for lv in level))
    if quantiles is not None:
        quantiles = sorted(quantiles)
        q_cols = [f"TimeGPT-q-{int(q * 100)}:double" for q in quantiles]
        schema.append(",".join(q_cols))
    return schema


def _distributed_setup(
    df: "AnyDataFrame",
    method: str,
    id_col: str,
    time_col: str,
    target_col: str,
    level: Optional[Union[int, float, list[Union[int, float]]]],
    quantiles: Optional[list[float]],
    num_partitions: Optional[int],
) -> tuple["triad.Schema", dict[str, Any]]:
    from fugue.execution import infer_execution_engine

    if infer_execution_engine([df]) is None:
        raise ValueError(
            f"Could not infer execution engine for type {type(df).__name__}. "
            "Expected a spark or dask DataFrame or a ray Dataset."
        )
    schema = _get_schema(
        df=df,
        method=method,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        level=level,
        quantiles=quantiles,
    )
    partition_config: dict[str, Any] = dict(by=id_col, algo="coarse")
    if num_partitions is not None:
        partition_config["num"] = num_partitions
    return schema, partition_config
