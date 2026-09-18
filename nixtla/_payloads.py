"""Request payload construction: DataFrame in, `(payload, parse_result)` out.

One builder per task, shared by the blocking `NixtlaClient` methods and the
`client.jobs` namespace so the two paths cannot drift. Batch
`detect_anomalies()` is the exception: its payload is still built inline in
`nixtla_client.py`, pending its removal in 1.0.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Union, get_args

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from utilsforecast.compat import DataFrame, DFType, pl_DataFrame

from ._http import logger
from ._preprocessing import (
    _align_future_categorical_exog,
    _array_tails,
    _build_exog_payload,
    _coerce_coupled_flag,
    _extract_categorical_exog,
    _extract_target_array,
    _features_with_missing_values,
    _is_numeric_column,
    _log_exog_features,
    _maybe_add_intervals,
    _maybe_convert_level_to_quantiles,
    _maybe_drop_id,
    _numeric_column_array,
    _parse_in_sample_output,
    _prepare_level_and_quantiles,
    _preprocess,
    _process_exog_features,
    _restrict_input_samples,
    _series_starts,
    _sort_categorical_values,
    _standardize_freq,
    _tail,
    _time_col_tz,
    _times_to_iso,
    _validate_exog,
    _validate_future_exog_keys,
    _validate_input_size,
)
from ._types import (
    _ExplainMethod,
    _ExtraParamDataType,
    _FeatureContributionsType,
    _FinetuneDepth,
    _Freq,
    _Loss,
    _Model,
    _NonNegativeInt,
    _PositiveInt,
    _ThresholdMethod,
)

if TYPE_CHECKING:
    from .nixtla_client import NixtlaClient


def prepare_forecast(
    client: "NixtlaClient",
    df: DFType,
    h: _PositiveInt,
    freq: Optional[_Freq],
    id_col: str,
    time_col: str,
    target_col: str,
    X_df: Optional[DFType],
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
    feature_contributions: bool,
    model_parameters: _ExtraParamDataType,
    multivariate: bool,
    feature_contributions_type: _FeatureContributionsType = "shapley",
) -> tuple[dict[str, Any], np.ndarray, int, int, Callable[..., Any]]:
    client.__dict__.pop("weights_x", None)
    client.__dict__.pop("feature_contributions", None)
    model = client._maybe_override_model(model)
    logger.info("Validating inputs...")
    df, X_df, drop_id, freq = client._run_validations(
        df=df,
        X_df=X_df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        validate_api_key=validate_api_key,
        freq=freq,
    )
    df, X_df, df_cat_vals, futr_cat_cols, hist_cat_cols, X_df_cat_future = (
        _extract_categorical_exog(
            df=df,
            categorical_exog_list=categorical_exog_list,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            X_df=X_df,
        )
    )
    # Exclude hist_cat_cols from hist_exog: they've been stripped from df
    # by _extract_categorical_exog, so _validate_exog must not look for them.
    num_hist_exog = (
        [c for c in hist_exog_list if c not in hist_cat_cols]
        if hist_exog_list
        else hist_exog_list
    )
    df, X_df = _validate_exog(
        df=df,
        X_df=X_df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        hist_exog=num_hist_exog,
    )

    level, quantiles = _prepare_level_and_quantiles(level, quantiles)

    logger.info("Preprocessing dataframes...")
    processed, X_future, x_cols, futr_cols = _preprocess(
        df=df,
        X_df=X_df,
        h=h,
        freq=freq,
        date_features=date_features,
        date_features_to_one_hot=date_features_to_one_hot,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    future_df = ufp.make_future_dataframe(
        uids=processed.uids,
        last_times=type(processed.uids)(processed.last_times),
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    _validate_future_exog_keys(
        X_df=X_df,
        expected=future_df,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    X_df_cat_future_values = _align_future_categorical_exog(
        X_df_cat_future=X_df_cat_future,
        uids=processed.uids,
        id_col=id_col,
        time_col=time_col,
    )

    sorted_df_cat: dict[str, np.ndarray] = {}
    if categorical_exog_list:
        sorted_df_cat = _sort_categorical_values(df_cat_vals, processed.sort_idxs)

    standard_freq = _standardize_freq(freq, processed)
    model_input_size, model_horizon = client._get_model_params(model, standard_freq)
    if finetune_steps > 0:
        _validate_input_size(processed, 1, 1)
    if add_history:
        _validate_input_size(processed, 1, 1)
    if h > model_horizon:
        logger.warning(
            'The specified horizon "h" exceeds the model horizon, '
            "this may lead to less accurate forecasts. "
            "Please consider using a smaller horizon."
        )
    restrict_input = (
        finetune_steps == 0
        and not x_cols
        and not categorical_exog_list
        and not add_history
    )
    orig_indptr: Optional[np.ndarray] = None
    orig_sort_idxs: Optional[np.ndarray] = None
    if restrict_input:
        logger.info("Restricting input...")
        new_input_size = _restrict_input_samples(
            level=level,
            input_size=model_input_size,
            model_horizon=model_horizon,
            h=h,
        )
        # _tail resets both of these, so keep them to map start times back to `df`
        orig_indptr = processed.indptr
        orig_sort_idxs = processed.sort_idxs
        processed = _tail(processed, new_input_size)

    X, X_future, categorical_exog_payload, weights_x_cols = _build_exog_payload(
        processed=processed,
        sorted_df_cat=sorted_df_cat,
        x_cols=x_cols,
        futr_cols=futr_cols,
        futr_cat_cols=futr_cat_cols,
        hist_cat_cols=hist_cat_cols,
        X_future=X_future,
        X_df_cat_future=X_df_cat_future_values,
        has_categorical=bool(categorical_exog_list),
    )
    if X is not None:
        _log_exog_features(
            futr_cols=futr_cols,
            futr_cat_cols=futr_cat_cols,
            hist_exog_list=hist_exog_list,
            hist_cat_cols=hist_cat_cols,
        )

    logger.info("Calling Forecast Endpoint...")
    sizes = np.diff(processed.indptr)
    series_payload: dict[str, Any] = {
        "y": processed.data[:, 0],
        "sizes": sizes,
        "X": X,
        "X_future": X_future,
    }
    start_datetime = _series_starts(
        df, processed, time_col, orig_indptr, orig_sort_idxs
    )
    if start_datetime is not None:
        series_payload["start_datetime"] = start_datetime
    if categorical_exog_payload is not None:
        series_payload["categorical_exog"] = categorical_exog_payload
    payload = {
        "series": series_payload,
        "model": model,
        "h": h,
        "freq": standard_freq,
        "clean_ex_first": clean_ex_first,
        "level": level,
        "finetune_steps": finetune_steps,
        "finetune_depth": finetune_depth,
        "finetune_loss": finetune_loss,
        "finetuned_model_id": finetuned_model_id,
        "feature_contributions": feature_contributions and X is not None,
        "multivariate": multivariate,
    }
    if feature_contributions:
        payload["feature_contributions_type"] = feature_contributions_type
    if model_parameters is not None:
        payload.update({"model_parameters": model_parameters})

    def parse_result(
        resp: dict[str, Any],
        in_sample_resp: Optional[dict[str, Any]] = None,
        insample_feat_contributions: Optional[Any] = None,
    ) -> Any:
        # assemble result
        out = ufp.assign_columns(future_df, "TimeGPT", resp["mean"])
        out = _maybe_add_intervals(out, resp["intervals"])
        if add_history:
            assert in_sample_resp is not None
            in_sample_df = _parse_in_sample_output(
                in_sample_output=in_sample_resp,
                df=df,
                processed=processed,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            in_sample_df = ufp.drop_columns(in_sample_df, target_col)
            out = ufp.vertical_concat([in_sample_df, out])
        out = _maybe_convert_level_to_quantiles(out, quantiles)
        client._maybe_assign_feature_contributions(
            expected_contributions=feature_contributions,
            resp=resp,
            x_cols=weights_x_cols,
            out_df=out[[id_col, time_col, "TimeGPT"]],
            insample_feat_contributions=insample_feat_contributions,
        )
        if add_history:
            sort_idxs = ufp.maybe_compute_sort_indices(
                out, id_col=id_col, time_col=time_col
            )
            if sort_idxs is not None:
                out = ufp.take_rows(out, sort_idxs)
                out = ufp.drop_index_if_pandas(out)
                if hasattr(client, "feature_contributions"):
                    client.feature_contributions = ufp.take_rows(
                        client.feature_contributions, sort_idxs
                    )
                    client.feature_contributions = ufp.drop_index_if_pandas(
                        client.feature_contributions
                    )
        out = _maybe_drop_id(df=out, id_col=id_col, drop=drop_id)
        client._maybe_assign_weights(
            weights=resp["weights_x"], df=df, x_cols=weights_x_cols
        )
        return out

    return payload, sizes, model_horizon, model_input_size, parse_result


def prepare_cross_validation(
    client: "NixtlaClient",
    df: DFType,
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
    model_parameters: _ExtraParamDataType,
    multivariate: bool,
    categorical_exog_list: Optional[list[str]],
) -> tuple[dict[str, Any], Callable[[dict[str, Any]], Any]]:
    model = client._maybe_override_model(model)
    logger.info("Validating inputs...")
    df, _, drop_id, freq = client._run_validations(
        df=df,
        X_df=None,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        validate_api_key=validate_api_key,
        freq=freq,
    )
    level, quantiles = _prepare_level_and_quantiles(level, quantiles)
    if step_size is None:
        step_size = h

    df, _, df_cat_vals, _, hist_cat_cols, _ = _extract_categorical_exog(
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

    sorted_df_cat: dict[str, np.ndarray] = {}
    if categorical_exog_list:
        for c, vals in df_cat_vals.items():
            sorted_df_cat[c] = (
                vals[processed.sort_idxs] if processed.sort_idxs is not None else vals
            )

    standard_freq = _standardize_freq(freq, processed)
    model_input_size, model_horizon = client._get_model_params(model, standard_freq)
    targets = _extract_target_array(df, target_col)
    times = df[time_col].to_numpy()
    if processed.sort_idxs is not None:
        targets = targets[processed.sort_idxs]
        times = times[processed.sort_idxs]
    restrict_input = finetune_steps == 0 and not x_cols and not categorical_exog_list
    if restrict_input:
        logger.info("Restricting input...")
        new_input_size = _restrict_input_samples(
            level=level,
            input_size=model_input_size,
            model_horizon=model_horizon,
            h=h,
        )
        new_input_size += h + step_size * (n_windows - 1)
        orig_indptr = processed.indptr
        processed = _tail(processed, new_input_size)
        times = _array_tails(times, orig_indptr, np.diff(processed.indptr))
        targets = _array_tails(targets, orig_indptr, np.diff(processed.indptr))
    _num_hist: Optional[list[str]] = None
    if hist_exog_list:
        _num_hist = [c for c in hist_exog_list if c not in hist_cat_cols] or None
    X_np, hist_exog = _process_exog_features(processed.data, x_cols, _num_hist)

    X: Optional[list[Any]] = None
    categorical_exog_payload: Optional[list[int]] = None
    if categorical_exog_list:
        n_num_cols = len(x_cols)
        cat_arrays = [sorted_df_cat[c].tolist() for c in hist_cat_cols]
        if X_np is not None:
            X = list(X_np) + cat_arrays
        else:
            X = cat_arrays
        cat_col_indices = list(range(n_num_cols, n_num_cols + len(hist_cat_cols)))
        categorical_exog_payload = cat_col_indices
        if hist_cat_cols:
            logger.info(
                f"Using historical categorical exogenous features: {hist_cat_cols}"
            )
    else:
        X = list(X_np) if X_np is not None else None

    series_payload: dict[str, Any] = {
        "y": targets,
        "sizes": np.diff(processed.indptr),
        "X": X,
    }
    # `times` is sorted and, when the input was restricted, trimmed alongside
    # `targets`, so it already matches the payload row order.
    start_datetime = _times_to_iso(
        times[processed.indptr[:-1]], _time_col_tz(df, time_col)
    )
    if start_datetime is not None:
        series_payload["start_datetime"] = start_datetime
    if categorical_exog_payload is not None:
        series_payload["categorical_exog"] = categorical_exog_payload

    logger.info("Calling Cross Validation Endpoint...")
    payload = {
        "series": series_payload,
        "model": model,
        "h": h,
        "n_windows": n_windows,
        "step_size": step_size,
        "freq": standard_freq,
        "clean_ex_first": clean_ex_first,
        "hist_exog": hist_exog,
        "level": level,
        "finetune_steps": finetune_steps,
        "finetune_depth": finetune_depth,
        "finetune_loss": finetune_loss,
        "finetuned_model_id": finetuned_model_id,
        "refit": refit,
        "multivariate": multivariate,
    }
    if model_parameters is not None:
        payload.update({"model_parameters": model_parameters})

    def parse_result(resp: dict[str, Any]) -> Any:
        # assemble result
        idxs = np.array(resp["idxs"], dtype=np.int64)
        sizes = np.array(resp["sizes"], dtype=np.int64)
        window_starts = np.arange(0, sizes.sum(), h)
        cutoff_idxs = np.repeat(idxs[window_starts] - 1, h)
        out = type(df)(
            {
                id_col: ufp.repeat(processed.uids, sizes),
                time_col: times[idxs],
                "cutoff": times[cutoff_idxs],
                target_col: targets[idxs],
            }
        )
        out = ufp.assign_columns(out, "TimeGPT", resp["mean"])
        out = _maybe_add_intervals(out, resp["intervals"])
        out = _maybe_drop_id(df=out, id_col=id_col, drop=drop_id)
        return _maybe_convert_level_to_quantiles(out, quantiles)

    return payload, parse_result


def prepare_anomaly_detection(
    client: "NixtlaClient",
    df: DFType,
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
    multivariate: bool,
) -> tuple[dict[str, Any], Callable[[dict[str, Any]], Any]]:
    """Build the payload and result parser shared by the sync and submit paths."""
    client.__dict__.pop("weights_x", None)
    model = client._maybe_override_model(model)
    logger.info("Validating inputs...")
    df, _, drop_id, freq = client._run_validations(
        df=df,
        X_df=None,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        validate_api_key=False,
        freq=freq,
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
    targets = _extract_target_array(df, target_col)
    times = df[time_col].to_numpy()
    if processed.sort_idxs is not None:
        targets = targets[processed.sort_idxs]
        times = times[processed.sort_idxs]
    else:
        # Own these: otherwise they are views into `df`, and `parse_result` can run
        # long after submit returned, by which time `df` may have been mutated.
        targets = targets.copy()
        times = times.copy()
    X, hist_exog = _process_exog_features(processed.data, x_cols, hist_exog_list)
    sizes = np.diff(processed.indptr)
    if np.all(sizes <= 6 * detection_size):
        logger.warn(
            "Detection size is large. Using the entire series to compute the anomaly threshold..."
        )
    online_series: dict[str, Any] = {
        "y": processed.data[:, 0],
        "sizes": sizes,
        "X": X,
    }
    # `times` is already sorted to match the payload row order
    start_datetime = _times_to_iso(
        times[processed.indptr[:-1]], _time_col_tz(df, time_col)
    )
    if start_datetime is not None:
        online_series["start_datetime"] = start_datetime
    payload = {
        "series": online_series,
        "h": h,
        "detection_size": detection_size,
        "threshold_method": threshold_method,
        "model": model,
        "freq": standard_freq,
        "clean_ex_first": clean_ex_first,
        "level": level,
        "step_size": step_size,
        "finetune_steps": finetune_steps,
        "finetune_loss": finetune_loss,
        "finetune_depth": finetune_depth,
        "finetuned_model_id": finetuned_model_id,
        "refit": refit,
        "hist_exog": hist_exog,
        "multivariate": multivariate,
    }
    if model_parameters is not None:
        payload.update({"model_parameters": model_parameters})

    # A `Job` holds `parse_result` until the caller drops it, so capturing
    # `df`/`processed` would pin the whole input for that long.
    df_cls = type(df)
    uids = processed.uids

    def parse_result(resp: dict[str, Any]) -> Any:
        # assemble result
        idxs = np.array(resp["idxs"], dtype=np.int64)
        sizes = np.array(resp["sizes"], dtype=np.int64)
        out = df_cls(
            {
                id_col: ufp.repeat(uids, sizes),
                time_col: times[idxs],
                target_col: targets[idxs],
            }
        )
        out = ufp.assign_columns(out, "TimeGPT", resp["mean"])
        out = ufp.assign_columns(out, "anomaly", resp["anomaly"])
        out = ufp.assign_columns(out, "anomaly_score", resp["anomaly_score"])
        if threshold_method == "multivariate":
            out = ufp.assign_columns(
                out, "accumulated_anomaly_score", resp["accumulated_anomaly_score"]
            )
        # Optional in the response schema; `_maybe_add_intervals` no-ops on None.
        return _maybe_add_intervals(out, resp.get("intervals"))

    return payload, parse_result


def prepare_simulate(
    client: "NixtlaClient",
    df: DataFrame,
    h: int,
    freq: Optional[_Freq],
    id_col: str,
    time_col: str,
    target_col: str,
    X_df: Optional[DataFrame],
    n_paths: int,
    quantiles: Optional[list[float]],
    seed: Optional[int],
    finetuned_model_id: Optional[str],
    clean_ex_first: bool,
    hist_exog_list: Optional[list[str]],
    categorical_exog_list: Optional[list[str]],
    validate_api_key: bool,
    date_features: Union[bool, list[Union[str, Callable]]],
    date_features_to_one_hot: Union[bool, list[str]],
    model: _Model,
    multivariate: bool,
    method_name: str,
) -> tuple[dict[str, Any], Callable[[dict[str, Any]], Any]]:
    """Build the payload and result parser shared by the sync and submit paths.

    `h`, `n_paths` and `seed` are expected to have gone through
    `_validate_simulate_args` already: the partitioned path needs the coerced
    values before it can derive its per-partition seeds.
    """
    if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
        raise ValueError(f"`{method_name}` only supports pandas and polars dataframes.")
    if X_df is not None and not isinstance(X_df, (pd.DataFrame, pl_DataFrame)):
        raise ValueError("`X_df` must be a pandas or polars dataframe.")
    level, _ = _prepare_level_and_quantiles(None, quantiles)

    model = client._maybe_override_model(model)
    logger.info("Validating inputs...")
    df, X_df, drop_id, freq = client._run_validations(
        df=df,
        X_df=X_df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        validate_api_key=validate_api_key,
        freq=freq,
    )
    (
        df,
        X_df,
        df_cat_vals,
        futr_cat_cols,
        hist_cat_cols,
        X_df_cat_future,
    ) = _extract_categorical_exog(
        df=df,
        categorical_exog_list=categorical_exog_list,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        X_df=X_df,
    )
    numeric_hist_exog = (
        [col for col in hist_exog_list if col not in hist_cat_cols]
        if hist_exog_list
        else hist_exog_list
    )
    df, X_df = _validate_exog(
        df=df,
        X_df=X_df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        hist_exog=numeric_hist_exog,
    )

    logger.info("Preprocessing dataframes...")
    processed, X_future, x_cols, futr_cols = _preprocess(
        df=df,
        X_df=X_df,
        h=h,
        freq=freq,
        date_features=date_features,
        date_features_to_one_hot=date_features_to_one_hot,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    future_df = ufp.make_future_dataframe(
        uids=processed.uids,
        last_times=type(processed.uids)(processed.last_times),
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    _validate_future_exog_keys(
        X_df=X_df,
        expected=future_df,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    X_df_cat_future_values = _align_future_categorical_exog(
        X_df_cat_future=X_df_cat_future,
        uids=processed.uids,
        id_col=id_col,
        time_col=time_col,
    )
    sorted_df_cat = _sort_categorical_values(df_cat_vals, processed.sort_idxs)
    standard_freq = _standardize_freq(freq, processed)
    model_input_size, model_horizon = client._get_model_params(model, standard_freq)
    if h > model_horizon:
        logger.warning(
            'The specified horizon "h" exceeds the model horizon, '
            "this may lead to less accurate sample paths. "
            "Please consider using a smaller horizon."
        )
    if not x_cols and not categorical_exog_list:
        logger.info("Restricting input...")
        new_input_size = _restrict_input_samples(
            level=level,
            input_size=model_input_size,
            model_horizon=model_horizon,
            h=h,
        )
        if multivariate:
            new_input_size = max(new_input_size, h)
        processed = _tail(processed, new_input_size)
    X, X_future, categorical_exog_payload, _ = _build_exog_payload(
        processed=processed,
        sorted_df_cat=sorted_df_cat,
        x_cols=x_cols,
        futr_cols=futr_cols,
        futr_cat_cols=futr_cat_cols,
        hist_cat_cols=hist_cat_cols,
        X_future=X_future,
        X_df_cat_future=X_df_cat_future_values,
        has_categorical=bool(categorical_exog_list),
    )
    if X is not None:
        _log_exog_features(
            futr_cols=futr_cols,
            futr_cat_cols=futr_cat_cols,
            hist_exog_list=hist_exog_list,
            hist_cat_cols=hist_cat_cols,
        )

    sizes = np.diff(processed.indptr)
    series_payload: dict[str, Any] = {
        "y": processed.data[:, 0],
        "sizes": sizes,
        "X": X,
        "X_future": X_future,
    }
    if categorical_exog_payload is not None:
        series_payload["categorical_exog"] = categorical_exog_payload
    payload = {
        "series": series_payload,
        "model": model,
        "h": h,
        "freq": standard_freq,
        "n_paths": n_paths,
        "quantiles": quantiles,
        "seed": seed,
        "finetuned_model_id": finetuned_model_id,
        "clean_ex_first": clean_ex_first,
        "multivariate": multivariate,
    }

    # A `Job` holds `parse_result` until the caller drops it, so only the
    # scaffolding the output needs is captured, never `df` or `processed`.
    n_series = len(sizes)
    output_values = n_paths * n_series * h
    expected_sizes = np.full(n_series, h)

    def parse_result(resp: dict[str, Any]) -> Any:
        response_n_paths = resp.get("n_paths")
        response_h = resp.get("h")
        try:
            response_sizes = np.asarray(resp.get("sizes"), dtype=np.int64)
            samples = np.asarray(resp.pop("samples", None), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Simulation response contains non-numeric sizes or samples."
            ) from exc
        if response_n_paths != n_paths or response_h != h:
            raise RuntimeError(
                "Simulation response metadata does not match the request."
            )
        if response_sizes.shape != expected_sizes.shape or not np.array_equal(
            response_sizes, expected_sizes
        ):
            raise RuntimeError("Simulation response contains unexpected series sizes.")
        if samples.ndim != 1 or samples.size != output_values:
            raise RuntimeError(
                f"Simulation response contains {samples.size:,} values; "
                f"expected {output_values:,}."
            )
        coupled = _coerce_coupled_flag(resp.get("coupled"))

        future_rows = len(future_df)
        out = ufp.take_rows(
            future_df, np.tile(np.arange(future_rows, dtype=np.int32), n_paths)
        )
        if isinstance(out, pd.DataFrame):
            out = out.set_axis(pd.RangeIndex(len(out)), axis=0, copy=False)
        out = ufp.assign_columns(
            out,
            "sample_id",
            np.repeat(np.arange(n_paths, dtype=np.int64), future_rows),
        )
        out = ufp.assign_columns(out, "TimeGPT", samples)
        out = ufp.assign_columns(out, "coupled", np.full(output_values, coupled))
        return _maybe_drop_id(df=out, id_col=id_col, drop=drop_id)

    return payload, parse_result


def prepare_explain(
    client: "NixtlaClient",
    df: DataFrame,
    method: _ExplainMethod,
    features: Optional[list[str]],
    freq: Optional[_Freq],
    id_col: str,
    time_col: str,
    target_col: str,
    categorical_exog_list: Optional[list[str]],
    validate_api_key: bool,
    method_name: str,
) -> tuple[dict[str, Any], Callable[[dict[str, Any]], Any]]:
    """Build the payload and result parser shared by the sync and submit paths."""
    if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
        raise ValueError(f"`{method_name}` only supports pandas and polars dataframes.")
    if method not in get_args(_ExplainMethod):
        raise ValueError("`method` must be 'granger' or 'transfer_entropy'.")
    df, _, _, freq = client._run_validations(
        df=df,
        X_df=None,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        validate_api_key=validate_api_key,
        freq=freq,
    )

    base_columns = {id_col, time_col, target_col}
    if features is None:
        features = [col for col in df.columns if col not in base_columns]
    else:
        features = list(features)
    if not features:
        raise ValueError(f"`{method_name}` requires at least one feature.")
    if len(features) != len(set(features)):
        raise ValueError("`features` must not contain duplicates.")
    invalid_features = set(features) - set(df.columns)
    if invalid_features:
        raise ValueError(
            f"The following features were not found in `df`: {invalid_features}."
        )
    reserved_features = set(features) & base_columns
    if reserved_features:
        raise ValueError(
            "ID, time, and target columns cannot be explanation features: "
            f"{reserved_features}."
        )

    categorical_exog_list = list(categorical_exog_list or [])
    invalid_categorical = set(categorical_exog_list) - set(features)
    if invalid_categorical:
        raise ValueError(
            "Every categorical feature must also be present in `features`: "
            f"{invalid_categorical}."
        )
    undeclared_non_numeric = sorted(
        col
        for col in features
        if col not in categorical_exog_list and not _is_numeric_column(df=df, col=col)
    )
    if undeclared_non_numeric:
        raise ValueError(
            "The following features are not numeric: "
            f"{undeclared_non_numeric}. Add them to `categorical_exog_list` "
            "to treat them as categorical, or exclude them via `features`."
        )
    features_with_missing = _features_with_missing_values(df, features)
    if features_with_missing:
        logger.warning(
            "The following features contain missing values: "
            f"{features_with_missing}. Rows with a missing value in a "
            "lagged column are excluded from that feature's weight "
            "estimation, which reduces the effective sample."
        )

    processed = ufp.process_df(
        df=df[[id_col, time_col, target_col]],
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    feature_rows: list[Any] = []
    for feature in features:
        is_categorical = feature in categorical_exog_list
        if is_categorical:
            values = df[feature].to_numpy()
            if isinstance(df, pd.DataFrame):
                missing = pd.isna(values)
                if missing.any():
                    values = values.astype(object, copy=True)
                    values[missing] = None
        else:
            values = _numeric_column_array(df, feature)
        if processed.sort_idxs is not None:
            values = values[processed.sort_idxs]
        feature_rows.append(values.tolist() if is_categorical else values)
    categorical_positions = [
        position
        for position, feature in enumerate(features)
        if feature in categorical_exog_list
    ]
    series_payload: dict[str, Any] = {
        "y": processed.data[:, 0],
        "sizes": np.diff(processed.indptr),
        "X": feature_rows,
    }
    if categorical_positions:
        series_payload["categorical_exog"] = categorical_positions
    payload = {"series": series_payload, "method": method}

    # A `Job` holds `parse_result` until the caller drops it, so the output
    # frame is rebuilt from the feature names alone, never from `df`.
    df_cls = type(df)

    def parse_result(resp: dict[str, Any]) -> Any:
        try:
            weights = np.asarray(resp.get("weights"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Explain response contains non-numeric weights."
            ) from exc
        if weights.ndim != 1 or weights.size != len(features):
            raise RuntimeError(
                f"Explain response contains {weights.size} weights; "
                f"expected {len(features)}."
            )
        response_method = resp.get("method")
        if response_method != method:
            raise RuntimeError("Explain response metadata does not match the request.")
        return df_cls(
            {
                "feature": features,
                "weight": weights,
                "method": [response_method] * len(features),
            }
        )

    return payload, parse_result


def prepare_finetune_payload(
    client: "NixtlaClient",
    df: DataFrame,
    freq: Optional[_Freq],
    id_col: str,
    time_col: str,
    target_col: str,
    finetune_steps: _NonNegativeInt,
    finetune_depth: _FinetuneDepth,
    finetune_loss: _Loss,
    output_model_id: Optional[str],
    finetuned_model_id: Optional[str],
    model: _Model,
) -> dict[str, Any]:
    if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
        raise ValueError("Can only fine-tune on pandas or polars dataframes.")
    model = client._maybe_override_model(model)
    logger.info("Validating inputs...")
    df, X_df, drop_id, freq = client._run_validations(
        df=df,
        X_df=None,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        validate_api_key=False,
        freq=freq,
    )

    logger.info("Preprocessing dataframes...")
    processed, *_ = _preprocess(
        df=df,
        X_df=None,
        h=0,
        freq=freq,
        date_features=False,
        date_features_to_one_hot=False,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    standard_freq = _standardize_freq(freq, processed)
    _validate_input_size(processed, 1, 1)
    logger.info("Calling Fine-tune Endpoint...")
    finetune_series: dict[str, Any] = {
        "y": processed.data[:, 0],
        "sizes": np.diff(processed.indptr),
    }
    start_datetime = _series_starts(df, processed, time_col)
    if start_datetime is not None:
        finetune_series["start_datetime"] = start_datetime
    return {
        "series": finetune_series,
        "model": model,
        "freq": standard_freq,
        "finetune_steps": finetune_steps,
        "finetune_depth": finetune_depth,
        "finetune_loss": finetune_loss,
        "output_model_id": output_model_id,
        "finetuned_model_id": finetuned_model_id,
    }
