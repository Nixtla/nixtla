"""Dataframe preparation: validation, frequency handling, exogenous features,
partitioning and result parsing.

Everything here is a plain function over dataframes and payload dicts -- no
client, no HTTP. `nixtla_client` calls into it on the way to a request and on
the way back from a response.
"""

import datetime
import functools
import math
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from utilsforecast.compat import DataFrame, DFType, pl_DataFrame
from utilsforecast.feature_engineering import _add_time_features, time_features
from utilsforecast.preprocessing import id_time_grid
from utilsforecast.processing import ensure_sorted
from utilsforecast.validation import ensure_time_dtype

from ._http import logger
from ._types import _Freq, _FreqType

_date_features_by_freq = {
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


def _coerce_positive_int(value: Any, name: str) -> int:
    """Return `value` as a plain `int`, rejecting anything that is not a positive integer.

    Numpy integers are accepted and narrowed so the payload and the partitioned path
    carry the same plain values; `bool` is not, being an `int` only by inheritance.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def _validate_simulate_args(
    h: Any,
    n_paths: Any,
    seed: Any,
    num_partitions: Any,
    multivariate: bool,
) -> tuple[int, int, Optional[int], Optional[int]]:
    """Coerce and check `simulate`'s numeric arguments, shared by the sync and submit paths.

    Returns them coerced to plain ints so the partitioned path and the payload agree on
    the exact values a numpy integer would otherwise carry through.
    """
    h = _coerce_positive_int(h, "h")
    n_paths = _coerce_positive_int(n_paths, "n_paths")
    if num_partitions is not None:
        num_partitions = _coerce_positive_int(num_partitions, "num_partitions")
        if multivariate:
            raise ValueError(
                "`num_partitions` cannot be combined with `multivariate=True`: "
                "cross-series coupling is computed across the series in a "
                "single request, so partitioning would silently return "
                "uncoupled paths."
            )
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("`seed` must be an integer.")
        seed = int(seed)
    return h, n_paths, seed, num_partitions

def _maybe_infer_freq(
    df: DataFrame,
    freq: Optional[_FreqType],
    id_col: str,
    time_col: str,
) -> _FreqType:
    if freq is not None:
        return freq
    if isinstance(df, pl_DataFrame):
        raise ValueError(
            "Cannot infer frequency for a polars DataFrame, please set the "
            "`freq` argument to a valid polars offset.\nYou can find them at "
            "https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.dt.offset_by.html"
        )
    assert isinstance(df, pd.DataFrame)
    sizes = df[id_col].value_counts(sort=True)
    times = df.loc[df[id_col] == sizes.index[0], time_col].sort_values()
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    inferred_freq = pd.infer_freq(times.values)
    if inferred_freq is None:
        raise RuntimeError(
            "Could not infer the frequency of the time column. This could be due "
            "to inconsistent intervals. Please check your data for missing, "
            "duplicated or irregular timestamps"
        )
    logger.info(f"Inferred freq: {inferred_freq}")
    return inferred_freq


def _is_numeric_column(df: DataFrame, col: str) -> bool:
    if isinstance(df, pd.DataFrame):
        dtype = df[col].dtype
        return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
            dtype
        )
    return df[col].dtype.is_numeric()


def _features_with_missing_values(df: DataFrame, features: list[str]) -> list[str]:
    if isinstance(df, pd.DataFrame):
        missing = df[features].isna().any()
        return [feature for feature in features if missing[feature]]
    out = []
    for feature in features:
        col = df[feature]
        n_missing = col.null_count()
        # polars keeps nulls and NaNs distinct; both break lag computations.
        if col.dtype.is_float():
            n_missing += int(col.is_nan().sum())
        if n_missing:
            out.append(feature)
    return out


def _numeric_column_array(df: DataFrame, col: str) -> np.ndarray:
    if not isinstance(df, pd.DataFrame):
        return df[col].to_numpy()
    target_dtype = df.dtypes[col].type
    if np.issubdtype(target_dtype, np.floating):
        return df[col].to_numpy(dtype=target_dtype, na_value=np.nan)
    if df[col].isna().any():
        return df[col].to_numpy(dtype=np.float64, na_value=np.nan)
    return df[col].to_numpy(dtype=target_dtype)


def _coerce_coupled_flag(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise RuntimeError(
            f"Simulation response reported a non-boolean `coupled` value: {value!r}."
        )
    return value


def _has_duplicate_keys(df: DataFrame, id_col: str, time_col: str) -> bool:
    if isinstance(df, pd.DataFrame):
        return bool(df.duplicated(subset=[id_col, time_col]).any())
    return df.n_unique(subset=[id_col, time_col]) != len(df)


def _validate_freq_regularity(
    df: DFType,
    freq: _Freq,
    id_col: str,
    time_col: str,
) -> None:
    if isinstance(freq, (str, int)):
        expected_ids_times = id_time_grid(
            df,
            freq=freq,
            start="per_serie",
            end="per_serie",
            id_col=id_col,
            time_col=time_col,
        )
        freq_ok = len(df) == len(expected_ids_times)
    elif isinstance(freq, pd.offsets.BaseOffset):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "pandas offsets are only supported for pandas dataframes, "
                "please provide `freq` as a string for polars dataframes."
            )
        times_by_id = df.groupby(id_col, observed=True)[time_col].agg(
            ["min", "max", "size"]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            expected_ends = times_by_id["min"] + freq * (times_by_id["size"] - 1)
        freq_ok = bool((expected_ends == times_by_id["max"]).all())
    else:
        raise ValueError(
            "`freq` should be a string, integer or pandas offset, "
            f"got {type(freq).__name__}."
        )
    if _has_duplicate_keys(df, id_col, time_col):
        raise ValueError(
            "Series contain duplicate timestamps. "
            f"Each (`{id_col}`, `{time_col}`) pair must be unique."
        )
    if not freq_ok:
        raise ValueError(
            "Series contain missing timestamps, or the timestamps "
            "do not match the provided frequency.\n"
            "Please make sure that all series have a single observation from the first "
            "to the last timestamp and that the provided frequency matches the timestamps'.\n"
            "You can refer to https://docs.nixtla.io/docs/tutorials-missing_values "
            "for an end to end example."
        )


def _dataframe_keys_match(
    actual: DFType,
    expected: DFType,
    id_col: str,
    time_col: str,
) -> bool:
    if len(actual) != len(expected):
        return False
    expected_codes, id_vocabulary = pd.factorize(expected[id_col].to_numpy())
    actual_codes = pd.Index(id_vocabulary).get_indexer(actual[id_col].to_numpy())
    if (actual_codes < 0).any():
        return False
    actual_times = actual[time_col].to_numpy()
    expected_times = expected[time_col].to_numpy()
    actual_order = np.lexsort((actual_times, actual_codes))
    expected_order = np.lexsort((expected_times, expected_codes))
    return bool(
        np.array_equal(actual_codes[actual_order], expected_codes[expected_order])
        and np.array_equal(actual_times[actual_order], expected_times[expected_order])
    )


def _standardize_freq(freq: _Freq, processed: ufp.ProcessedDF) -> str:
    if isinstance(freq, str):
        # polars uses 'mo' for months, all other strings are compatible with pandas
        freq = freq.replace("mo", "MS")
    elif isinstance(freq, pd.offsets.BaseOffset):
        freq = freq.freqstr
    elif isinstance(freq, int):
        freq = "MS"
    else:
        raise ValueError(
            f"`freq` must be a string, int or pandas offset, got {type(freq).__name__}"
        )
    return freq


def _array_tails(
    x: np.ndarray,
    indptr: np.ndarray,
    out_sizes: np.ndarray,
) -> np.ndarray:
    if (out_sizes > np.diff(indptr)).any():
        raise ValueError("out_sizes must be at most the original sizes.")
    idxs = np.hstack(
        [np.arange(end - size, end) for end, size in zip(indptr[1:], out_sizes)]
    )
    return x[idxs]


def _tail(proc: ufp.ProcessedDF, n: int) -> ufp.ProcessedDF:
    new_sizes = np.minimum(np.diff(proc.indptr), n)
    new_indptr = np.append(0, new_sizes.cumsum())
    new_data = _array_tails(proc.data, proc.indptr, new_sizes)
    return ufp.ProcessedDF(
        uids=proc.uids,
        last_times=proc.last_times,
        data=new_data,
        indptr=new_indptr,
        sort_idxs=None,
    )


def _time_col_tz(df: DataFrame, time_col: str) -> Optional[str]:
    """Time zone of a polars datetime column, or None.

    polars' `to_numpy()` UTC-normalizes tz-aware columns into plain datetime64,
    dropping the zone; tz-aware pandas needs no such handling because it
    converts to an object array of tz-aware Timestamps.
    """
    if isinstance(df, pl_DataFrame):
        return getattr(df[time_col].dtype, "time_zone", None)
    return None


@functools.lru_cache(maxsize=None)
def _is_constant_offset_timezone(tz: Any) -> bool:
    """Whether a timezone keeps one constant UTC offset, so an ISO 8601 offset is
    a lossless encoding of it.

    Probing actual offsets is necessary: pandas represents *every* IANA zone as a
    pytz `DstTzInfo`, whose `utcoffset(None)` is None whether or not the zone
    observes DST. Asking that would reject permanently-fixed zones such as
    Asia/Tokyo (+09:00) and Asia/Kolkata (+05:30).

    The probe window is fixed rather than relative to today so the result is
    deterministic and safe to cache.
    """
    if tz is None:
        return True
    try:
        probe = pd.date_range(
            "2010-01-01", "2030-01-01", freq="MS", tz="UTC"
        ).tz_convert(tz)
    except Exception:
        return False
    return len({t.utcoffset() for t in probe}) == 1


def _warn_non_constant_offset(tz: Any) -> None:
    logger.warning(
        "Omitting start_datetime because timezone %r changes its UTC offset "
        "(daylight saving), which a single ISO 8601 offset cannot describe. "
        "Convert the time column to UTC or a constant-offset timezone to "
        "register start_datetime.",
        str(tz),
    )


def _times_to_iso(times: np.ndarray, tz: Optional[str] = None) -> Optional[list[str]]:
    """Convert an array of per-series start times to ISO 8601 strings.

    `tz` restores the time zone that polars' `to_numpy()` drops (see
    `_time_col_tz`). Returns None when the values aren't datetimes (e.g. an
    integer time column), or when their timezone changes offset (DST), because
    reconstructing a daily-or-coarser index from a single offset would drift
    across the transition. In either case, `start_datetime` is omitted from the
    payload.
    """
    if times.size == 0:
        return None
    if times.dtype == object:
        # tz-aware pandas yields an object array of Timestamps.
        # isoformat keeps the UTC offset, which datetime64 cannot represent.
        if not isinstance(times[0], (pd.Timestamp, datetime.datetime)):
            return None
        tzinfo = times[0].tzinfo
        if not _is_constant_offset_timezone(tzinfo):
            _warn_non_constant_offset(tzinfo)
            return None
        return [t.isoformat() for t in times]
    if np.issubdtype(times.dtype, np.datetime64):
        if tz is not None:
            if not _is_constant_offset_timezone(tz):
                _warn_non_constant_offset(tz)
                return None
            # the values are UTC instants; re-attach the original zone's offset
            return [pd.Timestamp(t, tz="UTC").tz_convert(tz).isoformat() for t in times]
        return np.datetime_as_string(times, unit="auto").tolist()
    return None


def _series_starts(
    df: DataFrame,
    processed: ufp.ProcessedDF,
    time_col: str,
    orig_indptr: Optional[np.ndarray] = None,
    sort_idxs: Optional[np.ndarray] = None,
) -> Optional[list[str]]:
    """First timestamp of each series as it appears in the payload's `y`.

    When `processed` has been tail-truncated by `_tail`, its `indptr` no longer
    indexes `df` and its `sort_idxs` has been reset to None. Callers in that
    situation must pass the pre-truncation `orig_indptr` and `sort_idxs`; both are
    taken together, so `sort_idxs` is only read from `processed` when no
    `orig_indptr` was given.
    """
    if orig_indptr is None:
        sort_idxs = processed.sort_idxs
        pos = processed.indptr[:-1]
    else:
        # _tail keeps each series' last `size` rows, so the start is `end - size`
        pos = orig_indptr[1:] - np.diff(processed.indptr)
    if sort_idxs is not None:
        # map payload positions back to df rows instead of reordering the full
        # column: only one value per series is ever read
        pos = sort_idxs[pos]
    col = df[time_col]
    if isinstance(df, pd.DataFrame):
        # select before converting so tz-aware columns only box n_series values
        starts = col.iloc[pos].to_numpy()
    else:
        starts = col.gather(pos).to_numpy()
    return _times_to_iso(starts, _time_col_tz(df, time_col))


def _partition_series(
    payload: dict[str, Any], n_part: int, h: int
) -> list[dict[str, Any]]:
    parts = []
    series = payload.pop("series")
    n_series = len(series["sizes"])
    n_part = min(n_part, n_series)
    series_per_part = math.ceil(n_series / n_part)
    prev_size = 0
    for i in range(0, n_series, series_per_part):
        sizes = series["sizes"][i : i + series_per_part]
        curr_size = sum(sizes)
        part_idxs = slice(prev_size, prev_size + curr_size)
        prev_size += curr_size
        part_series = {
            "y": series["y"][part_idxs],
            "sizes": sizes,
        }
        if series.get("start_datetime") is not None:
            # one entry per series, so it slices like `sizes` (not like `y`)
            part_series["start_datetime"] = series["start_datetime"][
                i : i + series_per_part
            ]
        if series["X"] is None:
            part_series["X"] = None
            if h > 0:
                part_series["X_future"] = None
        else:
            part_series["X"] = [x[part_idxs] for x in series["X"]]
            if h > 0:
                if series["X_future"] is None:
                    part_series["X_future"] = None
                else:
                    part_series["X_future"] = [
                        x[i * h : (i + series_per_part) * h] for x in series["X_future"]
                    ]
        if "categorical_exog" in series:
            part_series["categorical_exog"] = series["categorical_exog"]
        parts.append({"series": part_series, **payload})
    return parts


def _maybe_add_date_features(
    df: DFType,
    X_df: Optional[DFType],
    features: Union[bool, Sequence[Union[str, Callable]]],
    one_hot: Union[bool, list[str]],
    freq: _Freq,
    h: int,
    id_col: str,
    time_col: str,
    target_col: str,
) -> tuple[DFType, Optional[DFType]]:
    if not features or not isinstance(freq, str):
        return df, X_df
    if isinstance(features, list):
        date_features: Sequence[Union[str, Callable]] = features
    else:
        date_features = _date_features_by_freq.get(freq, [])
        if not date_features:
            logger.warning(
                f"Non default date features for {freq} "
                "please provide a list of date features"
            )
    # add features
    if X_df is None:
        df, X_df = time_features(
            df=df,
            freq=freq,
            features=date_features,
            h=h,
            id_col=id_col,
            time_col=time_col,
        )
    else:
        df = _add_time_features(df, features=date_features, time_col=time_col)
        X_df = _add_time_features(X_df, features=date_features, time_col=time_col)
    # one hot
    if isinstance(one_hot, list):
        features_one_hot = one_hot
    elif one_hot:
        features_one_hot = [f for f in date_features if not callable(f)]
    else:
        features_one_hot = []
    if features_one_hot:
        X_df = ufp.assign_columns(X_df, target_col, 0)
        full_df = ufp.vertical_concat([df, X_df])
        if isinstance(full_df, pd.DataFrame):
            full_df = pd.get_dummies(full_df, columns=features_one_hot, dtype="float32")
        else:
            full_df = full_df.to_dummies(columns=features_one_hot)
        df = ufp.take_rows(full_df, slice(0, df.shape[0]))
        X_df = ufp.take_rows(full_df, slice(df.shape[0], full_df.shape[0]))
        X_df = ufp.drop_columns(X_df, target_col)
        X_df = ufp.drop_index_if_pandas(X_df)
    if h == 0:
        # time_features returns an empty df, we use it as None here
        X_df = None
    return df, X_df


def _validate_exog(
    df: DFType,
    X_df: Optional[DFType],
    id_col: str,
    time_col: str,
    target_col: str,
    hist_exog: Optional[list[str]],
) -> tuple[DFType, Optional[DFType]]:
    base_cols = {id_col, time_col, target_col}
    exogs = [c for c in df.columns if c not in base_cols]
    if hist_exog is None:
        hist_exog = []
    if X_df is None:
        # all exogs must be historic
        ignored_exogs = [c for c in exogs if c not in hist_exog]
        if ignored_exogs:
            logger.warning(
                f"`df` contains the following exogenous features: {ignored_exogs}, "
                "but `X_df` was not provided and they were not declared in `hist_exog_list`. "
                "They will be ignored."
            )
        exogs = [c for c in exogs if c in hist_exog]
        df = df[[id_col, time_col, target_col, *exogs]]
        return df, None

    # exogs in df that weren't declared as historic nor future
    futr_exog = [c for c in X_df.columns if c not in base_cols]
    declared_exogs = {*hist_exog, *futr_exog}
    ignored_exogs = [c for c in exogs if c not in declared_exogs]
    if ignored_exogs:
        logger.warning(
            f"`df` contains the following exogenous features: {ignored_exogs}, "
            "but they were not found in `X_df` nor declared in `hist_exog_list`. "
            "They will be ignored."
        )

    # future exogenous are provided in X_df that are not in df
    missing_futr = set(futr_exog) - set(exogs)
    if missing_futr:
        raise ValueError(
            "The following exogenous features are present in `X_df` "
            f"but not in `df`: {missing_futr}."
        )

    # features are provided through X_df but declared as historic
    futr_and_hist = set(futr_exog) & set(hist_exog)
    if futr_and_hist:
        logger.warning(
            "The following features were declared as historic but found in `X_df`: "
            f"{futr_and_hist}, they will be considered as historic."
        )
        futr_exog = [f for f in futr_exog if f not in hist_exog]

    # Make sure df and X_df are in right order
    df = df[[id_col, time_col, target_col, *futr_exog, *hist_exog]]
    X_df = X_df[[id_col, time_col, *futr_exog]]

    return df, X_df


def _extract_categorical_exog(
    df: DFType,
    categorical_exog_list: Optional[list[str]],
    id_col: str,
    time_col: str,
    target_col: str,
    X_df: Optional[DFType] = None,
) -> tuple[
    DFType,
    Optional[DFType],
    dict[str, np.ndarray],
    list[str],
    list[str],
    Optional[DFType],
]:
    """Validate, extract, and strip categorical exogenous columns from df/X_df.

    Returns:
        df: df with all categorical columns removed.
        X_df: X_df with future categorical columns removed (unchanged if None).
        df_cat_vals: mapping col → raw values array for every col in categorical_exog_list.
        futr_cat_cols: cat cols found in X_df (treated as future categoricals).
        hist_cat_cols: cat cols not in X_df.
        X_df_cat_future: future categorical values with their ID and time keys.
    """
    if not categorical_exog_list:
        return df, X_df, {}, [], [], None

    x_df_exog_cols = (
        {c for c in X_df.columns if c not in {id_col, time_col}}
        if X_df is not None
        else set()
    )
    df_exog_cols = {c for c in df.columns if c not in {id_col, time_col, target_col}}
    invalid_cats = set(categorical_exog_list) - df_exog_cols - x_df_exog_cols
    if invalid_cats:
        location = "`df` or `X_df`" if X_df is not None else "`df`"
        raise ValueError(
            "The following columns in `categorical_exog_list` were not "
            f"found in {location}: {invalid_cats}."
        )

    futr_cat_cols = [c for c in categorical_exog_list if c in x_df_exog_cols]
    hist_cat_cols = [c for c in categorical_exog_list if c not in futr_cat_cols]

    # futr_cat_cols must also exist in df to provide historical context rows for X.
    futr_cat_missing_from_df = set(futr_cat_cols) - df_exog_cols
    if futr_cat_missing_from_df:
        raise ValueError(
            "The following columns in `categorical_exog_list` were found in `X_df` but are "
            f"missing from `df`: {futr_cat_missing_from_df}. Future categorical features must "
            "also be present in `df` to provide historical context."
        )

    # Extract historical values for all cat cols from df.
    # futr_cat_cols appear in both df (history) and X_df (future); hist_cat_cols only in df.
    df_cat_vals: dict[str, np.ndarray] = {
        c: df[c].to_numpy() for c in categorical_exog_list
    }

    X_df_cat_future: Optional[DFType] = None
    if futr_cat_cols and X_df is not None:
        X_df_cat_future = X_df[[id_col, time_col, *futr_cat_cols]]
        X_df = X_df[[c for c in X_df.columns if c not in futr_cat_cols]]

    df = df[[c for c in df.columns if c not in set(categorical_exog_list)]]
    return df, X_df, df_cat_vals, futr_cat_cols, hist_cat_cols, X_df_cat_future


def _validate_input_size(
    processed: ufp.ProcessedDF,
    model_input_size: int,
    model_horizon: int,
) -> None:
    min_size = np.diff(processed.indptr).min().item()
    if min_size < model_input_size + model_horizon:
        raise ValueError(
            "Some series are too short. "
            "Please make sure that each series contains "
            f"at least {model_input_size + model_horizon} observations."
        )


def _ensure_local_dataframe(
    df: Any, *, method_name: str, sync_method_name: str
) -> None:
    if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
        raise ValueError(
            f"{method_name} only supports pandas or polars dataframes; "
            f"use {sync_method_name} for distributed (dask/spark/ray) dataframes."
        )


def _prepare_level_and_quantiles(
    level: Optional[list[Union[int, float]]],
    quantiles: Optional[list[float]],
) -> tuple[Optional[list[Union[int, float]]], Optional[list[float]]]:
    if level is not None and quantiles is not None:
        raise ValueError("You should provide `level` or `quantiles`, but not both.")
    if quantiles is None:
        return level, quantiles
    # we recover level from quantiles
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("`quantiles` should be floats between 0 and 1.")
    level = [abs(int(100 - 200 * q)) for q in quantiles]
    return level, quantiles


def _maybe_convert_level_to_quantiles(
    df: DFType,
    quantiles: Optional[list[float]],
) -> DFType:
    if quantiles is None:
        return df
    out_cols = [c for c in df.columns if "-lo-" not in c and "-hi-" not in c]
    df = ufp.copy_if_pandas(df, deep=False)
    for q in sorted(quantiles):
        if q == 0.5:
            col = "TimeGPT"
        else:
            lv = int(100 - 200 * q)
            hi_or_lo = "lo" if lv > 0 else "hi"
            lv = abs(lv)
            col = f"TimeGPT-{hi_or_lo}-{lv}"
        q_col = f"TimeGPT-q-{int(q * 100)}"
        df = ufp.assign_columns(df, q_col, df[col])
        out_cols.append(q_col)
    return df[out_cols]


def _align_future_exog_order(
    X_future: np.ndarray,
    X_indptr: np.ndarray,
    X_uids: Any,
    uids: Any,
    id_col: str,
) -> np.ndarray:
    if hasattr(X_uids, "to_numpy"):
        x_uids_list = X_uids.to_numpy().tolist()
    else:
        x_uids_list = np.asarray(X_uids).tolist()
    uids_list = uids.to_numpy().tolist()
    if x_uids_list == uids_list:
        return X_future
    if set(x_uids_list) != set(uids_list):
        missing = sorted(set(uids_list) - set(x_uids_list), key=str)
        unexpected = sorted(set(x_uids_list) - set(uids_list), key=str)
        raise ValueError(
            f"`X_df` must contain the same values of `{id_col}` as `df`. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )
    positions = {uid: pos for pos, uid in enumerate(x_uids_list)}
    row_idxs = np.concatenate(
        [
            np.arange(X_indptr[pos], X_indptr[pos + 1], dtype=np.int64)
            for pos in (positions[uid] for uid in uids_list)
        ]
    )
    return X_future[:, row_idxs]


def _align_future_categorical_exog(
    X_df_cat_future: Optional[DFType],
    uids: Any,
    id_col: str,
    time_col: str,
) -> list[list]:
    if X_df_cat_future is None:
        return []
    X_df_cat_future = ensure_time_dtype(X_df_cat_future, time_col=time_col)
    X_df_cat_future = ensure_sorted(
        X_df_cat_future,
        id_col=id_col,
        time_col=time_col,
    )
    ids = X_df_cat_future[id_col].to_numpy()
    starts = np.append(0, np.flatnonzero(ids[1:] != ids[:-1]) + 1)
    indptr = np.append(starts, len(ids))
    x_uids = ids[starts]
    cat_cols = [c for c in X_df_cat_future.columns if c not in (id_col, time_col)]
    values = np.empty((len(cat_cols), len(X_df_cat_future)), dtype=object)
    for position, col in enumerate(cat_cols):
        values[position] = X_df_cat_future[col].to_numpy()
    aligned = _align_future_exog_order(
        X_future=values,
        X_indptr=indptr,
        X_uids=x_uids,
        uids=uids,
        id_col=id_col,
    )
    return [row.tolist() for row in aligned]


def _preprocess(
    df: DFType,
    X_df: Optional[DFType],
    h: int,
    freq: str,
    date_features: Union[bool, Sequence[Union[str, Callable]]],
    date_features_to_one_hot: Union[bool, list[str]],
    id_col: str,
    time_col: str,
    target_col: str,
) -> tuple[ufp.ProcessedDF, Optional[DFType], list[str], Optional[list[str]]]:
    df, X_df = _maybe_add_date_features(
        df=df,
        X_df=X_df,
        features=date_features,
        one_hot=date_features_to_one_hot,
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    processed = ufp.process_df(
        df=df, id_col=id_col, time_col=time_col, target_col=target_col
    )
    if X_df is not None and X_df.shape[1] > 2:
        X_df = ensure_time_dtype(X_df, time_col=time_col)
        processed_X = ufp.process_df(
            df=X_df,
            id_col=id_col,
            time_col=time_col,
            target_col=None,
        )
        X_future = _align_future_exog_order(
            X_future=processed_X.data.T,
            X_indptr=processed_X.indptr,
            X_uids=processed_X.uids,
            uids=processed.uids,
            id_col=id_col,
        )
        futr_cols = [c for c in X_df.columns if c not in (id_col, time_col)]
    else:
        X_future = None
        futr_cols = None
    x_cols = [c for c in df.columns if c not in (id_col, time_col, target_col)]
    return processed, X_future, x_cols, futr_cols


def _validate_future_exog_keys(
    X_df: Optional[DFType],
    expected: DataFrame,
    h: int,
    id_col: str,
    time_col: str,
) -> None:
    if X_df is None:
        return
    X_df = ensure_time_dtype(X_df, time_col=time_col)
    if not _dataframe_keys_match(
        actual=X_df,
        expected=expected,
        id_col=id_col,
        time_col=time_col,
    ):
        raise ValueError(
            "`X_df` must contain exactly one row for every future "
            f"({id_col}, {time_col}) pair in the {h}-step forecast horizon."
        )


def _sort_categorical_values(
    df_cat_vals: dict[str, np.ndarray],
    sort_idxs: Optional[np.ndarray],
) -> dict[str, np.ndarray]:
    if sort_idxs is None:
        return dict(df_cat_vals)
    return {col: vals[sort_idxs] for col, vals in df_cat_vals.items()}


def _log_exog_features(
    futr_cols: Optional[list[str]],
    futr_cat_cols: list[str],
    hist_exog_list: Optional[list[str]],
    hist_cat_cols: list[str],
) -> None:
    if futr_cols is not None:
        logger.info(f"Using future exogenous features: {futr_cols}")
    if futr_cat_cols:
        logger.info(f"Using future categorical exogenous features: {futr_cat_cols}")
    if hist_exog_list:
        logger.info(f"Using historical exogenous features: {hist_exog_list}")
    if hist_cat_cols:
        logger.info(f"Using historical categorical exogenous features: {hist_cat_cols}")


def _build_exog_payload(
    processed: ufp.ProcessedDF,
    sorted_df_cat: dict[str, np.ndarray],
    x_cols: list[str],
    futr_cols: Optional[list[str]],
    futr_cat_cols: list[str],
    hist_cat_cols: list[str],
    X_future: Optional[list],
    X_df_cat_future: list[list],
    has_categorical: bool,
) -> tuple[Optional[list], Optional[list], Optional[list[int]], list[str]]:
    n_futr_num = len(futr_cols) if futr_cols is not None else 0
    n_futr_cat = len(futr_cat_cols)
    n_hist_num = len(x_cols) - n_futr_num
    n_hist_cat = len(hist_cat_cols)

    cat_hist_rows: list[list] = [
        sorted_df_cat[c].tolist() for c in futr_cat_cols + hist_cat_cols
    ]
    if processed.data.shape[1] > 1 or cat_hist_rows:
        num_rows = list(processed.data[:, 1:].T) if processed.data.shape[1] > 1 else []
        X: Optional[list] = (
            num_rows[:n_futr_num]
            + cat_hist_rows[:n_futr_cat]
            + num_rows[n_futr_num:]
            + cat_hist_rows[n_futr_cat:]
        )
    else:
        X = None

    if X_future is not None or X_df_cat_future:
        X_future = (list(X_future) if X_future is not None else []) + X_df_cat_future

    categorical_exog_payload: Optional[list[int]] = None
    if has_categorical:
        futr_cat_indices = list(range(n_futr_num, n_futr_num + n_futr_cat))
        hist_cat_start = n_futr_num + n_futr_cat + n_hist_num
        hist_cat_indices = list(range(hist_cat_start, hist_cat_start + n_hist_cat))
        categorical_exog_payload = futr_cat_indices + hist_cat_indices

    feature_names = (
        x_cols[:n_futr_num] + futr_cat_cols + x_cols[n_futr_num:] + hist_cat_cols
    )
    return X, X_future, categorical_exog_payload, feature_names


def _forecast_payload_to_in_sample(payload: dict, h: int, n_windows: int) -> dict:
    # No finetuning for in-sample
    payload["finetune_steps"] = 0

    # historic exogenous features
    hist_exog = None
    if payload["series"]["X"] is not None:
        n_features = len(payload["series"]["X"])
        hist_exog = list(range(n_features))
        if payload["series"]["X_future"] is not None:
            n_futr_exog = len(payload["series"]["X_future"])
            hist_exog = hist_exog[n_futr_exog:]
    payload["hist_exog"] = hist_exog
    del payload["series"]["X_future"]

    # in-sample horizon and number of windows
    payload["h"] = h
    payload["step_size"] = h
    payload["n_windows"] = n_windows

    # The in-sample forecast (add_history workflow) always runs cross_validation
    # in full_history mode: the server derives the horizon and number of windows,
    # so the values above are sent as placeholders and ignored.
    payload["full_history"] = True

    return payload


def _get_in_sample_horizon_and_windows(
    sizes: np.ndarray,
    model_horizon: int,
    model_input_size: int,
    clean_ex_first: bool,
    level: Optional[list[Union[int, float]]],
) -> tuple[int, int]:
    # in-sample horizon and number of windows
    min_size = min(sizes)
    h = min(model_horizon, min_size - 1)
    if clean_ex_first:
        n_windows = max((min_size - model_input_size) // model_horizon, 1)
    else:
        n_windows = max(
            (min_size - (model_input_size + model_horizon + 2 * h)) // model_horizon, 1
        )
    # In case of multiple windows, we reduce one to avoid errors when running with level argument
    if level is not None and n_windows > 1:
        n_windows -= 1
    return h, n_windows


def _maybe_add_intervals(
    df: DFType,
    intervals: Optional[dict[str, list[float]]],
) -> DFType:
    if intervals is None:
        return df
    first_key = next(iter(intervals), None)
    if first_key is None or intervals[first_key] is None:
        return df
    intervals_df = type(df)(
        {f"TimeGPT-{k}": intervals[k] for k in sorted(intervals.keys())}
    )
    return ufp.horizontal_concat([df, intervals_df])


def _maybe_drop_id(df: DFType, id_col: str, drop: bool) -> DFType:
    if drop:
        df = ufp.drop_columns(df, id_col)
    return df


def _parse_in_sample_output(
    in_sample_output: dict[str, Union[list[float], dict[str, list[float]]]],
    df: DataFrame,
    processed: ufp.ProcessedDF,
    id_col: str,
    time_col: str,
    target_col: str,
) -> DataFrame:
    times = df[time_col].to_numpy()
    targets = df[target_col].to_numpy()
    if processed.sort_idxs is not None:
        times = times[processed.sort_idxs]
        targets = targets[processed.sort_idxs]
    times = _array_tails(times, processed.indptr, in_sample_output["sizes"])
    targets = _array_tails(targets, processed.indptr, in_sample_output["sizes"])
    uids = ufp.repeat(processed.uids, in_sample_output["sizes"])
    out = type(df)(
        {
            id_col: uids,
            time_col: times,
            target_col: targets,
            "TimeGPT": in_sample_output["mean"],
        }
    )
    return _maybe_add_intervals(out, in_sample_output["intervals"])  # type: ignore


def _restrict_input_samples(level, input_size, model_horizon, h) -> int:
    if level is not None:
        # add sufficient info to compute
        # conformal interval
        # @AzulGarza
        #  this is an old opinionated decision
        #  about reducing the data sent to the api
        #  to reduce latency when
        #  a user passes level. since currently the model
        #  uses conformal prediction, we can change a minimum
        #  amount of data if the series are too large
        new_input_size = 3 * input_size + max(model_horizon, h)
    else:
        # we only want to forecast
        new_input_size = input_size
    return new_input_size


def _extract_target_array(df: DataFrame, target_col: str) -> np.ndarray:
    # in pandas<2.2 to_numpy can lead to an object array if
    # the type is a pandas nullable type, e.g. pd.Float64Dtype
    # we thus use the dtype's type as the target dtype
    if isinstance(df, pd.DataFrame):
        target_dtype = df.dtypes[target_col].type
        targets = df[target_col].to_numpy(dtype=target_dtype)
    else:
        targets = df[target_col].to_numpy()
    return targets


def _process_exog_features(
    processed_data: np.ndarray,
    x_cols: list[str],
    hist_exog_list: Optional[list[str]] = None,
) -> tuple[Optional[np.ndarray], Optional[list[int]]]:
    X = None
    hist_exog = None
    if processed_data.shape[1] > 1:
        X = processed_data[:, 1:].T
        if hist_exog_list is None:
            futr_exog = x_cols
        else:
            missing_hist: set[str] = set(hist_exog_list) - set(x_cols)
            if missing_hist:
                raise ValueError(
                    "The following exogenous features were declared as historic "
                    f"but were not found in `df`: {missing_hist}."
                )
            futr_exog = [c for c in x_cols if c not in hist_exog_list]
            # match the forecast method order [future, historic]
            fcst_features_order = futr_exog + hist_exog_list
            x_idxs = [x_cols.index(c) for c in fcst_features_order]
            X = X[x_idxs]
            hist_exog = [fcst_features_order.index(c) for c in hist_exog_list]
        if futr_exog and logger:
            logger.info(f"Using future exogenous features: {futr_exog}")
        if hist_exog_list and logger:
            logger.info(f"Using historical exogenous features: {hist_exog_list}")

    return X, hist_exog
