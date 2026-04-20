__all__ = ['generate_daily_series', 'generate_prices_for_series', 'PredictionIntervals']


from math import ceil, log10
from typing import Dict, List
import warnings

import narwhals as nw
import numpy as np
import pandas as pd
from joblib import cpu_count
from utilsforecast.compat import DataFrame, pl
from utilsforecast.data import generate_series


# Valid values for each date feature that can be dummy-encoded.
# Features not listed here (e.g. year) are not supported because they have an
# unbounded range that depends on the training data.
_DUMMY_FEATURE_VALUES: Dict[str, List[int]] = {
    "dayofweek":   list(range(7)),         # 0=Mon … 6=Sun (pandas convention)
    "day_of_week": list(range(7)),
    "weekday":     list(range(7)),
    "month":       list(range(1, 13)),
    "quarter":     list(range(1, 5)),
    "day":         list(range(1, 32)),
    "hour":        list(range(24)),
    "minute":      list(range(60)),
    "second":      list(range(60)),
    "dayofyear":   list(range(1, 367)),    # 366 columns (leap-year safe)
    "day_of_year": list(range(1, 367)),
    # no narwhals equivalent — computed via backend-specific fallback below
    "week":        list(range(1, 54)),     # ISO weeks 1-53
    "weekofyear":  list(range(1, 54)),
}

# narwhals dt method name when it differs from the mlforecast feature name
_NW_DT_ATTR: Dict[str, str] = {
    "dayofweek":   "weekday",
    "day_of_week": "weekday",
    "weekday":     "weekday",
    "quarter":     "month",      # special: quarter is derived from month
    "dayofyear":   "ordinal_day",
    "day_of_year": "ordinal_day",
    # "week" / "weekofyear" intentionally absent — handled via backend fallback
}

# additive offset applied after the narwhals call so values match the pandas convention
_NW_OFFSET: Dict[str, int] = {
    "dayofweek":   -1,   # narwhals weekday is 1-7; pandas dayofweek is 0-6
    "day_of_week": -1,
    "weekday":     -1,
}

# Features that narwhals does not expose and require a backend-specific path
_NW_MISSING = frozenset({"week", "weekofyear"})


def _extract_week(dates) -> np.ndarray:
    """Extract ISO week number via backend-specific calls.

    narwhals has no ``dt.week()`` equivalent, so we fall back to the native
    pandas / polars APIs directly.

    Parameters
    ----------
    dates : pd.Series or pl.Series
        Native (non-narwhals) datetime series.
    """
    if isinstance(dates, pd.Series):
        return dates.dt.isocalendar().week.to_numpy(dtype=np.int32)
    # polars — dt.week() returns ISO week 1-53
    return dates.dt.week().to_numpy()


def _compute_date_dummies(dates, feature: str) -> Dict[str, np.ndarray]:
    """Compute one-hot indicator columns for a categorical date feature.

    Most features are computed via narwhals for backend-agnostic datetime
    access.  ``week`` and ``weekofyear`` fall back to native pandas / polars
    calls because narwhals has no equivalent method.

    Parameters
    ----------
    dates : pd.DatetimeIndex, pd.Series, or pl.Series
        Datetime values (typically the unique timestamps in the DataFrame).
    feature : str
        Feature name — must be a key of :data:`_DUMMY_FEATURE_VALUES`.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``{f"{feature}_{value}": uint8_array}`` for each valid value.
    """
    if isinstance(dates, (pd.DatetimeIndex, pd.Index)):
        dates = pd.Series(dates.to_numpy())

    # Backend-specific path for features narwhals does not support
    if feature in _NW_MISSING:
        raw = _extract_week(dates)
        values = _DUMMY_FEATURE_VALUES[feature]
        return {f"{feature}_{v}": (raw == v).astype(np.uint8) for v in values}

    dates_nw = nw.from_native(dates, series_only=True)
    nw_attr = _NW_DT_ATTR.get(feature, feature)
    raw_nw = getattr(dates_nw.dt, nw_attr)()
    raw = raw_nw.to_numpy()

    if feature == "quarter":
        raw = ((raw - 1) // 3) + 1
    else:
        offset = _NW_OFFSET.get(feature, 0)
        if offset:
            raw = raw + offset

    values = _DUMMY_FEATURE_VALUES[feature]
    return {f"{feature}_{v}": (raw == v).astype(np.uint8) for v in values}


def _resolve_num_threads(num_threads: int) -> int:
    """Convert num_threads=-1 to actual CPU count.

    Args:
        num_threads: Number of threads. Use -1 for all available CPUs.

    Returns:
        int: Resolved number of threads (always >= 1)

    Note:
        Uses joblib.cpu_count() which respects CPU affinity and cgroup limits
        (Docker/Kubernetes resource constraints). Falls back to 1 if CPU
        count cannot be determined.
    """
    if num_threads == -1:
        try:
            resolved = cpu_count()
            if resolved is None:
                warnings.warn(
                    "Could not determine CPU count, using num_threads=1.",
                    UserWarning,
                    stacklevel=3
                )
                return 1
            return resolved
        except Exception as e:
            warnings.warn(
                f"Error determining CPU count ({e}), using num_threads=1.",
                UserWarning,
                stacklevel=3
            )
            return 1
    if num_threads < 1:
        raise ValueError(f"num_threads must be -1 or a positive integer, got {num_threads}.")
    return num_threads


def generate_daily_series(
    n_series: int,
    min_length: int = 50,
    max_length: int = 500,
    n_static_features: int = 0,
    equal_ends: bool = False,
    static_as_categorical: bool = True,
    with_trend: bool = False,
    seed: int = 0,
    engine: str = "pandas",
) -> DataFrame:
    """Generate Synthetic Panel Series.

    Args:
        n_series (int): Number of series for synthetic panel.
        min_length (int, default=50): Minimum length of synthetic panel's series.
        max_length (int, default=500): Maximum length of synthetic panel's series.
        n_static_features (int, default=0): Number of static exogenous variables for synthetic panel's series.
        equal_ends (bool, default=False): Series should end in the same date stamp `ds`.
        static_as_categorical (bool, default=True): Static features should have a categorical data type.
        with_trend (bool, default=False): Series should have a (positive) trend.
        seed (int, default=0): Random seed used for generating the data.
        engine (str, default='pandas'): Output Dataframe type.

    Returns:
        pandas or polars DataFrame: Synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous features.
    """
    series = generate_series(
        n_series=n_series,
        freq="D",
        min_length=min_length,
        max_length=max_length,
        n_static_features=n_static_features,
        equal_ends=equal_ends,
        static_as_categorical=static_as_categorical,
        with_trend=with_trend,
        seed=seed,
        engine=engine,
    )
    n_digits = ceil(log10(n_series))

    if engine == "pandas":
        series["unique_id"] = (
            "id_" + series["unique_id"].astype(str).str.rjust(n_digits, "0")
        ).astype("category")
    else:
        try:
            series = series.with_columns(
                ("id_" + pl.col("unique_id").cast(pl.Utf8).str.pad_start(n_digits, "0"))
                .alias("unique_id")
                .cast(pl.Categorical)
            )
        except AttributeError:
            series = series.with_columns(
                ("id_" + pl.col("unique_id").cast(pl.Utf8).str.rjust(n_digits, "0"))
                .alias("unique_id")
                .cast(pl.Categorical)
            )
    return series


def generate_prices_for_series(
    series: pd.DataFrame, horizon: int = 7, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    unique_last_dates = series.groupby("unique_id", observed=True)["ds"].max().nunique()
    if unique_last_dates > 1:
        raise ValueError("series must have equal ends.")
    day_offset = pd.tseries.frequencies.Day()
    starts_ends = series.groupby("unique_id", observed=True)["ds"].agg(["min", "max"])
    dfs = []
    for idx, (start, end) in starts_ends.iterrows():
        product_df = pd.DataFrame(
            {
                "unique_id": idx,
                "price": rng.rand((end - start).days + 1 + horizon),
            },
            index=pd.date_range(start, end + horizon * day_offset, name="ds"),
        )
        dfs.append(product_df)
    prices_catalog = pd.concat(dfs).reset_index()
    return prices_catalog


class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_error", "conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.h = h
        self.method = method

    def __repr__(self):
        return f"PredictionIntervals(n_windows={self.n_windows}, h={self.h}, method='{self.method}')"


class _ShortSeriesException(Exception):
    def __init__(self, idxs):
        self.idxs = idxs
