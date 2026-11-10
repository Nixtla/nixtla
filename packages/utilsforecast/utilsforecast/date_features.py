"""Backend-agnostic calendar features"""

__all__ = [
    "available",
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "day_of_week",
    "day_of_year",
    "days_in_month",
    "hour",
    "minute",
    "second",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
]


import pickle
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from packaging.version import Version

from .compat import pl, pl_Series
from .processing import _arrow_times_to_numpy
from .validation import _is_dt_dtype

_PANDAS_GE_2 = Version(pd.__version__) >= Version("2")
_PER_SECOND = {
    "ms": 10**3,
    "us": 10**6,
    "ns": 10**9,
    "ps": 10**12,
    "fs": 10**15,
    "as": 10**18,
}


class _Dates:
    """Calendar components of dates, computed once and shared by the features."""

    def __init__(self, datetimes: np.ndarray):
        self.datetimes = datetimes

    @cached_property
    def days(self) -> np.ndarray:
        return self.datetimes.astype("datetime64[D]")

    @cached_property
    def months(self) -> np.ndarray:
        return self.datetimes.astype("datetime64[M]")

    @cached_property
    def years(self) -> np.ndarray:
        return self.datetimes.astype("datetime64[Y]")

    @cached_property
    def month(self) -> np.ndarray:
        return self.months.astype(np.int64) % 12 + 1

    @cached_property
    def day(self) -> np.ndarray:
        return (self.days - self.months).astype(np.int64) + 1

    @cached_property
    def day_of_week(self) -> np.ndarray:
        # 1970-01-01 was a thursday
        return (self.days.astype(np.int64) + 3) % 7

    @cached_property
    def days_in_month(self) -> np.ndarray:
        next_month = (self.months + np.timedelta64(1, "M")).astype("datetime64[D]")
        return (next_month - self.months).astype(np.int64)

    @cached_property
    def seconds_of_day(self) -> np.ndarray:
        seconds = self.datetimes.astype("datetime64[s]", copy=False)
        return (seconds - self.days).astype(np.int64)

    @cached_property
    def is_month_end(self) -> np.ndarray:
        return self.day == self.days_in_month


@dataclass(frozen=True, eq=False)
class CalendarFeature:
    """Calendar feature with the same values and dtype for pandas and polars dates.

    For internal use only, custom features can be passed to `time_features` as functions.

    Attributes:
        name (str): Name of the feature, used as the column name.
        description (str): Description of the values.
        dtype (type): numpy dtype of the computed values.
        values (range, optional): Values to one-hot encode the feature with, None if it isn't one-hot encoded.
    """

    name: str
    description: str = field(repr=False)
    dtype: type = field(repr=False)
    values: Optional[range] = field(repr=False)
    _compute: Callable[[_Dates], np.ndarray] = field(repr=False)

    def compute(self, dates) -> np.ndarray:
        """Compute the feature values.

        Args:
            dates (pandas or polars Series, pandas DatetimeIndex or numpy datetime64 array): Dates to compute the feature from.
                Time zone aware dates use their local time.

        Returns:
            numpy.ndarray: Feature values, one per date.

        Raises:
            ValueError: If the dates aren't datetimes or there are null dates.
        """
        return _compute_features(dates, [self])[0]

    def __reduce__(self):
        if _FEATURES.get(self.name) is not self:
            raise pickle.PicklingError(
                f"Can't pickle {self.name!r}, only the features listed by `available()` can be pickled."
            )
        return _from_name, (self.name,)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _native_to_datetime64(dates) -> np.ndarray:
    if not isinstance(dates, (pd.Series, pl_Series)):
        dates = pd.Series(dates)
    if not _PANDAS_GE_2 and hasattr(dates.dtype, "pyarrow_dtype"):
        import pyarrow as pa

        arrow_type = dates.dtype.pyarrow_dtype
        if pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type):
            raise ValueError(
                "Can't compute calendar features, pyarrow backed dates require pandas>=2."
            )
    if not _is_dt_dtype(dates):
        raise ValueError(
            f"Can't compute calendar features, dates must be datetimes, got {dates.dtype}."
        )
    if isinstance(dates, pd.Series):
        if hasattr(dates.dtype, "pyarrow_dtype"):
            dates = _arrow_times_to_numpy(dates)
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
    elif dates.dtype == pl.Datetime:
        dates = dates.dt.replace_time_zone(None)
    return dates.to_numpy()


def _to_datetime64(dates) -> np.ndarray:
    if isinstance(dates, np.ndarray) and dates.dtype.kind == "M":
        # pandas<2 casts the dates to nanoseconds, which can't hold every date
        datetimes = dates
    else:
        datetimes = _native_to_datetime64(dates)
    if np.isnat(datetimes).any():
        raise ValueError("Can't compute calendar features, found null dates.")
    unit, count = np.datetime_data(datetimes.dtype)
    if count != 1:
        datetimes = datetimes.astype(f"datetime64[{unit}]")
    if unit not in _PER_SECOND:
        return datetimes
    # numpy's astype overflows when flooring dates close to the minimum int64
    per_second = _PER_SECOND[unit]
    return (datetimes.view(np.int64) // per_second).astype("datetime64[s]")


def _compute_features(dates, features: List[CalendarFeature]) -> List[np.ndarray]:
    if not features:
        return []
    components = _Dates(_to_datetime64(dates))
    return [f._compute(components).astype(f.dtype) for f in features]


def _week(d: _Dates) -> np.ndarray:
    # the ISO week is the week of the year of the thursday of that week
    thursday = d.days + (3 - d.day_of_week).astype("timedelta64[D]")
    return (thursday - thursday.astype("datetime64[Y]")).astype(np.int64) // 7 + 1


year = CalendarFeature(
    "year", "Year.", np.int16, None, lambda d: d.years.astype(np.int64) + 1970
)
quarter = CalendarFeature(
    "quarter",
    "Quarter of the year, from 1 to 4.",
    np.int8,
    range(1, 5),
    lambda d: (d.month - 1) // 3 + 1,
)
month = CalendarFeature(
    "month",
    "Month of the year, from 1 to 12.",
    np.int8,
    range(1, 13),
    lambda d: d.month,
)
week = CalendarFeature(
    "week", "ISO week of the year, from 1 to 53.", np.int8, range(1, 54), _week
)
day = CalendarFeature(
    "day", "Day of the month, from 1 to 31.", np.int8, range(1, 32), lambda d: d.day
)
day_of_week = CalendarFeature(
    "day_of_week",
    "Day of the week, from 0 (Monday) to 6 (Sunday).",
    np.int8,
    range(7),
    lambda d: d.day_of_week,
)
day_of_year = CalendarFeature(
    "day_of_year",
    "Day of the year, from 1 to 366.",
    np.int16,
    range(1, 367),
    lambda d: (d.days - d.years).astype(np.int64) + 1,
)
days_in_month = CalendarFeature(
    "days_in_month",
    "Number of days in the month, from 28 to 31.",
    np.int8,
    range(28, 32),
    lambda d: d.days_in_month,
)
hour = CalendarFeature(
    "hour",
    "Hour of the day, from 0 to 23.",
    np.int8,
    range(24),
    lambda d: d.seconds_of_day // 3600,
)
minute = CalendarFeature(
    "minute",
    "Minute of the hour, from 0 to 59.",
    np.int8,
    range(60),
    lambda d: d.seconds_of_day // 60 % 60,
)
second = CalendarFeature(
    "second",
    "Second of the minute, from 0 to 59.",
    np.int8,
    range(60),
    lambda d: d.seconds_of_day % 60,
)
is_month_start = CalendarFeature(
    "is_month_start",
    "1 on the first day of the month, 0 otherwise.",
    np.int8,
    None,
    lambda d: d.day == 1,
)
is_month_end = CalendarFeature(
    "is_month_end",
    "1 on the last day of the month, 0 otherwise.",
    np.int8,
    None,
    lambda d: d.is_month_end,
)
is_quarter_start = CalendarFeature(
    "is_quarter_start",
    "1 on the first day of the quarter, 0 otherwise.",
    np.int8,
    None,
    lambda d: (d.day == 1) & (d.month % 3 == 1),
)
is_quarter_end = CalendarFeature(
    "is_quarter_end",
    "1 on the last day of the quarter, 0 otherwise.",
    np.int8,
    None,
    lambda d: d.is_month_end & (d.month % 3 == 0),
)
is_year_start = CalendarFeature(
    "is_year_start",
    "1 on January 1st, 0 otherwise.",
    np.int8,
    None,
    lambda d: (d.day == 1) & (d.month == 1),
)
is_year_end = CalendarFeature(
    "is_year_end",
    "1 on December 31st, 0 otherwise.",
    np.int8,
    None,
    lambda d: (d.day == 31) & (d.month == 12),
)

_FEATURES: Dict[str, CalendarFeature] = {
    name: globals()[name] for name in __all__ if name != "available"
}


def _from_name(name: str) -> CalendarFeature:
    return _FEATURES[name]


def available() -> List[CalendarFeature]:
    """List the available calendar features.

    Returns:
        list: Available calendar features.
    """
    return list(_FEATURES.values())
