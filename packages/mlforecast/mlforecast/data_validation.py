__all__ = [
    'validate_update_start_dates',
    'validate_continuity',
    'validate_update_df',
    'validate_df',
]


import reprlib
from typing import Tuple, Union

import narwhals as nw
import pandas as pd
from utilsforecast.compat import DFType, pl
from utilsforecast.processing import offset_times


def validate_update_start_dates(
    df: DFType,
    id_col: str,
    time_col: str,
    last_dates_df: DFType,
    freq: Union[str, int],
) -> Tuple[bool, DFType]:
    """Check that each series in df starts at the expected timestamp (last_date + freq).

    Args:
        df: New observations DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column
        last_dates_df: DataFrame with id_col and '_last' columns containing the last
            known timestamp for each series
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.

    Returns:
        Tuple of (severity, bad_df):
            - has_issues: False if all series start at the expected timestamp, True otherwise
            - bad_df: DataFrame with id_col for series with invalid start dates (empty if False)
    """
    if df.shape[0] == 0:
        return (False, df[[id_col]].head(0))

    nw_df = nw.from_native(df, eager_only=True)
    nw_last = nw.from_native(last_dates_df, eager_only=True)

    df_sorted = nw_df.sort([id_col, time_col])
    stats = (
        df_sorted.group_by(id_col)
        .agg(nw.col(time_col).min().alias('_min'))
        .sort(id_col)
    )

    expected_start_native = offset_times(nw_last['_last'].to_native(), freq, 1)
    expected_start_nw = nw.from_native(expected_start_native, series_only=True).alias('_expected_start')

    expected_df = (
        nw_last
        .with_columns(expected_start_nw, nw.col(id_col).cast(nw.String))
        .select([id_col, '_expected_start'])
    )
    stats = stats.with_columns(nw.col(id_col).cast(nw.String))
    stats = stats.join(expected_df, on=id_col, how='left')

    bad = stats.filter(
        ~nw.col('_expected_start').is_null()
        & (nw.col('_min') != nw.col('_expected_start'))
    ).select([id_col])

    bad_native = nw.to_native(bad)
    if bad.shape[0] > 0:
        return (True, bad_native)
    return (False, bad_native)


def validate_continuity(
    df: DFType,
    id_col: str,
    time_col: str,
    freq: Union[str, int],
) -> Tuple[bool, DFType]:
    """Check for gaps or duplicate timestamps within time series data.

    For each series, verifies that n_unique timestamps equals the expected span
    (offset_times(min, freq, n_unique-1) == max) and that there are no duplicates
    (count == n_unique). This requires only one offset_times call per series
    instead of per row, and avoids the window function entirely.

    Args:
        df: Input DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.

    Returns:
        Tuple of (has_issues, bad_df):
            - has_issues: False if no gaps or duplicates found, True otherwise
            - bad_df: DataFrame with id_col for affected series (empty if False)
    """
    if df.shape[0] == 0:
        return (False, df[[id_col]].head(0))

    nw_df = nw.from_native(df, eager_only=True)

    stats = (
        nw_df.group_by(id_col)
        .agg([
            nw.col(time_col).min().alias('_min'),
            nw.col(time_col).max().alias('_max'),
            nw.col(time_col).count().alias('_count'),
            nw.col(time_col).n_unique().alias('_n_unique'),
        ])
    )

    # offset_times accepts Union[int, np.ndarray] for n; convert pandas Series to numpy
    n_steps = (stats['_n_unique'] - 1).to_native()
    if isinstance(n_steps, pd.Series):
        n_steps = n_steps.to_numpy()

    expected_max_native = offset_times(stats['_min'].to_native(), freq, n_steps)
    expected_max_nw = nw.from_native(expected_max_native, series_only=True).alias('_expected_max')

    stats = stats.with_columns(expected_max_nw)

    bad = stats.filter(
        (nw.col('_n_unique') != nw.col('_count'))      # duplicates
        | (nw.col('_max') != nw.col('_expected_max'))  # gaps
    ).select([id_col])

    bad_native = nw.to_native(bad)
    if bad.shape[0] > 0:
        return (True, bad_native)
    return (False, bad_native)


def validate_update_df(
    df: DFType,
    id_col: str,
    time_col: str,
    uids,
    last_dates,
    freq: Union[str, int],
) -> None:
    """Validate that new data is continuous with existing series.

    Checks that every series starts at last_date + freq and contains no internal
    gaps or duplicate timestamps. Raises ValueError on the first violation found.

    Args:
        df: New observations DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column
        uids: Known series identifiers (pd.Index or polars Series), parallel to last_dates
        last_dates: Last known timestamp per series (pd.Index or polars Series)
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.

    Raises:
        ValueError: If any series has an invalid start date or internal gaps/duplicates.
    """
    if isinstance(df, pd.DataFrame):
        last_dates_df = pd.DataFrame({id_col: uids, '_last': last_dates})
    else:
        last_dates_df = pl.DataFrame({id_col: uids, '_last': last_dates})

    has_issues, bad = validate_update_start_dates(df, id_col, time_col, last_dates_df, freq)
    if has_issues:
        bad_ids = nw.from_native(bad, eager_only=True)[id_col].to_list()
        raise ValueError(
            "Series have invalid start dates. "
            f"Expected start at last_date + freq for: {bad_ids}."
        )

    has_issues, bad = validate_continuity(df, id_col, time_col, freq)
    if has_issues:
        bad_ids = reprlib.repr(nw.from_native(bad, eager_only=True)[id_col].to_list())
        raise ValueError(
            f"Series contain missing or duplicate timestamps with the specified freq {freq}\n"
            f"Affected series: {bad_ids}\n"
            f"Consider using the fill_gaps parameter or preprocessing your data."
        )


def validate_df(
    df: DFType,
    id_col: str,
    time_col: str,
    freq: Union[str, int],
) -> None:
    """Run data quality validations and raise if problems are found.

    Checks for gaps or duplicate timestamps using a fast consecutive-comparison
    approach. Raises ValueError if problems are found.

    Args:
        df: Input DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.
    """
    has_issues, bad = validate_continuity(df, id_col, time_col, freq)
    if has_issues:
        bad_ids = reprlib.repr(nw.from_native(bad, eager_only=True)[id_col].to_list())
        raise ValueError(
            f"Series contain missing or duplicate timestamps with the specified freq {freq}\n"
            f"Affected series: {bad_ids}\n"
            f"Consider using the fill_gaps parameter or preprocessing your data."
        )
