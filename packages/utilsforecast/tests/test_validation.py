import datetime

import pandas as pd
import polars as pl
from conftest import assert_raises_with_message

from utilsforecast.compat import POLARS_INSTALLED
from utilsforecast.validation import (
    _is_dt_dtype,
    _is_int_dtype,
    ensure_time_dtype,
    validate_format,
    validate_freq,
)


def test_dtypes():
    assert _is_int_dtype(pd.Series([1, 2]))
    assert _is_int_dtype(pd.Index([1, 2], dtype='uint8'))
    assert not _is_int_dtype(pd.Series([1.0]))
    assert _is_dt_dtype(pd.to_datetime(['2000-01-01']))
    assert _is_dt_dtype(pd.to_datetime(['2000-01-01'], utc=True))


def test_dtypes_arrow():
    assert _is_dt_dtype(pd.to_datetime(['2000-01-01']).astype('datetime64[s]'))
    assert _is_int_dtype(pd.Series([1, 2], dtype='int32[pyarrow]'))
    assert _is_dt_dtype(pd.to_datetime(['2000-01-01']).astype('timestamp[ns][pyarrow]'))
    assert _is_int_dtype(pl.Series([1, 2]))
    assert _is_int_dtype(pl.Series([1, 2], dtype=pl.UInt8))


def test_dtypes_polars():
    assert not _is_int_dtype(pl.Series([1.0]))
    assert _is_dt_dtype(pl.Series([datetime.date(2000, 1, 1)]))
    assert _is_dt_dtype(pl.Series([datetime.datetime(2000, 1, 1)]))
    assert _is_dt_dtype(
        pl.Series([datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)])
    )


def test_ensure_time_dtype():
    pd.testing.assert_frame_equal(
        ensure_time_dtype(pd.DataFrame({'ds': ['2000-01-01']})),
        pd.DataFrame({'ds': pd.to_datetime(['2000-01-01'])}),
    )
    df = pd.DataFrame({'ds': [1, 2]})
    assert df is ensure_time_dtype(df)
    assert_raises_with_message(ensure_time_dtype, 'Please make sure that it contains valid timestamps', pd.DataFrame({'ds': ['2000-14-14']}))
    pl.testing.assert_frame_equal(
        ensure_time_dtype(pl.DataFrame({'ds': ['2000-01-01']})),
        pl.DataFrame().with_columns(ds=pl.datetime(2000, 1, 1)),
    )
    df = pl.DataFrame({'ds': [1, 2]})
    assert df is ensure_time_dtype(df)
    assert_raises_with_message(ensure_time_dtype, 'Please make sure that it contains valid timestamps', pl.DataFrame({'ds': ['hello']}))


def test_validate_format():
    assert_raises_with_message(validate_format, "got <class 'int'>", 1)
    constructors = [pd.DataFrame]
    if POLARS_INSTALLED:
        constructors.append(pl.DataFrame)
    for constructor in constructors:
        df = constructor({'unique_id': [1]})
        assert_raises_with_message(validate_format, "missing: ['ds', 'y']", df)
        df = constructor({'unique_id': [1], 'time': ['x'], 'y': [1]})
        assert_raises_with_message(validate_format,"('time') should have either timestamps or integers", df, time_col='time'),
        for time in [1, datetime.datetime(2000, 1, 1)]:
            df = constructor({'unique_id': [1], 'ds': [time], 'sales': ['x']})
            assert_raises_with_message(validate_format, "('sales') should have a numeric data type", df, target_col='sales')


def test_validate_freq():
    assert_raises_with_message(validate_freq, 'provide a valid integer',pd.Series([1, 2]), 'D')
    assert_raises_with_message(validate_freq, 'provide a valid pandas or polars offset', pd.to_datetime(['2000-01-01']).to_series(), 1),
    assert_raises_with_message(validate_freq, 'provide a valid integer', pl.Series([1, 2]), '1d')
    assert_raises_with_message(validate_freq, 'provide a valid pandas or polars offset', pl.Series([datetime.datetime(2000, 1, 1)]), 1),
    assert_raises_with_message(validate_freq, 'valid polars offset', pl.Series([datetime.datetime(2000, 1, 1)]), 'D'),

