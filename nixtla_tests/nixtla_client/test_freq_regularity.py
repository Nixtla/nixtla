"""Offline coverage for the shared frequency-regularity validation.

`_validate_freq_regularity` runs on every public method via `_run_validations`,
so its behaviour is asserted directly here rather than through a single endpoint.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nixtla.nixtla_client import _dataframe_keys_match, _validate_freq_regularity


def _regular_df(n_series=2, n=5):
    return pd.DataFrame(
        {
            "unique_id": np.repeat([f"id-{i}" for i in range(n_series)], n),
            "ds": list(pd.date_range("2024-01-01", periods=n, freq="D")) * n_series,
            "y": np.arange(n_series * n, dtype=float),
        }
    )


def _with_duplicate_cancelling_a_gap():
    df = _regular_df(n_series=1).drop(index=2).reset_index(drop=True)
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


FREQS = [pytest.param("D", id="string"), pytest.param(pd.offsets.Day(), id="offset")]


@pytest.mark.parametrize("freq", FREQS)
def test_accepts_regular_series(freq):
    _validate_freq_regularity(
        df=_regular_df(), freq=freq, id_col="unique_id", time_col="ds"
    )


@pytest.mark.parametrize("freq", FREQS)
def test_rejects_missing_timestamps(freq):
    df = _regular_df().drop(index=2).reset_index(drop=True)

    with pytest.raises(ValueError, match="missing or duplicate timestamps"):
        _validate_freq_regularity(df=df, freq=freq, id_col="unique_id", time_col="ds")


@pytest.mark.parametrize("freq", FREQS)
def test_rejects_duplicate_that_cancels_out_a_gap(freq):
    df = _with_duplicate_cancelling_a_gap()
    assert len(df) == 5

    with pytest.raises(ValueError, match="missing or duplicate timestamps"):
        _validate_freq_regularity(df=df, freq=freq, id_col="unique_id", time_col="ds")


def test_polars_accepts_regular_series():
    _validate_freq_regularity(
        df=pl.from_pandas(_regular_df()),
        freq="1d",
        id_col="unique_id",
        time_col="ds",
    )


def test_polars_rejects_duplicate_that_cancels_out_a_gap():
    with pytest.raises(ValueError, match="missing or duplicate timestamps"):
        _validate_freq_regularity(
            df=pl.from_pandas(_with_duplicate_cancelling_a_gap()),
            freq="1d",
            id_col="unique_id",
            time_col="ds",
        )


def test_polars_with_pandas_offset_raises_value_error():
    with pytest.raises(ValueError, match="pandas offsets are only supported"):
        _validate_freq_regularity(
            df=pl.from_pandas(_regular_df()),
            freq=pd.offsets.Day(),
            id_col="unique_id",
            time_col="ds",
        )


def test_rejects_unsupported_freq_type():
    with pytest.raises(ValueError, match="`freq` should be a string"):
        _validate_freq_regularity(
            df=_regular_df(), freq=1.5, id_col="unique_id", time_col="ds"
        )


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.from_pandas])
def test_dataframe_keys_match(constructor):
    expected = constructor(_regular_df()[["unique_id", "ds"]])
    keys = _regular_df()[["unique_id", "ds"]]

    assert _dataframe_keys_match(
        actual=constructor(keys), expected=expected, id_col="unique_id", time_col="ds"
    )
    shuffled = keys.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert _dataframe_keys_match(
        actual=constructor(shuffled),
        expected=expected,
        id_col="unique_id",
        time_col="ds",
    )
    assert not _dataframe_keys_match(
        actual=constructor(keys.iloc[:-1].reset_index(drop=True)),
        expected=expected,
        id_col="unique_id",
        time_col="ds",
    )
    swapped = pd.concat([keys.iloc[:-1], keys.iloc[[0]]], ignore_index=True)
    assert not _dataframe_keys_match(
        actual=constructor(swapped),
        expected=expected,
        id_col="unique_id",
        time_col="ds",
    )
