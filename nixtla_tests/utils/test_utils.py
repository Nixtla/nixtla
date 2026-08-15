import numpy as np
import orjson
import pandas as pd
import pytest

from nixtla.nixtla_client import _partition_series
from nixtla.nixtla_client import _preprocess
from nixtla.nixtla_client import _series_starts
from nixtla.nixtla_client import _tail
from nixtla.nixtla_client import _times_to_iso
from nixtla.nixtla_client import _audit_duplicate_rows
from nixtla.nixtla_client import _audit_categorical_variables
from nixtla.nixtla_client import _audit_leading_zeros
from nixtla.nixtla_client import _audit_missing_dates
from nixtla.nixtla_client import _audit_negative_values
from nixtla.nixtla_client import _forecast_payload_to_in_sample
from nixtla.nixtla_client import _maybe_add_date_features
from nixtla.nixtla_client import AuditDataSeverity
from nixtla.date_features import SpecialDates


def test_audit_duplicate_rows_pass(df_no_duplicates):
    audit, duplicates = _audit_duplicate_rows(df_no_duplicates)
    assert audit == AuditDataSeverity.PASS
    assert len(duplicates) == 0


def test_audit_duplicate_rows_fail(df_with_duplicates):
    audit, duplicates = _audit_duplicate_rows(df_with_duplicates)
    assert audit == AuditDataSeverity.FAIL
    assert len(duplicates) == 2


def test_audit_missing_dates_complete(df_complete):
    audit, missing = _audit_missing_dates(df_complete, freq="D")
    assert audit == AuditDataSeverity.PASS
    assert len(missing) == 0


def test_audit_missing_dates_with_missing(df_missing):
    audit, missing = _audit_missing_dates(df_missing, freq="D")
    assert audit == AuditDataSeverity.FAIL
    assert len(missing) == 2  # One missing date per unique_id


# --- Audit Categorical Variables ---
def test_audit_categorical_variables_no_cat(df_no_cat):
    audit, cat_df = _audit_categorical_variables(df_no_cat)
    assert audit == AuditDataSeverity.PASS
    assert len(cat_df) == 0


def test_audit_categorical_variables_with_cat(df_with_cat):
    audit, cat_df = _audit_categorical_variables(df_with_cat)
    assert audit == AuditDataSeverity.FAIL
    assert cat_df.shape[1] == 1  # Should include only 'cat_col'


def test_audit_categorical_variables_with_cat_dtype(df_with_cat_dtype):
    audit, cat_df = _audit_categorical_variables(df_with_cat_dtype)
    assert audit == AuditDataSeverity.FAIL
    assert cat_df.shape[1] == 1  # Should include only 'cat_col'


def test_audit_leading_zeros(df_leading_zeros):
    audit, leading_zeros_df = _audit_leading_zeros(df_leading_zeros)
    assert audit == AuditDataSeverity.CASE_SPECIFIC
    assert len(leading_zeros_df) == 3


def test_audit_negative_values(df_negative_values):
    audit, negative_values_df = _audit_negative_values(df_negative_values)
    assert audit == AuditDataSeverity.CASE_SPECIFIC
    assert len(negative_values_df) == 3


@pytest.mark.parametrize(
    "date_features,freq,one_hot,expected_date_features",
    [
        (["year", "month"], "MS", False, ["year", "month"]),
        (
            [
                SpecialDates(
                    {"first_dates": ["2021-01-1"], "second_dates": ["2021-01-01"]}
                )
            ],
            "D",
            False,
            ["first_dates", "second_dates"],
        ),
        (["year", "month"], "D", ["month"], ["month_" + str(i) for i in range(1, 13)]),
    ],
)
def test_maybe_add_date_features(
    air_passengers_df, date_features, freq, one_hot, expected_date_features
):
    df_copy = air_passengers_df.copy()
    df_copy.rename(columns={"timestamp": "ds", "value": "y"}, inplace=True)
    df_copy.insert(0, "unique_id", "AirPassengers")
    df_date_features, future_df = _maybe_add_date_features(
        df=df_copy,
        X_df=None,
        h=12,
        freq=freq,
        features=date_features,
        one_hot=one_hot,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    assert all(col in df_date_features for col in expected_date_features)
    assert all(col in future_df for col in expected_date_features)


@pytest.mark.parametrize(
    "date_features,one_hot,expected_date_features",
    [
        (["year", "month"], False, ["year", "month"]),
        (["month", "day"], ["month", "day"], ["month_" + str(i) for i in range(1, 13)]),
    ],
    ids=["no_one_hot", "with_one_hot"],
)
def test_add_date_features_with_exogenous_variables(
    air_passengers_df, date_features, one_hot, expected_date_features, request
):
    df_copy = air_passengers_df.copy()
    df_copy.rename(columns={"timestamp": "ds", "value": "y"}, inplace=True)
    df_copy.insert(0, "unique_id", "AirPassengers")

    df_actual_future = df_copy.tail(12)[["unique_id", "ds"]]
    df_date_features, future_df = _maybe_add_date_features(
        df=df_copy,
        X_df=df_actual_future,
        h=24,
        freq="H",
        features=date_features,
        one_hot=one_hot,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    assert all(col in df_date_features for col in expected_date_features)
    assert all(col in future_df for col in expected_date_features)
    pd.testing.assert_frame_equal(
        df_date_features[df_copy.columns],
        df_copy,
    )

    if request.node.callspec.id == "no_one_hot":
        expected_df_actual_future = df_actual_future.copy()
    elif request.node.callspec.id == "with_one_hot":
        expected_df_actual_future = df_actual_future.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        future_df[df_actual_future.columns],
        expected_df_actual_future,
    )


# --- _forecast_payload_to_in_sample (add_history workflow) ---
def test_forecast_payload_to_in_sample_always_sets_full_history(base_forecast_payload):
    payload = _forecast_payload_to_in_sample(base_forecast_payload, h=4, n_windows=2)
    # The add_history workflow always runs cross_validation in full_history mode.
    assert payload["full_history"] is True
    # h/step_size/n_windows are still populated as server-side-ignored placeholders.
    assert payload["h"] == 4
    assert payload["step_size"] == 4
    assert payload["n_windows"] == 2
    # No finetuning for in-sample, X_future is dropped, hist_exog excludes the future feature.
    assert payload["finetune_steps"] == 0
    assert "X_future" not in payload["series"]
    assert payload["hist_exog"] == [1]


# ---------------------------------------------------------------------------
# start_datetime helpers (_times_to_iso / _series_starts)
#
# The invariant under test: start_datetime[i] is the first timestamp of series i
# *as that series appears in the payload's y* -- not the customer's original
# first timestamp. It must track sorting, truncation and partitioning.
# ---------------------------------------------------------------------------
SPECS = [("a", "2020-01-01", 5), ("b", "2020-03-01", 3), ("c", "2020-06-01", 7)]


def _make_df(tz=None, shuffle=True, freq="D"):
    frames = []
    for uid, start, n in SPECS:
        ds = pd.date_range(start, periods=n, freq=freq, tz=tz)
        frames.append(
            pd.DataFrame({"unique_id": uid, "ds": ds, "y": np.arange(n, dtype=float)})
        )
    df = pd.concat(frames)
    if shuffle:
        df = df.sample(frac=1, random_state=0)
    return df.reset_index(drop=True)


def _process(df, freq="D"):
    processed, *_ = _preprocess(
        df=df,
        X_df=None,
        h=0,
        freq=freq,
        date_features=False,
        date_features_to_one_hot=False,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    return processed


def test_series_starts_unsorted_input():
    df = _make_df(shuffle=True)
    processed = _process(df)
    # rows were shuffled, so the sort_idxs branch must be exercised
    assert processed.sort_idxs is not None
    assert _series_starts(df, processed, "ds") == [
        "2020-01-01",
        "2020-03-01",
        "2020-06-01",
    ]


def test_series_starts_already_sorted_input():
    df = _make_df(shuffle=False)
    processed = _process(df)
    assert processed.sort_idxs is None
    assert _series_starts(df, processed, "ds") == [
        "2020-01-01",
        "2020-03-01",
        "2020-06-01",
    ]


def test_series_starts_uses_minimal_precision():
    # unit="auto" keeps daily data as plain dates instead of "...T00:00:00.000000000"
    starts = _series_starts(_make_df(shuffle=False), _process(_make_df(shuffle=False)), "ds")
    assert all("T" not in s for s in starts)


def test_series_starts_keeps_time_component():
    # a sub-daily start must not be truncated to a bare date by unit="auto"
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2020-01-01 09:30", periods=4, freq="30min"),
            "y": np.arange(4, dtype=float),
        }
    )
    processed, *_ = _preprocess(
        df=df,
        X_df=None,
        h=0,
        freq="30min",
        date_features=False,
        date_features_to_one_hot=False,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    assert _series_starts(df, processed, "ds") == ["2020-01-01T09:30"]


def test_series_starts_fixed_offset_timezone_preserves_offset():
    df = _make_df(tz="Etc/GMT+5", shuffle=True)
    processed = _process(df)
    starts = _series_starts(df, processed, "ds")
    assert starts == [
        "2020-01-01T00:00:00-05:00",
        "2020-03-01T00:00:00-05:00",
        "2020-06-01T00:00:00-05:00",
    ]
    # and the result must survive the orjson serialization used by _make_request
    assert orjson.loads(orjson.dumps(starts)) == starts


def test_series_starts_dst_timezone_returns_none(caplog):
    df = _make_df(tz="US/Eastern", shuffle=True)
    assert _series_starts(df, _process(df), "ds") is None
    assert "cannot be represented losslessly" in caplog.text


def test_series_starts_tz_aware_array_would_not_serialize():
    """Guards the design decision: pass the raw array and orjson fails."""
    times = _make_df(tz="US/Eastern", shuffle=False)["ds"].to_numpy()
    assert times.dtype == object
    with pytest.raises(TypeError):
        orjson.dumps(times, option=orjson.OPT_SERIALIZE_NUMPY)


# --- polars inputs ---
def test_series_starts_polars_matches_pandas():
    pl = pytest.importorskip("polars")
    df = _make_df(shuffle=True)
    pl_df = pl.from_pandas(df)
    assert _series_starts(pl_df, _process(pl_df, freq="1d"), "ds") == _series_starts(
        df, _process(df), "ds"
    )


def test_series_starts_polars_fixed_offset_timezone_restores_offset():
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(_make_df(tz="Etc/GMT+5", shuffle=True))
    # the premise: polars hands numpy naive UTC values, not tz-aware objects
    assert df["ds"].to_numpy().dtype != object
    starts = _series_starts(df, _process(df, freq="1d"), "ds")
    assert starts == [
        "2020-01-01T00:00:00-05:00",
        "2020-03-01T00:00:00-05:00",
        "2020-06-01T00:00:00-05:00",
    ]
    assert orjson.loads(orjson.dumps(starts)) == starts


def test_series_starts_polars_dst_timezone_returns_none(caplog):
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(_make_df(tz="US/Eastern", shuffle=True))
    assert _series_starts(df, _process(df, freq="1d"), "ds") is None
    assert "cannot be represented losslessly" in caplog.text


def test_series_starts_polars_fixed_offset_timezone_truncated():
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(_make_df(tz="Etc/GMT+5", shuffle=True))
    processed = _process(df, freq="1d")
    orig_indptr, orig_sort_idxs = processed.indptr, processed.sort_idxs
    starts = _series_starts(
        df, _tail(processed, 4), "ds", orig_indptr, orig_sort_idxs
    )
    assert starts == [
        "2020-01-02T00:00:00-05:00",
        "2020-03-01T00:00:00-05:00",
        "2020-06-04T00:00:00-05:00",
    ]


def test_series_starts_polars_date_dtype():
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(_make_df(shuffle=False)).with_columns(pl.col("ds").cast(pl.Date))
    starts = _series_starts(df, _process(df, freq="1d"), "ds")
    assert starts == ["2020-01-01", "2020-03-01", "2020-06-01"]


def test_times_to_iso_tz_restores_offset():
    times = np.array(["2019-12-31T23:00"], dtype="datetime64[ns]")
    assert _times_to_iso(times, tz="Etc/GMT-1") == [
        "2020-01-01T00:00:00+01:00"
    ]


def test_times_to_iso_dst_timezone_returns_none(caplog):
    times = np.array(["2019-12-31T23:00"], dtype="datetime64[ns]")
    assert _times_to_iso(times, tz="Europe/Amsterdam") is None
    assert "cannot be represented losslessly" in caplog.text


def test_series_starts_integer_time_col_returns_none(integer_freq_series):
    df = integer_freq_series
    processed, *_ = _preprocess(
        df=df,
        X_df=None,
        h=0,
        freq=1,
        date_features=False,
        date_features_to_one_hot=False,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    assert _series_starts(df, processed, "ds") is None


def test_times_to_iso_empty_returns_none():
    assert _times_to_iso(np.array([], dtype="datetime64[ns]")) is None


def test_times_to_iso_non_datetime_returns_none():
    assert _times_to_iso(np.array([1.0, 2.0])) is None
    assert _times_to_iso(np.array(["a", "b"], dtype=object)) is None


def test_series_starts_after_truncation_tracks_y():
    """_tail drops leading rows, so each start must move forward with y."""
    df = _make_df(shuffle=True)
    processed = _process(df)
    orig_indptr, orig_sort_idxs = processed.indptr, processed.sort_idxs
    truncated = _tail(processed, 4)
    # _tail resets sort_idxs, which is why the caller must pass it explicitly
    assert truncated.sort_idxs is None

    starts = _series_starts(df, truncated, "ds", orig_indptr, orig_sort_idxs)
    sizes = np.diff(truncated.indptr)
    assert list(sizes) == [4, 3, 4]
    # series 'a' (5 rows -> 4) and 'c' (7 -> 4) shift; 'b' (3) is untouched
    assert starts == ["2020-01-02", "2020-03-01", "2020-06-04"]

    # verify positionally against the sorted frame: each start is the
    # size-th-from-last timestamp of its own series
    sorted_df = df.sort_values(["unique_id", "ds"])
    for uid, start, size in zip(truncated.uids, starts, sizes):
        expected = sorted_df.loc[sorted_df["unique_id"] == uid, "ds"].iloc[-size]
        assert pd.Timestamp(start) == expected


def test_series_starts_truncation_differs_from_original_start():
    df = _make_df(shuffle=False)
    processed = _process(df)
    orig_indptr, orig_sort_idxs = processed.indptr, processed.sort_idxs
    untruncated = _series_starts(df, processed, "ds")
    truncated = _series_starts(
        df, _tail(processed, 4), "ds", orig_indptr, orig_sort_idxs
    )
    assert truncated != untruncated


def test_series_starts_truncation_no_op_when_series_are_short():
    df = _make_df(shuffle=False)
    processed = _process(df)
    orig_indptr, orig_sort_idxs = processed.indptr, processed.sort_idxs
    # n larger than every series -> _tail changes nothing
    starts = _series_starts(
        df, _tail(processed, 100), "ds", orig_indptr, orig_sort_idxs
    )
    assert starts == _series_starts(df, processed, "ds")


# --- _partition_series carries start_datetime ---
def _payload(n_series, with_start=True):
    sizes = np.array([2] * n_series)
    series = {
        "y": np.arange(2 * n_series, dtype=float),
        "sizes": sizes,
        "X": None,
        "X_future": None,
    }
    if with_start:
        series["start_datetime"] = [f"2020-01-{i + 1:02d}" for i in range(n_series)]
    return {"series": series, "h": 0}


def test_partition_series_slices_start_datetime():
    parts = _partition_series(_payload(5), n_part=2, h=0)
    assert len(parts) == 2
    # sliced per series (like sizes), not per row (like y)
    assert parts[0]["series"]["start_datetime"] == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert parts[1]["series"]["start_datetime"] == ["2020-01-04", "2020-01-05"]
    # the server's hard requirement: one entry per series in every partition
    for part in parts:
        assert len(part["series"]["start_datetime"]) == len(part["series"]["sizes"])


def test_partition_series_start_datetime_concatenates_to_whole():
    expected = _payload(7)["series"]["start_datetime"]
    parts = _partition_series(_payload(7), n_part=3, h=0)
    rejoined = [s for part in parts for s in part["series"]["start_datetime"]]
    assert rejoined == expected


def test_partition_series_without_start_datetime():
    parts = _partition_series(_payload(4, with_start=False), n_part=2, h=0)
    assert all("start_datetime" not in part["series"] for part in parts)


def test_partition_series_single_series_is_never_split():
    # n_part = min(n_part, n_series), so one series always stays whole
    parts = _partition_series(_payload(1), n_part=10, h=0)
    assert len(parts) == 1
    assert parts[0]["series"]["start_datetime"] == ["2020-01-01"]
