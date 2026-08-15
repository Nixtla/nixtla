"""End-to-end payload tests for the `start_datetime` field.

These assert the invariant the server depends on: `start_datetime[i]` is the first
timestamp of series `i` *as that series appears in the payload's `y`*. The three
fields `y`, `sizes` and `start_datetime` must always agree, regardless of sorting,
input truncation, or `num_partitions`.

Everything here is offline -- `_make_request` is replaced with a recorder that
synthesizes plausible responses, so no API key or network access is needed.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nixtla.nixtla_client import NixtlaClient


# ---------------------------------------------------------------------------
# Payload capture harness
# ---------------------------------------------------------------------------
# Stand-in for the /model_params response, so restrict_input truncates to a
# predictable length without a network call.
MODEL_INPUT_SIZE = 100
MODEL_HORIZON = 10


def _fake_response(endpoint, payload):
    """Synthesize a response with the keys each endpoint's assembly code reads."""
    series = payload["series"]
    sizes = np.asarray(series["sizes"], dtype=np.int64)
    indptr = np.append(0, sizes.cumsum())

    if endpoint == "v2/finetune":
        return {"finetuned_model_id": "test-model-id"}

    if endpoint == "v2/forecast":
        h = payload["h"]
        n = h * len(sizes)
        return {
            "mean": np.zeros(n, dtype=np.float32),
            "intervals": None,
            "weights_x": None,
            "feature_contributions": None,
        }

    # in-sample endpoints return a row per input point, addressed by `idxs`
    if endpoint == "v2/cross_validation":
        h, n_windows = payload["h"], payload["n_windows"]
        out_sizes, idxs = [], []
        for i, size in enumerate(sizes):
            total = h * n_windows
            # the last `total` points of each series, as real windows would be
            start = indptr[i] + size - total
            idxs.append(np.arange(start, start + total))
            out_sizes.append(total)
        idxs = np.concatenate(idxs)
    else:  # anomaly_detection / online_anomaly_detection
        idxs = np.concatenate(
            [np.arange(indptr[i], indptr[i + 1]) for i in range(len(sizes))]
        )
        out_sizes = list(sizes)

    n = len(idxs)
    resp = {
        "mean": np.zeros(n, dtype=np.float32),
        "idxs": idxs,
        "sizes": np.asarray(out_sizes, dtype=np.int64),
        "intervals": None,
        "weights_x": None,
        "feature_contributions": None,
    }
    if "anomaly" in endpoint:
        resp["anomaly"] = np.zeros(n, dtype=bool)
        resp["anomaly_score"] = np.zeros(n, dtype=np.float32)
    return resp


@pytest.fixture
def capture(monkeypatch):
    """A client whose requests are recorded instead of sent.

    `payloads` is ordered by series, not by wall-clock: partitioned requests are
    dispatched on a thread pool, so the recorder alone would see them in completion
    order. `_partition_series` is wrapped to record the ordered partition list,
    mirroring how production restores order via `results[pos]`.
    """
    import nixtla.nixtla_client as nc

    client = NixtlaClient(api_key="test-key")
    payloads = []
    ordered_parts = []

    real_partition_series = nc._partition_series

    def _spy_partition_series(payload, n_part, h):
        parts = real_partition_series(payload, n_part, h)
        ordered_parts.append([deepcopy(p) for p in parts])
        return parts

    monkeypatch.setattr(nc, "_partition_series", _spy_partition_series)

    def _record(client, endpoint, payload, multithreaded_compress=True):
        payloads.append((endpoint, deepcopy(payload)))
        return _fake_response(endpoint, payload)

    client._make_request = _record
    # pre-seed the (model, freq) cache so _get_model_params makes no network call
    for freq in ("D", "h", "MS", "30min", "1d", "1h"):
        client._model_params[("timegpt-2.1", freq)] = (MODEL_INPUT_SIZE, MODEL_HORIZON)
    return SimpleNamespace(client=client, payloads=payloads, parts=ordered_parts)


def _ordered_starts(cap):
    """start_datetime of every series, in original series order."""
    if cap.parts:
        # partitioned: use the ordered partition list from the spy
        return [
            s
            for part in cap.parts[0]
            for s in part["series"].get("start_datetime", [])
        ]
    return [
        s for _, p in cap.payloads for s in p["series"].get("start_datetime", [])
    ]


def assert_consistent(payload, freq, expected=None):
    """The core invariant: start_datetime agrees with y and sizes."""
    series = payload["series"]
    assert "start_datetime" in series, "start_datetime missing from payload"
    starts = series["start_datetime"]
    sizes = np.asarray(series["sizes"], dtype=np.int64)

    # the server's hard check
    assert len(starts) == len(sizes)
    # y and sizes agree, so start_datetime describes the same rows
    assert len(series["y"]) == int(sizes.sum())
    # every entry is a parseable ISO 8601 timestamp
    for start in starts:
        assert isinstance(start, str) and start.strip()
        assert pd.Timestamp(start) is not pd.NaT
    # the reconstructed index -- what the server does with these three fields --
    # covers exactly the rows in y
    for start, size in zip(starts, sizes):
        idx = pd.date_range(start=start, periods=int(size), freq=freq)
        assert len(idx) == size
    if expected is not None:
        assert starts == expected


def _series(n_series=6, n=60, freq="D", tz=None, start="2020-01-01"):
    frames = []
    for i in range(n_series):
        ds = pd.date_range(start, periods=n + i, freq=freq, tz=tz)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"id_{i}",
                    "ds": ds,
                    "y": np.arange(len(ds), dtype=float) + 1.0,
                }
            )
        )
    return pd.concat(frames).reset_index(drop=True)


def _staggered_series(n_series=6, n=40):
    """Series with a distinct start each, so a mis-slice cannot pass by luck."""
    frames = [
        pd.DataFrame(
            {
                "unique_id": f"id_{i}",
                "ds": pd.date_range(f"2020-{i + 1:02d}-01", periods=n, freq="D"),
                "y": np.arange(n, dtype=float) + 1.0,
            }
        )
        for i in range(n_series)
    ]
    return pd.concat(frames).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Requirement 1: every endpoint registers start_datetime
# ---------------------------------------------------------------------------
def _call(client, endpoint, df, freq="D", **kw):
    if endpoint == "forecast":
        return client.forecast(df=df, h=5, freq=freq, **kw)
    if endpoint == "cross_validation":
        return client.cross_validation(df=df, h=5, n_windows=2, freq=freq, **kw)
    if endpoint == "detect_anomalies":
        return client.detect_anomalies(df=df, freq=freq, **kw)
    if endpoint == "detect_anomalies_online":
        return client.detect_anomalies_online(
            df=df, h=5, detection_size=10, freq=freq, **kw
        )
    if endpoint == "finetune":
        return client.finetune(df=df, freq=freq, finetune_steps=1, **kw)
    raise AssertionError(f"unknown endpoint {endpoint}")


ENDPOINTS = [
    "forecast",
    "cross_validation",
    "detect_anomalies",
    "detect_anomalies_online",
    "finetune",
]

# endpoints that accept num_partitions
PARTITIONED = ENDPOINTS[:-1]

# keeps the whole history in y, so starts equal the real first timestamps
NO_TRUNCATE = {"finetune_steps": 1}


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_start_datetime_registered(capture, endpoint):
    """Requirement 1: the field reaches the payload for every endpoint."""
    _call(capture.client, endpoint, _series())
    assert capture.payloads, f"{endpoint} made no request"
    for _, payload in capture.payloads:
        assert_consistent(payload, "D")


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_start_datetime_matches_input_when_untruncated(capture, endpoint):
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}
    _call(capture.client, endpoint, _series(n_series=4, n=40), **kw)
    for _, payload in capture.payloads:
        assert_consistent(payload, "D", expected=["2020-01-01"] * 4)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_start_datetime_with_unsorted_input(capture, endpoint):
    """Rows shuffled on the way in must not scramble the starts."""
    df = _series(n_series=4, n=40).sample(frac=1, random_state=0).reset_index(drop=True)
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}
    _call(capture.client, endpoint, df, **kw)
    for _, payload in capture.payloads:
        assert_consistent(payload, "D", expected=["2020-01-01"] * 4)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_start_datetime_distinct_per_series(capture, endpoint):
    """Each series gets its own start, in payload (uid-sorted) order."""
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}
    _call(capture.client, endpoint, _staggered_series(6), **kw)
    expected = [f"2020-{m + 1:02d}-01" for m in range(6)]
    for _, payload in capture.payloads:
        assert_consistent(payload, "D", expected=expected)


# ---------------------------------------------------------------------------
# Requirement 2: correct regardless of num_partitions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("endpoint", PARTITIONED)
@pytest.mark.parametrize("num_partitions", [1, 3, 10])
def test_start_datetime_per_partition(capture, endpoint, num_partitions):
    """Every partition must independently satisfy the server's contract."""
    _call(capture.client, endpoint, _series(n_series=6, n=60), num_partitions=num_partitions)
    assert capture.payloads
    for _, payload in capture.payloads:
        assert_consistent(payload, "D")
    # the parts rejoin into the whole: one entry per series overall
    assert len(_ordered_starts(capture)) == 6


@pytest.mark.parametrize("endpoint", PARTITIONED)
@pytest.mark.parametrize("num_partitions", [1, 3, 10])
def test_start_datetime_partitions_preserve_series_order(
    capture, endpoint, num_partitions
):
    """start_datetime slices per series while y slices per row -- they stay paired.

    Distinct starts per series mean a mis-slice cannot pass by coincidence.
    """
    _call(
        capture.client,
        endpoint,
        _staggered_series(6),
        num_partitions=num_partitions,
        **(NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}),
    )
    expected = [f"2020-{m + 1:02d}-01" for m in range(6)]
    assert _ordered_starts(capture) == expected
    for _, payload in capture.payloads:
        assert_consistent(payload, "D")


@pytest.mark.parametrize("endpoint", PARTITIONED)
def test_start_datetime_partitioned_matches_unpartitioned(capture, endpoint):
    """Partitioning changes only how the field is split, never its contents."""
    df = _staggered_series(6)
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}

    _call(capture.client, endpoint, df, **kw)
    whole = _ordered_starts(capture)
    capture.payloads.clear()
    capture.parts.clear()

    _call(capture.client, endpoint, df, num_partitions=3, **kw)
    assert _ordered_starts(capture) == whole


# ---------------------------------------------------------------------------
# Truncation: start_datetime must follow y, not the original input
# ---------------------------------------------------------------------------
def _assert_starts_are_tails(df, payload):
    """Each start is the (size)th-from-last timestamp of its own series."""
    sizes = np.asarray(payload["series"]["sizes"], dtype=np.int64)
    for uid, start, size in zip(sorted(df["unique_id"].unique()),
                               payload["series"]["start_datetime"], sizes):
        series_ds = df.loc[df["unique_id"] == uid, "ds"].sort_values()
        assert pd.Timestamp(start) == series_ds.iloc[-int(size)]


def test_forecast_restrict_input_moves_start_datetime(capture):
    """restrict_input drops leading rows, so the start moves forward with y."""
    n = 500
    df = _series(n_series=3, n=n)
    # no exog, no finetuning, no add_history -> restrict_input is active
    capture.client.forecast(df=df, h=5, freq="D")

    _, payload = capture.payloads[0]
    sizes = np.asarray(payload["series"]["sizes"])
    assert sizes.max() < n, "expected the input to be truncated"
    assert_consistent(payload, "D")
    # the registered start is NOT the customer's original first timestamp
    assert payload["series"]["start_datetime"][0] != "2020-01-01"
    _assert_starts_are_tails(df, payload)


def test_cross_validation_restrict_input_moves_start_datetime(capture):
    n = 500
    df = _series(n_series=3, n=n)
    capture.client.cross_validation(df=df, h=5, n_windows=2, freq="D")

    _, payload = capture.payloads[0]
    assert np.asarray(payload["series"]["sizes"]).max() < n
    assert_consistent(payload, "D")
    assert payload["series"]["start_datetime"][0] != "2020-01-01"
    _assert_starts_are_tails(df, payload)


@pytest.mark.parametrize("endpoint", ["forecast", "cross_validation"])
def test_restrict_input_with_partitions(capture, endpoint):
    """Truncation and partitioning compose."""
    df = _series(n_series=6, n=500)
    _call(capture.client, endpoint, df, num_partitions=3)

    for _, payload in capture.payloads:
        assert_consistent(payload, "D")
        assert np.asarray(payload["series"]["sizes"]).max() < 500
        assert payload["series"]["start_datetime"][0] != "2020-01-01"
    assert len(_ordered_starts(capture)) == 6


def test_restrict_input_unsorted_and_truncated(capture):
    """The regression that motivated passing sort_idxs explicitly.

    _tail resets sort_idxs, so a shuffled input plus truncation used to map
    start times through the wrong row positions.
    """
    df = _series(n_series=3, n=500).sample(frac=1, random_state=0).reset_index(drop=True)
    capture.client.forecast(df=df, h=5, freq="D")
    _, payload = capture.payloads[0]
    assert_consistent(payload, "D")
    _assert_starts_are_tails(df, payload)


def test_forecast_add_history_reuses_start_datetime(capture):
    """The derived cross_validation call covers the same rows, so same starts."""
    capture.client.forecast(df=_series(n_series=3, n=60), h=5, freq="D", add_history=True)

    starts = {e: p["series"]["start_datetime"] for e, p in capture.payloads}
    assert "v2/forecast" in starts and "v2/cross_validation" in starts
    assert starts["v2/forecast"] == starts["v2/cross_validation"]


# ---------------------------------------------------------------------------
# Contract edges
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_integer_time_col_omits_start_datetime(capture, endpoint, integer_freq_series):
    """No timestamps exist, so the field is absent rather than inconsistent."""
    _call(capture.client, endpoint, integer_freq_series, freq=1)
    assert capture.payloads
    for _, payload in capture.payloads:
        assert "start_datetime" not in payload["series"]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_integer_time_col_with_partitions(capture, endpoint, integer_freq_series):
    if endpoint == "finetune":
        pytest.skip("finetune has no num_partitions argument")
    _call(capture.client, endpoint, integer_freq_series, freq=1, num_partitions=2)
    assert capture.payloads
    for _, payload in capture.payloads:
        assert "start_datetime" not in payload["series"]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_tz_aware_offset_reaches_payload(capture, endpoint):
    """datetime64 cannot hold an offset, so this pins the ISO-string decision."""
    df = _series(n_series=3, n=40, tz="Etc/GMT+5")
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}
    _call(capture.client, endpoint, df, **kw)
    for _, payload in capture.payloads:
        starts = payload["series"]["start_datetime"]
        assert all(s.endswith("-05:00") for s in starts), starts


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_dst_timezone_omits_start_datetime(capture, endpoint, caplog):
    """An offset alone cannot describe an IANA timezone across DST changes."""
    df = _series(
        n_series=3,
        n=40,
        freq="h",
        tz="US/Eastern",
        start="2020-03-07",
    )
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}
    _call(capture.client, endpoint, df, freq="h", **kw)
    for _, payload in capture.payloads:
        assert "start_datetime" not in payload["series"]
    assert "cannot be represented losslessly" in caplog.text


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_polars_input_matches_pandas(capture, endpoint):
    """The backend must not change the payload: polars emits the same starts."""
    pl = pytest.importorskip("polars")
    df = _staggered_series(4)
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}
    _call(capture.client, endpoint, df, **kw)
    pandas_starts = [p["series"]["start_datetime"] for _, p in capture.payloads]
    capture.payloads.clear()
    capture.parts.clear()
    _call(capture.client, endpoint, pl.from_pandas(df), freq="1d", **kw)
    polars_starts = [p["series"]["start_datetime"] for _, p in capture.payloads]
    assert polars_starts == pandas_starts


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_polars_tz_aware_offset_reaches_payload(capture, endpoint):
    """polars' to_numpy() drops the tz; the client must restore the offset."""
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(_series(n_series=3, n=40, tz="Etc/GMT+5"))
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}
    _call(capture.client, endpoint, df, freq="1d", **kw)
    for _, payload in capture.payloads:
        starts = payload["series"]["start_datetime"]
        assert all(s.endswith("-05:00") for s in starts), starts


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_polars_dst_timezone_omits_start_datetime(capture, endpoint, caplog):
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(
        _series(
            n_series=3,
            n=40,
            freq="h",
            tz="US/Eastern",
            start="2020-03-07",
        )
    )
    kw = NO_TRUNCATE if endpoint in ("forecast", "cross_validation") else {}
    _call(capture.client, endpoint, df, freq="1h", **kw)
    for _, payload in capture.payloads:
        assert "start_datetime" not in payload["series"]
    assert "cannot be represented losslessly" in caplog.text


def test_sub_daily_freq_keeps_time_component(capture):
    df = _series(n_series=2, n=40, freq="h", start="2020-01-01 09:30")
    capture.client.forecast(df=df, h=5, freq="h", **NO_TRUNCATE)
    starts = capture.payloads[0][1]["series"]["start_datetime"]
    assert all(s.startswith("2020-01-01T09:30") for s in starts), starts


def test_payload_is_json_serializable(capture):
    """start_datetime must survive the orjson path used by _make_request."""
    import orjson

    capture.client.forecast(
        df=_series(n_series=3, n=40, tz="Etc/GMT+5"), h=5, freq="D"
    )
    starts = capture.payloads[0][1]["series"]["start_datetime"]
    assert orjson.loads(orjson.dumps(starts)) == starts
