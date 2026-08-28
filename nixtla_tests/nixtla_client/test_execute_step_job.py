"""Tests for `submit_execute_step_job` and the `nixtla.steps` codec.

These live apart from `test_async_jobs.py` rather than joining its `SUBMIT_JOB_CASES` /
`WAIT_JOB_CASES` tables: those tables monkeypatch `NixtlaClient._submit_job`, and execute_step goes
through `_submit_binary_job` instead because its request is a zip body plus a header rather than a
JSON payload. Folding it in would mean branching inside the shared tests. The three shared
behaviours (returns a Job, wait returns the result, job_timeout_seconds is threaded through) are
reproduced here for the binary path.

Fully mocked, like `test_async_jobs.py` — no network.
"""

import json
import logging
import zipfile
from io import BytesIO
from unittest.mock import MagicMock

import httpx
import orjson
import pandas as pd
import pyarrow as pa
import pytest

from nixtla.nixtla_client import ApiError, Job, NixtlaClient, _is_retriable_error
from nixtla.steps import (
    CONTENT_TYPE,
    HEADER_BUDGET,
    MAX_MEMBERS,
    MAX_METADATA_DEPTH,
    METADATA_HEADER,
    StepResult,
    _collect_refs,
    _pack,
    _unpack,
    build_request,
    build_result,
    ref,
    to_arrow,
)


def _client(**kwargs):
    return NixtlaClient(api_key="dummy", **kwargs)


def _mock_json_response(status_code, body):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = orjson.dumps(body)
    return resp


def _stub_submit_binary(monkeypatch, job_id="es-1", capture=None):
    """Make `_submit_binary_job` succeed without HTTP, optionally recording its args."""

    def fake_submit(self, client, endpoint, metadata, body):
        if capture is not None:
            capture.append({"endpoint": endpoint, "metadata": metadata, "body": body})
        return job_id

    monkeypatch.setattr(NixtlaClient, "_submit_binary_job", fake_submit)


def _small_df(n=5):
    return pd.DataFrame(
        {
            "unique_id": "id_0",
            "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
            "y": range(n),
        }
    )


def _tagged_table():
    """A table carrying the resource identity the server stamps into schema metadata."""
    return pa.table({"a": [1, 2, 3]}).replace_schema_metadata(
        {b"tsmp_resource_meta": b'{"resource": "arrowtsi", "freq": "D"}'}
    )


def _call_kwargs(**overrides):
    kwargs = {
        "func_name": "make_forecast_input",
        "params": {"data": ref("panel"), "freq": "D"},
        "data": {"panel": _small_df()},
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------


class TestCodec:
    def test_pack_unpack_round_trip(self):
        tables = {"a": pa.table({"x": [1, 2]}), "b": pa.table({"y": ["p", "q"]})}
        restored = _unpack(_pack(tables))
        assert set(restored) == {"a", "b"}
        assert restored["a"].column("x").to_pylist() == [1, 2]

    def test_schema_metadata_survives(self):
        # The whole reason pyarrow is required: a pandas round-trip would drop this, and a
        # chained step would then misread the previous step's output.
        table = _tagged_table()
        restored = _unpack(_pack({"t": table}))["t"]
        assert restored.schema.metadata[b"tsmp_resource_meta"] == (
            b'{"resource": "arrowtsi", "freq": "D"}'
        )

    def test_members_are_named_for_their_key(self):
        with zipfile.ZipFile(BytesIO(_pack({"sales": pa.table({"x": [1]})}))) as zf:
            assert zf.namelist() == ["sales.parquet"]

    def test_layout_is_deterministic(self):
        tables = {"b": pa.table({"x": [1]}), "a": pa.table({"y": [2]})}
        assert _pack(tables) == _pack({"a": tables["a"], "b": tables["b"]})

    def test_empty_data_map(self):
        assert _unpack(_pack({})) == {}

    def test_to_arrow_passes_a_table_through_untouched(self):
        table = _tagged_table()
        assert to_arrow(table) is table

    def test_to_arrow_converts_pandas(self):
        assert to_arrow(_small_df()).num_rows == 5

    def test_to_arrow_converts_polars(self):
        pl = pytest.importorskip("polars")
        assert to_arrow(pl.DataFrame({"x": [1, 2, 3]})).num_rows == 3

    def test_to_arrow_rejects_unsupported_input(self):
        with pytest.raises(TypeError, match="pyarrow Tables or eager pandas/polars"):
            to_arrow([1, 2, 3])

    def test_to_arrow_drops_a_positional_pandas_index(self):
        # Arrow serializes any non-default index as a column, so an ordinary filter would
        # otherwise upload a phantom `__index_level_0__` that TSMP rebuilds the resource around.
        filtered = _small_df()[lambda df: df["y"] > 1]
        assert list(filtered.index) != list(range(len(filtered)))
        assert to_arrow(filtered).column_names == ["unique_id", "ds", "y"]

    def test_to_arrow_folds_a_named_pandas_index_into_a_column(self):
        # Dropping it would silently cost the caller the id column they meant to send.
        indexed = _small_df().set_index("unique_id")
        assert set(to_arrow(indexed).column_names) == {"unique_id", "ds", "y"}

    def test_to_arrow_leaves_a_default_range_index_alone(self):
        assert to_arrow(_small_df()).column_names == ["unique_id", "ds", "y"]

    def test_collect_refs_finds_every_depth(self):
        params = {
            "resource": ref("a"),
            "extras": [ref("b"), {"nested": ref("c")}],
            "h": 7,
        }
        assert _collect_refs(params) == {"a", "b", "c"}

    def test_collect_refs_rejects_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            _collect_refs({"data_ref": {"nested": "evil"}})

    def test_collect_refs_stops_at_an_envelope(self):
        # The server replaces the whole envelope with the one table `data_ref` names, so a step's
        # own `result` envelope -- which carries `resource`/`schema_expr` siblings -- feeds straight
        # back in without those being mistaken for anything.
        envelope = {
            "data_ref": "result",
            "resource": "arrowtsi",
            "schema_expr": {"y": "f64"},
        }
        assert _collect_refs({"resource": envelope}) == {"result"}

    def test_collect_refs_rejects_a_ref_nested_under_an_envelope(self):
        # The server never reads it, so allowing it through would upload `deep` and leave the
        # param it was meant for unset -- a quietly wrong result rather than an error.
        with pytest.raises(ValueError, match="never reads a nested reference"):
            _collect_refs({"data": {**ref("panel"), "opts": ref("deep")}})

    def test_build_result_reads_header_case_insensitively(self):
        # HTTP header names are not case-sensitive and proxies rewrite their case freely.
        # `httpx.Headers` is what the client hands over, and it handles the folding.
        res = build_result(
            httpx.Headers({"Nixtla-Metadata": '{"func_name": "forecast"}'}),
            _pack({"r": _tagged_table()}),
        )
        assert res.metadata["func_name"] == "forecast"

    def test_build_result_without_a_metadata_header(self):
        res = build_result(httpx.Headers({}), _pack({"r": _tagged_table()}))
        assert res.metadata == {}
        assert res["r"].num_rows == 3

    @pytest.mark.parametrize("value", ["not json at all", '["a", "list"]'])
    def test_build_result_tolerates_an_unusable_metadata_header(self, value, caplog):
        # A result whose tables came back intact should not be discarded because the metadata
        # describing it is unusable -- the caller can still read everything from `.data`.
        with caplog.at_level(logging.WARNING, logger="nixtla.steps"):
            res = build_result(
                httpx.Headers({METADATA_HEADER: value}), _pack({"r": _tagged_table()})
            )

        assert res.metadata == {}
        assert res["r"].num_rows == 3
        assert METADATA_HEADER in caplog.text

    def test_unpack_warns_about_a_member_that_is_not_a_table(self, caplog):
        packed = _pack({"result": pa.table({"x": [1]})})
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            with zipfile.ZipFile(BytesIO(packed)) as src:
                zf.writestr("result.parquet", src.read("result.parquet"))
            zf.writestr("manifest.json", b"{}")

        with caplog.at_level(logging.WARNING, logger="nixtla.steps"):
            tables = _unpack(buf.getvalue())

        assert set(tables) == {"result"}
        assert "manifest.json" in caplog.text

    def test_step_result_accessors(self):
        res = StepResult(
            data={"result": _tagged_table()}, metadata={"func_name": "forecast"}
        )
        assert "result" in res
        assert list(res.keys()) == ["result"]
        assert res["result"].num_rows == 3
        assert isinstance(res.to_pandas()["result"], pd.DataFrame)
        assert "forecast" in repr(res)


# ---------------------------------------------------------------------------
# client-side validation, before anything is uploaded
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.fixture(autouse=True)
    def no_http(self, monkeypatch):
        """Any HTTP attempt is a test failure: these must all fail locally."""

        def boom(*args, **kwargs):
            raise AssertionError("validation should fail before any HTTP call")

        monkeypatch.setattr(NixtlaClient, "_submit_binary_job", boom)

    def test_rejects_a_ref_with_no_table(self):
        with pytest.raises(ValueError, match="not supplied"):
            _client().submit_execute_step_job(
                **_call_kwargs(params={"data": ref("absent")})
            )

    def test_rejects_a_ref_nested_under_an_envelope(self):
        # `decode` replaces the envelope with the table `data_ref` names and never recurses into
        # its siblings, so accepting this would upload `deep` and leave `opts` unset server-side.
        with pytest.raises(ValueError, match=r"\['deep'\]"):
            _client().submit_execute_step_job(
                **_call_kwargs(
                    params={"data": {**ref("panel"), "opts": ref("deep")}},
                    data={"panel": _small_df(), "deep": _small_df()},
                )
            )

    @pytest.mark.parametrize("func_name", ["", "f" * 129])
    def test_rejects_a_func_name_outside_the_servers_bounds(self, func_name):
        with pytest.raises(ValueError, match="func_name must be a string"):
            _client().submit_execute_step_job(**_call_kwargs(func_name=func_name))

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_rejects_a_non_positive_job_timeout(self, timeout):
        with pytest.raises(ValueError, match="job_timeout_seconds must be positive"):
            _client().submit_execute_step_job(
                **_call_kwargs(job_timeout_seconds=timeout)
            )

    def test_rejects_more_tables_than_the_server_accepts(self):
        data = {f"t{i}": _small_df(1) for i in range(MAX_MEMBERS + 1)}
        with pytest.raises(ValueError, match=f"over the {MAX_MEMBERS}"):
            _client().submit_execute_step_job(
                **_call_kwargs(params={"data": ref("t0")}, data=data)
            )

    def test_rejects_metadata_nested_deeper_than_the_server_parses(self):
        deep = inner = {}
        for _ in range(MAX_METADATA_DEPTH + 2):
            inner["k"] = inner = {}
        with pytest.raises(ValueError, match=f"deeper than {MAX_METADATA_DEPTH}"):
            _client().submit_execute_step_job(
                **_call_kwargs(params={"data": ref("panel"), "deep": deep})
            )

    def test_rejects_a_body_over_the_server_limit(self, monkeypatch):
        # Without this the request is accepted and only reported as a failed job later, so the
        # error would surface long after the call that caused it.
        monkeypatch.setattr("nixtla.steps.MAX_BODY_BYTES", 128)
        with pytest.raises(ValueError, match="over the 128-byte limit"):
            _client().submit_execute_step_job(**_call_kwargs())

    @pytest.mark.parametrize("key", ["../evil", "/abs", "nested/path", "", " lead"])
    def test_rejects_unsafe_data_keys(self, key):
        with pytest.raises(ValueError, match="data key"):
            _client().submit_execute_step_job(
                **_call_kwargs(params={"data": ref(key)}, data={key: _small_df()})
            )

    def test_rejects_metadata_over_the_header_budget(self):
        # The server answers 431 and deliberately does not spill metadata into the body.
        with pytest.raises(ValueError, match="over the .* budget"):
            _client().submit_execute_step_job(
                **_call_kwargs(
                    params={"data": ref("panel"), "sql": "x" * (HEADER_BUDGET + 1)}
                )
            )

    def test_accepts_a_call_with_no_data(self, monkeypatch):
        _stub_submit_binary(monkeypatch)
        job = _client().submit_execute_step_job(
            func_name="select_by_sql", params={"query": "SELECT 1"}
        )
        assert job.job_id == "es-1"


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_returns_a_job_for_the_execute_step_endpoint(self, monkeypatch):
        calls = []

        def fake_submit(self, client, endpoint, metadata, body):
            calls.append(endpoint)
            return "es-abc123"

        monkeypatch.setattr(NixtlaClient, "_submit_binary_job", fake_submit)
        monkeypatch.setattr(
            NixtlaClient, "_get_job_data", lambda self, c, e, j: {"status": "pending"}
        )

        job = _client().submit_execute_step_job(**_call_kwargs())

        assert isinstance(job, Job)
        assert job.job_id == "es-abc123"
        assert job.status == "pending"
        assert calls == ["v2/execute_step"]

    def test_sends_zip_body_and_metadata_header_uncompressed(self):
        http_client = MagicMock()
        http_client.post.return_value = _mock_json_response(202, {"job_id": "es-1"})

        metadata, body = build_request(**_call_kwargs())
        job_id = _client()._submit_binary_job(
            http_client, "v2/execute_step", metadata, body
        )

        assert job_id == "es-1"
        _, kwargs = http_client.post.call_args
        assert kwargs["url"] == "v2/execute_step/async"
        assert kwargs["headers"]["content-type"] == CONTENT_TYPE
        # Overrides the client-level application/json default.
        assert kwargs["headers"][METADATA_HEADER] == metadata
        # The zip is already deflated, so it must not be zstd-compressed on top.
        assert "content-encoding" not in kwargs["headers"]
        assert kwargs["content"] is body
        assert kwargs["content"][:2] == b"PK"

    def test_metadata_carries_only_func_name_and_params(self):
        # No `model`: the step names its models in `params`, and the server rejects an unknown
        # metadata key outright.
        metadata, _ = build_request(**_call_kwargs())
        decoded = json.loads(metadata)
        assert decoded["func_name"] == "make_forecast_input"
        assert decoded["params"]["data"] == {"data_ref": "panel"}
        assert "model" not in decoded
        assert "job_options" not in decoded

    def test_job_timeout_seconds_goes_into_the_header_not_the_body(self, monkeypatch):
        # Unlike the JSON tasks, this task's request metadata travels in a header. Goes
        # through the public method so it also covers the client forwarding the argument.
        sent = []
        _stub_submit_binary(monkeypatch, capture=sent)

        _client().submit_execute_step_job(**_call_kwargs(), job_timeout_seconds=120)

        assert json.loads(sent[0]["metadata"])["job_options"] == {
            "timeout_seconds": 120
        }
        assert sent[0]["body"][:2] == b"PK"

    def test_an_unreferenced_table_warns_and_is_still_sent(self, monkeypatch, caplog):
        # Chaining passes a whole `.data` map, and a step can return more tables than the
        # next one consumes -- warn, don't reject the call.
        _stub_submit_binary(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="nixtla.steps"):
            job = _client().submit_execute_step_job(
                **_call_kwargs(data={"panel": _small_df(), "spare": _small_df()})
            )
        assert job.job_id == "es-1"
        assert "spare" in caplog.text

    def test_raises_api_error_on_a_bad_status(self):
        http_client = MagicMock()
        http_client.post.return_value = _mock_json_response(422, {"detail": "nope"})
        with pytest.raises(ApiError) as exc:
            _client()._submit_binary_job(http_client, "v2/execute_step", "{}", b"PK")
        assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# result retrieval
# ---------------------------------------------------------------------------


class TestResult:
    def test_wait_takes_the_fetch_result_path(self, monkeypatch):
        body = _pack({"result": _tagged_table()})
        headers = {
            METADATA_HEADER: '{"func_name": "make_forecast_input", "profile": {"num_rows": 3}}'
        }

        _stub_submit_binary(monkeypatch)
        # The status response carries no result for a binary task; it must not be parsed.
        monkeypatch.setattr(
            NixtlaClient,
            "_poll_job",
            lambda self, c, e, j, pi, pt: {"status": "succeeded", "result": None},
        )
        monkeypatch.setattr(
            NixtlaClient,
            "_get_job_result_bytes",
            lambda self, client, endpoint, job_id: (headers, body),
        )

        job = _client().submit_execute_step_job(**_call_kwargs())
        result = job.wait(poll_interval=0, poll_timeout=1)

        assert isinstance(result, StepResult)
        assert result["result"].num_rows == 3
        assert result.metadata["profile"]["num_rows"] == 3
        assert job.status == "succeeded"
        assert job.result is result

    def test_result_tables_can_be_chained_back_in(self, monkeypatch):
        body = _pack({"result": _tagged_table()})
        _stub_submit_binary(monkeypatch)
        monkeypatch.setattr(
            NixtlaClient,
            "_poll_job",
            lambda self, c, e, j, pi, pt: {"status": "succeeded"},
        )
        monkeypatch.setattr(
            NixtlaClient,
            "_get_job_result_bytes",
            lambda self, client, endpoint, job_id: ({}, body),
        )

        res = _client().submit_execute_step_job(**_call_kwargs()).wait(poll_interval=0)

        # Feeding `.data` straight back must preserve the resource tag end to end.
        _, chained_body = build_request(
            "forecast", {"resource": ref("result")}, res.data
        )
        assert _unpack(chained_body)["result"].schema.metadata[
            b"tsmp_resource_meta"
        ] == (b'{"resource": "arrowtsi", "freq": "D"}')

    def test_result_endpoint_hits_the_right_url(self):
        http_client = MagicMock()
        http_client.get.return_value = MagicMock(
            status_code=200, headers={}, content=b"PK"
        )
        _client()._get_job_result_bytes(http_client, "v2/execute_step", "es-1")
        http_client.get.assert_called_once_with("v2/execute_step/jobs/es-1/result")

    def test_a_bad_status_from_the_result_endpoint_raises(self):
        http_client = MagicMock()
        resp = MagicMock(status_code=422)
        resp.json.return_value = {"detail": "nope"}
        http_client.get.return_value = resp
        with pytest.raises(ApiError) as exc:
            _client()._get_job_result_bytes(http_client, "v2/execute_step", "es-1")
        assert exc.value.status_code == 422

    @pytest.mark.parametrize("not_ready_status", [202, 409])
    def test_a_not_ready_response_raises_a_retriable_error(self, not_ready_status):
        # Goes through the real status check rather than a hand-made ApiError: the retry above
        # only helps if `_get_job_result_bytes` actually raises on this status *and*
        # `_is_retriable_error` accepts what it raised. Together these pin the wire contract.
        http_client = MagicMock()
        resp = MagicMock(status_code=not_ready_status, content=b"PK")
        resp.json.return_value = {"detail": "result not ready, poll again"}
        http_client.get.return_value = resp

        with pytest.raises(ApiError) as exc:
            _client()._get_job_result_bytes(http_client, "v2/execute_step", "es-1")

        assert exc.value.status_code == not_ready_status
        assert _is_retriable_error(exc.value)

    @pytest.mark.parametrize("not_ready_status", [202, 409])
    def test_not_ready_from_the_result_endpoint_is_retried(
        self, monkeypatch, not_ready_status
    ):
        # The job says succeeded but the zip has not materialized yet: giving up here would
        # discard work the server has already done and billed. 202 is what current servers
        # answer, 409 what older ones did, and both must keep the retry going.
        body = _pack({"result": _tagged_table()})
        attempts = []

        def flaky_result(self, client, endpoint, job_id):
            attempts.append(job_id)
            if len(attempts) == 1:
                raise ApiError(
                    status_code=not_ready_status, body={"detail": "not succeeded"}
                )
            return {}, body

        _stub_submit_binary(monkeypatch)
        monkeypatch.setattr(
            NixtlaClient,
            "_poll_job",
            lambda self, c, e, j, pi, pt: {"status": "succeeded", "result": None},
        )
        monkeypatch.setattr(NixtlaClient, "_get_job_result_bytes", flaky_result)

        res = (
            _client(retry_interval=0)
            .submit_execute_step_job(**_call_kwargs())
            .wait(poll_interval=0)
        )

        assert len(attempts) == 2
        assert res["result"].num_rows == 3


# ---------------------------------------------------------------------------
# the shared Job surface still works for this job type
# ---------------------------------------------------------------------------


class TestJobSurface:
    def test_cancel_uses_the_task_agnostic_endpoint(self, monkeypatch):
        calls = []
        _stub_submit_binary(monkeypatch)
        monkeypatch.setattr(
            NixtlaClient,
            "_cancel_job",
            lambda self, client, job_id: calls.append(job_id),
        )

        job = _client().submit_execute_step_job(**_call_kwargs())
        job.cancel()

        assert calls == ["es-1"]
        assert job.status == "cancelled"

    def test_json_jobs_still_take_the_parse_result_path(self, monkeypatch):
        """Regression guard on the `fetch_result` hook added to `Job`."""
        monkeypatch.setattr(
            NixtlaClient,
            "_submit_job",
            lambda self, c, e, p, multithreaded_compress=True: "ft-1",
        )
        monkeypatch.setattr(
            NixtlaClient,
            "_poll_job",
            lambda self, c, e, j, pi, pt: {
                "status": "succeeded",
                "result": {"finetuned_model_id": "model-abc"},
            },
        )

        job = _client().submit_finetune_job(df=_small_df(n=20), freq="D")

        assert job.wait(poll_interval=0, poll_timeout=1) == "model-abc"
