"""Client-side codec for the `execute_step` endpoint.

`execute_step` runs one TSMP top-level API call server-side. Unlike the other endpoints it does not
exchange JSON: the request body is a zip of `<key>.parquet` members and all of its metadata rides in
a `nixtla-metadata` header. The invariant is that the zip is data and the header is metadata, so
there is exactly one place to look for each.

Nothing here understands TSMP semantics. Tables returned by the server carry their resource
identity in arrow schema metadata, and this module passes that through untouched, which is what
makes chaining one step's output into the next lossless. That is also why this endpoint needs
pyarrow: a pandas round-trip drops schema metadata, so a chained call would silently misread
the previous step's output as an untyped table.
"""

import io
import json
import logging
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Optional

import narwhals as nw
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["StepResult", "ref"]

logger = logging.getLogger(__name__)

METADATA_HEADER = "nixtla-metadata"
CONTENT_TYPE = "application/zip"
_PARQUET_SUFFIX = ".parquet"
_REF_KEY = "data_ref"

# The largest metadata header the API accepts. Metadata is never moved into the body -- the archive
# holds tables and nothing else -- so an oversized header is a bad request rather than something to
# route around. Checked here to fail before uploading rather than after.
HEADER_BUDGET = 8192

# The other limits the API places on a request. Exceeding one of these is reported as a failed job
# rather than as an error when the job is submitted, so it would otherwise surface long after the
# call that caused it. Checking locally turns that back into an immediate ValueError.
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 512
MAX_METADATA_DEPTH = 32
MAX_FUNC_NAME_LENGTH = 128

# The zip format's minimum timestamp, stamped on every member so one data map always serializes to
# the same bytes. See `_pack`.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def ref(key: str) -> dict[str, str]:
    """Build the params envelope naming a table in the data map.

    Args:
        key (str): Key of the table in the `data` mapping.

    Returns:
        dict: `{"data_ref": key}`, the envelope `params` uses to reference a table.
    """
    return {_REF_KEY: key}


def _collect_refs(obj: Any) -> set[str]:
    """Every `data_ref` in a params tree, so a bad reference is caught before uploading.

    An envelope terminates the walk, exactly as it does server-side: the server replaces the whole
    dict with the one table `data_ref` names and never looks at its siblings. A ref nested under an
    envelope would therefore be silently ignored -- the table gets uploaded and the param it was
    meant for stays unset -- so it is rejected here rather than left to produce a quietly wrong
    result. Siblings that are not refs (the `resource`/`schema_expr` a step's own `result` envelope
    carries) are passed through untouched, so a result envelope can be fed straight back in.
    """
    if isinstance(obj, dict):
        if _REF_KEY in obj:
            key = obj[_REF_KEY]
            if not isinstance(key, str):
                raise ValueError(
                    f"{_REF_KEY} must be a string, got {type(key).__name__}"
                )
            ignored: set[str] = set()
            for name, value in obj.items():
                if name != _REF_KEY:
                    ignored |= _collect_refs(value)
            if ignored:
                raise ValueError(
                    f"the {_REF_KEY} envelope naming {key!r} also nests {sorted(ignored)}; the "
                    f"server replaces the whole envelope with that one table and never reads a "
                    f"nested reference. Pass the nested table as its own param instead."
                )
            return {key}
        return {r for value in obj.values() for r in _collect_refs(value)}
    if isinstance(obj, (list, tuple)):
        return {r for value in obj for r in _collect_refs(value)}
    return set()


def _validate_member_names(names: Iterable[str]) -> None:
    """Require every data-map key to be a bare name.

    Keys become archive member names, so a nested or absolute path would be a traversal attempt.
    The server enforces this too; checking here turns a post-upload 400 into a local error.
    """
    for name in names:
        if not name or name != name.strip():
            raise ValueError(f"invalid data key: {name!r}")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 1
            or "\\" in name
        ):
            raise ValueError(f"unsafe data key: {name!r} (must be a bare name)")


def _normalize_pandas_index(obj: Any, key: Optional[str] = None) -> Any:
    """Drop a pandas index that carries no meaning, and refuse one whose meaning is ambiguous.

    Arrow serializes any non-default index as a column, so a filtered frame would otherwise send a
    phantom `__index_level_0__`. This endpoint serializes the whole frame, so the exact column set
    is what the step runs on.

    A named level is therefore rejected rather than guessed at: the name says the values matter but
    not whether they should travel as a column, and guessing either fabricates one or loses one. An
    unnamed index makes no such claim and is dropped.

    A no-op for non-pandas input.
    """
    # Local import to keep pandas out of this module's import graph; it is a hard SDK dependency.
    import pandas as pd

    if not isinstance(obj, pd.DataFrame):
        return obj
    index = obj.index
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        # What arrow already stores as metadata rather than as a column; nothing to do.
        return obj
    named = [name for name in index.names if name is not None]
    if named:
        where = f"data[{key!r}]" if key is not None else "a data value"
        raise ValueError(
            f"{where} has a named index level {named[0]!r}, so it is unclear whether those values "
            f"should be sent as a column. Call .reset_index() to send them, or "
            f".reset_index(drop=True) to discard them."
        )
    return obj.reset_index(drop=True)


def to_arrow(obj: Any, key: Optional[str] = None) -> pa.Table:
    """Coerce a supported table-like object to a `pyarrow.Table`.

    A `pa.Table` is returned unchanged so that a table received from a previous step keeps its
    schema metadata. pandas and polars frames go through narwhals, which is already a hard
    dependency and reaches arrow in one hop.

    A pandas index is never sent as data -- see `_normalize_pandas_index`.

    Args:
        obj: The table-like object to convert.
        key (str, optional): The `data` key `obj` came from, used only to point error messages at
            the offending entry. Defaults to None.
    """
    if isinstance(obj, pa.Table):
        return obj

    try:
        frame = nw.from_native(_normalize_pandas_index(obj, key), eager_only=True)
    except TypeError as exc:
        raise TypeError(
            "execute_step data values must be pyarrow Tables or eager pandas/polars DataFrames, "
            f"got {type(obj).__name__}"
        ) from exc
    return frame.to_arrow()


def _pack(tables: dict[str, pa.Table]) -> bytes:
    """Serialize each table to a `<key>.parquet` member and zip them.

    Byte-for-byte reproducible: members are written in sorted order, and each carries `_ZIP_EPOCH`.
    A bare name would make `writestr` stamp `time.localtime()` into the member header, so one data
    map would serialize differently from one second to the next.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(tables):
            member = io.BytesIO()
            pq.write_table(tables[key], member)
            info = zipfile.ZipInfo(key + _PARQUET_SUFFIX, date_time=_ZIP_EPOCH)
            # Not inherited from the ZipFile when a ZipInfo is passed.
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, member.getvalue())
    return buf.getvalue()


def _unpack(body: bytes) -> dict[str, pa.Table]:
    """Parse a response archive back into a data map keyed the way the caller will reference it."""
    tables: dict[str, pa.Table] = {}
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        for name in zf.namelist():
            if not name.endswith(_PARQUET_SUFFIX):
                # Not a table; the archive is documented to hold nothing else, so say so
                # rather than returning a quietly incomplete result.
                logger.warning("ignoring unexpected member %r in step result", name)
                continue
            tables[name[: -len(_PARQUET_SUFFIX)]] = pq.read_table(
                io.BytesIO(zf.read(name))
            )
    return tables


def _check_metadata_depth(obj: Any) -> None:
    """Reject metadata nested deeper than the server will parse.

    Iterative on purpose: a recursive walk would hit the very limit it exists to enforce.
    """
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_METADATA_DEPTH:
            raise ValueError(
                f"{METADATA_HEADER} nests deeper than {MAX_METADATA_DEPTH} levels; flatten the "
                f"params."
            )
        if isinstance(node, dict):
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend((x, depth + 1) for x in node)


def _encode_metadata(metadata: dict[str, Any]) -> str:
    """Serialize the request metadata, refusing anything the server would not parse.

    `json.dumps` defaults to `ensure_ascii=True`, so the value is pure ASCII -- one byte per
    character, and safe as an HTTP header. Metadata is never moved into the body: the archive
    holds tables and nothing else, so an oversized header is a bad request rather than
    something to route around.
    """
    _check_metadata_depth(metadata)
    payload = json.dumps(metadata)
    size = len(payload)
    if size > HEADER_BUDGET:
        raise ValueError(
            f"{METADATA_HEADER} would be {size} bytes, over the {HEADER_BUDGET}-byte budget. "
            f"Shrink the request (a shorter SQL string, fewer models, fewer quantiles); metadata "
            f"is never moved into the body."
        )
    return payload


def _decode_metadata(headers: Mapping[str, str]) -> dict[str, Any]:
    """Read the response metadata header, or `{}` when the server sent none.

    A header that will not parse is warned about rather than raised on: a result whose tables came
    back intact should not be discarded because the metadata describing it is unusable. The tables
    are in `.data` either way.
    """
    value = headers.get(METADATA_HEADER)
    if not value:
        return {}
    try:
        metadata = json.loads(value)
    except json.JSONDecodeError:
        logger.warning(
            "ignoring unparseable %s header on step result: %r", METADATA_HEADER, value
        )
        return {}
    if not isinstance(metadata, dict):
        logger.warning(
            "ignoring %s header on step result: expected a JSON object, got %s",
            METADATA_HEADER,
            type(metadata).__name__,
        )
        return {}
    return metadata


class StepResult(Mapping):
    """Result of an `execute_step` job.

    A read-only mapping of table name to `pyarrow.Table`, so `res["result"]`, `len(res)`,
    iteration and `dict(res)` all work.

    Attributes:
        data (dict): Result tables keyed by name, as `pyarrow.Table`. Pass this straight back as
            the `data` argument of the next step to chain calls without losing the tables'
            resource identity.
        metadata (dict): What the server reported about the call — `func_name`, the `result`
            envelope, and a `profile` of the output when the result is a dataframe.
    """

    def __init__(self, *, data: dict[str, pa.Table], metadata: dict[str, Any]):
        self.data = data
        self.metadata = metadata

    def __getitem__(self, key: str) -> pa.Table:
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def to_pandas(self) -> dict[str, "pd.DataFrame"]:
        """Convert every result table to a pandas DataFrame.

        Convenience for inspecting a final result. Do not feed the output of this back into a
        chained step: the conversion drops the arrow schema metadata carrying the table's resource
        identity. Pass `.data` instead.
        """
        return {key: table.to_pandas() for key, table in self.data.items()}

    def __repr__(self) -> str:
        func_name = self.metadata.get("func_name", "?")
        shapes = ", ".join(
            f"{k}: {v.num_rows}x{v.num_columns}" for k, v in self.data.items()
        )
        return f"StepResult(func_name={func_name!r}, data={{{shapes}}})"


def build_result(headers: Mapping[str, str], body: bytes) -> StepResult:
    """Assemble a `StepResult` from a raw `(headers, body)` response pair."""
    return StepResult(data=_unpack(body), metadata=_decode_metadata(headers))


def build_request(
    func_name: str,
    params: dict[str, Any],
    data: Optional[dict[str, Any]] = None,
    job_timeout_seconds: Optional[int] = None,
) -> tuple[str, bytes]:
    """Validate and encode one execute_step call into `(metadata_header, zip_body)`.

    Validation mirrors the limits the API enforces so a malformed request fails locally rather than
    after a full upload, and runs cheapest-first: whatever `params` alone decides is settled before
    any table is converted, so a bad reference costs no arrow work.

    Only referenced tables are packed. The API ignores the rest, so uploading them would spend
    bandwidth and body-size budget for nothing -- which makes chaining on a previous step's whole
    `.data` map free.
    """
    if (
        not isinstance(func_name, str)
        or not 1 <= len(func_name) <= MAX_FUNC_NAME_LENGTH
    ):
        raise ValueError(
            f"func_name must be a string of 1 to {MAX_FUNC_NAME_LENGTH} characters, got "
            f"{func_name!r}"
        )
    if job_timeout_seconds is not None and job_timeout_seconds <= 0:
        raise ValueError(
            f"job_timeout_seconds must be positive, got {job_timeout_seconds!r}"
        )

    # Bounds the recursive `_collect_refs` below, so params nested past the limit (or referring to
    # themselves) raise a ValueError rather than a RecursionError.
    _check_metadata_depth(params)
    refs = _collect_refs(params)

    supplied = data or {}
    _validate_member_names(supplied)
    if len(supplied) > MAX_MEMBERS:
        raise ValueError(
            f"data holds {len(supplied)} tables, over the {MAX_MEMBERS} the server accepts. Send "
            f"fewer tables per request."
        )
    missing = refs - set(supplied)
    if missing:
        raise ValueError(
            f"params reference data keys that were not supplied: {sorted(missing)}; "
            f"supplied: {sorted(supplied)}"
        )

    tables = {key: to_arrow(supplied[key], key) for key in refs}

    metadata: dict[str, Any] = {
        "func_name": func_name,
        "params": params,
    }
    if job_timeout_seconds is not None:
        metadata["job_options"] = {"timeout_seconds": job_timeout_seconds}
    header = _encode_metadata(metadata)

    body = _pack(tables)
    if len(body) > MAX_BODY_BYTES:
        raise ValueError(
            f"the request body is {len(body)} bytes, over the {MAX_BODY_BYTES}-byte limit the "
            f"server accepts. Split the request into smaller batches (fewer ids, a shorter "
            f"history)."
        )
    return header, body
