"""Client-side codec for the `execute_step` endpoint.

`execute_step` runs one TSMP top-level API call server-side in a fresh sandbox. Unlike the other
endpoints it does not exchange JSON: the request body is a zip of `<key>.parquet` members and all
of its metadata rides in a `nixtla-metadata` header. The invariant is that the zip is data and the
header is metadata, so there is exactly one place to look for each.

Nothing here understands TSMP semantics. Tables returned by the server carry their resource
identity in arrow schema metadata, and this module passes that through untouched, which is what
makes chaining one step's output into the next lossless. That is also why this endpoint needs
pyarrow at all, and hence the `nixtla[steps]` extra: a pandas round-trip drops schema metadata,
so a chained call would silently misread the previous step's output as an untyped table.
"""

import io
import json
import logging
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

__all__ = ["StepResult", "ref"]

logger = logging.getLogger(__name__)

METADATA_HEADER = "nixtla-metadata"
CONTENT_TYPE = "application/zip"
_PARQUET_SUFFIX = ".parquet"
_REF_KEY = "data_ref"

# Tightest common proxy limit (nginx large_client_header_buffers 8k, CloudFront 8k). The server
# rejects anything larger with 431 and deliberately does not spill metadata into the body, so the
# client checks the same budget to fail before uploading rather than after.
HEADER_BUDGET = 8192

_PYARROW_HINT = (
    "execute_step requires pyarrow. Install it with: pip install 'nixtla[steps]'"
)


def _require_pyarrow() -> tuple[Any, Any]:
    """Import pyarrow on demand so the rest of the SDK stays installable without it."""
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via the error path only
        raise ImportError(_PYARROW_HINT) from exc
    return pa, pq


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

    An envelope's siblings are still walked: a param can hold both a `data_ref` and nested
    params that reference further tables, and missing one of those would reject a valid call.
    """
    refs: set[str] = set()
    if isinstance(obj, dict):
        if _REF_KEY in obj:
            key = obj[_REF_KEY]
            if not isinstance(key, str):
                raise ValueError(
                    f"{_REF_KEY} must be a string, got {type(key).__name__}"
                )
            refs.add(key)
        for value in obj.values():
            refs |= _collect_refs(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            refs |= _collect_refs(value)
    return refs


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


def to_arrow(obj: Any) -> "pa.Table":
    """Coerce a supported table-like object to a `pyarrow.Table`.

    A `pa.Table` is returned unchanged so that a table received from a previous step keeps its
    schema metadata. pandas and polars frames go through narwhals, which is already a hard
    dependency and reaches arrow in one hop; the rest of the client converts via utilsforecast.
    """
    pa, _ = _require_pyarrow()
    if isinstance(obj, pa.Table):
        return obj

    import narwhals as nw

    try:
        frame = nw.from_native(obj, eager_only=True)
    except TypeError as exc:
        raise TypeError(
            "execute_step data values must be pyarrow Tables or eager pandas/polars DataFrames, "
            f"got {type(obj).__name__}"
        ) from exc
    return frame.to_arrow()


def _pack(tables: dict[str, "pa.Table"]) -> bytes:
    """Serialize each table to a `<key>.parquet` member and zip them.

    Members are written in sorted order so the same data map always produces the same bytes.
    """
    _, pq = _require_pyarrow()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(tables):
            member = io.BytesIO()
            pq.write_table(tables[key], member)
            zf.writestr(key + _PARQUET_SUFFIX, member.getvalue())
    return buf.getvalue()


def _unpack(body: bytes) -> dict[str, "pa.Table"]:
    """Parse a response archive back into a data map keyed the way the caller will reference it."""
    _, pq = _require_pyarrow()
    tables: dict[str, "pa.Table"] = {}
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


def _encode_metadata(metadata: dict[str, Any]) -> str:
    """Serialize the request metadata, refusing anything that will not fit the header.

    `json.dumps` defaults to `ensure_ascii=True`, so the value is pure ASCII -- one byte per
    character, and safe as an HTTP header. Metadata is never moved into the body: the archive
    holds tables and nothing else, so an oversized header is a bad request rather than
    something to route around.
    """
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
    """Read the response metadata header, or `{}` when the server sent none."""
    value = headers.get(METADATA_HEADER)
    return json.loads(value) if value else {}


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

    def __init__(self, *, data: dict[str, "pa.Table"], metadata: dict[str, Any]):
        self.data = data
        self.metadata = metadata

    def __getitem__(self, key: str) -> "pa.Table":
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

    Validation mirrors the server's own checks so a malformed request fails locally rather than
    after a full upload. A table no param references is only warned about: passing a previous
    step's whole `.data` map is the intended way to chain calls, and a step can return more
    tables than the next one consumes.
    """
    tables = {key: to_arrow(value) for key, value in (data or {}).items()}
    _validate_member_names(tables)

    refs = _collect_refs(params)
    missing = refs - set(tables)
    if missing:
        raise ValueError(
            f"params reference data keys that were not supplied: {sorted(missing)}; "
            f"supplied: {sorted(tables)}"
        )
    unused = set(tables) - refs
    if unused:
        logger.warning(
            "data contains tables no param references: %s; they will be uploaded and ignored",
            sorted(unused),
        )

    metadata: dict[str, Any] = {
        "func_name": func_name,
        "params": params,
    }
    if job_timeout_seconds is not None:
        metadata["job_options"] = {"timeout_seconds": job_timeout_seconds}
    return _encode_metadata(metadata), _pack(tables)
