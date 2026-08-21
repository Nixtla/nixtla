"""Client-side codec for the `execute_step` endpoint.

`execute_step` runs one TSMP top-level API call server-side in a fresh sandbox. Unlike the other
endpoints it does not exchange JSON: the request body is a zip of `<key>.parquet` members and all
of its metadata rides in a `nixtla-metadata` header. The invariant is that the zip is data and the
header is metadata, so there is exactly one place to look for each.

Nothing here understands TSMP semantics. Tables returned by the server carry their resource
identity in arrow schema metadata, and this module passes that through untouched, which is what
makes chaining one step's output into the next lossless. That is also why pyarrow is required
rather than optional: a pandas round-trip drops schema metadata, so a chained call would silently
misread the previous step's output as an untyped table.
"""

import io
import json
import zipfile
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

METADATA_HEADER = "nixtla-metadata"
CONTENT_TYPE = "application/zip"
PARQUET_SUFFIX = ".parquet"
REF_KEY = "data_ref"

# Tightest common proxy limit (nginx large_client_header_buffers 8k, CloudFront 8k). The server
# rejects anything larger with 431 and deliberately does not spill metadata into the body, so the
# client checks the same budget to fail before uploading rather than after.
HEADER_BUDGET = 8192

_PYARROW_HINT = "submit_execute_step_job requires pyarrow. Install it with: pip install 'nixtla[steps]'"


def _require_pyarrow():
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
    return {REF_KEY: key}


def collect_refs(obj: Any) -> set[str]:
    """Every `data_ref` in a params tree, so a bad reference is caught before uploading."""
    if isinstance(obj, dict):
        if REF_KEY in obj:
            key = obj[REF_KEY]
            if not isinstance(key, str):
                raise ValueError(
                    f"{REF_KEY} must be a string, got {type(key).__name__}"
                )
            return {key}
        return set().union(*(collect_refs(v) for v in obj.values())) if obj else set()
    if isinstance(obj, (list, tuple)):
        return set().union(*(collect_refs(v) for v in obj)) if obj else set()
    return set()


def validate_member_names(names: Iterable[str]) -> None:
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
    schema metadata. pandas and polars frames go through narwhals, the compatibility layer the rest
    of the SDK already uses.
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


def pack(tables: dict[str, "pa.Table"]) -> bytes:
    """Serialize each table to a `<key>.parquet` member and zip them.

    Members are written in sorted order so the same data map always produces the same bytes.
    """
    _, pq = _require_pyarrow()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(tables):
            member = io.BytesIO()
            pq.write_table(tables[key], member)
            zf.writestr(key + PARQUET_SUFFIX, member.getvalue())
    return buf.getvalue()


def unpack(body: bytes) -> dict[str, "pa.Table"]:
    """Parse a response archive back into a data map keyed the way the caller will reference it."""
    _, pq = _require_pyarrow()
    tables: dict[str, "pa.Table"] = {}
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        for name in zf.namelist():
            if not name.endswith(PARQUET_SUFFIX):
                continue
            tables[name[: -len(PARQUET_SUFFIX)]] = pq.read_table(
                io.BytesIO(zf.read(name))
            )
    return tables


def encode_metadata(metadata: dict[str, Any]) -> str:
    """Serialize the request metadata, refusing anything that will not fit the header.

    `json.dumps` defaults to `ensure_ascii=True`, so the value is pure ASCII and therefore safe as
    an HTTP header. Metadata is never moved into the body: the archive holds tables and nothing
    else, so an oversized header is a bad request rather than something to route around.
    """
    payload = json.dumps(metadata)
    size = len(payload.encode())
    if size > HEADER_BUDGET:
        raise ValueError(
            f"{METADATA_HEADER} would be {size} bytes, over the {HEADER_BUDGET}-byte budget. "
            f"Shrink the request (a shorter SQL string, fewer models, fewer quantiles); metadata "
            f"is never moved into the body."
        )
    return payload


def decode_metadata(headers: Any) -> dict[str, Any]:
    """Read the response metadata header. HTTP header names are not case-sensitive."""
    for key, value in dict(headers).items():
        if key.lower() == METADATA_HEADER:
            return json.loads(value)
    return {}


class StepResult:
    """Result of an `execute_step` job.

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

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self):
        return self.data.keys()

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


def build_result(headers: Any, body: bytes) -> StepResult:
    """Assemble a `StepResult` from a raw `(headers, body)` response pair."""
    return StepResult(data=unpack(body), metadata=decode_metadata(headers))


def build_request(
    func_name: str,
    params: dict[str, Any],
    data: Optional[dict[str, Any]] = None,
    job_timeout_seconds: Optional[int] = None,
) -> tuple[str, bytes]:
    """Validate and encode one execute_step call into `(metadata_header, zip_body)`.

    Validation mirrors the server's own checks so a malformed request fails locally rather than
    after a full upload.
    """
    tables = {key: to_arrow(value) for key, value in (data or {}).items()}
    validate_member_names(tables)

    refs = collect_refs(params)
    missing = refs - set(tables)
    if missing:
        raise ValueError(
            f"params reference data keys that were not supplied: {sorted(missing)}; "
            f"supplied: {sorted(tables)}"
        )
    unused = set(tables) - refs
    if unused:
        raise ValueError(
            f"data contains tables no param references: {sorted(unused)}. "
            f"They would be uploaded and ignored."
        )

    metadata: dict[str, Any] = {
        "func_name": func_name,
        "params": params,
    }
    if job_timeout_seconds is not None:
        metadata["job_options"] = {"timeout_seconds": job_timeout_seconds}
    return encode_metadata(metadata), pack(tables)
