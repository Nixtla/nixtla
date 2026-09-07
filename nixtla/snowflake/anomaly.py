"""Plot Snowflake TimeGPT anomaly results with ``nixtla_client``.

``NIXTLA_DETECT_ANOMALIES`` returns the columns::

    UNIQUE_ID  DS  Y  TIMEGPT  ANOMALY  TIMEGPT_LO  TIMEGPT_HI

which differ from what ``NixtlaClient.detect_anomalies`` returns in three ways:
Snowflake uppercases the names, ``ANOMALY`` comes back as a string, and the
level is stripped from the interval columns (the client names them
``TimeGPT-lo-99`` / ``TimeGPT-hi-99``). Because the level is gone from the
data, it has to be supplied by the caller -- it is the level the SQL ran at.

Renaming is all that is needed. ``nixtla_client.plot`` recognises an
``anomaly`` column and reads the level, the flags and the model name straight
off the frame, so no plotting wrapper is required here.

Examples
--------
>>> from anomaly import detect_anomalies, load_actuals
>>> table = "DEMO.PUBLIC.EXAMPLE_ANOMALY_DATA"
>>> actuals = load_actuals(session, table)
>>> nixtla_client.plot(actuals)
>>> anomalies = detect_anomalies(session, table, level=99, freq="D")
>>> nixtla_client.plot(actuals, anomalies)

For a result you obtained yourself -- a SQL cell's ``dataframe_1``, say -- skip
``detect_anomalies`` and rename in place:

>>> nixtla_client.plot(actuals, to_anomalies_df(dataframe_1, level=99))

"""

from __future__ import annotations

import json
from typing import Any, Union

import pandas as pd

# Union (not `|`) because these aliases are evaluated at runtime and
# warehouse-runtime notebooks default to Python 3.9.
Level = Union[int, float]  # noqa: UP007

_TRUTHY = {"true", "t", "1", "yes"}
_ACTUALS = ["unique_id", "ds", "y"]
_PROCEDURE = "NIXTLA_DETECT_ANOMALIES"


def _to_pandas(result: Any) -> pd.DataFrame:
    """Convert a Snowpark result to pandas and strip quoted identifiers.

    Snowpark leaves the literal quotes on quoted column names, so ``DS`` can
    arrive as ``'"DS"'``.
    """
    df = result.to_pandas() if hasattr(result, "to_pandas") else result.copy()
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]
    return df


def _finalize(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Check the renamed columns, type ``ds``, and sort by series and time."""
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"missing {missing}; got {list(df.columns)}")
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values(["unique_id", "ds"], ignore_index=True)


def _qualify(session: Any, procedure: str | None) -> str:
    """Qualify the procedure with the session's current database and schema."""
    if procedure is not None:
        return procedure
    database, schema = session.get_current_database(), session.get_current_schema()
    if not (database and schema):
        raise ValueError(
            "the session has no current database and schema, so the procedure "
            f"cannot be located; pass procedure='<db>.<schema>.{_PROCEDURE}'"
        )
    # Snowpark returns both already quoted, which keeps a lowercase name intact.
    return f"{database}.{schema}.{_PROCEDURE}"


def load_actuals(session: Any, table: str) -> pd.DataFrame:
    """Read the input table as actuals, named the way ``plot`` expects.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session.
    table : str
        Fully qualified table name.

    Returns
    -------
    pandas.DataFrame
        Columns ``unique_id``, ``ds``, ``y``, sorted by series and time.

    """
    df = _to_pandas(session.table(table))
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return _finalize(df, _ACTUALS)


def to_anomalies_df(result: Any, level: Level) -> pd.DataFrame:
    """Rename a ``NIXTLA_DETECT_ANOMALIES`` result to the client's own names.

    Parameters
    ----------
    result : snowflake.snowpark.DataFrame or pandas.DataFrame
        The procedure's output, however you obtained it.
    level : int or float
        The level the SQL ran at. The procedure drops it from the interval
        column names, so it cannot be recovered from the data; passing the
        wrong one mislabels the band that gets plotted.

    Returns
    -------
    pandas.DataFrame
        Columns ``unique_id``, ``ds``, ``y``, ``TimeGPT``, ``anomaly`` (bool)
        and ``TimeGPT-lo-<level>`` / ``TimeGPT-hi-<level>``.

    Raises
    ------
    KeyError
        If the frame is not a detection result -- usually a sign the procedure
        returned a status row rather than its table.

    """
    renames = {
        "UNIQUE_ID": "unique_id",
        "DS": "ds",
        "Y": "y",
        "TIMEGPT": "TimeGPT",
        "ANOMALY": "anomaly",
        "TIMEGPT_LO": f"TimeGPT-lo-{level}",
        "TIMEGPT_HI": f"TimeGPT-hi-{level}",
    }
    df = _to_pandas(result)
    df = df.rename(
        columns={c: renames[c.upper()] for c in df.columns if c.upper() in renames}
    )
    bounds = [renames["TIMEGPT_LO"], renames["TIMEGPT_HI"]]
    df = _finalize(df, [*_ACTUALS, "TimeGPT", "anomaly", *bounds])
    df["anomaly"] = df["anomaly"].astype(str).str.strip().str.lower().isin(_TRUTHY)
    return df


def detect_anomalies(
    session: Any,
    table: str,
    level: Level = 99,
    *,
    procedure: str | None = None,
    **params: Any,
) -> pd.DataFrame:
    """Run ``NIXTLA_DETECT_ANOMALIES`` and return a client-shaped frame.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session.
    table : str
        Fully qualified name of the input table.
    level : int or float, default 99
        Confidence level. Whole floats are sent as integers so that the
        procedure's interval columns come back populated.
    procedure : str, optional
        Fully qualified procedure name. Defaults to ``NIXTLA_DETECT_ANOMALIES``
        in the session's current database and schema; pass it explicitly when
        the procedure was installed somewhere other than where the notebook is
        pointed.
    **params
        Further entries for the procedure's ``PARAMS`` object, e.g. ``freq``,
        ``model`` or ``finetuned_model_id``.

    Returns
    -------
    pandas.DataFrame
        Ready for ``nixtla_client.plot(actuals, anomalies)``.

    """
    if isinstance(level, float) and level.is_integer():
        level = int(level)
    payload = json.dumps({"level": level, **params}).replace("'", "''")
    sql = (
        f"CALL {_qualify(session, procedure)}("
        f"INPUT_DATA => '{table}', PARAMS => PARSE_JSON('{payload}'))"
    )
    try:
        raw = session.sql(sql).to_pandas()
    except Exception:
        # A procedure returning a table sometimes has to be read back out of
        # the result cache instead. This re-runs the CALL, so detection is
        # billed twice on the (rare) attempts that land here.
        session.sql(sql).collect()
        raw = session.sql(
            "SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))"
        ).to_pandas()
    return to_anomalies_df(raw, level)
