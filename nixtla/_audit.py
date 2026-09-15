"""Data quality checks and repairs, run entirely on the caller's machine.

Five named checks -- D001 duplicate rows, D002 missing dates, F001 categorical
columns, V001 negative values, V002 leading zeros -- plus the repairs for them.
`audit_data` reports what it finds, `clean_data` fixes it and re-audits.

Nothing here touches the API: `NixtlaClient.audit_data` and
`NixtlaClient.clean_data` are the documented entry points, but they only forward
to these functions.
"""

import datetime
from enum import Enum
from typing import Callable, Optional, Union

import pandas as pd
from utilsforecast.compat import DataFrame
from utilsforecast.preprocessing import fill_gaps
from utilsforecast.processing import ensure_sorted
from utilsforecast.validation import ensure_time_dtype

from ._http import logger
from ._types import AnyDFType, _Freq


class AuditDataSeverity(Enum):
    """Enum class to indicate audit data severity levels"""

    FAIL = "Fail"  # Indicates a critical issue that requires immediate attention
    CASE_SPECIFIC = "Case Specific"  # Indicates an issue that may be acceptable in specific contexts
    PASS = "Pass"  # Indicates that the data is acceptable


def _audit_duplicate_rows(
    df: AnyDFType,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> tuple[AuditDataSeverity, AnyDFType]:
    if isinstance(df, pd.DataFrame):
        duplicates = df.duplicated(subset=[id_col, time_col], keep=False)
        if duplicates.any():
            return AuditDataSeverity.FAIL, df[duplicates]
        return AuditDataSeverity.PASS, pd.DataFrame()
    else:
        raise ValueError(f"Dataframe type {type(df)} is not supported yet.")


def _audit_missing_dates(
    df: AnyDFType,
    freq: _Freq,
    id_col: str = "unique_id",
    time_col: str = "ds",
    start: Union[str, int, datetime.date, datetime.datetime] = "per_serie",
    end: Union[str, int, datetime.date, datetime.datetime] = "global",
) -> tuple[AuditDataSeverity, AnyDFType]:
    if isinstance(df, pd.DataFrame):
        # Fill gaps in data
        # Convert time_col to datetime if it's string/object type
        df = ensure_time_dtype(df, time_col=time_col)
        df_complete = fill_gaps(
            df, freq=freq, id_col=id_col, time_col=time_col, start=start, end=end
        )

        # Find missing dates by comparing df_complete with df
        df_missing = pd.merge(
            df_complete, df, on=[id_col, time_col], how="outer", indicator=True
        )
        df_missing = df_missing.query("_merge == 'left_only'")[[id_col, time_col]]
        if len(df_missing) > 0:
            return AuditDataSeverity.FAIL, df_missing
        return AuditDataSeverity.PASS, pd.DataFrame()
    else:
        raise ValueError(f"Dataframe type {type(df)} is not supported yet.")


def _audit_categorical_variables(
    df: AnyDFType,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> tuple[AuditDataSeverity, AnyDFType]:
    if isinstance(df, pd.DataFrame):
        # Check categorical variables in df except id_col and time_col
        categorical_cols = (
            df.select_dtypes(include=["category", "object"])
            .columns.drop([id_col, time_col], errors="ignore")
            .tolist()
        )

        if categorical_cols:
            return AuditDataSeverity.FAIL, df[categorical_cols]
        return AuditDataSeverity.PASS, pd.DataFrame()
    else:
        raise ValueError(f"Dataframe type {type(df)} is not supported yet.")


def _audit_leading_zeros(
    df: pd.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> tuple[AuditDataSeverity, pd.DataFrame]:
    df = ensure_sorted(df, id_col=id_col, time_col=time_col)
    if isinstance(df, pd.DataFrame):
        group_info = df.groupby(id_col).agg(
            first_index=(target_col, lambda s: s.index[0]),
            first_nonzero_index=(
                target_col,
                lambda s: s.ne(0).idxmax() if s.ne(0).any() else s.index[0],
            ),
        )
        leading_zeros_df = group_info[
            group_info["first_index"] != group_info["first_nonzero_index"]
        ].reset_index()
        if len(leading_zeros_df) > 0:
            return AuditDataSeverity.CASE_SPECIFIC, leading_zeros_df
        return AuditDataSeverity.PASS, pd.DataFrame()
    else:
        raise ValueError(f"Dataframe type {type(df)} is not supported yet.")


def _audit_negative_values(
    df: AnyDFType,
    target_col: str = "y",
) -> tuple[AuditDataSeverity, AnyDFType]:
    if isinstance(df, pd.DataFrame):
        negative_values = df.loc[df[target_col] < 0]
        if len(negative_values) > 0:
            return AuditDataSeverity.CASE_SPECIFIC, negative_values
        return AuditDataSeverity.PASS, pd.DataFrame()
    else:
        raise ValueError(f"Dataframe type {type(df)} is not supported yet.")


def audit_data(
    df: AnyDFType,
    freq: _Freq,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    start: Union[str, int, datetime.date, datetime.datetime] = "per_serie",
    end: Union[str, int, datetime.date, datetime.datetime] = "global",
) -> tuple[bool, dict[str, DataFrame], dict[str, DataFrame]]:
    """Audit data quality.

    Args:
        df (pandas or polars DataFrame): The dataframe to be audited.
        freq (str, int or pandas offset): Frequency of the timestamps.
            Must be specified. See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
        id_col (str): Column that identifies each series. Defaults to
            'unique_id'.
        time_col (str): Column that identifies each timestep, its values
            can be timestamps or integers. Defaults to 'ds'.
        target_col (str): Column that contains the target. Defaults to 'y'.
        start (Union[str, int, datetime.date, datetime.datetime], optional):
            Initial timestamp for the series.
                * 'per_serie' uses each series first timestamp
                * 'global' uses the first timestamp seen in the data
                * Can also be a specific timestamp or integer,
                e.g. '2000-01-01', 2000 or datetime(2000, 1, 1)
            , by default "per_serie"
        end (Union[str, int, datetime.date, datetime.datetime], optional):
            Final timestamp for the series.
                * 'per_serie' uses each series last timestamp
                * 'global' uses the last timestamp seen in the data
                * Can also be a specific timestamp or integer,
                e.g. '2000-01-01', 2000 or datetime(2000, 1, 1)
            , by default "global"

    Returns:
        tuple[bool, dict[str, DataFrame], dict[str, DataFrame]]:
            Tuple containing:
            - bool: True if all tests pass, False otherwise
            - dict: Dictionary mapping test IDs to error DataFrames for failed
                    tests or None if the test could not be performed.
            - dict: Dictionary mapping test IDs to error DataFrames for
                    case-specific tests.

            Test IDs:
            - D001: Test for duplicate rows
            - D002: Test for missing dates
            - F001: Test for presence of categorical feature columns
            - V001: Test for negative values
            - V002: Test for leading zeros
    """
    df = ensure_time_dtype(df, time_col=time_col)

    logger.info("Running data quality tests...")
    pass_D001, error_df_D001 = _audit_duplicate_rows(df, id_col, time_col)
    pass_D002, error_df_D002 = AuditDataSeverity.FAIL, None
    if pass_D001 != AuditDataSeverity.FAIL:
        # If data has duplicate rows, missing dates can not be added by fill_gaps.
        # Duplicate rows issue needs to be resolved first.
        pass_D002, error_df_D002 = _audit_missing_dates(
            df, freq, id_col, time_col, start, end
        )
    pass_F001, error_df_F001 = _audit_categorical_variables(df, id_col, time_col)
    pass_V001, error_df_V001 = _audit_negative_values(df, target_col)
    pass_V002, error_df_V002 = _audit_leading_zeros(df, id_col, time_col, target_col)

    fail_dict, case_specific_dict = {}, {}
    test_ids = ["D001", "D002", "F001", "V001", "V002"]
    pass_vals = [pass_D001, pass_D002, pass_F001, pass_V001, pass_V002]
    error_dfs = [
        error_df_D001,
        error_df_D002,
        error_df_F001,
        error_df_V001,
        error_df_V002,
    ]
    all_pass = True

    for test_id, pass_val, error_df in zip(test_ids, pass_vals, error_dfs):
        # Only include errors for failed or case specific tests
        if pass_val == AuditDataSeverity.FAIL:
            all_pass = False
            if error_df is not None:
                logger.warning(f"Failure {test_id} detected with critical severity.")
            else:
                logger.warning(f"Test {test_id} could not be performed.")
            fail_dict[test_id] = error_df

        if pass_val == AuditDataSeverity.CASE_SPECIFIC:
            all_pass = False
            logger.warning(
                f"Failure {test_id} detected which could cause issue depending on the use case."
            )
            case_specific_dict[test_id] = error_df

    if all_pass:
        logger.info("All checks passed...")
    return all_pass, fail_dict, case_specific_dict


def clean_data(
    df: AnyDFType,
    fail_dict: dict[str, DataFrame],
    case_specific_dict: dict[str, DataFrame],
    freq: _Freq,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    clean_case_specific: bool = False,
    agg_dict: Optional[dict[str, Union[str, Callable]]] = None,
) -> tuple[AnyDFType, bool, dict[str, DataFrame], dict[str, DataFrame]]:
    """Clean the data. This should be run after running `audit_data`.

    Args:
        df (AnyDFType): The dataframe to be cleaned
        fail_dict (dict[str, DataFrame]): The failure dictionary from the
            audit_data method.
        case_specific_dict (dict[str, DataFrame]): The case specific
            dictionary from the audit_data method.
        freq (str, int or pandas offset): Frequency of the timestamps.
            Must be specified. See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
        id_col (str): Column that identifies each series. Defaults to
            'unique_id'.
        time_col (str): Column that identifies each timestep, its values
            can be timestamps or integers. Defaults to 'ds'.
        target_col (str): Column that contains the target. Defaults to 'y'.
        clean_case_specific (bool, optional): If True, clean case
            specific issues. Defaults to False.
        agg_dict (Optional[dict[str, Union[str, Callable]]], optional):
            The aggregation methods to use when there are duplicate rows (D001),
            by default None

    Returns:
        tuple[AnyDFType, bool, dict[str, DataFrame], dict[str, DataFrame]]:
            Tuple containing:
            - AnyDFType: The cleaned dataframe
            - The three outputs from audit_data that are run at the end of cleansing.

    Raises:
        ValueError: Any exceptions during the cleaning process.
    """
    df = ensure_time_dtype(df, time_col=time_col)
    logger.info("Running data cleansing...")

    if fail_dict:
        if "D001" in fail_dict:
            try:
                logger.info("Fixing D001: Cleaning duplicate rows...")

                if agg_dict is None:
                    raise ValueError(
                        "agg_dict must be provided to resolve D001 failure."
                    )

                # Get all columns except id_col and time_col
                other_cols = [
                    col for col in df.columns if col not in [id_col, time_col]
                ]

                # Verify all columns have aggregation rules
                missing_cols = [col for col in other_cols if col not in agg_dict]
                if missing_cols:
                    raise ValueError(
                        f"D001: Missing aggregation rules for columns: {missing_cols}. "
                        "Please provide aggregation rules for all columns in agg_dict."
                    )
                df = df.groupby([id_col, time_col], as_index=False).agg(agg_dict)
            except Exception as e:
                raise ValueError(f"Error cleaning duplicate rows D001: {e}")
        if "D002" in fail_dict:
            try:
                missing = fail_dict.get("D002")
                if missing is None:
                    logger.warning(
                        "D002: Missing dates could not be checked by audit_data. "
                        "Hence not filling missing dates..."
                    )
                else:
                    logger.info("Fixing D002: Filling missing dates...")
                    df = pd.concat([df, fail_dict.get("D002")])
            except Exception as e:
                raise ValueError(f"Error filling missing dates D002: {e}")

    if case_specific_dict and clean_case_specific:
        if "V001" in case_specific_dict:
            try:
                logger.info("Fixing V001: Removing negative values...")
                df.loc[df[target_col] < 0, target_col] = 0
            except Exception as e:
                raise ValueError(f"Error removing negative values V001: {e}")

        if "V002" in case_specific_dict:
            try:
                logger.info("Fixing V002: Removing leading zeros...")
                leading_zeros_df = case_specific_dict["V002"]
                leading_zeros_dict = leading_zeros_df.set_index(id_col)[
                    "first_nonzero_index"
                ].to_dict()
                df = df.groupby(id_col, group_keys=False).apply(
                    lambda group: group.loc[
                        group.index
                        >= leading_zeros_dict.get(group.name, group.index[0])
                    ]
                )
            except Exception as e:
                raise ValueError(f"Error removing leading zeros V002: {e}")

    # Run data quality checks on the cleaned data
    all_pass, error_dfs, case_specific_dfs = audit_data(
        df=df, freq=freq, id_col=id_col, time_col=time_col
    )

    return df, all_pass, error_dfs, case_specific_dfs
