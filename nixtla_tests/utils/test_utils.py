import re
import pandas as pd
import pytest

from nixtla.nixtla_client import _audit_duplicate_rows
from nixtla.nixtla_client import _audit_categorical_variables
from nixtla.nixtla_client import _audit_leading_zeros
from nixtla.nixtla_client import _audit_missing_dates
from nixtla.nixtla_client import _audit_negative_values
from nixtla.nixtla_client import _model_in_list
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
