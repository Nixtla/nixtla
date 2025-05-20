import re
import pytest
from nixtla.nixtla_client import _model_in_list
from nixtla.nixtla_client import _audit_duplicate_rows
from nixtla.nixtla_client import _audit_categorical_variables
from nixtla.nixtla_client import _audit_leading_zeros
from nixtla.nixtla_client import _audit_missing_dates
from nixtla.nixtla_client import _audit_negative_values
from nixtla.nixtla_client import AuditDataSeverity

@pytest.mark.parametrize(
    "name, patterns, expected",
    [
        ("a", ("a", "b"), True),
        ("a", ("b", "c"), False),
        ("axb", ("x", re.compile("a.*b")), True),
        ("axb", ("x", re.compile("^a.*b$")), True),
        ("a-b", ("x", re.compile("^a-.*b$")), True),
        ("a-dfdfb", ("x", re.compile("^a-.*b$")), True),
        ("abc", ("x", re.compile("ab"), re.compile("abcd")), False),
    ]
)
def test_model_in_list(name, patterns, expected):
    assert _model_in_list(name, patterns) is expected


def test_audit_duplicate_rows_pass(df_no_duplicates):
    audit, duplicates = _audit_duplicate_rows(df_no_duplicates)
    assert audit == AuditDataSeverity.PASS
    assert len(duplicates) == 0

def test_audit_duplicate_rows_fail(df_with_duplicates):
    audit, duplicates = _audit_duplicate_rows(df_with_duplicates)
    assert audit == AuditDataSeverity.FAIL
    assert len(duplicates) == 2

def test_audit_missing_dates_complete(df_complete):
    audit, missing = _audit_missing_dates(df_complete, freq='D')
    assert audit == AuditDataSeverity.PASS
    assert len(missing) == 0

def test_audit_missing_dates_with_missing(df_missing):
    audit, missing = _audit_missing_dates(df_missing, freq='D')
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