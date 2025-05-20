import pandas as pd
import re
import pytest
from nixtla.nixtla_client import _model_in_list
from nixtla.nixtla_client import _audit_duplicate_rows, AuditDataSeverity

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