"""Guards on the `integration` marker, which decides what CI runs where.

Only the Linux / Python 3.11 matrix cell runs live-API tests; every other cell
runs `-m "not integration ..."`. That makes the marker load-bearing: a live test
that slips through unmarked burns API quota on six runners instead of one, and
fails on the five that have no credentials configured for it.
"""

import pathlib

import pytest

from nixtla_tests.conftest import (
    _LIVE_CLIENT_FIXTURES,
    pytest_collection_modifyitems,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class _StubItem:
    """The slice of a collected item the hook actually touches."""

    def __init__(self, *fixturenames):
        self.fixturenames = list(fixturenames)
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


def _marks(*fixturenames):
    item = _StubItem(*fixturenames)
    pytest_collection_modifyitems(config=None, items=[item])
    return item.markers


@pytest.mark.parametrize("fixture_name", sorted(_LIVE_CLIENT_FIXTURES))
def test_each_live_fixture_earns_the_marker(fixture_name):
    assert "integration" in _marks(fixture_name, "df_ok")


def test_offline_tests_are_left_alone():
    """A test that takes only local fixtures must stay in the offline
    selection -- over-marking is the quieter failure, and it silently stops
    real coverage from running anywhere but one matrix cell."""
    assert _marks("df_ok", "common_kwargs") == []


def test_marker_is_declared_in_pyproject():
    """`-m integration` must not warn about an unknown marker.

    Read as text rather than parsed: `tomllib` only exists on 3.11+ and the
    matrix still covers 3.10, which is exactly the cell this marker gates.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert '"integration: mark test as requiring a live Nixtla API"' in pyproject


def test_audit_tests_are_not_integration():
    """`audit_data` / `clean_data` are pure local pandas.

    They dropped the `custom_client` fixture when `_audit` was extracted, so
    they must keep running on every matrix cell. If this fails, someone
    reintroduced a live-client fixture into `test_audit_data.py`.

    Checks the functions' parameter lists rather than the file's raw text. A
    substring scan fires on any docstring or comment that merely *names* a
    fixture -- the docstring in that file names `custom_client` precisely to
    explain why it is not used -- while still missing a live fixture reached
    under a local alias (`def client(custom_client): return custom_client`),
    which is the more realistic reintroduction. Parameters catch the alias and
    ignore the prose.
    """
    import ast

    tree = ast.parse(
        (REPO_ROOT / "nixtla_tests/nixtla_client/test_audit_data.py").read_text()
    )
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = {a.arg for a in node.args.args} | {
                a.arg for a in node.args.kwonlyargs
            }
            offenders += [
                f"{node.name}({bad})" for bad in sorted(_LIVE_CLIENT_FIXTURES & params)
            ]
    assert offenders == [], offenders
