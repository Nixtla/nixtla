"""Tests for the package version attribute."""

import importlib
from importlib.metadata import PackageNotFoundError

import synforecast


def test_version_falls_back_without_metadata(monkeypatch):
    def missing(distribution_name):
        raise PackageNotFoundError(distribution_name)

    monkeypatch.setattr("importlib.metadata.version", missing)
    try:
        assert importlib.reload(synforecast).__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(synforecast)
    assert synforecast.__version__ != "0+unknown"
