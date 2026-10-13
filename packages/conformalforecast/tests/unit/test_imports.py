"""Sanity tests: the public import surface, and the dead surface staying dead."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import conformalforecast


def test_package_imports() -> None:
    assert conformalforecast.__version__


@pytest.mark.parametrize(
    "module",
    [
        "conformalforecast.utils",
        "conformalforecast.adapters.base",
        "conformalforecast.adapters.darts",
        "conformalforecast.adapters.sktime",
    ],
)
def test_deleted_stub_modules_are_gone(module: str) -> None:
    """The 0-byte stubs were deleted; importing them must fail cleanly."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_array_surface_import_surface() -> None:
    """The Nixtla-facing surface: methods + panel, nothing else needed.

    ``fit`` lives on the method; ``interval`` and ``update`` on the fitted
    class each method's ``fit`` returns.
    """
    from conformalforecast import panel
    from conformalforecast.methods import (
        ACIForecaster,
        AdaptiveConformalInference,
        AgACIForecaster,
        AggregatedAdaptiveConformalInference,
        ConformalizedQuantileRegression,
        FixedOffsetForecaster,
        NexCPForecaster,
        NonexchangeableConformalPrediction,
        SequentialPredictiveConformalInference,
        SPCIForecaster,
        SplitConformal,
        WeightedSplit,
        WeightedSplitForecaster,
    )

    pairs = (
        (SplitConformal, FixedOffsetForecaster),
        (ConformalizedQuantileRegression, FixedOffsetForecaster),
        (AdaptiveConformalInference, ACIForecaster),
        (AggregatedAdaptiveConformalInference, AgACIForecaster),
        (NonexchangeableConformalPrediction, NexCPForecaster),
        (SequentialPredictiveConformalInference, SPCIForecaster),
        (WeightedSplit, WeightedSplitForecaster),
    )
    for method_cls, fitted_cls in pairs:
        assert hasattr(method_cls, "fit")
        assert not hasattr(method_cls, "interval")
        assert hasattr(fitted_cls, "interval") and hasattr(fitted_cls, "update")
    assert hasattr(panel, "long_to_panel")


def test_fitted_classes_are_importable_from_the_root() -> None:
    for name in (
        "ConformalForecaster",
        "FixedOffsetForecaster",
        "WeightedSplitForecaster",
        "ACIForecaster",
        "AgACIForecaster",
        "NexCPForecaster",
        "SPCIForecaster",
        "ScoreField",
    ):
        assert name in conformalforecast.__all__
        assert getattr(conformalforecast, name) is not None


def test_weighting_surface_is_importable_from_the_root() -> None:
    """DensityRatio is what a user must import for adapt() to do anything, so
    it belongs at the root alongside WeightedSplit and SupportsTransfer."""
    for name in (
        "DensityRatio",
        "DensityRatioEstimator",
        "PrecomputedWeights",
        "UniformWeights",
        "WeightFunction",
    ):
        assert name in conformalforecast.__all__
        assert getattr(conformalforecast, name) is not None


def test_kernels_package_is_gone() -> None:
    """The two-layer split was collapsed; ``conformalforecast.kernels`` must fail."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("conformalforecast.kernels")


_NUMPY_ONLY = ["methods", "nonconformity", "diagnostics", "panel.py", "_quantiles.py"]
_FRAME_LIBS = {"pandas", "polars", "narwhals"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("target", _NUMPY_ONLY)
def test_numpy_only_modules_import_no_frame_library(target: str) -> None:
    root = Path(conformalforecast.__file__).parent / target
    files = [root] if root.suffix == ".py" else sorted(root.rglob("*.py"))
    offenders = {
        str(f.relative_to(root.parent)): sorted(_imported_roots(f) & _FRAME_LIBS)
        for f in files
        if _imported_roots(f) & _FRAME_LIBS
    }
    assert offenders == {}
