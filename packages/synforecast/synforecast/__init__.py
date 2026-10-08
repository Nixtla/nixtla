"""SynForecast: Synthetic Time Series Generation."""

from importlib.metadata import version

__version__: str = version(distribution_name="synforecast")

try:
    from synforecast import _lib as _lib
except ImportError as exc:
    raise ImportError(
        "SynForecast requires its native Rust extension. Reinstall the package "
        "using a supported wheel, or install a Rust toolchain before building "
        "from source."
    ) from exc

from synforecast.base import BaseGenerator
from synforecast.dataset import SynAugment, SynSet
from synforecast.evaluation import (
    CoverageDiagnostics,
    CoverageResult,
    compare_feature_coverage,
    compute_features,
    feature_coverage,
)
from synforecast.multivariatize import Multivariatizer
from synforecast.presets import (
    balanced_pool,
    interpretable_pool,
    pretraining_pool,
    seasonal_pool,
)
from synforecast.utils import generate_series

__all__ = [
    "BaseGenerator",
    "CoverageDiagnostics",
    "CoverageResult",
    "Multivariatizer",
    "SynAugment",
    "SynSet",
    "balanced_pool",
    "compare_feature_coverage",
    "compute_features",
    "feature_coverage",
    "generate_series",
    "interpretable_pool",
    "pretraining_pool",
    "seasonal_pool",
]
