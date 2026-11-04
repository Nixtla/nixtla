"""SynForecast: Synthetic Time Series Generation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(distribution_name="synforecast")
except PackageNotFoundError:
    __version__ = "0+unknown"

try:
    from synforecast import _lib as _lib
except ImportError as exc:
    raise ImportError(
        "SynForecast requires its native Rust extension. Reinstall the package "
        "using a supported wheel, or install a Rust toolchain before building "
        "from source."
    ) from exc

from synforecast.augment import (
    DBA,
    BaseAugmenter,
    FitAndSimulate,
    Mixup,
    MovingBlockBootstrap,
)
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
from synforecast.overlays import (
    Anomalies,
    Calendar,
    CalendarCycle,
    Changepoints,
    Missing,
    Observation,
    OverlayConfig,
    Waveform,
)
from synforecast.presets import (
    balanced_pool,
    calendar_overlays,
    interpretable_pool,
    observability_overlays,
    pretraining_overlays,
    pretraining_pool,
    seasonal_pool,
)
from synforecast.utils import generate_series

__all__ = [
    "DBA",
    "Anomalies",
    "BaseAugmenter",
    "BaseGenerator",
    "Calendar",
    "CalendarCycle",
    "Changepoints",
    "CoverageDiagnostics",
    "CoverageResult",
    "FitAndSimulate",
    "Missing",
    "Mixup",
    "MovingBlockBootstrap",
    "Multivariatizer",
    "Observation",
    "OverlayConfig",
    "SynAugment",
    "SynSet",
    "Waveform",
    "balanced_pool",
    "calendar_overlays",
    "compare_feature_coverage",
    "compute_features",
    "feature_coverage",
    "generate_series",
    "interpretable_pool",
    "observability_overlays",
    "pretraining_overlays",
    "pretraining_pool",
    "seasonal_pool",
]
