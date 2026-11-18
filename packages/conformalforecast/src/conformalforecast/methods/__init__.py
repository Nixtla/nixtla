"""Conformal prediction methods and the fitted forecasters they return."""

from __future__ import annotations

from .aci import ACIForecaster, AdaptiveConformalInference
from .agaci import AgACIForecaster, AggregatedAdaptiveConformalInference
from .cqr import ConformalizedQuantileRegression
from .fixed_offset import FixedOffsetForecaster
from .nexcp import NexCPForecaster, NonexchangeableConformalPrediction
from .spci import SequentialPredictiveConformalInference, SPCIForecaster
from .split import SplitConformal
from .weighted_split import WeightedSplit, WeightedSplitForecaster

__all__ = [
    "ACIForecaster",
    "AdaptiveConformalInference",
    "AgACIForecaster",
    "AggregatedAdaptiveConformalInference",
    "ConformalizedQuantileRegression",
    "FixedOffsetForecaster",
    "NexCPForecaster",
    "NonexchangeableConformalPrediction",
    "SPCIForecaster",
    "SequentialPredictiveConformalInference",
    "SplitConformal",
    "WeightedSplit",
    "WeightedSplitForecaster",
]
