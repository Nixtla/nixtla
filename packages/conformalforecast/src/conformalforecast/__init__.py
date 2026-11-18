"""Conformal prediction for time series forecasting."""

from __future__ import annotations

from . import panel
from .adapters import CallableAdapter
from .aggregators import EWA, OnlineAggregator
from .base import (
    CalibrationError,
    CalibrationResult,
    ConformalForecastError,
    ForecasterAdapter,
    InsufficientCalibrationSamples,
    PredictionResult,
    ScoreFunction,
    UnsupportedCapability,
)
from .capabilities import (
    SupportsBootstrap,
    SupportsCrossValidation,
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
    SupportsRefit,
)
from .diagnostics import Report, evaluate, evaluate_calibration
from .methods import (
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
from .methods.base import ConformalForecaster, ConformalMethod
from .methods.transfer import SupportsTransfer
from .nonconformity import AbsoluteResidual, QuantileScore, ScoreField, SignedResidual
from .quantile_regressors import QRFQuantileRegressor, QuantileRegressor
from .weighting import (
    DensityRatio,
    DensityRatioEstimator,
    PrecomputedWeights,
    UniformWeights,
    WeightFunction,
)

__version__ = "0.1.0"

__all__ = [
    "ACIForecaster",
    "AbsoluteResidual",
    "AdaptiveConformalInference",
    "AgACIForecaster",
    "AggregatedAdaptiveConformalInference",
    "CalibrationError",
    "CalibrationResult",
    "CallableAdapter",
    "ConformalForecastError",
    "ConformalForecaster",
    "ConformalMethod",
    "ConformalizedQuantileRegression",
    "DensityRatio",
    "DensityRatioEstimator",
    "EWA",
    "FixedOffsetForecaster",
    "ForecasterAdapter",
    "InsufficientCalibrationSamples",
    "NexCPForecaster",
    "NonexchangeableConformalPrediction",
    "OnlineAggregator",
    "PrecomputedWeights",
    "PredictionResult",
    "QRFQuantileRegressor",
    "QuantileRegressor",
    "QuantileScore",
    "Report",
    "SPCIForecaster",
    "ScoreField",
    "ScoreFunction",
    "SequentialPredictiveConformalInference",
    "SignedResidual",
    "SplitConformal",
    "SupportsBootstrap",
    "SupportsCrossValidation",
    "SupportsCrossValidationQuantiles",
    "SupportsQuantiles",
    "SupportsRefit",
    "SupportsTransfer",
    "UniformWeights",
    "UnsupportedCapability",
    "WeightFunction",
    "WeightedSplit",
    "WeightedSplitForecaster",
    "evaluate",
    "evaluate_calibration",
    "panel",
]
