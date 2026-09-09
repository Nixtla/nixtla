"""Minimal native feature computation for feature-targeted generation.

Strength formulas: Hyndman and Athanasopoulos, Forecasting: Principles and
Practice, https://otexts.com/fpp3/stlfeatures.html. We apply them to classical
moving-average decomposition with endpoint extension, rather than STL.
"""

import numpy as np

from synforecast._analysis import _autocorrelation
from synforecast._lib import augmentation as _rs_augmentation


def _validate_values(values: np.ndarray) -> np.ndarray:
    """Return a finite one-dimensional float array with at least three values."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if len(values) < 3:
        raise ValueError("values must contain at least 3 observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite observations")
    return values


def classical_decompose(
    values: np.ndarray, period: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return classical moving-average trend, seasonal, and remainder arrays.

    Even periods use the standard centered 2 x period moving average. When a
    usable period is unavailable, the trend uses an odd window near one tenth
    of the series length and the seasonal component is zero. The remainder is
    computed last, so the three returned components reconstruct the input.
    """
    values = _validate_values(values)
    if period is not None and period < 2:
        raise ValueError("period must be >= 2 when provided")
    trend, seasonal, remainder = _rs_augmentation.classical_decompose(
        np.ascontiguousarray(values), period
    )
    return np.asarray(trend), np.asarray(seasonal), np.asarray(remainder)


def spectral_entropy(values: np.ndarray) -> float:
    """Return normalized periodogram entropy in [0, 1].

    A constant series has zero spectral uncertainty and therefore returns 0.0.
    """
    values = _validate_values(values)
    centered = values - values.mean()
    if np.var(centered) <= np.finfo(float).eps:
        return 0.0
    power = np.abs(np.fft.rfft(centered))[1:] ** 2
    total = float(power.sum())
    if total <= np.finfo(float).eps or len(power) <= 1:
        return 0.0
    probabilities = power / total
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return float(np.clip(entropy / np.log(len(power)), 0.0, 1.0))


def trend_strength(values: np.ndarray, period: int | None) -> float:
    """Return decomposition-based trend strength in [0, 1] (see module sources)."""
    trend, _, remainder = classical_decompose(values, period)
    denominator = float(np.var(trend + remainder))
    if denominator <= np.finfo(float).eps:
        return 0.0
    strength = 1.0 - float(np.var(remainder)) / denominator
    return float(np.clip(strength, 0.0, 1.0))


def seasonal_strength(values: np.ndarray, period: int | None) -> float:
    """Return decomposition-based seasonal strength in [0, 1] (see module sources)."""
    if period is None or len(values) < 2 * period:
        _validate_values(values)
        return 0.0
    _, seasonal, remainder = classical_decompose(values, period)
    if not np.any(seasonal):
        return 0.0
    denominator = float(np.var(seasonal + remainder))
    if denominator <= np.finfo(float).eps:
        return 0.0
    strength = 1.0 - float(np.var(remainder)) / denominator
    return float(np.clip(strength, 0.0, 1.0))


def acf1(values: np.ndarray) -> float:
    """Return lag-one autocorrelation, mapping undefined values to zero."""
    values = _validate_values(values)
    result = _autocorrelation(values, 1)
    return 0.0 if np.isnan(result) else float(result)


def compute_features(
    values: np.ndarray, seasonal_period: int | None
) -> dict[str, float]:
    """Compute the minimal tsfeatures-style set used by MAR targeting.

    This set is scoped to GRATIS-style feature targeting from Kang, Hyndman,
    and Li (2020, https://arxiv.org/abs/1903.02787). Its exact decomposition,
    feature definitions, and numerical guards are SynForecast's own design
    rather than a reproduction of the reference code.
    """
    values = _validate_values(values)
    if seasonal_period is not None and seasonal_period < 2:
        raise ValueError("period must be >= 2 when provided")
    entropy, trend, seasonal, autocorrelation = _rs_augmentation.compute_features(
        np.ascontiguousarray(values), seasonal_period
    )
    return {
        "spectral_entropy": float(entropy),
        "trend_strength": float(trend),
        "seasonal_strength": float(seasonal),
        "acf1": float(autocorrelation),
    }
