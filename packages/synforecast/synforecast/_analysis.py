"""Statistical analysis functions for time series classification."""

import numpy as np


def compute_basic_stats(series: np.ndarray) -> dict:
    """Compute basic statistical properties of a time series.

    Args:
        series: 1D array of time series values

    Returns:
        dict with mean, std, min, max, and first few autocorrelations
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 2:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "autocorr_1": np.nan,
            "autocorr_5": np.nan,
            "autocorr_10": np.nan,
        }

    # Compute autocorrelations
    autocorr_1 = _autocorrelation(valid, 1)
    autocorr_5 = _autocorrelation(valid, 5) if len(valid) > 5 else np.nan
    autocorr_10 = _autocorrelation(valid, 10) if len(valid) > 10 else np.nan

    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "autocorr_1": autocorr_1,
        "autocorr_5": autocorr_5,
        "autocorr_10": autocorr_10,
    }


def _autocorrelation(series: np.ndarray, lag: int) -> float:
    """Compute autocorrelation at a given lag.

    Args:
        series: 1D array of time series values (no NaN)
        lag: Lag for autocorrelation

    Returns:
        Autocorrelation coefficient
    """
    if len(series) <= lag:
        return np.nan

    n = len(series)
    mean = np.mean(series)
    var = np.var(series)

    if var == 0:
        return np.nan

    cov = np.sum((series[: n - lag] - mean) * (series[lag:] - mean)) / n
    return float(cov / var)


def detect_seasonality(
    series: np.ndarray, max_period: int = 365, min_period: int = 2
) -> dict:
    """Detect seasonality in a time series using autocorrelation peaks.

    Args:
        series: 1D array of time series values
        max_period: Maximum period to check
        min_period: Minimum period to check

    Returns:
        dict with 'period' (detected period or None), 'strength' (0-1),
        and 'has_seasonality' (bool)
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < max_period:
        max_period = len(valid) // 2

    if max_period < min_period:
        return {"period": None, "strength": 0.0, "has_seasonality": False}

    # Compute autocorrelations for all lags
    lags = list(range(min_period, max_period + 1))
    acs = np.array(
        [
            ac if not np.isnan(ac := _autocorrelation(valid, lag)) else 0.0
            for lag in lags
        ]
    )

    if len(acs) < 3 or np.all(acs <= 0):
        return {"period": None, "strength": 0.0, "has_seasonality": False}

    # Seasonality shows up as a *local peak* in the ACF at the period, not as a
    # high value on a monotonically decaying ACF. A random walk, AR(1), or a
    # single level shift has a monotone ACF (its argmax sits at the smallest
    # lag) and must not be read as seasonal. So restrict candidates to interior
    # local maxima and pick the strongest.
    peaks = [
        (lags[i], acs[i])
        for i in range(1, len(acs) - 1)
        if acs[i] > acs[i - 1] and acs[i] >= acs[i + 1] and acs[i] > 0
    ]
    if not peaks:
        return {"period": None, "strength": 0.0, "has_seasonality": False}

    detected_period, max_ac = max(peaks, key=lambda p: p[1])

    # Consider seasonality significant if the peak autocorrelation > 0.3
    has_seasonality = max_ac > 0.3

    return {
        "period": int(detected_period) if has_seasonality else None,
        "strength": float(max(0, max_ac)),
        "has_seasonality": has_seasonality,
    }


def detect_trend(series: np.ndarray) -> dict:
    """Detect linear trend in a time series.

    Args:
        series: 1D array of time series values

    Returns:
        dict with 'slope', 'intercept', 'has_trend' (bool),
        and 'r_squared' (goodness of fit)
    """
    series = np.asarray(series, dtype=np.float64)
    valid_mask = ~np.isnan(series)
    valid = series[valid_mask]

    if len(valid) < 3:
        return {
            "slope": 0.0,
            "intercept": np.nan,
            "has_trend": False,
            "r_squared": 0.0,
        }

    # Use indices where we have valid values
    t = np.arange(len(series))[valid_mask]

    # Linear regression: y = slope * t + intercept
    n = len(valid)
    sum_t = np.sum(t)
    sum_y = np.sum(valid)
    sum_ty = np.sum(t * valid)
    sum_t2 = np.sum(t * t)

    denom = n * sum_t2 - sum_t * sum_t
    if denom == 0:
        return {
            "slope": 0.0,
            "intercept": float(np.mean(valid)),
            "has_trend": False,
            "r_squared": 0.0,
        }

    slope = (n * sum_ty - sum_t * sum_y) / denom
    intercept = (sum_y - slope * sum_t) / n

    # Compute R-squared
    y_pred = slope * t + intercept
    ss_res = np.sum((valid - y_pred) ** 2)
    ss_tot = np.sum((valid - np.mean(valid)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Normalize slope by series std to make it scale-independent
    std = np.std(valid)
    normalized_slope = abs(slope * len(series) / std) if std > 0 else 0.0

    # Consider trend significant if R² > 0.1 and normalized slope is meaningful
    has_trend = r_squared > 0.1 and normalized_slope > 0.5

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "has_trend": has_trend,
        "r_squared": float(max(0, r_squared)),
    }


def detect_stationarity(series: np.ndarray) -> dict:
    """Approximate stationarity test using variance ratio.

    Uses a simple heuristic based on comparing variance in first and second
    half of the series, plus checking for unit root behavior.

    Args:
        series: 1D array of time series values

    Returns:
        dict with 'is_stationary' (bool) and 'confidence' (0-1)
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 20:
        return {"is_stationary": True, "confidence": 0.5}

    # Split into halves
    mid = len(valid) // 2
    first_half = valid[:mid]
    second_half = valid[mid:]

    # Compare means and variances
    mean1, mean2 = np.mean(first_half), np.mean(second_half)
    var1, var2 = np.var(first_half), np.var(second_half)

    # Variance ratio test
    var_ratio = max(var1, var2) / (min(var1, var2) + 1e-10)

    # Mean difference relative to std
    pooled_std = np.std(valid)
    mean_diff = abs(mean1 - mean2) / (pooled_std + 1e-10)

    # Check autocorrelation decay (stationary series should have decaying ACF)
    ac1 = _autocorrelation(valid, 1)
    ac5 = _autocorrelation(valid, 5) if len(valid) > 5 else 0.0

    # Heuristics for stationarity
    # - Variance should be relatively stable (ratio < 3)
    # - Mean shouldn't drift too much (mean_diff < 1)
    # - ACF should decay (ac1 > ac5 or both small)
    variance_stable = var_ratio < 3.0
    mean_stable = mean_diff < 1.0
    acf_decays = (ac1 > ac5) or (abs(ac1) < 0.5)

    is_stationary = variance_stable and mean_stable and acf_decays

    # Confidence based on how well criteria are met
    confidence = (
        (1.0 if variance_stable else 0.0)
        + (1.0 if mean_stable else 0.0)
        + (1.0 if acf_decays else 0.0)
    ) / 3.0

    return {"is_stationary": is_stationary, "confidence": float(confidence)}


def detect_volatility_clustering(series: np.ndarray) -> dict:
    """Detect volatility clustering (ARCH effects) in a time series.

    Uses a simple test based on autocorrelation of squared returns.

    Args:
        series: 1D array of time series values

    Returns:
        dict with 'has_clustering' (bool) and 'strength' (0-1)
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 20:
        return {"has_clustering": False, "strength": 0.0}

    # Compute returns (differences)
    returns = np.diff(valid)

    if len(returns) < 10:
        return {"has_clustering": False, "strength": 0.0}

    # Demean returns and square them
    returns_centered = returns - np.mean(returns)
    squared_returns = returns_centered**2

    # Compute autocorrelation of squared returns
    ac1_sq = _autocorrelation(squared_returns, 1)
    ac2_sq = _autocorrelation(squared_returns, 2) if len(squared_returns) > 2 else 0.0
    ac5_sq = _autocorrelation(squared_returns, 5) if len(squared_returns) > 5 else 0.0

    # Average autocorrelation of squared returns
    avg_ac_sq = np.mean([ac for ac in [ac1_sq, ac2_sq, ac5_sq] if not np.isnan(ac)])

    if np.isnan(avg_ac_sq):
        return {"has_clustering": False, "strength": 0.0}

    # ARCH effects present if squared returns are autocorrelated
    has_clustering = avg_ac_sq > 0.1

    return {
        "has_clustering": has_clustering,
        "strength": float(max(0, min(1, avg_ac_sq))),
    }


def detect_hurst_exponent(series: np.ndarray) -> dict:
    """Estimate the Hurst exponent using R/S analysis.

    H < 0.5: Anti-persistent (mean-reverting)
    H = 0.5: Random walk
    H > 0.5: Persistent (trending)

    Args:
        series: 1D array of time series values

    Returns:
        dict with 'hurst' (float) and 'behavior' (str)
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 20:
        return {"hurst": 0.5, "behavior": "random_walk"}

    n = len(valid)
    max_k = n // 4

    segment_sizes = []
    rs_values = []

    for k in range(10, max_k + 1, max(1, max_k // 20)):
        n_segments = n // k
        if n_segments < 2:
            continue

        rs_k = []
        for i in range(n_segments):
            segment = valid[i * k : (i + 1) * k]
            mean = np.mean(segment)
            cumdev = np.cumsum(segment - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(segment, ddof=1)
            if S > 0:
                rs_k.append(R / S)

        if rs_k:
            segment_sizes.append(k)
            rs_values.append(np.mean(rs_k))

    if len(segment_sizes) < 2:
        return {"hurst": 0.5, "behavior": "random_walk"}

    # Linear regression on log-log scale
    log_n = np.log(segment_sizes)
    log_rs = np.log(rs_values)
    slope, _ = np.polyfit(log_n, log_rs, 1)

    # Clip to valid range
    hurst = float(np.clip(slope, 0.01, 0.99))

    if hurst < 0.4:
        behavior = "mean_reverting"
    elif hurst > 0.6:
        behavior = "persistent"
    else:
        behavior = "random_walk"

    return {"hurst": hurst, "behavior": behavior}


def detect_intermittency(series: np.ndarray) -> dict:
    """Detect if a series has intermittent/sparse demand patterns.

    Args:
        series: 1D array of time series values

    Returns:
        dict with 'is_intermittent' (bool), 'zero_fraction' (float),
        and 'adi' (average demand interval)
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 5:
        return {"is_intermittent": False, "zero_fraction": 0.0, "adi": 1.0}

    # Count zeros or near-zeros
    threshold = np.std(valid) * 0.01 if np.std(valid) > 0 else 0.01
    zero_count = np.sum(np.abs(valid) < threshold)
    zero_fraction = zero_count / len(valid)

    # Average demand interval (ADI)
    # Number of periods between non-zero demands
    non_zero_indices = np.where(np.abs(valid) >= threshold)[0]
    if len(non_zero_indices) > 1:
        intervals = np.diff(non_zero_indices)
        adi = float(np.mean(intervals))
    else:
        adi = float(len(valid))

    # Intermittent if more than 30% zeros and ADI > 1.32 (Syntetos-Boylan)
    is_intermittent = zero_fraction > 0.3 and adi > 1.32

    return {
        "is_intermittent": is_intermittent,
        "zero_fraction": float(zero_fraction),
        "adi": adi,
    }


# A split is accepted only if two constant-level segments explain the data at
# least this many times better (residual-variance ratio) than a single linear
# fit over the same span. Scoring against a *linear* null means a smooth trend
# is the thing to beat, so trends are not mistaken for regimes; a sharp level
# shift, which a line fits terribly but two means fit almost perfectly, scores
# far above the threshold. The old detector scored a raw mean shift against the
# pooled std and fired at 0.5, which nearly every wandering series cleared.
_REGIME_VARIANCE_RATIO = 3.0


def detect_regime_changes(series: np.ndarray, max_regimes: int = 3) -> dict:
    """Detect regime changes: level shifts between otherwise stable segments.

    Uses binary segmentation, scoring each candidate split by how much better a
    piecewise-constant (two-mean) fit explains the segment than a single linear
    fit — the residual-variance ratio. A split is accepted only when that ratio
    exceeds ``_REGIME_VARIANCE_RATIO``, so smooth trends and random-walk wander
    (both well or comparably explained by the linear null) are not mistaken for
    regimes, while sharp level shifts are.

    Args:
        series: 1D array of time series values
        max_regimes: Maximum number of regimes to detect

    Returns:
        dict with 'n_regimes' (int), 'change_points' (list of indices),
        and 'has_regimes' (bool)
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 40:
        return {"n_regimes": 1, "change_points": [], "has_regimes": False}

    n = len(valid)
    min_segment = max(15, n // 8)

    def find_change_point(start: int, end: int) -> tuple[int | None, float]:
        """Best split of valid[start:end], scored by the residual-variance
        ratio of a linear null over a piecewise-constant (two-mean) fit."""
        seg = valid[start:end]
        m = len(seg)
        if m < 2 * min_segment:
            return None, 0.0

        # Linear null: residual variance of a single straight-line fit.
        tt = np.arange(m)
        slope, intercept = np.polyfit(tt, seg, 1)
        var_linear = float(np.var(seg - (slope * tt + intercept))) + 1e-10

        best_score = 0.0
        best_idx = None
        for i in range(min_segment, m - min_segment):
            left, right = seg[:i], seg[i:]
            rss = np.sum((left - left.mean()) ** 2) + np.sum(
                (right - right.mean()) ** 2
            )
            var_piecewise = rss / m + 1e-10
            score = var_linear / var_piecewise
            if score > best_score:
                best_score = score
                best_idx = start + i
        return best_idx, best_score

    # Binary segmentation: accept only splits clearing the variance-ratio bar.
    change_points: list[int] = []
    segments = [(0, n)]
    for _ in range(max_regimes - 1):
        best_cp = None
        best_segment_idx = None
        best_score = _REGIME_VARIANCE_RATIO

        for seg_idx, (start, end) in enumerate(segments):
            cp, score = find_change_point(start, end)
            if cp is not None and score > best_score:
                best_score = score
                best_cp = cp
                best_segment_idx = seg_idx

        if best_cp is None or best_segment_idx is None:
            break
        change_points.append(best_cp)
        start, end = segments[best_segment_idx]
        segments[best_segment_idx] = (start, best_cp)
        segments.insert(best_segment_idx + 1, (best_cp, end))

    change_points = sorted(change_points)
    n_regimes = len(change_points) + 1

    return {
        "n_regimes": n_regimes,
        "change_points": change_points,
        "has_regimes": n_regimes > 1,
    }


def classify_series(series: np.ndarray) -> dict:
    """Classify a time series and recommend the best generator.

    Analyzes multiple properties and returns the most suitable generator
    along with all detected properties.

    Args:
        series: 1D array of time series values

    Returns:
        dict with 'recommended_generator' (str) and 'properties' (dict)
    """
    series = np.asarray(series, dtype=np.float64)

    # Run all detections
    basic_stats = compute_basic_stats(series)
    seasonality = detect_seasonality(series)
    trend = detect_trend(series)
    stationarity = detect_stationarity(series)
    volatility = detect_volatility_clustering(series)
    hurst = detect_hurst_exponent(series)
    intermittency = detect_intermittency(series)
    regimes = detect_regime_changes(series)

    properties = {
        "basic_stats": basic_stats,
        "seasonality": seasonality,
        "trend": trend,
        "stationarity": stationarity,
        "volatility_clustering": volatility,
        "hurst": hurst,
        "intermittency": intermittency,
        "regimes": regimes,
    }

    # Decision logic for generator selection. Ordered most-distinctive first,
    # with regime switching demoted to a specific fallback (below) so it no
    # longer shadows cleaner explanations like seasonality or volatility.

    # 1. Intermittent data — many zeros, unmistakable
    if intermittency["is_intermittent"]:
        return {
            "recommended_generator": "IntermittentDemandGenerator",
            "properties": properties,
        }

    # 2. Strong seasonality
    if seasonality["has_seasonality"] and seasonality["strength"] > 0.5:
        if stationarity["is_stationary"]:
            return {
                "recommended_generator": "SeasonalGenerator",
                "properties": properties,
            }
        else:
            return {
                "recommended_generator": "SARIMAGenerator",
                "properties": properties,
            }

    # 3. Volatility clustering (GARCH effects)
    if volatility["has_clustering"] and volatility["strength"] > 0.2:
        return {
            "recommended_generator": "GARCHGenerator",
            "properties": properties,
        }

    # 4. Regime switching — genuine level shifts between stable segments.
    #    A specific, well-identified pattern, so it ranks above the noisier
    #    Hurst-based branches that a level shift would otherwise trigger.
    if regimes["has_regimes"] and regimes["n_regimes"] >= 2:
        return {
            "recommended_generator": "RegimeSwitchingGenerator",
            "properties": properties,
        }

    # 5. Non-stationary with trend (linear, or exponential -> GBM)
    if trend["has_trend"] and trend["r_squared"] > 0.3:
        valid = series[~np.isnan(series)]
        if len(valid) > 10 and np.all(valid > 0):
            log_series = np.log(valid + 1e-10)
            log_trend = detect_trend(log_series)
            if log_trend["r_squared"] > trend["r_squared"]:
                return {
                    "recommended_generator": "GeometricBrownianMotionGenerator",
                    "properties": properties,
                }
        return {
            "recommended_generator": "RandomWalkGenerator",
            "properties": properties,
        }

    # 6. Stationary with autocorrelation structure
    if stationarity["is_stationary"] and abs(basic_stats["autocorr_1"]) > 0.3:
        return {
            "recommended_generator": "SARIMAGenerator",
            "properties": properties,
        }

    # 7. Mean-reverting behavior (Hurst-based; noisier, so below the specific
    #    detectors above)
    if hurst["behavior"] == "mean_reverting" and hurst["hurst"] < 0.3:
        return {
            "recommended_generator": "OrnsteinUhlenbeckGenerator",
            "properties": properties,
        }

    # 8. Long-range dependence / persistent memory (Hurst-based fallback)
    if hurst["behavior"] == "persistent" and hurst["hurst"] > 0.7:
        return {
            "recommended_generator": "FractionalBrownianMotionGenerator",
            "properties": properties,
        }

    # 9. Weak seasonality
    if seasonality["has_seasonality"]:
        return {
            "recommended_generator": "SeasonalGenerator",
            "properties": properties,
        }

    # 10. Default: Random walk
    return {
        "recommended_generator": "RandomWalkGenerator",
        "properties": properties,
    }
