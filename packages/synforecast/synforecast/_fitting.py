"""Parameter fitting functions for time series generators."""

import numpy as np

from synforecast._analysis import (
    _autocorrelation,
    detect_hurst_exponent,
    detect_regime_changes,
    detect_seasonality,
    detect_trend,
)


def fit_random_walk(series: np.ndarray) -> dict:
    """Fit RandomWalkGenerator parameters to a time series.

    Estimates drift and volatility from the series differences.

    Args:
        series: 1D array of time series values

    Returns:
        dict with fitted parameters for RandomWalkGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 2:
        return {"drift": 0.0, "volatility": 1.0, "start_value": 0.0}

    # Compute differences (steps)
    steps = np.diff(valid)

    drift = float(np.mean(steps))
    volatility = float(np.std(steps, ddof=1)) if len(steps) > 1 else 1.0
    start_value = float(valid[0])

    # Ensure volatility is positive
    volatility = max(volatility, 1e-6)

    return {
        "drift": drift,
        "volatility": volatility,
        "start_value": start_value,
    }


def fit_seasonal(series: np.ndarray, period: int | None = None) -> dict:
    """Fit SeasonalGenerator parameters to a time series.

    Estimates seasonality period, amplitude, trend, and noise level.

    Args:
        series: 1D array of time series values
        period: Optional known period; if None, auto-detect

    Returns:
        dict with fitted parameters for SeasonalGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 5:
        return {
            "seasonality_period": 24,
            "seasonality_amplitude": 1.0,
            "trend": 0.0,
            "noise_level": 1.0,
            "base_level": 0.0,
        }

    # Detect period if not provided
    if period is None:
        seasonality_info = detect_seasonality(valid)
        period = seasonality_info["period"] if seasonality_info["period"] else 24

    # Ensure period is valid
    period = max(2, min(period, len(valid) // 2))

    # Estimate trend
    trend_info = detect_trend(valid)
    trend = trend_info["slope"]

    # Remove trend to isolate seasonal + noise
    t = np.arange(len(valid))
    detrended = valid - trend * t

    # Estimate base level
    base_level = float(np.mean(detrended))

    # Remove base level
    centered = detrended - base_level

    # Estimate amplitude using the average of period maxima
    n_periods = len(valid) // period
    if n_periods >= 1:
        amplitudes = []
        for i in range(n_periods):
            segment = centered[i * period : (i + 1) * period]
            if len(segment) > 0:
                amplitudes.append((np.max(segment) - np.min(segment)) / 2)
        amplitude = float(np.mean(amplitudes)) if amplitudes else 1.0
    else:
        amplitude = float((np.max(centered) - np.min(centered)) / 2)

    # Estimate noise level from residuals
    # Create a simple seasonal fit
    seasonal_fit = amplitude * np.sin(2 * np.pi * t / period)
    residuals = centered - seasonal_fit
    noise_level = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0

    # Ensure positive values
    amplitude = max(amplitude, 1e-6)
    noise_level = max(noise_level, 1e-6)

    return {
        "seasonality_period": int(period),
        "seasonality_amplitude": amplitude,
        "trend": float(trend),
        "noise_level": noise_level,
        "base_level": base_level,
    }


def fit_sarima(series: np.ndarray, max_order: int = 2) -> dict:
    """Fit SARIMAGenerator parameters to a time series.

    Uses simple heuristics to estimate ARIMA parameters.

    Args:
        series: 1D array of time series values
        max_order: Maximum AR/MA order to consider

    Returns:
        dict with fitted parameters for SARIMAGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 10:
        return {
            "p": 1,
            "d": 0,
            "q": 1,
            "P": 0,
            "D": 0,
            "Q": 0,
            "seasonal_period": 12,
            "mean": 0.0,
            "noise_std": 1.0,
        }

    # Check for differencing need
    from synforecast._analysis import detect_stationarity

    stationarity = detect_stationarity(valid)
    d = 0 if stationarity["is_stationary"] else 1

    # Apply differencing if needed
    diff_series = np.diff(valid) if d > 0 else valid

    if len(diff_series) < 5:
        diff_series = valid
        d = 0

    # Estimate mean and noise std
    mean = float(np.mean(diff_series))
    noise_std = float(np.std(diff_series, ddof=1)) if len(diff_series) > 1 else 1.0

    # Estimate AR order from PACF decay
    pacf_values = []
    for lag in range(1, min(max_order + 1, len(diff_series) // 4)):
        pacf = _partial_autocorrelation(diff_series, lag)
        pacf_values.append(abs(pacf) if not np.isnan(pacf) else 0.0)

    # AR order is where PACF drops below threshold
    p = 0
    threshold = 2.0 / np.sqrt(len(diff_series))  # Approximate significance
    for i, pacf in enumerate(pacf_values):
        if pacf > threshold:
            p = i + 1
        else:
            break
    p = min(p, max_order)

    # Estimate MA order from ACF decay after removing AR effect
    q = 1 if abs(_autocorrelation(diff_series, 1)) > threshold else 0
    q = min(q, max_order)

    # Detect seasonality for seasonal orders
    seasonality = detect_seasonality(valid)
    if seasonality["has_seasonality"] and seasonality["period"]:
        seasonal_period = seasonality["period"]
        P = 1 if seasonality["strength"] > 0.5 else 0
        Q = 1 if seasonality["strength"] > 0.5 else 0
        D = 0  # Keep simple
    else:
        seasonal_period = 12
        P = 0
        Q = 0
        D = 0

    return {
        "p": max(1, p),  # At least AR(1)
        "d": d,
        "q": q,
        "P": P,
        "D": D,
        "Q": Q,
        "seasonal_period": seasonal_period,
        "mean": mean if d == 0 else 0.0,
        "drift": mean if d > 0 else 0.0,
        "noise_std": max(noise_std, 1e-6),
    }


def _partial_autocorrelation(series: np.ndarray, lag: int) -> float:
    """Compute partial autocorrelation using Durbin-Levinson recursion.

    Args:
        series: 1D array of time series values
        lag: Lag for PACF

    Returns:
        Partial autocorrelation coefficient
    """
    if len(series) <= lag:
        return np.nan

    # Compute autocorrelations up to lag
    acf = [_autocorrelation(series, k) for k in range(lag + 1)]
    if any(np.isnan(a) for a in acf):
        return np.nan

    # Durbin-Levinson recursion
    if lag == 1:
        return acf[1]

    phi = np.zeros((lag + 1, lag + 1))
    phi[1, 1] = acf[1]

    for k in range(2, lag + 1):
        num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
        den = 1 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
        if abs(den) < 1e-10:
            return 0.0
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return float(phi[lag, lag])


def fit_garch(series: np.ndarray) -> dict:
    """Fit GARCHGenerator parameters to a time series.

    Estimates GARCH(1,1) parameters using method of moments.

    Args:
        series: 1D array of time series values

    Returns:
        dict with fitted parameters for GARCHGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 20:
        return {
            "omega": 0.1,
            "alpha": [0.1],  # GARCHGenerator expects lists
            "beta": [0.8],
            "mean": 0.0,
        }

    # Compute returns
    returns = np.diff(valid)
    mean = float(np.mean(returns))
    returns_centered = returns - mean

    # Estimate unconditional variance
    sigma2 = np.var(returns_centered)

    # Estimate GARCH parameters using squared returns autocorrelation
    squared_returns = returns_centered**2
    ac1 = _autocorrelation(squared_returns, 1)
    ac2 = _autocorrelation(squared_returns, 2) if len(squared_returns) > 2 else ac1

    if np.isnan(ac1):
        ac1 = 0.1
    if np.isnan(ac2):
        ac2 = ac1 * 0.8

    # For GARCH(1,1): E[r_t^2 * r_{t-1}^2] / E[r_t^4] relates to alpha + beta
    # Use simplified method of moments
    alpha_plus_beta = min(0.99, max(0.1, ac1))

    # Typical ratio for financial data
    alpha = min(0.3, alpha_plus_beta * 0.3)
    beta = alpha_plus_beta - alpha

    # Omega from unconditional variance: sigma2 = omega / (1 - alpha - beta)
    omega = sigma2 * (1 - alpha - beta)
    omega = max(1e-6, omega)

    # GARCHGenerator expects alpha and beta as lists (for GARCH(p,q))
    return {
        "omega": float(omega),
        "alpha": [float(max(0.01, alpha))],
        "beta": [float(max(0.01, min(0.98, beta)))],
        "mean": float(mean),
    }


def fit_ornstein_uhlenbeck(series: np.ndarray) -> dict:
    """Fit OrnsteinUhlenbeckGenerator parameters to a time series.

    Estimates mean-reversion speed, long-term mean, and volatility.

    Args:
        series: 1D array of time series values

    Returns:
        dict with fitted parameters for OrnsteinUhlenbeckGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 10:
        return {
            "theta": 0.5,
            "mu": 0.0,
            "sigma": 1.0,
            "initial_value": 0.0,
        }

    # Estimate long-term mean
    mu = float(np.mean(valid))

    # Estimate theta (mean-reversion speed) from the AR(1) coefficient.
    # OrnsteinUhlenbeckGenerator uses the Euler-Maruyama scheme
    #   X_t = X_{t-1} + theta*(mu - X_{t-1})*dt + sigma*sqrt(dt)*z_t,
    # an AR(1) with phi = 1 - theta*dt, so with dt = 1 (one model step per
    # observation, the generator default) the consistent estimate is
    # theta = 1 - acf1. Clip inside the generator's stability bound
    # theta*dt < 2 (phi in (-0.9, 0.9999]).
    ac1 = _autocorrelation(valid, 1)
    if np.isnan(ac1) or ac1 >= 1:
        ac1 = 0.9
    theta = float(np.clip(1.0 - ac1, 1e-4, 1.9))

    # Euler stationary variance: Var(X) = sigma^2*dt / (1 - phi^2)
    #                                   = sigma^2 / (theta*(2 - theta*dt)),
    # so with dt = 1: sigma = sqrt(Var(X) * theta * (2 - theta)).
    var_x = np.var(valid)
    sigma = np.sqrt(var_x * theta * (2.0 - theta))
    sigma = max(1e-6, sigma)

    initial_value = float(valid[0])

    return {
        "theta": float(theta),
        "mu": mu,
        "sigma": float(sigma),
        "initial_value": initial_value,
    }


def fit_fbm(series: np.ndarray) -> dict:
    """Fit FractionalBrownianMotionGenerator parameters to a time series.

    Estimates Hurst exponent and volatility.

    Args:
        series: 1D array of time series values

    Returns:
        dict with fitted parameters for FractionalBrownianMotionGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 20:
        return {
            "hurst": 0.5,
            "sigma": 1.0,
            "initial_value": 0.0,
        }

    # Estimate Hurst exponent
    hurst_info = detect_hurst_exponent(valid)
    hurst = hurst_info["hurst"]

    # Estimate sigma from increments
    increments = np.diff(valid)
    sigma = float(np.std(increments, ddof=1)) if len(increments) > 1 else 1.0
    sigma = max(1e-6, sigma)

    initial_value = float(valid[0])

    return {
        "hurst": float(hurst),
        "sigma": sigma,
        "initial_value": initial_value,
    }


def fit_regime_switching(series: np.ndarray, n_regimes: int | None = None) -> dict:
    """Fit RegimeSwitchingGenerator parameters to a time series.

    Estimates regime parameters using simple segmentation.

    Args:
        series: 1D array of time series values
        n_regimes: Number of regimes; if None, auto-detect

    Returns:
        dict with fitted parameters for RegimeSwitchingGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 30:
        return {
            "n_regimes": 2,
            "regime_means": [0.0, 1.0],
            "regime_variances": [1.0, 1.0],
        }

    # Detect regimes if not specified
    if n_regimes is None:
        regime_info = detect_regime_changes(valid)
        n_regimes = regime_info["n_regimes"]
        change_points = regime_info["change_points"]
    else:
        # Simple equal-length segmentation
        segment_length = len(valid) // n_regimes
        change_points = [segment_length * i for i in range(1, n_regimes)]

    n_regimes = max(2, min(n_regimes, 5))

    # Compute segment boundaries
    boundaries = [0] + change_points + [len(valid)]

    # Estimate parameters for each regime
    regime_means = []
    regime_variances = []

    for i in range(len(boundaries) - 1):
        segment = valid[boundaries[i] : boundaries[i + 1]]
        if len(segment) > 0:
            regime_means.append(float(np.mean(segment)))
            # Compute variance (not std) as expected by RegimeSwitchingGenerator
            var = float(np.var(segment, ddof=1)) if len(segment) > 1 else 1.0
            regime_variances.append(var)

    # Pad if we have fewer segments than n_regimes
    while len(regime_means) < n_regimes:
        regime_means.append(regime_means[-1] if regime_means else 0.0)
        regime_variances.append(regime_variances[-1] if regime_variances else 1.0)

    # Truncate if we have more
    regime_means = regime_means[:n_regimes]
    regime_variances = regime_variances[:n_regimes]

    # Ensure positive variances
    regime_variances = [max(v, 1e-6) for v in regime_variances]

    return {
        "n_regimes": n_regimes,
        "regime_means": regime_means,
        "regime_variances": regime_variances,
    }


def fit_intermittent(series: np.ndarray) -> dict:
    """Fit IntermittentDemandGenerator parameters to a time series.

    Estimates demand probability and demand size distribution.

    Args:
        series: 1D array of time series values

    Returns:
        dict with fitted parameters for IntermittentDemandGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    if len(valid) < 5:
        return {
            "demand_probability": 0.3,
            "demand_mean": 1.0,
            "demand_std": 0.5,
        }

    # Determine threshold for "zero" demand
    threshold = np.std(valid) * 0.01 if np.std(valid) > 0 else 0.01

    # Compute demand probability
    non_zero_mask = np.abs(valid) >= threshold
    demand_probability = float(np.mean(non_zero_mask))
    demand_probability = max(0.01, min(0.99, demand_probability))

    # Compute demand size statistics (for non-zero values)
    non_zero_values = valid[non_zero_mask]
    if len(non_zero_values) > 0:
        demand_mean = float(np.mean(np.abs(non_zero_values)))
        demand_std = (
            float(np.std(np.abs(non_zero_values), ddof=1))
            if len(non_zero_values) > 1
            else demand_mean * 0.5
        )
    else:
        demand_mean = 1.0
        demand_std = 0.5

    demand_mean = max(1e-6, demand_mean)
    demand_std = max(1e-6, demand_std)

    return {
        "demand_probability": demand_probability,
        "demand_mean": demand_mean,
        "demand_std": demand_std,
    }


def fit_gbm(series: np.ndarray) -> dict:
    """Fit GeometricBrownianMotionGenerator parameters to a time series.

    Estimates drift (mu) and volatility (sigma) assuming log-normal returns.

    Args:
        series: 1D array of time series values

    Returns:
        dict with fitted parameters for GeometricBrownianMotionGenerator
    """
    series = np.asarray(series, dtype=np.float64)
    valid = series[~np.isnan(series)]

    # Filter to positive values for GBM
    valid = valid[valid > 0]

    if len(valid) < 10:
        return {
            "mu": 0.0,
            "sigma": 0.2,
            "initial_value": 1.0,
        }

    # Compute log returns
    log_returns = np.diff(np.log(valid))

    # GBM parameters: dS/S = mu*dt + sigma*dW
    # log returns: r = (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z
    # For dt=1: E[r] = mu - sigma^2/2, Var[r] = sigma^2

    mean_log_return = float(np.mean(log_returns))
    var_log_return = (
        float(np.var(log_returns, ddof=1)) if len(log_returns) > 1 else 0.04
    )

    # sigma = sqrt(Var[r])
    sigma = np.sqrt(var_log_return)
    sigma = max(0.01, min(2.0, sigma))

    # mu = E[r] + sigma^2/2
    mu = mean_log_return + sigma**2 / 2

    initial_value = float(valid[0])

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "initial_value": initial_value,
    }


def fit_generator_params(
    series: np.ndarray, generator_name: str, properties: dict | None = None
) -> dict:
    """Fit parameters for a specific generator.

    Args:
        series: 1D array of time series values
        generator_name: Name of the generator to fit
        properties: Optional pre-computed properties from classify_series

    Returns:
        dict with fitted parameters for the specified generator
    """
    fitting_functions = {
        "RandomWalkGenerator": fit_random_walk,
        "SeasonalGenerator": lambda s: fit_seasonal(
            s,
            period=properties["seasonality"]["period"]
            if properties and properties.get("seasonality", {}).get("period")
            else None,
        ),
        "SARIMAGenerator": fit_sarima,
        "GARCHGenerator": fit_garch,
        "OrnsteinUhlenbeckGenerator": fit_ornstein_uhlenbeck,
        "FractionalBrownianMotionGenerator": fit_fbm,
        "RegimeSwitchingGenerator": lambda s: fit_regime_switching(
            s,
            n_regimes=properties["regimes"]["n_regimes"]
            if properties and properties.get("regimes", {}).get("n_regimes")
            else None,
        ),
        "IntermittentDemandGenerator": fit_intermittent,
        "GeometricBrownianMotionGenerator": fit_gbm,
    }

    if generator_name in fitting_functions:
        return fitting_functions[generator_name](series)
    else:
        # Fallback to random walk fitting
        return fit_random_walk(series)
