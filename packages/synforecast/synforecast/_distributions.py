"""Pure NumPy implementations of statistical distribution functions.

This module provides CDF and PPF (inverse CDF) functions for common distributions
without requiring scipy. All implementations are numerically accurate and vectorized.
"""

import numpy as np

try:
    from synforecast._lib import distributions as _rs_dist

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# Constants
_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)


# =============================================================================
# Error Function Implementation
# =============================================================================


def _erf(x: np.ndarray) -> np.ndarray:
    """Compute the error function using a high-accuracy approximation.

    Uses the approximation from Abramowitz and Stegun (1964), formula 7.1.26,
    which has maximum error of 1.5e-7.

    Args:
        x: Input array

    Returns:
        Error function values
    """
    x = np.asarray(x, dtype=np.float64)
    # Save sign and work with absolute values
    sign = np.sign(x)
    x = np.abs(x)

    # Constants for the approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


# =============================================================================
# Normal Distribution
# =============================================================================


def norm_cdf(x: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Cumulative distribution function for the normal distribution.

    Args:
        x: Input values
        loc: Mean of the distribution (default: 0.0)
        scale: Standard deviation (default: 1.0)

    Returns:
        CDF values in [0, 1]
    """
    if _HAS_RUST:
        x = np.asarray(x, dtype=np.float64)
        z = (x - loc) / scale
        shape = z.shape
        result = _rs_dist.norm_cdf(z.ravel())
        return result.reshape(shape)
    x = np.asarray(x, dtype=np.float64)
    z = (x - loc) / scale
    return 0.5 * (1.0 + _erf(z / _SQRT2))


def norm_ppf(u: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Percent point function (inverse CDF) for the normal distribution.

    Uses a high-accuracy rational approximation.

    Args:
        u: Probability values in (0, 1)
        loc: Mean of the distribution (default: 0.0)
        scale: Standard deviation (default: 1.0)

    Returns:
        Quantile values
    """
    if _HAS_RUST:
        u = np.asarray(u, dtype=np.float64)
        shape = u.shape
        result = _rs_dist.norm_ppf(u.ravel())
        return loc + scale * result.reshape(shape)
    u = np.asarray(u, dtype=np.float64)
    result = np.zeros_like(u, dtype=np.float64)

    # Handle edge cases
    result[u <= 0] = -np.inf
    result[u >= 1] = np.inf

    # Work with values in valid range
    mask = (u > 0) & (u < 1)
    if not np.any(mask):
        return loc + scale * result

    p = u[mask]

    # Rational approximation for the normal quantile function
    # Wichura (1988), Algorithm AS 241, doi:10.2307/2347330.

    # Constants
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    # Split points
    p_low = 0.02425
    p_high = 1 - p_low

    x = np.zeros_like(p)

    # Lower region
    mask_low = p < p_low
    if np.any(mask_low):
        q = np.sqrt(-2.0 * np.log(p[mask_low]))
        x[mask_low] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if np.any(mask_mid):
        q = p[mask_mid] - 0.5
        r = q * q
        x[mask_mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    # Upper region
    mask_high = p > p_high
    if np.any(mask_high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[mask_high]))
        x[mask_high] = -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    result[mask] = x

    return loc + scale * result


# =============================================================================
# Student's t Distribution
# =============================================================================


def _lgamma(x: np.ndarray) -> np.ndarray:
    """Log-gamma function using Lanczos approximation."""
    x = np.asarray(x, dtype=np.float64)

    # Lanczos approximation coefficients (g=7)
    g = 7
    c = np.array(
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]
    )

    # Reflection formula for x < 0.5
    reflect = x < 0.5
    x_work = np.where(reflect, 1.0 - x, x)

    x_work = x_work - 1
    a = c[0] * np.ones_like(x_work)
    for i in range(1, len(c)):
        a = a + c[i] / (x_work + i)

    t = x_work + g + 0.5
    result = 0.5 * np.log(2 * np.pi) + (x_work + 0.5) * np.log(t) - t + np.log(a)

    # Apply reflection formula
    result = np.where(
        reflect,
        np.log(np.pi) - np.log(np.abs(np.sin(np.pi * x)) + 1e-300) - result,
        result,
    )

    return result


def _betainc_scalar(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) for scalar inputs."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry for better convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc_scalar(b, a, 1.0 - x)

    # Compute the beta function via log-gamma
    log_beta = (
        _lgamma(np.array([a]))[0]
        + _lgamma(np.array([b]))[0]
        - _lgamma(np.array([a + b]))[0]
    )

    # Front factor
    front = np.exp(a * np.log(x) + b * np.log(1 - x) - log_beta) / a

    # Continued fraction using modified Lentz's algorithm
    max_iter = 200
    eps = 1e-14
    tiny = 1e-30

    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m

        # Even step
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return front * h


def t_cdf(x: np.ndarray, df: float) -> np.ndarray:
    """Cumulative distribution function for Student's t distribution.

    Args:
        x: Input values
        df: Degrees of freedom (positive)

    Returns:
        CDF values in [0, 1]
    """
    if _HAS_RUST:
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape
        result = _rs_dist.t_cdf(x.ravel(), float(df))
        return result.reshape(shape)
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)

    # F(x) = 0.5 + x * Gamma((df+1)/2) / (sqrt(df*pi) * Gamma(df/2)) * 2F1(...)
    # Using the incomplete beta relation:
    # F(x) = 1 - 0.5 * I_{df/(df+x^2)}(df/2, 1/2) for x > 0
    # F(x) = 0.5 * I_{df/(df+x^2)}(df/2, 1/2) for x < 0

    t_sq = x * x
    z = df / (df + t_sq)

    # Compute incomplete beta for each element
    for i in range(len(x.flat)):
        idx = np.unravel_index(i, x.shape)
        zi = z[idx]
        ibeta = _betainc_scalar(df / 2.0, 0.5, zi)

        if x[idx] > 0:
            result[idx] = 1.0 - 0.5 * ibeta
        elif x[idx] < 0:
            result[idx] = 0.5 * ibeta
        else:
            result[idx] = 0.5

    return result


# =============================================================================
# Exponential Distribution
# =============================================================================


def expon_ppf(u: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Percent point function (inverse CDF) for the exponential distribution.

    Args:
        u: Probability values in (0, 1)
        scale: Scale parameter (1/rate), default 1.0

    Returns:
        Quantile values
    """
    if _HAS_RUST:
        u = np.asarray(u, dtype=np.float64)
        shape = u.shape
        return _rs_dist.expon_ppf(u.ravel(), float(scale)).reshape(shape)
    u = np.asarray(u, dtype=np.float64)
    result = np.zeros_like(u)

    # Handle edge cases
    result[u <= 0] = 0.0
    result[u >= 1] = np.inf

    mask = (u > 0) & (u < 1)
    result[mask] = -scale * np.log(1.0 - u[mask])

    return result


# =============================================================================
# Lognormal Distribution
# =============================================================================


def lognorm_ppf(u: np.ndarray, s: float, scale: float = 1.0) -> np.ndarray:
    """Percent point function (inverse CDF) for the lognormal distribution.

    Args:
        u: Probability values in (0, 1)
        s: Shape parameter (standard deviation of log)
        scale: Scale parameter (exp of mean of log), default 1.0

    Returns:
        Quantile values
    """
    if _HAS_RUST:
        u = np.asarray(u, dtype=np.float64)
        u_shape = u.shape
        return _rs_dist.lognorm_ppf(u.ravel(), float(s), float(scale)).reshape(u_shape)
    return scale * np.exp(s * norm_ppf(u))


# =============================================================================
# Uniform Distribution
# =============================================================================


def uniform_ppf(u: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Percent point function (inverse CDF) for the uniform distribution.

    Args:
        u: Probability values in [0, 1]
        loc: Lower bound of the distribution (default: 0.0)
        scale: Width of the distribution (default: 1.0)

    Returns:
        Quantile values in [loc, loc + scale]
    """
    if _HAS_RUST:
        u = np.asarray(u, dtype=np.float64)
        u_shape = u.shape
        return _rs_dist.uniform_ppf(u.ravel(), float(loc), float(scale)).reshape(
            u_shape
        )
    u = np.asarray(u, dtype=np.float64)
    return loc + scale * u


# =============================================================================
# Gamma Distribution
# =============================================================================


def _gammainc_lower(a: float, x: float) -> float:
    """Lower regularized incomplete gamma function P(a, x) = gamma(a,x)/Gamma(a)."""
    if x <= 0:
        return 0.0
    if x == np.inf:
        return 1.0

    # Use series for x < a+1, continued fraction otherwise
    if x < a + 1:
        return _gammainc_series(a, x)
    else:
        return 1.0 - _gammainc_cf(a, x)


def _gammainc_series(a: float, x: float) -> float:
    """Lower incomplete gamma via series expansion."""
    max_iter = 200
    eps = 1e-14

    ap = a
    sum_val = 1.0 / a
    delta = sum_val

    for _ in range(max_iter):
        ap += 1
        delta *= x / ap
        sum_val += delta
        if abs(delta) < abs(sum_val) * eps:
            break

    log_prefix = a * np.log(x) - x - _lgamma(np.array([a]))[0]
    return sum_val * np.exp(log_prefix)


def _gammainc_cf(a: float, x: float) -> float:
    """Upper incomplete gamma via continued fraction."""
    max_iter = 200
    eps = 1e-14
    tiny = 1e-30

    b = x + 1 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d

    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    log_prefix = a * np.log(x) - x - _lgamma(np.array([a]))[0]
    return np.exp(log_prefix) * h


def _gamma_ppf_scalar(u: float, shape: float) -> float:
    """Compute gamma PPF for a single value using Newton-Raphson."""
    if u <= 0:
        return 0.0
    if u >= 1:
        return np.inf

    # Initial guess using Wilson-Hilferty approximation for larger shapes
    if shape >= 1:
        z = norm_ppf(np.array([u]))[0]
        # Wilson-Hilferty transformation
        tmp = 1.0 - 2.0 / (9.0 * shape) + z * np.sqrt(2.0 / (9.0 * shape))
        x = shape * tmp**3 if tmp > 0 else 0.01
    else:
        # For small shape, use power law approximation
        x = (u * np.exp(_lgamma(np.array([shape + 1]))[0])) ** (1.0 / shape)

    # Ensure positive
    x = max(x, 1e-10)

    # Newton-Raphson
    max_iter = 50
    tol = 1e-10

    for _ in range(max_iter):
        # CDF
        cdf_x = _gammainc_lower(shape, x)

        # PDF = x^(a-1) * exp(-x) / Gamma(a)
        log_pdf = (shape - 1) * np.log(x) - x - _lgamma(np.array([shape]))[0]
        pdf_x = np.exp(log_pdf)

        if pdf_x < 1e-100:
            # Avoid division by tiny number
            if cdf_x < u:
                x *= 2
            else:
                x /= 2
            continue

        # Newton step
        delta = (cdf_x - u) / pdf_x
        x_new = x - delta

        # Keep x positive
        if x_new <= 0:
            x /= 2
        else:
            x = x_new

        if abs(delta) < tol * max(1.0, x):
            break

    return x


def gamma_ppf(u: np.ndarray, shape: float, scale: float = 1.0) -> np.ndarray:
    """Percent point function (inverse CDF) for the gamma distribution.

    Args:
        u: Probability values in (0, 1)
        shape: Shape parameter (a > 0)
        scale: Scale parameter (default: 1.0)

    Returns:
        Quantile values
    """
    if _HAS_RUST:
        u = np.asarray(u, dtype=np.float64)
        u_shape = u.shape
        return _rs_dist.gamma_ppf(u.ravel(), float(shape), float(scale)).reshape(
            u_shape
        )
    u = np.asarray(u, dtype=np.float64)
    result = np.zeros_like(u)

    result[u <= 0] = 0.0
    result[u >= 1] = np.inf

    mask = (u > 0) & (u < 1)
    u_valid = u[mask]

    result_valid = np.array([_gamma_ppf_scalar(ui, shape) for ui in u_valid])
    result[mask] = result_valid * scale

    return result
