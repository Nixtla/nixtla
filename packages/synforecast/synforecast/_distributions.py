"""Statistical distribution functions backed by the native extension."""

import numpy as np

from synforecast._lib import distributions as _rs_dist


def norm_cdf(x: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Return the normal cumulative distribution function."""
    values = np.asarray(x, dtype=np.float64)
    standardized = (values - loc) / scale
    return _rs_dist.norm_cdf(standardized.ravel()).reshape(values.shape)


def norm_ppf(u: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Return the normal percent-point function."""
    probabilities = np.asarray(u, dtype=np.float64)
    quantiles = _rs_dist.norm_ppf(probabilities.ravel()).reshape(probabilities.shape)
    return loc + scale * quantiles


def t_cdf(x: np.ndarray, df: float) -> np.ndarray:
    """Return Student's t cumulative distribution function."""
    values = np.asarray(x, dtype=np.float64)
    return _rs_dist.t_cdf(values.ravel(), float(df)).reshape(values.shape)


def expon_ppf(u: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Return the exponential percent-point function."""
    probabilities = np.asarray(u, dtype=np.float64)
    return _rs_dist.expon_ppf(probabilities.ravel(), float(scale)).reshape(
        probabilities.shape
    )


def lognorm_ppf(u: np.ndarray, s: float, scale: float = 1.0) -> np.ndarray:
    """Return the lognormal percent-point function."""
    probabilities = np.asarray(u, dtype=np.float64)
    return _rs_dist.lognorm_ppf(probabilities.ravel(), float(s), float(scale)).reshape(
        probabilities.shape
    )


def uniform_ppf(u: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Return the uniform percent-point function."""
    probabilities = np.asarray(u, dtype=np.float64)
    return _rs_dist.uniform_ppf(
        probabilities.ravel(), float(loc), float(scale)
    ).reshape(probabilities.shape)


def gamma_ppf(u: np.ndarray, shape: float, scale: float = 1.0) -> np.ndarray:
    """Return the gamma percent-point function."""
    probabilities = np.asarray(u, dtype=np.float64)
    return _rs_dist.gamma_ppf(
        probabilities.ravel(), float(shape), float(scale)
    ).reshape(probabilities.shape)
