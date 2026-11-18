"""Shared test helpers: schema checks and statistical assertions.

Statistical assertions use z-bounds derived from the sampling error of the
estimator, with Z set so a correct implementation fails with probability
< 1e-6 per assertion. Callers must seed their generators.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import numpy as np

# One-sided tail beyond z=5 is ~2.9e-7, so a correct implementation
# essentially never trips these bounds while real bugs (wrong scale,
# wrong sign, mis-parameterized distribution) still do.
Z = 5.0


def to_pandas(df):
    """Convert any supported native frame to pandas for assertions."""
    return nw.from_native(df).to_pandas()


def series_values(
    df, id_col: str = "unique_id", time_col: str = "ds", target_col: str = "y"
) -> dict[str, np.ndarray]:
    """Split a long-format frame into {series_id: values sorted by time}."""
    pdf = to_pandas(df).sort_values(time_col)
    return {
        str(uid): group[target_col].to_numpy()
        for uid, group in pdf.groupby(id_col, observed=True)
    }


def assert_long_format(
    df,
    n_series: int | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    min_length: int | None = None,
    max_length: int | None = None,
    allow_nan: bool = False,
    extra_cols: set[str] | None = None,
) -> None:
    """Assert the Nixtla long-format contract on a generator's output."""
    nw_df = nw.from_native(df)
    expected = {id_col, time_col, target_col} | (extra_cols or set())
    assert set(nw_df.columns) == expected, (
        f"columns {nw_df.columns} != expected {sorted(expected)}"
    )

    schema = nw_df.schema
    assert schema[id_col] == nw.Categorical(), f"{id_col} is {schema[id_col]}"
    assert schema[time_col] in (
        nw.Datetime("ns"),
        nw.Datetime("us"),
        nw.Int64(),
    ), f"{time_col} is {schema[time_col]}"
    assert schema[target_col] == nw.Float64(), f"{target_col} is {schema[target_col]}"

    pdf = to_pandas(df)
    if n_series is not None:
        assert pdf[id_col].nunique() == n_series
    lengths = pdf.groupby(id_col, observed=True).size()
    if min_length is not None:
        assert (lengths >= min_length).all(), f"lengths {lengths.min()} < {min_length}"
    if max_length is not None:
        assert (lengths <= max_length).all(), f"lengths {lengths.max()} > {max_length}"
    if not allow_nan:
        values = pdf[target_col].to_numpy()
        assert np.isfinite(values).all(), "non-finite values in target column"


def assert_mean(samples: np.ndarray, expected: float, std: float, z: float = Z) -> None:
    """Assert the sample mean is within z standard errors of `expected`.

    `std` is the theoretical standard deviation of one sample.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    se = std / np.sqrt(n)
    err = abs(samples.mean() - expected)
    assert err <= z * se, (
        f"mean {samples.mean():.6g} deviates from {expected:.6g} "
        f"by {err / se if se > 0 else np.inf:.1f} SEs (allowed {z})"
    )


def assert_std(
    samples: np.ndarray, expected: float, z: float = Z, kurtosis: float = 3.0
) -> None:
    """Assert the sample standard deviation matches `expected`.

    `kurtosis` is the theoretical kurtosis of the samples (3 for normal);
    it widens the bound for heavy-tailed distributions where the variance
    estimator itself has higher variance.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    var = samples.var()
    expected_var = expected**2
    # Var(s^2) ~= (kurtosis - 1) * sigma^4 / n
    se_var = expected_var * np.sqrt(max(kurtosis - 1.0, 0.1) / n)
    err = abs(var - expected_var)
    assert err <= z * se_var, (
        f"std {np.sqrt(var):.6g} deviates from {expected:.6g} "
        f"({err / se_var if se_var > 0 else np.inf:.1f} SEs on the variance, "
        f"allowed {z})"
    )


def sample_acf(values: np.ndarray, lag: int) -> float:
    """Lag-k sample autocorrelation."""
    values = np.asarray(values, dtype=float)
    demeaned = values - values.mean()
    denom = float(demeaned @ demeaned)
    if denom == 0.0:
        return 0.0
    return float(demeaned[:-lag] @ demeaned[lag:] / denom)


def assert_acf(values: np.ndarray, lag: int, expected: float, z: float = Z) -> None:
    """Assert the lag-k sample autocorrelation matches `expected`.

    Uses the ~1/sqrt(n) standard error of the sample ACF; adequate for the
    moderate correlations these tests check.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    se = 1.0 / np.sqrt(n)
    got = sample_acf(values, lag)
    err = abs(got - expected)
    assert err <= z * se, (
        f"acf(lag={lag}) {got:.4f} deviates from {expected:.4f} "
        f"by {err / se:.1f} SEs (allowed {z})"
    )


def assert_distribution(samples: np.ndarray, scipy_dist, alpha: float = 1e-6) -> None:
    """Kolmogorov-Smirnov test of `samples` against a frozen scipy dist."""
    from scipy import stats

    result = stats.kstest(np.asarray(samples, dtype=float), scipy_dist.cdf)
    assert result.pvalue > alpha, (
        f"KS test rejects the distribution (p={result.pvalue:.3g}, "
        f"stat={result.statistic:.4f})"
    )
