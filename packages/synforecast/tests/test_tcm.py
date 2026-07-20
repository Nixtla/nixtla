"""Tests for TCMGenerator (temporal causal model)."""

from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pytest

from synforecast.generators.tcm import (
    _MAX_REDRAWS,
    TCMGenerator,
)
from tests.helpers import (
    assert_acf,
    assert_long_format,
    sample_acf,
    series_values,
    to_pandas,
)


def _make(engine: str = "pandas", **overrides) -> TCMGenerator:
    params = {
        "min_length": 64,
        "max_length": 128,
        "freq": "h",
        "engine": engine,
        "seed": 42,
    }
    params.update(overrides)
    return TCMGenerator(**params)


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-9)


@lru_cache(maxsize=1)
def _default_pool() -> tuple[TCMGenerator, list[np.ndarray]]:
    """300 series of length 512 from default parameters (shared by tests)."""
    gen = TCMGenerator(min_length=512, max_length=512, freq="h", seed=42)
    series = [gen.generate_single_series(512) for _ in range(300)]
    return gen, series


class TestAPI:
    def test_long_format(self, engine: str) -> None:
        df = _make(engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=64, max_length=128)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = _make(engine).generate(n_series=3)
        df2 = _make(engine).generate(n_series=3)
        vals1 = series_values(df1)
        vals2 = series_values(df2)
        assert vals1.keys() == vals2.keys()
        for uid in vals1:
            np.testing.assert_array_equal(vals1[uid], vals2[uid])

    def test_per_series_variation(self, engine: str) -> None:
        """Each series gets a fresh random SCM, so series differ."""
        vals = series_values(_make(engine, min_length=100).generate(n_series=2))
        a, b = (vals[uid] for uid in sorted(vals))
        n = min(len(a), len(b))
        assert not np.array_equal(a[:n], b[:n])

    def test_bounded_only_edge_kinds(self) -> None:
        """A pool without linear-part kinds (zero-radius path) still works."""
        gen = _make(edge_kinds=["product", "threshold"], n_vars_range=(2, 2))
        values = gen.generate_single_series(128)
        assert values.shape == (128,)
        assert np.isfinite(values).all()


class TestMultivariate:
    """multivariate=True: generate(n_series) observes n_series nodes of one
    shared SCM; multivariate=False keeps the univariate behavior intact."""

    def test_default_univariate_output_unchanged(self) -> None:
        """multivariate=False (the default) preserves the pre-existing
        univariate behavior: golden values captured before the multivariate
        mode was added.

        generate() now dispatches to the Rust batch kernel (GEN_TCM), whose
        RNG streams intentionally differ from numpy; the golden values pin
        the *Python* univariate reference path, so it is forced here by
        disabling the batch params. Compared with rtol=1e-9 (not bitwise):
        the rollout uses libm transcendentals, which differ in the last ulp
        across platforms."""
        with patch.object(TCMGenerator, "_get_batch_params", lambda _self: None):
            df = _make().generate(n_series=3, n_jobs=1)
        vals = np.ascontiguousarray(to_pandas(df)["y"].to_numpy())
        assert len(vals) == 289
        rtol = 1e-9
        np.testing.assert_allclose(
            vals[:5],
            [
                0.00998910969344513,
                -0.23736418532889036,
                0.41136990963985187,
                0.7219779071653518,
                1.6341145762843,
            ],
            rtol=rtol,
        )
        np.testing.assert_allclose(
            vals[-3:],
            [-0.4964426704020574, -1.6551682341882803, -1.0539902051795074],
            rtol=rtol,
        )
        np.testing.assert_allclose(vals.mean(), -0.4391180463501229, rtol=rtol)
        np.testing.assert_allclose(vals.std(), 1.8746817354144987, rtol=rtol)

        single = _make(seed=7).generate_single_series(100)
        np.testing.assert_allclose(
            single[:3],
            [1.3090638958916325, 0.8043032527828881, 0.17325017083576916],
            rtol=rtol,
        )
        np.testing.assert_allclose(single.mean(), 0.5509774146303598, rtol=rtol)

    def test_multivariate_false_equals_default(self) -> None:
        """Passing multivariate=False explicitly matches the default path."""
        df_default = _make(seed=11).generate(n_series=2, n_jobs=1)
        df_flag = _make(seed=11, multivariate=False).generate(n_series=2, n_jobs=1)
        np.testing.assert_array_equal(
            to_pandas(df_default)["y"].to_numpy(),
            to_pandas(df_flag)["y"].to_numpy(),
        )

    def test_multivariate_long_format(self, engine: str) -> None:
        df = _make(engine, multivariate=True).generate(n_series=4)
        assert_long_format(df, n_series=4, min_length=64, max_length=128)

    def test_multivariate_shared_length(self) -> None:
        df = _make(multivariate=True).generate(n_series=4)
        lengths = to_pandas(df).groupby("unique_id", observed=True).size()
        assert lengths.nunique() == 1

    def test_multivariate_start_id(self) -> None:
        vals = series_values(_make(multivariate=True).generate(n_series=2, start_id=3))
        assert set(vals.keys()) == {"3", "4"}

    def test_multivariate_seed_determinism(self, engine: str) -> None:
        df1 = _make(engine, multivariate=True).generate(n_series=3)
        df2 = _make(engine, multivariate=True).generate(n_series=3)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for uid in v1:
            np.testing.assert_array_equal(v1[uid], v2[uid])

    def test_multivariate_series_differ(self) -> None:
        vals = series_values(_make(multivariate=True).generate(n_series=3))
        arrays = list(vals.values())
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                assert not np.array_equal(arrays[i], arrays[j])

    def test_multivariate_clamps_n_vars(self) -> None:
        """When n_series exceeds n_vars_range, the sampled SCM gets at least
        n_series variables."""
        gen = _make(multivariate=True, n_vars_range=(1, 2))
        df = gen.generate(n_series=5)
        assert to_pandas(df)["unique_id"].nunique() == 5
        assert gen._last_scm["n_vars"] >= 5

    def test_multivariate_guards_hold(self) -> None:
        gen = _make(multivariate=True, min_length=256, max_length=256)
        for _ in range(5):
            for values in series_values(gen.generate(n_series=3)).values():
                assert np.isfinite(values).all()
                assert np.abs(values).max() < 1e8
                assert values.std() > 1e-8


class TestValidation:
    def test_bad_n_vars_range(self) -> None:
        with pytest.raises(ValueError, match="n_vars_range"):
            _make(n_vars_range=(0, 2))
        with pytest.raises(ValueError, match="n_vars_range"):
            _make(n_vars_range=(3, 1))

    def test_bad_max_lag_range(self) -> None:
        with pytest.raises(ValueError, match="max_lag_range"):
            _make(max_lag_range=(0, 3))
        with pytest.raises(ValueError, match="max_lag_range"):
            _make(max_lag_range=(5, 2))

    def test_bad_edge_probability_range(self) -> None:
        for bad in [(-0.1, 0.5), (0.5, 0.2), (0.5, 1.5)]:
            with pytest.raises(ValueError, match="edge_probability_range"):
                _make(edge_probability_range=bad)

    def test_bad_coef_and_noise_scale_ranges(self) -> None:
        with pytest.raises(ValueError, match="coef_range"):
            _make(coef_range=(0.0, 0.5))
        with pytest.raises(ValueError, match="coef_range"):
            _make(coef_range=(0.5, 0.2))
        with pytest.raises(ValueError, match="noise_scale_range"):
            _make(noise_scale_range=(0.0, 1.0))

    def test_empty_or_unknown_edge_kinds(self) -> None:
        with pytest.raises(ValueError, match="edge_kinds"):
            _make(edge_kinds=[])
        with pytest.raises(ValueError, match="edge_kinds"):
            _make(edge_kinds=["linear", "sigmoid"])

    def test_empty_or_unknown_noise_types(self) -> None:
        with pytest.raises(ValueError, match="noise_types"):
            _make(noise_types=[])
        with pytest.raises(ValueError, match="noise_types"):
            _make(noise_types=["cauchy"])

    def test_stability_margin_bounds(self) -> None:
        for bad in [0.0, 1.0, 1.5, -0.1]:
            with pytest.raises(ValueError):
                _make(stability_margin=bad)

    def test_heteroscedastic_prob_bounds(self) -> None:
        for bad in [-0.1, 1.1]:
            with pytest.raises(ValueError):
                _make(heteroscedastic_prob=bad)

    def test_clamp_threshold_positive(self) -> None:
        with pytest.raises(ValueError):
            _make(clamp_threshold=0.0)


class TestStability:
    def test_small_pool_guards(self) -> None:
        """Quick guard check kept in the fast suite."""
        gen = _make(min_length=256, max_length=256)
        for _ in range(20):
            values = gen.generate_single_series(256)
            assert np.isfinite(values).all()
            assert np.abs(values).max() < 1e8
            assert values.std() > 1e-8

    def test_fallback_after_redraw_exhaustion(self) -> None:
        """When every rollout fails the guard, the stable AR(1) fallback
        is returned and the counters record the redraws."""
        gen = _make()
        with patch.object(TCMGenerator, "_series_ok", staticmethod(lambda _v: False)):
            values = gen.generate_single_series(100)
        assert values.shape == (100,)
        assert np.isfinite(values).all()
        assert values.std() > 1e-8
        assert gen._redraw_total == _MAX_REDRAWS + 1
        assert gen._fallback_total == 1

    def test_fallback_series_is_stable(self) -> None:
        values = _make()._fallback_series(2000)
        assert np.isfinite(values).all()
        assert np.abs(values).max() < 1e8
        assert values.std() > 1e-8

    @pytest.mark.stats
    def test_many_random_scms_stay_bounded(self) -> None:
        """300 random SCMs at default parameters: all finite, bounded,
        non-constant, with no redraws exhausted."""
        gen, series = _default_pool()
        assert len(series) == 300
        for values in series:
            assert np.isfinite(values).all()
            assert np.abs(values).max() < 1e8
            assert values.std() > 1e-8
        assert gen._fallback_total == 0


@pytest.mark.stats
class TestStructure:
    def test_linear_self_lag_behaves_like_ar1(self) -> None:
        """With one variable, one lag, and linear edges only, the SCM is an
        AR(1); its sample lag-1 ACF matches the sampled coefficient."""
        gen = _make(
            min_length=4000,
            max_length=4000,
            seed=7,
            n_vars_range=(1, 1),
            max_lag_range=(1, 1),
            edge_kinds=["linear"],
            noise_types=["gaussian"],
            heteroscedastic_prob=0.0,
        )
        values = gen.generate_single_series(4000)
        node = gen._last_scm["nodes"][0]
        self_edge = (node["pvar"] == 0) & (node["plag"] == 1)
        phi = float(node["coef"][self_edge][0])
        assert 0.0 < phi < 0.95
        assert_acf(values, lag=1, expected=phi)

    def test_multivar_config_has_lag_structure(self) -> None:
        """A multi-variable, multi-lag linear SCM produces significant
        autocorrelation at the sampled lags."""
        gen = _make(
            min_length=4000,
            max_length=4000,
            seed=42,
            n_vars_range=(3, 3),
            max_lag_range=(6, 6),
            edge_kinds=["linear"],
            noise_types=["gaussian"],
            heteroscedastic_prob=0.0,
        )
        values = gen.generate_single_series(4000)
        scm = gen._last_scm
        assert scm["n_vars"] == 3
        assert scm["max_lag"] == 6
        assert all(node["pvar"].size >= 1 for node in scm["nodes"])
        max_acf = max(abs(sample_acf(values, lag)) for lag in range(1, 7))
        # 5 SEs is ~0.08 at n=4000; the sampled graph gives far more
        assert max_acf > 0.2, f"max |acf(1..6)| = {max_acf:.3f}"

    def test_nonlinear_edges_leave_structured_residuals(self) -> None:
        """Fitting a linear AR model and then adding relu(lag) regressors:
        the RSS reduction is substantial for a relu/threshold SCM and
        negligible for a linear SCM."""

        def nonlinearity_score(values: np.ndarray, order: int = 3) -> float:
            z = _standardize(values)
            n = len(z)
            y = z[order:]
            lags = np.column_stack([z[order - k : n - k] for k in range(1, order + 1)])
            x_lin = np.column_stack([np.ones(len(y)), lags])
            x_aug = np.column_stack([x_lin, np.maximum(lags, 0.0)])
            res_lin = y - x_lin @ np.linalg.lstsq(x_lin, y, rcond=None)[0]
            res_aug = y - x_aug @ np.linalg.lstsq(x_aug, y, rcond=None)[0]
            return 1.0 - (res_aug @ res_aug) / (res_lin @ res_lin)

        config = {
            "min_length": 4000,
            "max_length": 4000,
            "n_vars_range": (1, 2),
            "max_lag_range": (1, 2),
            "coef_range": (0.8, 1.5),
            "noise_types": ["gaussian"],
            "heteroscedastic_prob": 0.0,
        }
        gen_nl = _make(seed=1, edge_kinds=["relu", "threshold"], **config)
        gen_lin = _make(seed=1, edge_kinds=["linear"], **config)
        score_nl = nonlinearity_score(gen_nl.generate_single_series(4000))
        score_lin = nonlinearity_score(gen_lin.generate_single_series(4000))
        assert score_nl > 0.02, f"nonlinear score {score_nl:.4f}"
        assert score_nl > 10.0 * score_lin, f"{score_nl:.4f} vs {score_lin:.4f}"

    def test_multivariate_nodes_cross_correlated(self) -> None:
        """Observed nodes of one shared SCM are cross-correlated at some lag.

        Under independence, max |cross-corr| over lags -8..8 at n=2000 has
        mean ~0.05; nodes sharing a causal graph aggregate far above that
        (with several pairs strongly linked at their sampled edge lags)."""

        def max_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
            best = abs(np.corrcoef(a, b)[0, 1])
            for lag in range(1, max_lag + 1):
                best = max(
                    best,
                    abs(np.corrcoef(a[:-lag], b[lag:])[0, 1]),
                    abs(np.corrcoef(b[:-lag], a[lag:])[0, 1]),
                )
            return best

        pair_vals = []
        for seed in range(6):
            gen = _make(
                seed=seed,
                min_length=2000,
                max_length=2000,
                multivariate=True,
                n_vars_range=(3, 3),
                max_lag_range=(1, 8),
                edge_probability_range=(0.25, 0.35),
                edge_kinds=["linear"],
                noise_types=["gaussian"],
                heteroscedastic_prob=0.0,
            )
            wide = (
                to_pandas(gen.generate(n_series=3))
                .pivot_table(index="ds", columns="unique_id", values="y", observed=True)
                .to_numpy()
            )
            pair_vals.extend(
                max_xcorr(wide[:, i], wide[:, j], 8)
                for i in range(3)
                for j in range(i + 1, 3)
            )
        pair_vals = np.array(pair_vals)
        # measured with these fixed seeds: mean 0.19, max 0.76
        assert pair_vals.mean() > 0.12, f"mean max|xcorr| {pair_vals.mean():.3f}"
        assert pair_vals.max() > 0.4, f"max max|xcorr| {pair_vals.max():.3f}"


@pytest.mark.stats
class TestPoolDiversity:
    """Diversity audit on 300 default-parameter series (tsfm audit metrics)."""

    def test_pool_diversity_metrics(self) -> None:
        from scipy.signal import periodogram

        _, series = _default_pool()
        roughness, entropy, acf12 = [], [], []
        for values in series:
            z = _standardize(values)
            roughness.append(np.std(np.diff(z)) / (np.std(z) + 1e-9))
            _, power = periodogram(z)
            power = power[1:]  # drop the zero-frequency bin
            power = power / power.sum()
            entropy.append(-(power * np.log(power + 1e-12)).sum() / np.log(len(power)))
            acf12.append(sample_acf(z, 12))
        roughness = np.array(roughness)
        entropy = np.array(entropy)
        acf12 = np.array(acf12)

        assert np.median(roughness) >= 1.0, f"roughness {np.median(roughness):.3f}"
        assert np.median(np.abs(acf12)) <= 0.5, (
            f"|acf12| {np.median(np.abs(acf12)):.3f}"
        )
        assert np.median(entropy) >= 0.8, f"entropy {np.median(entropy):.3f}"
        spread = np.percentile(entropy, 90) - np.percentile(entropy, 10)
        assert spread >= 0.2, f"entropy p90-p10 {spread:.3f}"


# --------------------------------------------------------------------------
# Rust batch path (used by generate() when _lib is installed).
# RNG streams differ from numpy, so parity with the Python reference is
# statistical, not bitwise. Appended with the GEN_TCM = 29 port.
# --------------------------------------------------------------------------

from synforecast.base import _GEN_TYPE_MAP, _rs_batch  # noqa: E402

rust_batch = pytest.mark.skipif(
    _rs_batch is None, reason="Rust batch extension (_lib) not available"
)


def _pool_metrics(values: np.ndarray) -> tuple[float, float, float]:
    """(roughness, lag-12 acf, spectral entropy) — the same metric code as
    TestPoolDiversity.test_pool_diversity_metrics, on the standardized
    series."""
    from scipy.signal import periodogram

    z = _standardize(values)
    roughness = float(np.std(np.diff(z)) / (np.std(z) + 1e-9))
    _, power = periodogram(z)
    power = power[1:]  # drop the zero-frequency bin
    power = power / power.sum()
    entropy = float(-(power * np.log(power + 1e-12)).sum() / np.log(len(power)))
    return roughness, sample_acf(z, 12), entropy


@rust_batch
class TestTCMRustBatch:
    def test_batch_path_is_wired(self) -> None:
        gen = _make()
        assert gen._batch_gen_type == _GEN_TYPE_MAP["TCMGenerator"] == 29
        scalars, arrays = gen._get_batch_params()
        assert scalars.shape == (13,)
        # [edge kind ids, noise type ids]
        assert [len(a) for a in arrays] == [5, 3]

    def test_multivariate_disables_batch_params(self) -> None:
        """The multivariate generate() override bypasses batching, and
        _get_batch_params is defensive about it too."""
        assert _make(multivariate=True)._get_batch_params() is None
        assert _make()._get_batch_params() is not None

    def test_generate_seed_determinism(self) -> None:
        df1 = _make().generate(n_series=5)
        df2 = _make().generate(n_series=5)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for uid in v1:
            np.testing.assert_array_equal(v1[uid], v2[uid])

    def test_n_jobs_invariance(self) -> None:
        df1 = _make().generate(n_series=8, n_jobs=1)
        df2 = _make().generate(n_series=8, n_jobs=4)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for uid in v1:
            np.testing.assert_array_equal(v1[uid], v2[uid])

    def test_batch_output_guards(self) -> None:
        """Finite, |x| < 1e8 and non-constant over many random SCMs and
        lengths through the batch path."""
        gen = TCMGenerator(min_length=32, max_length=512, freq="h", seed=123)
        for values in series_values(gen.generate(n_series=100)).values():
            assert np.isfinite(values).all()
            assert np.abs(values).max() < 1e8
            assert values.std() > 1e-8

    def test_bounded_only_edge_kinds_batch(self) -> None:
        """Zero-radius pools (no linear-part kinds) through the batch path."""
        gen = _make(edge_kinds=["product", "threshold"], n_vars_range=(2, 2))
        for values in series_values(gen.generate(n_series=10)).values():
            assert np.isfinite(values).all()
            assert values.std() > 1e-8

    def test_multivariate_generate_unaffected(self) -> None:
        """multivariate=True keeps its Python joint-rollout path: valid
        long-format output, shared length, deterministic per seed."""
        df1 = _make(multivariate=True).generate(n_series=3)
        df2 = _make(multivariate=True).generate(n_series=3)
        assert_long_format(df1, n_series=3, min_length=64, max_length=128)
        lengths = to_pandas(df1).groupby("unique_id", observed=True).size()
        assert lengths.nunique() == 1
        v1, v2 = series_values(df1), series_values(df2)
        for uid in v1:
            np.testing.assert_array_equal(v1[uid], v2[uid])


@pytest.mark.stats
@rust_batch
class TestTCMRustBatchStats:
    """Statistical parity of the Rust batch path with the Python reference."""

    def test_pool_diversity_metrics(self) -> None:
        """Rust-generated pool (through generate()) meets the same diversity
        targets as the Python pool test above."""
        gen = TCMGenerator(min_length=512, max_length=512, freq="h", seed=0)
        pool = series_values(gen.generate(n_series=300))
        assert len(pool) == 300
        rows = np.array([_pool_metrics(v) for v in pool.values()])
        roughness, acf12, entropy = rows[:, 0], rows[:, 1], rows[:, 2]

        assert np.median(roughness) >= 1.0, f"roughness {np.median(roughness):.3f}"
        assert np.median(np.abs(acf12)) <= 0.5, (
            f"|acf12| {np.median(np.abs(acf12)):.3f}"
        )
        assert np.median(entropy) >= 0.8, f"entropy {np.median(entropy):.3f}"
        spread = np.percentile(entropy, 90) - np.percentile(entropy, 10)
        assert spread >= 0.2, f"entropy p90-p10 {spread:.3f}"

    def test_forced_linear_self_lag1_behaves_like_ar1(self) -> None:
        """One variable, one lag, linear edges only through the batch path:
        the SCM is an AR(1) with phi = the per-series radius target, so the
        lag-1 ACF is significantly positive (null SE at n=4000 is ~0.016)
        and below the stability margin."""
        gen = TCMGenerator(
            min_length=4000,
            max_length=4000,
            freq="h",
            seed=7,
            n_vars_range=(1, 1),
            max_lag_range=(1, 1),
            edge_kinds=["linear"],
            noise_types=["gaussian"],
            heteroscedastic_prob=0.0,
        )
        for uid, values in series_values(gen.generate(n_series=5)).items():
            acf1 = sample_acf(values, 1)
            assert 0.05 < acf1 < 0.96, f"series {uid}: lag-1 acf {acf1:.3f}"

    def test_batch_pool_matches_python_pool_medians(self) -> None:
        """Median diversity metrics of the Rust pool land near the Python
        reference pool's (loose bands: statistical, not bitwise, parity)."""
        _, py_series = _default_pool()
        py_rows = np.array([_pool_metrics(v) for v in py_series])
        gen = TCMGenerator(min_length=512, max_length=512, freq="h", seed=1)
        rs_pool = series_values(gen.generate(n_series=300))
        rs_rows = np.array([_pool_metrics(v) for v in rs_pool.values()])
        for col, tol, name in ((0, 0.25, "roughness"), (2, 0.1, "entropy")):
            py_med = np.median(py_rows[:, col])
            rs_med = np.median(rs_rows[:, col])
            assert abs(py_med - rs_med) <= tol, (
                f"{name}: python median {py_med:.3f} vs rust median {rs_med:.3f}"
            )
