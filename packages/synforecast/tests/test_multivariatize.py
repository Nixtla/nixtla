"""Tests for Multivariatizer (generic cross-series coupling wrapper)."""

import numpy as np
import pytest

from synforecast import Multivariatizer
from synforecast.generators import RandomWalkGenerator, TSIGenerator
from tests.helpers import assert_long_format, series_values, to_pandas
from tests.test_tsi import pool_metrics

BASE = {"min_length": 30, "max_length": 60, "freq": "D"}


def _stationary_base(seed: int | None = None, **overrides) -> TSIGenerator:
    """TSI base without trend/seasonality: independent draws of this base
    have near-zero cross-correlation, giving a clean dependence baseline."""
    params = {
        "min_length": 512,
        "max_length": 512,
        "freq": "h",
        "trend_types": ["none"],
        "n_seasonal_range": (0, 0),
        "irregular_types": ["gaussian"],
        "seed": seed,
    }
    params.update(overrides)
    return TSIGenerator(**params)


def _wide(df) -> np.ndarray:
    """Pivot a long frame to a (length, n_series) array, ids sorted as ints."""
    pdf = to_pandas(df).pivot_table(
        index="ds", columns="unique_id", values="y", observed=True
    )
    cols = sorted(pdf.columns, key=lambda c: int(c))
    return pdf[cols].to_numpy()


def _mean_abs_offdiag_corr(wide: np.ndarray) -> float:
    """Mean absolute cotemporaneous correlation across channel pairs."""
    k = wide.shape[1]
    corr = np.corrcoef(wide.T)
    return float(np.abs(corr[np.triu_indices(k, 1)]).mean())


class TestAPI:
    def test_long_format(self, engine: str) -> None:
        mv = Multivariatizer(base=TSIGenerator(**BASE, engine=engine), seed=42)
        df = mv.generate(n_series=4)
        assert_long_format(df, n_series=4, min_length=30, max_length=60)

    def test_channels_share_one_length(self, engine: str) -> None:
        mv = Multivariatizer(base=TSIGenerator(**BASE, engine=engine), seed=0)
        pdf = to_pandas(mv.generate(n_series=5))
        lengths = pdf.groupby("unique_id", observed=True).size()
        assert lengths.nunique() == 1

    def test_start_id_offsets(self, engine: str) -> None:
        mv = Multivariatizer(base=TSIGenerator(**BASE, engine=engine), seed=1)
        vals = series_values(mv.generate(n_series=3, start_id=5))
        assert set(vals.keys()) == {"5", "6", "7"}

    def test_seed_determinism(self, engine: str) -> None:
        base = TSIGenerator(**BASE, engine=engine)
        df1 = Multivariatizer(base=base, seed=7).generate(n_series=3)
        df2 = Multivariatizer(base=base, seed=7).generate(n_series=3)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for uid in v1:
            np.testing.assert_array_equal(v1[uid], v2[uid])

    def test_different_seeds_differ(self) -> None:
        base = TSIGenerator(**BASE)
        df1 = Multivariatizer(base=base, seed=1).generate(n_series=2)
        df2 = Multivariatizer(base=base, seed=2).generate(n_series=2)
        assert not np.array_equal(
            to_pandas(df1)["y"].to_numpy(), to_pandas(df2)["y"].to_numpy()
        )

    def test_output_independent_of_base_seed(self) -> None:
        """The multivariatizer reseeds the base per call, so the base's own
        seed never influences the output."""
        df1 = Multivariatizer(base=TSIGenerator(**BASE, seed=3), seed=5).generate(
            n_series=3
        )
        df2 = Multivariatizer(base=TSIGenerator(**BASE, seed=99), seed=5).generate(
            n_series=3
        )
        np.testing.assert_array_equal(
            to_pandas(df1)["y"].to_numpy(), to_pandas(df2)["y"].to_numpy()
        )

    def test_original_base_not_mutated(self) -> None:
        """generate() works on a reseeded copy; the wrapped generator's rng
        state is untouched."""
        base = TSIGenerator(**BASE, seed=3)
        Multivariatizer(base=base, seed=0).generate(n_series=4)
        after = base.generate_single_series(50)
        fresh = TSIGenerator(**BASE, seed=3).generate_single_series(50)
        np.testing.assert_array_equal(after, fresh)

    def test_single_channel(self) -> None:
        df = Multivariatizer(base=TSIGenerator(**BASE), seed=0).generate(n_series=1)
        assert_long_format(df, n_series=1, min_length=30, max_length=60)

    def test_wraps_any_base_generator(self) -> None:
        base = RandomWalkGenerator(min_length=50, max_length=50, freq="D")
        df = Multivariatizer(base=base, seed=0).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=50)

    def test_couplings_compose(self) -> None:
        mv = Multivariatizer(
            base=_stationary_base(), couplings=["mixing", "leadlag"], seed=0
        )
        df = mv.generate(n_series=4)
        assert mv.last_recipe["couplings"] == ["mixing", "leadlag"]
        assert len(mv.last_recipe["leadlag"]) >= 1
        assert np.isfinite(to_pandas(df)["y"].to_numpy()).all()

    def test_guards_hold_on_output(self) -> None:
        mv = Multivariatizer(base=TSIGenerator(**BASE), seed=11)
        for _ in range(5):
            vals = series_values(mv.generate(n_series=4))
            for v in vals.values():
                assert np.isfinite(v).all()
                assert np.abs(v).max() < 1e8
                assert v.std() > 1e-8


class TestValidation:
    def test_unknown_coupling(self) -> None:
        with pytest.raises(ValueError, match="couplings"):
            Multivariatizer(base=TSIGenerator(**BASE), couplings=["bogus"])

    def test_empty_couplings(self) -> None:
        with pytest.raises(ValueError, match="couplings"):
            Multivariatizer(base=TSIGenerator(**BASE), couplings=[])

    def test_bad_mixing_strength_range(self) -> None:
        for bad in [(-0.1, 0.5), (0.5, 0.2), (0.5, 1.0)]:
            with pytest.raises(ValueError, match="mixing_strength_range"):
                Multivariatizer(base=TSIGenerator(**BASE), mixing_strength_range=bad)

    def test_bad_lag_range(self) -> None:
        for bad in [(0, 5), (5, 2)]:
            with pytest.raises(ValueError, match="lag_range"):
                Multivariatizer(base=TSIGenerator(**BASE), lag_range=bad)

    def test_bad_noise_scale_range(self) -> None:
        for bad in [(0.0, 0.1), (0.2, 0.1)]:
            with pytest.raises(ValueError, match="noise_scale_range"):
                Multivariatizer(base=TSIGenerator(**BASE), noise_scale_range=bad)

    def test_bad_n_series(self) -> None:
        with pytest.raises(ValueError, match="n_series"):
            Multivariatizer(base=TSIGenerator(**BASE), seed=0).generate(n_series=0)


@pytest.mark.stats
class TestDependence:
    def test_mixing_correlates_channels_above_baseline(self) -> None:
        """Mean absolute cotemporaneous cross-correlation of mixed channels
        is far above the independent-channel baseline of the same base."""
        mv = Multivariatizer(
            base=_stationary_base(),
            couplings=["mixing"],
            mixing_strength_range=(0.5, 0.9),
            seed=0,
        )
        coupled = np.mean(
            [_mean_abs_offdiag_corr(_wide(mv.generate(n_series=4))) for _ in range(10)]
        )

        indep = _stationary_base(seed=1)
        baseline = np.mean(
            [
                _mean_abs_offdiag_corr(
                    np.column_stack(
                        [indep.generate_single_series(512) for _ in range(4)]
                    )
                )
                for _ in range(10)
            ]
        )
        # measured: coupled ~0.31, baseline ~0.04
        assert coupled > 0.15, f"coupled mean |corr| {coupled:.3f}"
        assert coupled > 4.0 * baseline, f"{coupled:.3f} vs baseline {baseline:.3f}"

    def test_leadlag_correlation_peaks_at_sampled_lag(self) -> None:
        """For lead-lag coupled pairs, |cross-correlation| at the sampled lag
        far exceeds the cotemporaneous (lag-0) correlation."""
        mv = Multivariatizer(base=_stationary_base(), couplings=["leadlag"], seed=3)
        lag_corrs, zero_corrs = [], []
        for _ in range(10):
            wide = _wide(mv.generate(n_series=4))
            pairs = mv.last_recipe["leadlag"]
            assert len(pairs) >= 1
            for pair in pairs:
                src, dst, lag = pair["src"], pair["dst"], pair["lag"]
                zero_corrs.append(abs(np.corrcoef(wide[:, src], wide[:, dst])[0, 1]))
                lag_corrs.append(
                    abs(np.corrcoef(wide[:-lag, src], wide[lag:, dst])[0, 1])
                )
        lag_corrs, zero_corrs = np.array(lag_corrs), np.array(zero_corrs)
        # measured: lag corr ~0.99 per pair (noise <= 0.2), lag-0 ~0.03
        assert lag_corrs.min() > 0.8, f"min lag corr {lag_corrs.min():.3f}"
        margin = np.median(lag_corrs) - np.median(zero_corrs)
        assert margin > 0.5, f"lag vs lag-0 margin {margin:.3f}"


@pytest.mark.stats
class TestDiversityPreserved:
    """Multivariatized TSI channels still meet the synthetic-pool diversity
    targets (real-data audit: roughness 1.24, |lag-12 acf| 0.26, entropy
    0.88); coupling must not wash the univariate diversity out."""

    def test_multivariatized_tsi_pool_diversity(self) -> None:
        base = TSIGenerator(min_length=512, max_length=512, freq="D")
        mv = Multivariatizer(base=base, seed=0)
        rows = []
        for _ in range(60):
            wide = _wide(mv.generate(n_series=5))
            for j in range(wide.shape[1]):
                rows.append(pool_metrics(wide[:, j]))
        rows = np.array(rows)
        roughness = rows[:, 0]
        abs_acf12 = np.abs(rows[:, 1])
        entropy = rows[:, 2]

        assert np.median(roughness) >= 1.0, f"roughness {np.median(roughness):.3f}"
        assert np.median(abs_acf12) <= 0.5, f"|acf12| {np.median(abs_acf12):.3f}"
        assert np.median(entropy) >= 0.8, f"entropy {np.median(entropy):.3f}"
