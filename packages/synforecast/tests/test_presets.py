"""Tests for the pool presets."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from synforecast import SynSet, interpretable_pool, pretraining_pool
from synforecast.base import BaseGenerator
from synforecast.generators import (
    ETSGenerator,
    GaussianProcessGenerator,
    IntermittentDemandGenerator,
    SARIMAGenerator,
)
from synforecast.presets import _seasonal_period_from_freq


class TestInterpretablePool:
    """Tests for the interpretable_pool function."""

    def test_returns_list_of_generators(self) -> None:
        """Test that interpretable_pool returns a list of BaseGenerator instances."""
        generators = interpretable_pool()
        assert isinstance(generators, list)
        assert len(generators) == 42
        for gen in generators:
            assert isinstance(gen, BaseGenerator)

    def test_custom_length_and_freq(self) -> None:
        """Test that custom base parameters are applied to all generators."""
        generators = interpretable_pool(
            min_length=50, max_length=75, freq="h", engine="polars"
        )
        for gen in generators:
            assert gen.min_length == 50
            assert gen.max_length == 75
            assert gen.freq == "h"

    def test_custom_seed(self) -> None:
        """Test that seeds are offset per generator."""
        generators = interpretable_pool(seed=100)
        seeds = [gen.seed for gen in generators]
        assert seeds[0] == 100
        # Every generator gets a distinct offset from the base seed; the
        # list order interleaves niches, so offsets are unique but not
        # consecutive along the list.
        assert set(seeds) == set(range(100, 142))

    def test_none_seed(self) -> None:
        """Test that seed=None produces None seeds for all generators."""
        generators = interpretable_pool(seed=None)
        for gen in generators:
            assert gen.seed is None

    def test_base_kwargs_passed(self) -> None:
        """Test that extra kwargs are forwarded to all generators."""
        generators = interpretable_pool(engine="pandas")
        for gen in generators:
            assert gen.engine == "pandas"

    def test_with_synset(self) -> None:
        """Test integration with SynSet."""
        generators = interpretable_pool(
            min_length=30, max_length=30, freq="D", engine="polars"
        )
        dataset = SynSet(generators)
        df = dataset.generate(n_series_per_generator=1)
        assert isinstance(df, pl.DataFrame)
        n_series = df["unique_id"].n_unique()
        assert n_series == 42

    def test_all_generators_produce_output(self) -> None:
        """Test that every generator in the pool produces valid time series."""
        generators = interpretable_pool(
            min_length=50, max_length=50, freq="D", engine="polars"
        )
        for i, gen in enumerate(generators):
            df = gen.generate(n_series=1)
            assert isinstance(df, pl.DataFrame), (
                f"Generator {i} ({gen.alias}) did not return a DataFrame"
            )
            assert len(df) > 0, f"Generator {i} ({gen.alias}) returned empty DataFrame"

    def test_reproducibility(self) -> None:
        """Test that the same seed produces identical results."""
        gen1 = interpretable_pool(
            min_length=50, max_length=50, freq="D", engine="polars", seed=42
        )
        gen2 = interpretable_pool(
            min_length=50, max_length=50, freq="D", engine="polars", seed=42
        )

        df1 = gen1[0].generate(n_series=1)
        df2 = gen2[0].generate(n_series=1)

        assert df1["y"].to_list() == df2["y"].to_list()

    def test_diverse_generator_types(self) -> None:
        """Test that the pool contains generators from multiple classes."""
        generators = interpretable_pool()
        class_names = {type(gen).__name__ for gen in generators}
        assert len(class_names) == 15
        expected = {
            "SARIMAGenerator",
            "ETSGenerator",
            "FractionalBrownianMotionGenerator",
            "RegimeSwitchingGenerator",
            "GARCHGenerator",
            "CyclicGenerator",
            "IntermittentDemandGenerator",
            "EnergyLoadGenerator",
            "IoTSensorGenerator",
            "VitalSignsGenerator",
            "GaussianProcessGenerator",
            "ChaoticSystemGenerator",
            "INARGenerator",
            "BoundedProcessGenerator",
            "LevyProcessGenerator",
        }
        assert class_names == expected

    def test_prefix_spans_all_niches(self) -> None:
        """The pool is interleaved round-robin across niches: the first 15
        entries cover all 15 generator classes, so any prefix is as
        behaviorally diverse as possible."""
        generators = interpretable_pool()
        first_15_classes = [type(gen).__name__ for gen in generators[:15]]
        assert len(set(first_15_classes)) == 15
        # And every prefix of length k <= 15 has k distinct classes.
        for k in range(1, 16):
            assert len({type(g).__name__ for g in generators[:k]}) == k

    def test_interleaving_preserves_pool_contents(self) -> None:
        """Interleaving reorders the pool but keeps the same 42 configured
        variants (same class/seed-offset pairs)."""
        generators = interpretable_pool(seed=0)
        assert len(generators) == 42
        seeds_by_class: dict[str, set[int]] = {}
        for gen in generators:
            seeds_by_class.setdefault(type(gen).__name__, set()).add(gen.seed)
        # Variant counts per niche are unchanged.
        expected_counts = {
            "SARIMAGenerator": 5,
            "ETSGenerator": 4,
            "FractionalBrownianMotionGenerator": 3,
            "RegimeSwitchingGenerator": 2,
            "GARCHGenerator": 2,
            "CyclicGenerator": 2,
            "IntermittentDemandGenerator": 3,
            "EnergyLoadGenerator": 2,
            "IoTSensorGenerator": 3,
            "VitalSignsGenerator": 3,
            "GaussianProcessGenerator": 4,
            "ChaoticSystemGenerator": 3,
            "INARGenerator": 2,
            "BoundedProcessGenerator": 2,
            "LevyProcessGenerator": 2,
        }
        assert {c: len(s) for c, s in seeds_by_class.items()} == expected_counts

    def test_interpretable_pool_with_patterns(self) -> None:
        """Test interpretable_pool with anomalies, changepoints, and missing data."""
        generators = interpretable_pool(
            min_length=100,
            max_length=100,
            freq="D",
            engine="polars",
            seed=42,
            anomalies=True,
            changepoints=True,
            missing_data=True,
        )
        assert len(generators) > 0
        for gen in generators:
            df = gen.generate(n_series=1)
            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0


class TestPretrainingPool:
    """Tests for the pretraining_pool function."""

    META = {"TSIGenerator", "TCMGenerator", "KernelSynthGenerator", "MARGenerator"}

    def test_includes_meta_and_balanced_by_default(self) -> None:
        from synforecast import pretraining_pool

        pool = pretraining_pool(min_length=64, max_length=64, engine="polars")
        classes = {type(g).__name__ for g in pool}
        # Meta-generators that interpretable_pool excludes are present...
        assert classes >= self.META
        # ...alongside the full balanced pool.
        assert len(pool) == 42 + 4 * 3

    def test_meta_only(self) -> None:
        from synforecast import pretraining_pool

        pool = pretraining_pool(
            include_interpretable=False,
            n_meta_variants=2,
            min_length=64,
            max_length=64,
            engine="polars",
        )
        assert len(pool) == 8
        assert {type(g).__name__ for g in pool} == self.META

    def test_all_seeds_unique(self) -> None:
        from synforecast import pretraining_pool

        seeds = [g.seed for g in pretraining_pool(seed=42)]
        assert len(set(seeds)) == len(seeds)

    def test_none_seed(self) -> None:
        from synforecast import pretraining_pool

        assert all(g.seed is None for g in pretraining_pool(seed=None))

    def test_rejects_zero_variants(self) -> None:
        import pytest

        from synforecast import pretraining_pool

        with pytest.raises(ValueError):
            pretraining_pool(n_meta_variants=0)

    def test_with_synset(self) -> None:
        from synforecast import SynSet, pretraining_pool

        pool = pretraining_pool(
            min_length=48, max_length=48, freq="D", engine="polars", seed=1
        )
        df = SynSet(pool).generate(n_series_per_generator=1)
        assert isinstance(df, pl.DataFrame)
        assert df["unique_id"].n_unique() == len(pool)
        assert len(df) > 0


def _skip_if_alias_unsupported(freq: str) -> None:
    """Skip when the installed pandas does not know this offset alias.

    Aliases such as "ME"/"YE" appear in pandas 2.2 and legacy ones such as
    "H"/"T"/"A" are removed later, so each parametrization is only checked
    where the alias parses.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            pd.tseries.frequencies.to_offset(freq)
        except ValueError:
            pytest.skip(f"pandas {pd.__version__} does not support alias {freq!r}")


class TestSeasonalPeriodFromFreq:
    """Tests for deriving the seasonal period from a frequency."""

    @pytest.mark.parametrize(
        ("freq", "expected"),
        [
            ("s", 60),
            ("min", 60),
            ("5min", 12),
            ("15min", 4),
            ("30min", 2),
            ("h", 24),
            ("H", 24),
            ("2h", 12),
            ("D", 7),
            ("1d", 7),
            ("B", 5),
            ("W", 52),
            ("W-MON", 52),
            ("MS", 12),
            ("ME", 12),
            ("3MS", 4),
            ("QS", 4),
            ("QE-DEC", 4),
            ("YS", 1),
            ("YE", 1),
        ],
    )
    def test_calendar_conventions(self, freq: str, expected: int) -> None:
        _skip_if_alias_unsupported(freq)
        assert _seasonal_period_from_freq(freq) == expected

    @pytest.mark.parametrize(
        ("freq", "expected"),
        [
            ("S", 60),
            ("T", 60),
            ("15T", 4),
            ("H", 24),
            ("2H", 12),
            ("M", 12),
            ("Q", 4),
            ("Q-DEC", 4),
            ("A", 1),
            ("Y", 1),
            ("A-DEC", 1),
        ],
    )
    def test_legacy_aliases(self, freq: str, expected: int) -> None:
        # Aliases used by pandas 2.0-2.1 and deprecated since; the lookup must
        # not depend on the version-specific rule code.
        _skip_if_alias_unsupported(freq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert _seasonal_period_from_freq(freq) == expected

    def test_integer_freq_uses_default(self) -> None:
        assert _seasonal_period_from_freq(1) == 12
        assert _seasonal_period_from_freq(7, default=4) == 4

    def test_offset_without_convention_uses_default(self) -> None:
        assert _seasonal_period_from_freq("ms") == 12

    def test_never_below_one(self) -> None:
        assert _seasonal_period_from_freq("36h") == 1
        assert _seasonal_period_from_freq("5YS") == 1


def _seasonal_period_of(gen: BaseGenerator) -> int | None:
    """Seasonal period of a pool member, or None for non-seasonal variants."""
    if isinstance(gen, SARIMAGenerator) and (gen.P or gen.D or gen.Q):
        return gen.seasonal_period
    if isinstance(gen, ETSGenerator) and gen.seasonal_type is not None:
        return gen.seasonal_period
    if (
        isinstance(gen, IntermittentDemandGenerator)
        and gen.intermittent_pattern == "seasonal"
    ):
        return gen.seasonal_period
    if isinstance(gen, GaussianProcessGenerator) and gen.kernel == "periodic":
        return int(gen.period)
    return None


class TestInterpretablePoolSeasonalPeriod:
    """Tests that interpretable_pool's seasonal variants follow freq."""

    @staticmethod
    def _seasonal_periods(generators: list[BaseGenerator]) -> set[int]:
        periods = {_seasonal_period_of(gen) for gen in generators}
        periods.discard(None)
        return periods  # type: ignore[return-value]

    @pytest.mark.parametrize(
        ("freq", "expected"),
        [("h", 24), ("H", 24), ("D", 7), ("W", 52), ("MS", 12), ("QS", 4), (1, 12)],
    )
    def test_derived_from_freq(self, freq: str | int, expected: int) -> None:
        if isinstance(freq, str):
            _skip_if_alias_unsupported(freq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            generators = interpretable_pool(freq=freq)
        assert self._seasonal_periods(generators) == {expected}

    def test_covers_every_seasonal_variant(self) -> None:
        generators = interpretable_pool(freq="h")
        seasonal = [g for g in generators if _seasonal_period_of(g) is not None]
        # 2 SARIMA + 2 ETS + 1 intermittent + 1 GP
        assert len(seasonal) == 6

    def test_non_seasonal_variants_unchanged(self) -> None:
        generators = interpretable_pool(freq="h")
        plain = [
            gen
            for gen in generators
            if isinstance(gen, SARIMAGenerator) and not (gen.P or gen.D or gen.Q)
        ]
        assert len(plain) == 3
        assert all(gen.seasonal_period == 1 for gen in plain)

    def test_explicit_override(self) -> None:
        generators = interpretable_pool(freq="h", seasonal_period=168)
        assert self._seasonal_periods(generators) == {168}

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError, match="seasonal_period"):
            interpretable_pool(seasonal_period=0)

    def test_generates_with_derived_period(self) -> None:
        generators = interpretable_pool(
            min_length=60, max_length=60, freq="h", engine="polars"
        )
        for gen in generators:
            df = gen.generate(n_series=1)
            assert len(df) == 60, gen.alias

    def test_pretraining_pool_forwards_override(self) -> None:
        pool = pretraining_pool(
            min_length=64, max_length=64, freq="h", seasonal_period=168
        )
        assert self._seasonal_periods(pool) == {168}

    def test_pretraining_pool_derives_from_freq(self) -> None:
        pool = pretraining_pool(min_length=64, max_length=64, freq="MS")
        assert self._seasonal_periods(pool) == {12}


class TestSeasonalPool:
    """Tests for the opt-in seasonal_pool preset."""

    def test_eight_configured_public_generators(self) -> None:
        from synforecast import seasonal_pool
        from synforecast.generators import ETSGenerator, SeasonalGenerator, TSIGenerator

        pool = seasonal_pool(min_length=48, max_length=48, freq="MS", engine="polars")
        assert len(pool) == 8
        aliases = [g.alias for g in pool]
        assert len(set(aliases)) == 8
        assert all(
            isinstance(g, (TSIGenerator, ETSGenerator, SeasonalGenerator)) for g in pool
        )
        seeds = [g.seed for g in pool]
        assert seeds == [42 + i for i in range(8)]

    @pytest.mark.parametrize(("freq", "expected"), [("QS", 4), ("MS", 12), ("h", 24)])
    def test_period_derived_from_freq(self, freq: str, expected: int) -> None:
        from synforecast import seasonal_pool
        from synforecast.generators import ETSGenerator, SeasonalGenerator, TSIGenerator

        for gen in seasonal_pool(freq=freq):
            if isinstance(gen, TSIGenerator):
                assert gen.seasonal_periods == [float(expected)]
            elif isinstance(gen, ETSGenerator):
                assert gen.seasonal_period == expected
                assert len(gen.seasonal) == expected
            else:
                assert isinstance(gen, SeasonalGenerator)
                assert gen.seasonality_period == expected

    def test_explicit_period_and_yearly_rejection(self) -> None:
        from synforecast import seasonal_pool

        pool = seasonal_pool(freq="D", seasonal_period=5)
        assert pool[0].seasonal_periods == [5.0]
        with pytest.raises(ValueError, match="at least 2"):
            seasonal_pool(freq="YS")

    def test_multiplicative_factors_have_unit_mean(self) -> None:
        from synforecast import seasonal_pool

        ets = [g for g in seasonal_pool(freq="QS") if g.alias == "ets_MAdM_noisy"][0]
        assert sum(ets.seasonal) / len(ets.seasonal) == pytest.approx(1.0)
        assert max(ets.seasonal) > min(ets.seasonal)

    def test_generates_finite_series_and_adds_to_interpretable_pool(self) -> None:
        from synforecast import SynSet, interpretable_pool, seasonal_pool

        pool = interpretable_pool(
            min_length=60, max_length=60, freq="QS", engine="polars"
        )
        extra = seasonal_pool(min_length=60, max_length=60, freq="QS", engine="polars")
        assert len(pool + extra) == 50
        df = SynSet(extra).generate(n_series_per_generator=2, n_jobs=1)
        values = df["y"].to_numpy()
        assert df["unique_id"].n_unique() == 16
        assert np.isfinite(values).all()

    def test_none_seed_and_reproducibility(self) -> None:
        from synforecast import seasonal_pool

        assert all(g.seed is None for g in seasonal_pool(freq="MS", seed=None))
        first = seasonal_pool(freq="MS", seed=3)[0].generate(1, n_jobs=1)
        second = seasonal_pool(freq="MS", seed=3)[0].generate(1, n_jobs=1)
        assert first["y"].tolist() == second["y"].tolist()


class TestDeprecatedBalancedPool:
    def test_alias_warns_and_returns_identical_generators(self) -> None:
        from synforecast import balanced_pool, interpretable_pool

        with pytest.warns(DeprecationWarning, match="interpretable_pool") as record:
            legacy = balanced_pool(min_length=30, max_length=30, freq="QS", seed=5)
        assert record[0].filename == __file__
        current = interpretable_pool(min_length=30, max_length=30, freq="QS", seed=5)
        assert [g.model_dump(mode="json") for g in legacy] == [
            g.model_dump(mode="json") for g in current
        ]

    def test_include_balanced_keyword_warns_and_still_works(self) -> None:
        with pytest.warns(DeprecationWarning, match="include_interpretable") as record:
            legacy = pretraining_pool(include_balanced=False, n_meta_variants=1)
        assert record[0].filename == __file__
        current = pretraining_pool(include_interpretable=False, n_meta_variants=1)
        assert [g.model_dump(mode="json") for g in legacy] == [
            g.model_dump(mode="json") for g in current
        ]

    def test_current_names_emit_no_deprecation_warning(self) -> None:
        from synforecast import seasonal_pool

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            interpretable_pool()
            pretraining_pool(freq="MS", include_seasonal=True)
            seasonal_pool(freq="MS")


class TestPretrainingPoolSeasonalFlag:
    def test_default_unchanged_and_flag_appends_seasonal_pool(self) -> None:
        from synforecast import pretraining_pool, seasonal_pool

        default = pretraining_pool(
            min_length=64, max_length=64, freq="MS", engine="polars"
        )
        extended = pretraining_pool(
            min_length=64,
            max_length=64,
            freq="MS",
            engine="polars",
            include_seasonal=True,
        )
        assert len(default) == 54 and len(extended) == 62
        assert [g.model_dump(mode="json") for g in extended[:54]] == [
            g.model_dump(mode="json") for g in default
        ]
        expected = seasonal_pool(
            min_length=64, max_length=64, freq="MS", seed=42 + 2000, engine="polars"
        )
        assert [g.model_dump(mode="json") for g in extended[54:]] == [
            g.model_dump(mode="json") for g in expected
        ]
        seeds = [g.seed for g in extended]
        assert len(set(seeds)) == len(seeds)

    def test_flag_skips_yearly_and_works_meta_only(self) -> None:
        from synforecast import pretraining_pool

        with pytest.warns(UserWarning, match="include_seasonal=True was ignored"):
            yearly = pretraining_pool(freq="YS", include_seasonal=True)
        assert len(yearly) == 54
        meta = pretraining_pool(
            freq="QS",
            include_interpretable=False,
            include_seasonal=True,
            n_meta_variants=1,
        )
        assert len(meta) == 4 + 8


class TestSeasonalPoolIsActuallySeasonal:
    """The preset exists to fill a seasonal gap; pin that it does."""

    @staticmethod
    def _median_seasonal_strength(pool) -> float:
        from synforecast import SynSet, compute_features

        df = SynSet(pool).generate(n_series_per_generator=2, n_jobs=1)
        # Some pool members emit missing values by design; feature extraction
        # requires finite panels, so drop those series rather than impute.
        bad = set(df.loc[~np.isfinite(df["y"]), "unique_id"])
        features = compute_features(df[~df["unique_id"].isin(bad)], seasonal_period=4)
        return float(np.nanmedian(features["seasonal_strength"].to_numpy()))

    def test_far_more_seasonal_than_the_pool_it_supplements(self) -> None:
        from synforecast import interpretable_pool, seasonal_pool

        shared = {
            "min_length": 64,
            "max_length": 64,
            "freq": "QS",
            "seed": 1,
            "engine": "pandas",
        }
        seasonal = self._median_seasonal_strength(seasonal_pool(**shared))
        interpretable = self._median_seasonal_strength(interpretable_pool(**shared))
        # Measured 0.62 vs 0.06; the margin guards against a generator whose
        # seasonal amplitude gets swamped by its own noise range.
        assert seasonal > 0.35
        assert interpretable < 0.20
        assert seasonal > 3 * interpretable

    def test_every_instance_generates_at_its_default_length(self) -> None:
        from synforecast import seasonal_pool

        for generator in seasonal_pool(freq="QS"):
            values = generator.generate(1, n_jobs=1)["y"].to_numpy()
            assert len(values) == 200
            assert np.isfinite(values).all(), generator.alias


class TestPretrainingPoolSeasonalPeriodForwarding:
    def test_explicit_period_reaches_the_seasonal_instances(self) -> None:
        from synforecast import pretraining_pool

        pool = pretraining_pool(
            min_length=64,
            max_length=64,
            freq="MS",
            seasonal_period=4,
            include_seasonal=True,
            engine="polars",
        )
        seasonal = [g for g in pool if g.alias == "ets_MAdM_noisy"]
        assert len(seasonal) == 1
        assert len(seasonal[0].seasonal) == 4

    def test_skip_on_yearly_warns_rather_than_silently_no_op(self) -> None:
        from synforecast import pretraining_pool

        with pytest.warns(UserWarning, match="include_seasonal=True was ignored"):
            yearly = pretraining_pool(freq="YS", include_seasonal=True)
        assert len(yearly) == len(pretraining_pool(freq="YS"))


class TestSeasonalPoolArguments:
    def test_freq_is_required(self) -> None:
        from synforecast import seasonal_pool

        with pytest.raises(TypeError, match="freq"):
            seasonal_pool()  # type: ignore[call-arg]

    @pytest.mark.parametrize("key", ["alias", "level", "phi"])
    def test_per_generator_keys_are_rejected(self, key: str) -> None:
        from synforecast import seasonal_pool

        with pytest.raises(ValueError, match=f"seasonal_pool sets .*{key}"):
            seasonal_pool(freq="MS", **{key: "x" if key == "alias" else 1.0})

    def test_alias_through_pretraining_pool_is_rejected_clearly(self) -> None:
        # alias is accepted by every pretraining generator, so it reaches
        # seasonal_pool, which assigns its own aliases.
        with pytest.raises(ValueError, match="seasonal_pool sets .*alias"):
            pretraining_pool(freq="MS", include_seasonal=True, alias="x")

    def test_explicit_period_one_is_rejected(self) -> None:
        from synforecast import seasonal_pool

        with pytest.raises(ValueError, match="at least 2"):
            seasonal_pool(freq="MS", seasonal_period=1)


class TestPretrainingPoolSeasonalPeriodValidation:
    @pytest.mark.parametrize("include_interpretable", [True, False])
    def test_non_positive_period_raises_regardless_of_composition(
        self, include_interpretable: bool
    ) -> None:
        with pytest.raises(ValueError, match="seasonal_period must be >= 1"):
            pretraining_pool(
                include_interpretable=include_interpretable,
                include_seasonal=True,
                seasonal_period=0,
            )

    def test_explicit_period_one_warning_does_not_suggest_passing_it(self) -> None:
        with pytest.warns(
            UserWarning, match="seasonal_period=1 has no seasonal cycle"
        ) as record:
            pool = pretraining_pool(freq="MS", include_seasonal=True, seasonal_period=1)
        assert "explicitly" not in str(record[0].message)
        assert len(pool) == 54
