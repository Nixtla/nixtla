"""Tests for ClickstreamGenerator."""

import numpy as np
import pytest
from pydantic import ValidationError

from synforecast.generators import ClickstreamGenerator
from tests.helpers import assert_long_format, sample_acf, series_values


def make_gen(**kwargs):
    params = {
        "min_length": 150,
        "max_length": 200,
        "freq": "h",
        "seed": 42,
    }
    params.update(kwargs)
    return ClickstreamGenerator(**params)


class TestClickstreamAPI:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=150, max_length=200)

    def test_seed_determinism(self) -> None:
        v1 = series_values(make_gen(seed=7).generate(n_series=3))
        v2 = series_values(make_gen(seed=7).generate(n_series=3))
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    @pytest.mark.parametrize(
        "output_type", ["sessions", "pageviews", "conversions", "bounce_rate"]
    )
    def test_output_types_valid(self, output_type: str) -> None:
        values = make_gen(output_type=output_type).generate_single_series(200)
        assert values.shape == (200,)
        assert np.isfinite(values).all()
        assert (values >= 0).all()
        if output_type == "bounce_rate":
            assert (values <= 1).all()

    @pytest.mark.parametrize(
        "source", ["organic", "paid", "direct", "referral", "mixed"]
    )
    def test_traffic_sources_valid(self, source: str) -> None:
        values = make_gen(traffic_source=source).generate_single_series(150)
        assert np.isfinite(values).all()

    def test_generate_full_metrics(self) -> None:
        metrics = make_gen().generate_full_metrics(n_series=2)
        expected = {
            "series_id",
            "sessions",
            "pageviews",
            "conversions",
            "bounces",
            "bounce_rate",
            "conversion_rate",
            "pages_per_session",
        }
        assert set(metrics) == expected
        n = len(metrics["series_id"])
        assert all(len(v) == n for v in metrics.values())
        assert (metrics["conversions"] <= metrics["sessions"]).all()
        assert (metrics["bounce_rate"] <= 1).all()

    def test_generate_funnel_monotone(self) -> None:
        funnel = make_gen().generate_funnel(n_sessions=10_000)
        counts = list(funnel.values())
        assert counts[0] == 10_000
        assert all(a >= b for a, b in zip(counts, counts[1:], strict=False))
        assert counts[-1] > 0

    def test_get_model_info(self) -> None:
        info = make_gen(traffic_source="paid", output_type="pageviews").get_model_info()
        assert info["traffic_source"] == "paid"
        assert info["output_type"] == "pageviews"
        assert info["source_params"]["conversion_mult"] == 1.5

    def test_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            make_gen(bot_fraction=1.0)
        with pytest.raises(ValidationError):
            make_gen(avg_session_depth=0.0)
        with pytest.raises(ValidationError):
            make_gen(conversion_rate=1.5)
        with pytest.raises(ValidationError):
            make_gen(traffic_source="carrier_pigeon")
        with pytest.raises(ValidationError):
            make_gen(output_type="revenue")


@pytest.mark.stats
class TestClickstreamStats:
    def test_sessions_mean_scale(self) -> None:
        """Session counts fluctuate around base_sessions (short horizon)."""
        gen = make_gen(
            min_length=300,
            max_length=300,
            base_sessions=200.0,
            include_seasonality=False,
            include_bots=False,
            seed=1,
        )
        means = [gen.generate_single_series(300).mean() for _ in range(6)]
        # The log-random-walk trend spreads realized levels; bounds are loose
        assert 100 < np.mean(means) < 400

    def test_conversions_within_sessions(self) -> None:
        """Same seed pairs the simulation: conversions <= sessions per bin."""
        conv = make_gen(seed=3, output_type="conversions").generate_single_series(300)
        sess = make_gen(seed=3, output_type="sessions").generate_single_series(300)
        assert (conv <= sess).all()
        ratio = conv.sum() / sess.sum()
        # ~ conversion_rate * (1 - bounce_rate) = 0.018 (bots dilute slightly)
        assert 0.005 < ratio < 0.05

    def test_bounce_rate_level(self) -> None:
        gen = make_gen(
            min_length=500,
            max_length=500,
            base_sessions=500.0,
            output_type="bounce_rate",
            include_bots=False,
            bounce_rate=0.4,
            seed=4,
        )
        values = gen.generate_single_series(500)
        assert 0.3 < values.mean() < 0.5

    def test_pageviews_exceed_sessions(self) -> None:
        """With depth > 1, pageviews outnumber sessions (paired seed)."""
        pv = make_gen(
            seed=5, output_type="pageviews", include_bots=False
        ).generate_single_series(300)
        sess = make_gen(
            seed=5, output_type="sessions", include_bots=False
        ).generate_single_series(300)
        assert pv.sum() > sess.sum()

    def test_daily_seasonality_acf(self) -> None:
        """Hourly sessions show a daily cycle at lag 24."""
        gen = make_gen(
            min_length=24 * 14,
            max_length=24 * 14,
            base_sessions=1000.0,
            include_bots=False,
            seed=6,
        )
        values = gen.generate_single_series(24 * 14)
        assert sample_acf(values, 24) > 0.3

    def test_no_seasonality_flattens_acf(self) -> None:
        gen = make_gen(
            min_length=24 * 14,
            max_length=24 * 14,
            base_sessions=1000.0,
            include_seasonality=False,
            include_bots=False,
            seed=6,
        )
        values = gen.generate_single_series(24 * 14)
        seasonal = make_gen(
            min_length=24 * 14,
            max_length=24 * 14,
            base_sessions=1000.0,
            include_bots=False,
            seed=6,
        ).generate_single_series(24 * 14)
        assert sample_acf(values, 24) < sample_acf(seasonal, 24)

    def test_bots_add_traffic(self) -> None:
        """Bots only add sessions on top of the (paired) human traffic."""
        with_bots = make_gen(
            seed=7, include_bots=True, bot_fraction=0.2
        ).generate_single_series(300)
        without_bots = make_gen(seed=7, include_bots=False).generate_single_series(300)
        assert (with_bots >= without_bots).all()
        assert with_bots.sum() > without_bots.sum()

    def test_direct_traffic_converts_more_than_referral(self) -> None:
        """conversion_mult: direct (1.8) converts more than referral (0.8)."""

        def total_conversions(source: str) -> float:
            gen = make_gen(
                min_length=500,
                max_length=500,
                base_sessions=500.0,
                traffic_source=source,
                output_type="conversions",
                include_bots=False,
                seed=8,
            )
            return float(
                np.sum([gen.generate_single_series(500).sum() for _ in range(3)])
            )

        assert total_conversions("direct") > total_conversions("referral")
