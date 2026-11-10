"""Relative overlay magnitudes, the pretraining overlay prior and the extended pretraining pool."""

import numpy as np
import pytest

from synforecast import pretraining_overlays, pretraining_pool
from synforecast._core import _overlay_scale
from synforecast._lib import pattern_injection as rs_pi
from synforecast.overlays import Anomalies, Changepoints, OverlayConfig

EMPTY = np.array([], dtype=np.float64)


def _series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n)) + 0.1 * np.arange(n)


@pytest.mark.parametrize("factor", [1e-3, 1.0, 250.0])
def test_relative_anomalies_scale_with_the_series(factor: float) -> None:
    base = _series()
    out = []
    for values in (base.copy(), factor * base):
        injected, _ = rs_pi.add_anomalies(
            values,
            7,
            ["spike", "dip", "level_shift"],
            0.03,
            5.0,
            -5.0,
            2.0,
            10,
            _overlay_scale(values),
        )
        out.append(injected)
    np.testing.assert_allclose(out[1], factor * out[0], rtol=1e-12, atol=1e-12 * factor)


@pytest.mark.parametrize("kind", ["level", "trend", "variance", "mixed"])
def test_relative_changepoints_scale_with_the_series(kind: str) -> None:
    base = _series(seed=3)
    out = []
    for values in (base.copy(), 40.0 * base):
        injected, _ = rs_pi.add_changepoints(
            values,
            11,
            2,
            EMPTY,
            kind,
            EMPTY,
            EMPTY,
            EMPTY,
            True,
            _overlay_scale(values),
        )
        out.append(injected)
    np.testing.assert_allclose(out[1], 40.0 * out[0], rtol=1e-10, atol=1e-9)


def test_relative_trend_change_is_total_drift_in_std_units() -> None:
    values = np.zeros(200)
    values[::2] = 1.0
    scale = _overlay_scale(values)
    injected, meta = rs_pi.add_changepoints(
        values.copy(),
        0,
        1,
        np.array([0.5]),
        "trend",
        EMPTY,
        np.array([2.0]),
        EMPTY,
        True,
        scale,
    )
    pos = int(meta["changepoint_indices"][0])
    drift = (injected - values)[-1]
    assert drift == pytest.approx(
        2.0 * scale * (len(values) - pos - 1) / (len(values) - pos)
    )


def test_absolute_mode_is_the_default_and_unscaled() -> None:
    values = _series()
    a, _ = rs_pi.add_anomalies(values.copy(), 3, ["spike"], 0.05, 10.0, -10.0, 20.0, 10)
    b, _ = rs_pi.add_anomalies(
        values.copy(), 3, ["spike"], 0.05, 10.0, -10.0, 20.0, 10, 1.0
    )
    np.testing.assert_array_equal(a, b)
    assert sorted(set(np.round(a - values, 9)) - {0.0}) == [10.0]


@pytest.mark.parametrize(
    "values", [np.full(50, 3.0), np.array([1.0]), np.array([np.nan, 2.0])]
)
def test_overlay_scale_falls_back_to_one(values: np.ndarray) -> None:
    assert _overlay_scale(values) == 1.0


def test_overlay_prior_is_deterministic_and_documented_rates() -> None:
    first = [pretraining_overlays(np.random.default_rng([5, i])) for i in range(50)]
    again = [pretraining_overlays(np.random.default_rng([5, i])) for i in range(50)]
    assert first == again
    draws = [pretraining_overlays(np.random.default_rng([9, i])) for i in range(20000)]
    assert np.mean(["innovation_distribution" in d for d in draws]) == pytest.approx(
        0.3, abs=0.015
    )
    stages = [d.get("overlays", OverlayConfig()) for d in draws]
    for stage, rate in {"changepoints": 0.3, "anomalies": 0.25, "missing": 0.1}.items():
        drawn = [getattr(o, stage) for o in stages]
        assert np.mean([x is not None for x in drawn]) == pytest.approx(rate, abs=0.015)
    assert all(
        spec.scale == "relative"
        for o in stages
        for spec in (o.changepoints, o.anomalies)
        if spec is not None
    )
    assert any(not d for d in draws)


def test_member_built_alone_matches_full_pool_with_overlays() -> None:
    kwargs = {
        "min_length": 64,
        "max_length": 64,
        "freq": "D",
        "seed": 4,
        "include_extended": True,
        "overlay_seed": 123,
    }
    full = pretraining_pool(**kwargs)
    for i in (0, 17, 45, 50, 60, len(full) - 1):
        (alone,) = pretraining_pool(**kwargs, members=[i])
        assert (
            type(alone) is type(full[i]) and alone.model_dump() == full[i].model_dump()
        )


def test_extended_entries_are_appended() -> None:
    plain = pretraining_pool(min_length=32, max_length=32, freq="h", seed=1)
    extended = pretraining_pool(
        min_length=32, max_length=32, freq="h", seed=1, include_extended=True
    )
    assert len(extended) == len(plain) + 11
    assert [g.model_dump() for g in extended[: len(plain)]] == [
        g.model_dump() for g in plain
    ]


def test_extended_pool_generates_through_synset() -> None:
    from synforecast import SynSet

    pool = pretraining_pool(
        min_length=48, max_length=48, freq="h", seed=1, include_extended=True
    )
    df = SynSet(pool).generate(n_series_per_generator=2, n_jobs=1)
    assert df["unique_id"].nunique() == 2 * len(pool)
    assert {"unique_id", "ds", "y"} <= set(df.columns)


def test_entries_with_own_changepoints_keep_them() -> None:
    kwargs = {
        "min_length": 96,
        "max_length": 96,
        "freq": "MS",
        "seed": 2,
        "include_seasonal": True,
    }
    plain = pretraining_pool(**kwargs)
    own = [
        i
        for i, g in enumerate(plain)
        if g.overlays.changepoints is not None or g.overlays.anomalies is not None
    ]
    assert own
    for overlay_seed in range(40):
        drawn = pretraining_pool(**kwargs, overlay_seed=overlay_seed, members=own)
        for g, ref in zip(drawn, (plain[i] for i in own), strict=True):
            assert g.overlays.changepoints == ref.overlays.changepoints
            assert g.overlays.anomalies == ref.overlays.anomalies


@pytest.mark.parametrize(
    "freq,length", [("D", 400), ("h", 1024), ("YS", 2273), ("MS", 128)]
)
def test_every_member_draws_valid_series_with_overlays(freq: str, length: int) -> None:
    n = len(
        pretraining_pool(min_length=8, max_length=8, freq=freq, include_extended=True)
    )
    rng = np.random.default_rng(0)
    for draw in range(3 * n):
        (gen,) = pretraining_pool(
            min_length=length,
            max_length=length,
            freq=freq,
            seed=int(rng.integers(1 << 40)),
            include_extended=True,
            overlay_seed=int(rng.integers(1 << 40)),
            members=[draw % n],
        )
        (values,) = gen.generate_values(1, n_jobs=1)
        assert values.shape == (length,) and not np.isinf(values).any()
        assert np.isfinite(values).sum() >= 2, (draw % n, type(gen).__name__)


@pytest.mark.parametrize("kind", ["level", "trend", "variance", "mixed"])
def test_relative_overlays_stay_on_the_series_scale(kind: str) -> None:
    overlays = OverlayConfig(
        changepoints=Changepoints(count=3, kind=kind, scale="relative"),
        anomalies=Anomalies(
            types=["spike", "dip", "level_shift"],
            fraction=0.03,
            spike_magnitude=10.0,
            dip_magnitude=-10.0,
            level_shift_magnitude=4.0,
            level_shift_duration=50,
            scale="relative",
        ),
    )
    n = len(
        pretraining_pool(min_length=8, max_length=8, freq="D", include_extended=True)
    )
    for member in range(n):
        kwargs = {
            "min_length": 512,
            "max_length": 512,
            "freq": "D",
            "seed": 11,
            "include_extended": True,
            "members": [member],
        }
        (base,) = pretraining_pool(**kwargs)[0].generate_values(1, n_jobs=1)
        (both,) = pretraining_pool(**kwargs, overlays=overlays)[0].generate_values(
            1, n_jobs=1
        )
        sigma = _overlay_scale(base)
        if np.nanmax(np.abs(base)) >= 1e11:
            continue
        # Spikes reach 10 sigma, drift and level changes 3 sigma each, variance changes at most 2x per break.
        assert np.nanmax(np.abs(both - base)) <= 200 * sigma, (member, kind)


@pytest.mark.parametrize("freq", ["h", "D", "MS"])
def test_extended_entries_stay_bounded_on_long_series(freq: str) -> None:
    n = len(pretraining_pool(min_length=8, max_length=8, freq=freq))
    rng = np.random.default_rng(1)
    for member in range(n, n + 11):
        for _ in range(25):
            (gen,) = pretraining_pool(
                min_length=2273,
                max_length=2273,
                freq=freq,
                seed=int(rng.integers(1 << 40)),
                include_extended=True,
                members=[member],
            )
            (values,) = gen.generate_values(1, n_jobs=1)
            assert np.nanmax(np.abs(values)) < 1e8, (member, type(gen).__name__)


@pytest.mark.parametrize("scale", ["absolute", "relative"])
def test_variance_changepoints_keep_sensor_gaps_local(scale: str) -> None:
    # IoTSensor series have failure gaps; variance changepoints must not turn the rest of the series into NaN.
    member = next(
        i
        for i, g in enumerate(pretraining_pool(min_length=8, max_length=8, freq="D"))
        if type(g).__name__ == "IoTSensorGenerator"
    )
    rng = np.random.default_rng(2)
    for _ in range(60):
        kwargs = {
            "min_length": 128,
            "max_length": 128,
            "freq": "D",
            "seed": int(rng.integers(1 << 40)),
            "members": [member],
        }
        (base,) = pretraining_pool(**kwargs)[0].generate_values(1, n_jobs=1)
        (both,) = pretraining_pool(
            **kwargs,
            overlays=OverlayConfig(
                changepoints=Changepoints(count=3, kind="variance", scale=scale)
            ),
        )[0].generate_values(1, n_jobs=1)
        np.testing.assert_array_equal(np.isnan(both), np.isnan(base))
