"""The overlay pipeline: OverlayConfig, its stages, the deprecated flat arguments and the priors."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from synforecast import (
    Multivariatizer,
    SynSet,
    calendar_overlays,
    observability_overlays,
    pretraining_pool,
)
from synforecast.base import BaseGenerator
from synforecast.exogenous import ExogenousConfig
from synforecast.generators import (
    ETSGenerator,
    PoissonProcessGenerator,
    RandomWalkGenerator,
    StateSpaceGenerator,
)
from synforecast.overlays import (
    Anomalies,
    Calendar,
    CalendarCycle,
    Changepoints,
    Missing,
    Observation,
    OverlayConfig,
    Waveform,
)

_FLAGS = ExogenousConfig(anomaly_flags=True, changepoint_flags=True, missing_flags=True)


class _ThreadedWalk(BaseGenerator):
    """No native batch kernel, so generation takes the per-series Python path."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.cumsum(self.rng.normal(size=length))


def _values(gen: BaseGenerator, n: int = 3) -> list[np.ndarray]:
    return gen.generate_values(n, n_jobs=1)


def _pair(cls: type, overlays: OverlayConfig, **kwargs: Any) -> tuple[list, list]:
    """Draws without and with the overlays, from the same seed."""
    base = {"min_length": 24 * 21, "max_length": 24 * 21, "freq": "h", "seed": 3}
    base.update(kwargs)
    return _values(cls(**base)), _values(cls(**base, overlays=overlays))


# ---------------------------------------------------------------------------
# Deprecated flat arguments
# ---------------------------------------------------------------------------

LEGACY = {
    "changepoints": True,
    "num_changepoints": 3,
    "changepoint_type": "mixed",
    "anomalies": True,
    "anomaly_types": ["spike", "level_shift"],
    "anomaly_fraction": 0.03,
    "missing_data": True,
    "missing_pattern": "block",
    "missing_block_size": 4,
    "overlay_scale": "relative",
}
NEW = OverlayConfig(
    changepoints=Changepoints(count=3, kind="mixed", scale="relative"),
    anomalies=Anomalies(
        types=["spike", "level_shift"], fraction=0.03, scale="relative"
    ),
    missing=Missing(pattern="block", block_size=4),
)


@pytest.mark.parametrize("cls", [RandomWalkGenerator, _ThreadedWalk])
def test_flat_arguments_warn_and_draw_as_the_new_api(cls: type) -> None:
    base = {
        "min_length": 80,
        "max_length": 120,
        "freq": "h",
        "seed": 5,
        "exogenous": _FLAGS,
    }
    with pytest.warns(
        DeprecationWarning, match="flat overlay arguments are deprecated"
    ):
        legacy = cls(**base, **LEGACY)
    new = cls(**base, overlays=NEW)
    assert legacy.model_dump() == new.model_dump()
    pd.testing.assert_frame_equal(
        legacy.generate(4, n_jobs=2), new.generate(4, n_jobs=2)
    )


def test_flat_arguments_fill_stages_left_off_and_clash_otherwise() -> None:
    base = {"min_length": 10, "max_length": 10, "freq": "D"}
    calendar = Calendar(cycles=[CalendarCycle(phase="week")])
    with pytest.warns(DeprecationWarning):
        gen = RandomWalkGenerator(
            **base, overlays=OverlayConfig(calendar=calendar), missing_data=True
        )
    assert gen.overlays.calendar == calendar and gen.overlays.missing == Missing()
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="missing set both"),
    ):
        RandomWalkGenerator(
            **base, overlays=OverlayConfig(missing=Missing()), missing_data=True
        )


def test_deprecated_attributes_read_from_overlays() -> None:
    gen = RandomWalkGenerator(min_length=10, max_length=10, freq="D", overlays=NEW)
    with pytest.warns(
        DeprecationWarning, match="read it from the generator's overlays"
    ):
        assert gen.missing_data and gen.missing_block_size == 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert gen.overlay_scale == "relative" and gen.num_changepoints == 3
        assert gen.anomaly_types == ["spike", "level_shift"]
        plain = RandomWalkGenerator(min_length=10, max_length=10, freq="D")
        assert not plain.anomalies and plain.missing_rate == 0.1


def test_new_api_emits_no_deprecation_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        RandomWalkGenerator(min_length=10, max_length=10, freq="D", overlays=NEW)
        pretraining_pool(
            freq="h", overlay_seed=1, calendar_rate=0.5, observability_rate=0.5
        )


# ---------------------------------------------------------------------------
# One pipeline: stage order, streams and scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [RandomWalkGenerator, _ThreadedWalk])
@pytest.mark.parametrize(
    "extra",
    [
        {"observation": Observation()},
        {"waveform": Waveform(amplitude=0.0)},
        {"calendar": Calendar(cycles=[CalendarCycle(phase="day")])},
    ],
    ids=["noop_observation", "flat_waveform", "calendar"],
)
def test_switching_a_stage_on_never_moves_another(cls: type, extra: dict) -> None:
    released = OverlayConfig(
        changepoints=Changepoints(count=2),
        anomalies=Anomalies(fraction=0.02),
        missing=Missing(rate=0.1),
    )
    base = {
        "min_length": 200,
        "max_length": 200,
        "freq": "h",
        "seed": 9,
        "exogenous": _FLAGS,
    }
    before = cls(**base, overlays=released).generate(3, n_jobs=1)
    after_gen = cls(**base, overlays=released.model_copy(update=extra))
    after = after_gen.generate(3, n_jobs=1)
    for flag in ("changepoint_flag", "anomaly_flag", "missing_flag"):
        np.testing.assert_array_equal(before[flag], after[flag])
    np.testing.assert_array_equal(before["y"].isna(), after["y"].isna())
    if "observation" in extra or "waveform" in extra:
        np.testing.assert_array_equal(before["y"], after["y"])


def test_overlays_leave_the_generator_stream_untouched() -> None:
    config = OverlayConfig(
        calendar=Calendar(cycles=[CalendarCycle(phase="day")]),
        waveform=Waveform(),
        observation=Observation(quantum=0.1, hold=3, stuck_fraction=0.1),
    )
    for freq in ("h", "MS", 7):
        kwargs = {"min_length": 50, "max_length": 60, "freq": freq, "seed": 2}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plain, dressed = (
                RandomWalkGenerator(**kwargs),
                RandomWalkGenerator(**kwargs, overlays=config),
            )
        plain.generate_values(4)
        dressed.generate_values(4)
        assert plain.rng.integers(1 << 62) == dressed.rng.integers(1 << 62)


def test_relative_sizes_are_measured_on_the_base_series() -> None:
    anomalies = Anomalies(
        types=["spike"], fraction=0.02, spike_magnitude=5.0, scale="relative"
    )
    calendar = Calendar(cycles=[CalendarCycle(phase="day")], amplitude=8.0)
    base = {
        "min_length": 400,
        "max_length": 400,
        "freq": "h",
        "seed": 4,
        "exogenous": _FLAGS,
    }
    walk = RandomWalkGenerator(**base).generate_values(1, n_jobs=1)[0]
    spiked = RandomWalkGenerator(**base, overlays=OverlayConfig(anomalies=anomalies))
    both = RandomWalkGenerator(
        **base, overlays=OverlayConfig(anomalies=anomalies, calendar=calendar)
    )
    cal_only = RandomWalkGenerator(**base, overlays=OverlayConfig(calendar=calendar))
    jump = spiked.generate_values(1, n_jobs=1)[0] - walk
    jump_with_calendar = (
        both.generate_values(1, n_jobs=1)[0] - cal_only.generate_values(1, n_jobs=1)[0]
    )
    np.testing.assert_allclose(jump, jump_with_calendar, atol=1e-9)


def test_base_weight_shrinks_the_base_but_not_the_calendar() -> None:
    calendar = Calendar(cycles=[CalendarCycle(phase="day", harmonics=2)], amplitude=2.0)
    plain, shaped = _pair(
        RandomWalkGenerator,
        OverlayConfig(calendar=calendar),
        min_length=96,
        max_length=96,
    )
    _, exact = _pair(
        RandomWalkGenerator,
        OverlayConfig(calendar=calendar, base_weight=0.0),
        min_length=96,
        max_length=96,
    )
    for a, b, e in zip(plain, shaped, exact, strict=True):
        np.testing.assert_allclose(e - a.mean(), b - a, atol=1e-9)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def test_daily_and_weekly_cycles_repeat_exactly() -> None:
    calendar = Calendar(
        cycles=[
            CalendarCycle(phase="day", harmonics=4),
            CalendarCycle(phase="week", profile="levels", weight=0.5),
        ],
        amplitude=3.0,
    )
    plain, shaped = _pair(RandomWalkGenerator, OverlayConfig(calendar=calendar))
    for a, b in zip(plain, shaped, strict=True):
        profile = b - a
        np.testing.assert_allclose(profile[: 24 * 14], profile[24 * 7 :], atol=1e-9)
        assert not np.allclose(profile[:24], profile[24:48])
        assert np.abs(profile).max() <= 3.0 * np.std(a) + 1e-9


def test_cycles_follow_the_emitted_timestamps() -> None:
    # A fixed hour-of-day profile: the value added at each step is its hour in the ds column.
    hours = CalendarCycle(phase="day", profile="levels", values=list(range(24)))
    calendar = Calendar(cycles=[hours], amplitude=1.0, scale="absolute")
    for start in ("2024-03-01 05:00", "2024-01-01 00:00+02:00"):
        kwargs = {
            "min_length": 72,
            "max_length": 72,
            "freq": "h",
            "seed": 1,
            "start_datetime": start,
        }
        plain = RandomWalkGenerator(**kwargs).generate(1)
        shaped = RandomWalkGenerator(
            **kwargs, overlays=OverlayConfig(calendar=calendar)
        ).generate(1)
        added = (shaped["y"] - plain["y"]).to_numpy()
        hour = pd.DatetimeIndex(shaped["ds"]).hour.to_numpy()
        expected = (hour - 11.5) / 11.5
        np.testing.assert_allclose(added, expected, atol=1e-9)


def test_year_cycle_follows_calendar_months_on_monthly_data() -> None:
    months = CalendarCycle(phase="year", profile="levels")
    plain, shaped = _pair(
        RandomWalkGenerator,
        OverlayConfig(calendar=Calendar(cycles=[months], amplitude=4.0)),
        freq="MS",
        min_length=60,
        max_length=60,
        start_datetime="2021-03-01",
    )
    profile = shaped[0] - plain[0]
    np.testing.assert_allclose(profile[:48], profile[12:], atol=1e-9)
    assert len(np.unique(np.round(profile[:12], 9))) == 12


def test_step_cycles_work_on_integer_frequencies() -> None:
    cycle = CalendarCycle(phase="step", period=10)
    plain, shaped = _pair(
        RandomWalkGenerator,
        OverlayConfig(calendar=Calendar(cycles=[cycle])),
        freq=1,
        min_length=100,
        max_length=100,
    )
    profile = shaped[0] - plain[0]
    np.testing.assert_allclose(profile[:90], profile[10:], atol=1e-9)


@pytest.mark.parametrize("freq", ["D", "MS", 1])
def test_unsupported_cycles_warn_skip_or_raise(freq: str | int) -> None:
    kwargs = {"min_length": 30, "max_length": 30, "freq": freq, "seed": 1}
    day = [CalendarCycle(phase="day")]
    with pytest.warns(UserWarning, match="cannot show"):
        gen = RandomWalkGenerator(
            **kwargs, overlays=OverlayConfig(calendar=Calendar(cycles=day))
        )
    np.testing.assert_array_equal(
        gen.generate_values(2)[1], RandomWalkGenerator(**kwargs).generate_values(2)[1]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RandomWalkGenerator(
            **kwargs,
            overlays=OverlayConfig(
                calendar=Calendar(cycles=day, on_unsupported="skip")
            ),
        )
    with pytest.raises(ValueError, match="cannot show"):
        RandomWalkGenerator(
            **kwargs,
            overlays=OverlayConfig(
                calendar=Calendar(cycles=day, on_unsupported="raise")
            ),
        )


def test_weekly_cycle_on_business_days_uses_the_real_weekdays() -> None:
    weekdays = CalendarCycle(
        phase="week", profile="levels", values=[0, 1, 2, 3, 4, 5, 6]
    )
    calendar = Calendar(cycles=[weekdays], scale="absolute")
    kwargs = {"min_length": 20, "max_length": 20, "freq": "B", "seed": 1}
    plain = RandomWalkGenerator(**kwargs).generate(1)
    shaped = RandomWalkGenerator(
        **kwargs, overlays=OverlayConfig(calendar=calendar)
    ).generate(1)
    added = (shaped["y"] - plain["y"]).to_numpy()
    expected = (pd.DatetimeIndex(shaped["ds"]).dayofweek.to_numpy() - 3.0) / 3.0
    np.testing.assert_allclose(added, expected, atol=1e-9)


def test_multiplicative_keeps_zeros_and_falls_back_on_negative_series() -> None:
    calendar = Calendar(
        cycles=[CalendarCycle(phase="day")], mode="multiplicative", swing=0.8
    )
    plain, shaped = _pair(PoissonProcessGenerator, OverlayConfig(calendar=calendar))
    for a, b in zip(plain, shaped, strict=True):
        assert (a == 0).any()
        np.testing.assert_array_equal(b[a == 0], 0.0)
        assert (b >= 0).all()
    plain, shaped = _pair(RandomWalkGenerator, OverlayConfig(calendar=calendar))
    assert any((a < 0).any() for a in plain)
    assert all(not np.array_equal(a, b) for a, b in zip(plain, shaped, strict=True))
    skip = calendar.model_copy(update={"fallback": "none"})
    for a, b in zip(
        *_pair(RandomWalkGenerator, OverlayConfig(calendar=skip)), strict=True
    ):
        if (a < 0).any():
            np.testing.assert_array_equal(a, b)


def test_calendar_keeps_missing_values_missing() -> None:
    config = OverlayConfig(
        calendar=Calendar(cycles=[CalendarCycle(phase="day")]),
        missing=Missing(rate=0.2),
    )
    with_missing = RandomWalkGenerator(
        min_length=200,
        max_length=200,
        freq="h",
        seed=8,
        overlays=OverlayConfig(missing=Missing(rate=0.2)),
    )
    both = RandomWalkGenerator(
        min_length=200, max_length=200, freq="h", seed=8, overlays=config
    )
    for a, b in zip(_values(with_missing), _values(both), strict=True):
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


# ---------------------------------------------------------------------------
# Waveform and observation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", ["pulse", "spike", "sawtooth", "trapezoid", "levels", "staircase"]
)
@pytest.mark.parametrize("timing", ["periodic", "jitter", "renewal"])
def test_waveform_template_stays_within_its_amplitude(shape: str, timing: str) -> None:
    waveform = Waveform(
        shape=shape, timing=timing, period=40, cv=1.0, amplitude=2.0, scale="absolute"
    )
    plain, shaped = _pair(
        RandomWalkGenerator,
        OverlayConfig(waveform=waveform, base_weight=0.0),
        min_length=800,
        max_length=800,
    )
    for a, b in zip(plain, shaped, strict=True):
        template = b - a.mean()
        assert template.min() >= -1e-9 and template.max() <= 2.0 + 1e-9
        assert template.max() > 0


def test_periodic_waveform_repeats_and_dips_fall() -> None:
    waveform = Waveform(
        period=24, duty=0.25, direction="dip", amplitude=1.0, scale="absolute"
    )
    plain, shaped = _pair(RandomWalkGenerator, OverlayConfig(waveform=waveform))
    added = shaped[0] - plain[0]
    np.testing.assert_allclose(added[:-24], added[24:], atol=1e-12)
    assert set(np.round(np.unique(added), 9)) == {-1.0, 0.0}
    assert np.isclose(added[:24].sum(), -6.0)


def test_observation_saturates_quantizes_and_holds() -> None:
    config = OverlayConfig(
        observation=Observation(floor=0.1, cap=0.9, quantum=0.25, hold=4)
    )
    plain, shaped = _pair(RandomWalkGenerator, config, min_length=400, max_length=400)
    for a, b in zip(plain, shaped, strict=True):
        lo, hi = a.min(), a.max()
        q = 0.25 * np.std(a)
        assert (
            b.min() >= lo + 0.1 * (hi - lo) - q and b.max() <= lo + 0.9 * (hi - lo) + q
        )
        np.testing.assert_allclose(np.round(b / q) * q, b, atol=1e-9)
        # Readings only change at sample-and-hold block edges, every 4 steps after an offset.
        changes = np.flatnonzero(np.diff(b) != 0) + 1
        assert len(set(changes % 4)) == 1


def test_observation_keeps_missing_values_local() -> None:
    observation = Observation(hold=5, stuck_fraction=0.2, stuck_length=10)
    missing = Missing(rate=0.1)
    _, both = _pair(
        RandomWalkGenerator,
        OverlayConfig(observation=observation, missing=missing),
        min_length=400,
        max_length=400,
    )
    _, missing_only = _pair(
        RandomWalkGenerator,
        OverlayConfig(missing=missing),
        min_length=400,
        max_length=400,
    )
    for b, m in zip(both, missing_only, strict=True):
        np.testing.assert_array_equal(np.isnan(b), np.isnan(m))


# ---------------------------------------------------------------------------
# Every path gets the overlays once
# ---------------------------------------------------------------------------

_CALENDAR_ONLY = OverlayConfig(
    calendar=Calendar(cycles=[CalendarCycle(phase="day")], amplitude=5.0)
)


def test_synset_and_generate_agree_on_the_calendar() -> None:
    kwargs = {
        "min_length": 96,
        "max_length": 96,
        "freq": "h",
        "seed": 6,
        "overlays": _CALENDAR_ONLY,
    }
    alone = RandomWalkGenerator(**kwargs).generate(2)
    mixed = SynSet(
        [RandomWalkGenerator(**kwargs), _ThreadedWalk(**{**kwargs, "seed": 7})]
    ).generate(n_series_per_generator=2)
    np.testing.assert_allclose(
        mixed["y"].to_numpy()[: len(alone)], alone["y"].to_numpy()
    )


@pytest.mark.parametrize("cls", [ETSGenerator, StateSpaceGenerator])
def test_states_stay_clean_while_observations_get_every_overlay(cls: type) -> None:
    config = OverlayConfig(
        calendar=Calendar(cycles=[CalendarCycle(phase="day")], amplitude=4.0),
        anomalies=Anomalies(types=["spike"], fraction=0.05, spike_magnitude=50.0),
    )
    kwargs = {"min_length": 96, "max_length": 96, "freq": "h", "seed": 3}
    obs_plain, states_plain = cls(**kwargs).generate_with_states(2)
    obs, states = cls(**kwargs, overlays=config).generate_with_states(2)
    pd.testing.assert_frame_equal(states_plain, states)
    assert not np.allclose(obs_plain["y"], obs["y"])


def test_multivariatizer_overlays_are_opt_in() -> None:
    base = RandomWalkGenerator(
        min_length=100,
        max_length=100,
        freq="h",
        overlays=OverlayConfig(missing=Missing(rate=0.2)),
    )
    plain = Multivariatizer(
        base=RandomWalkGenerator(min_length=100, max_length=100, freq="h"), seed=1
    )
    off = Multivariatizer(base=base, seed=1)
    on = Multivariatizer(base=base, seed=1, apply_overlays=True)
    pd.testing.assert_frame_equal(plain.generate(3), off.generate(3))
    drawn = on.generate(3)
    assert drawn["y"].isna().mean() == pytest.approx(0.2)
    fresh = Multivariatizer(base=base, seed=1).generate(3)["y"]
    np.testing.assert_allclose(drawn["y"].dropna(), fresh[drawn["y"].notna()])


# ---------------------------------------------------------------------------
# Priors and validation
# ---------------------------------------------------------------------------


def test_calendar_prior_rates_and_determinism() -> None:
    draws = [calendar_overlays(np.random.default_rng([3, i])) for i in range(4000)]
    assert draws[:20] == [
        calendar_overlays(np.random.default_rng([3, i])) for i in range(20)
    ]
    assert np.mean([d.mode == "multiplicative" for d in draws]) == pytest.approx(
        0.5, abs=0.03
    )
    assert np.mean([len(d.cycles) == 1 for d in draws]) == pytest.approx(0.3, abs=0.03)
    assert {d.cycles[0].harmonics for d in draws} == {1, 2, 3, 4, 5}
    assert all(1.25 <= d.amplitude <= 10.0 and 0.2 <= d.swing <= 0.9 for d in draws)
    assert all(d.on_unsupported == "skip" for d in draws)


def test_observability_prior_components() -> None:
    draws = [
        observability_overlays(np.random.default_rng([4, i]), 200) for i in range(3000)
    ]
    assert all(d.waveform is not None and 12 <= d.waveform.period <= 200 for d in draws)
    assert all(d.calendar is None for d in draws)
    clean = [
        d
        for d in draws
        if d.base_weight <= 0.1 and d.anomalies is None and d.missing is None
    ]
    assert len(clean) / len(draws) > 0.35
    layered = [d for d in draws if d.changepoints is not None]
    assert len(layered) / len(draws) == pytest.approx(0.25, abs=0.03)
    assert all(d.observation is not None for d in layered)
    assert all(
        d.observation.hold <= d.waveform.on_steps
        for d in draws
        if d.observation is not None and d.observation.hold > 1
    )


@pytest.mark.parametrize("rate", ["calendar_rate", "observability_rate"])
def test_pool_rates_validate_and_keep_members_consistent(rate: str) -> None:
    with pytest.raises(ValueError, match="needs an overlay_seed"):
        pretraining_pool(**{rate: 0.5})
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        pretraining_pool(overlay_seed=1, **{rate: 1.5})
    kwargs = {
        "min_length": 96,
        "max_length": 96,
        "freq": "h",
        "seed": 2,
        "overlay_seed": 5,
        rate: 0.6,
    }
    full = pretraining_pool(**kwargs, include_extended=True)
    for i in (0, 9, 33, len(full) - 1):
        (alone,) = pretraining_pool(**kwargs, include_extended=True, members=[i])
        assert alone.model_dump() == full[i].model_dump()
    assert any(
        getattr(g.overlays, "calendar" if rate == "calendar_rate" else "waveform")
        is not None
        for g in full
    )


@pytest.mark.parametrize(
    ("spec", "kwargs", "message"),
    [
        (CalendarCycle, {"phase": "step"}, "period is required"),
        (CalendarCycle, {"phase": "day", "period": 4}, "only allowed"),
        (
            CalendarCycle,
            {"phase": "week", "profile": "levels", "values": [1.0]},
            "7 entries",
        ),
        (CalendarCycle, {"phase": "day", "harmonics": 13}, "less than or equal"),
        (Calendar, {"cycles": []}, "at least 1"),
        (
            Calendar,
            {"cycles": [CalendarCycle(phase="day")], "swing": 1.0},
            "less than 1",
        ),
        (
            Calendar,
            {"cycles": [CalendarCycle(phase="day")], "amplitude": float("inf")},
            "finite",
        ),
        (Waveform, {"period": 1}, "greater than or equal"),
        (Waveform, {"duty": 1.0}, "less than 1"),
        (
            Waveform,
            {"mode": "multiplicative", "direction": "dip", "swing": 1.5},
            "<= 1",
        ),
        (Observation, {"floor": 0.5, "cap": 0.5}, "below cap"),
        (Observation, {"hold": 0}, "greater than or equal"),
        (Changepoints, {"locations": [1.2]}, r"\[0, 1\]"),
        (Anomalies, {"types": []}, "at least 1"),
        (OverlayConfig, {"base_weight": 1.5}, "less than or equal"),
        (OverlayConfig, {"unknown": 1}, "Extra inputs"),
    ],
)
def test_specs_validate(spec: type, kwargs: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        spec(**kwargs)
