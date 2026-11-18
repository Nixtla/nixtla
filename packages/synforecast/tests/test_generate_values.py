"""generate_values returns exactly the series generate() draws, on native and threaded generators alike."""

import numpy as np
import pytest

from synforecast.base import BaseGenerator
from synforecast.exogenous import CorrelatedExogConfig, ExogenousConfig
from synforecast.generators import (
    CopulaGenerator,
    DailyActiveUsersGenerator,
    IoTSensorGenerator,
    KernelSynthGenerator,
    TCMGenerator,
    VARGenerator,
)
from synforecast.generators.sarima import SARIMAGenerator
from synforecast.presets import pretraining_pool

POOL = pretraining_pool(
    min_length=40, max_length=90, freq="D", seed=3, include_seasonal=True
)


@pytest.mark.parametrize("member", range(len(POOL)))
def test_values_match_generate(member: int) -> None:
    kwargs = {
        "min_length": 40,
        "max_length": 90,
        "freq": "D",
        "seed": 3,
        "include_seasonal": True,
        "members": [member],
    }
    df = pretraining_pool(**kwargs)[0].generate(n_series=3, n_jobs=1)
    values = pretraining_pool(**kwargs)[0].generate_values(n_series=3, n_jobs=1)
    assert len(values) == 3
    for i, v in enumerate(values):
        assert v.dtype == np.float64
        expected = df.loc[
            df["unique_id"] == df["unique_id"].unique()[i], "y"
        ].to_numpy()
        np.testing.assert_array_equal(v, expected)


class _ThreadedWalk(BaseGenerator):
    """No Rust batch kernel, so generation takes the threaded path."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.cumsum(self.rng.standard_normal(length))


def _pool_member(cls: type[BaseGenerator]) -> BaseGenerator:
    kwargs = {"min_length": 64, "max_length": 64, "freq": "h", "seed": 8}
    index = next(
        i for i, g in enumerate(pretraining_pool(**kwargs)) if isinstance(g, cls)
    )
    return pretraining_pool(**kwargs, members=[index])[0]


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _pool_member(KernelSynthGenerator), id="native"),
        pytest.param(lambda: _pool_member(TCMGenerator), id="override"),
        pytest.param(
            lambda: _ThreadedWalk(min_length=64, max_length=64, freq="h", seed=8),
            id="threaded",
        ),
    ],
)
def test_values_do_not_depend_on_workers(make) -> None:
    one = make().generate_values(6, n_jobs=1)
    many = make().generate_values(6, n_jobs=4)
    for a, b in zip(one, many, strict=True):
        np.testing.assert_array_equal(a, b)


def _frame_values(df) -> list[np.ndarray]:
    return [
        group["y"].to_numpy()
        for _, group in df.groupby("unique_id", observed=True, sort=False)
    ]


def test_repeated_calls_follow_generate_with_correlated_exogenous() -> None:
    kwargs = {
        "min_length": 30,
        "max_length": 50,
        "freq": "D",
        "seed": 4,
        "exogenous": ExogenousConfig(
            correlated=[CorrelatedExogConfig(name="x", correlation=0.7)]
        ),
    }
    values_gen, frame_gen = SARIMAGenerator(**kwargs), SARIMAGenerator(**kwargs)
    for _ in range(3):
        values = values_gen.generate_values(2, n_jobs=1)
        expected = _frame_values(frame_gen.generate(2, n_jobs=1))
        for a, b in zip(values, expected, strict=True):
            np.testing.assert_array_equal(a, b)


def test_grouped_generators_return_one_array_per_series() -> None:
    kwargs = {"min_length": 40, "max_length": 40, "freq": "h", "seed": 6}
    values = IoTSensorGenerator(**kwargs, n_sensors=3).generate_values(2, n_jobs=1)
    expected = _frame_values(
        IoTSensorGenerator(**kwargs, n_sensors=3).generate(2, n_jobs=1)
    )
    assert len(values) == len(expected) == 6
    for a, b in zip(values, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "cls", [CopulaGenerator, VARGenerator, DailyActiveUsersGenerator]
)
def test_overriding_generators_outside_the_pool(cls: type[BaseGenerator]) -> None:
    kwargs = {"min_length": 30, "max_length": 50, "freq": "D", "seed": 9}
    values = cls(**kwargs).generate_values(3, n_jobs=1)
    expected = _frame_values(cls(**kwargs).generate(3, n_jobs=1))
    assert len(values) == len(expected) == 3
    for a, b in zip(values, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_overlays_and_missingness_are_kept() -> None:
    kwargs = {
        "min_length": 60,
        "max_length": 60,
        "freq": "D",
        "seed": 5,
        "missing_data": True,
        "missing_rate": 0.3,
    }
    values = SARIMAGenerator(**kwargs).generate_values(4, n_jobs=1)
    df = SARIMAGenerator(**kwargs).generate(4, n_jobs=1)
    assert all(np.isnan(v).any() for v in values)
    np.testing.assert_array_equal(np.concatenate(values), df["y"].to_numpy())


@pytest.mark.parametrize("n_series", [0, -1])
def test_rejects_nonpositive_n_series(n_series: int) -> None:
    with pytest.raises(ValueError):
        POOL[0].generate_values(n_series)


@pytest.mark.parametrize(
    "member",
    range(len(pretraining_pool(min_length=8, max_length=8, freq="YS", seed=1))),
)
def test_long_yearly_series_need_no_calendar(member: int) -> None:
    # 2,273 yearly steps run past pandas' datetime range.
    gen = pretraining_pool(
        min_length=2273,
        max_length=2273,
        freq="YS",
        seed=1,
        members=[member],
    )
    (values,) = gen[0].generate_values(1, n_jobs=1)
    assert values.shape == (2273,)
    # No missing-data overlay; only the failing IoT sensor drops values (~35%).
    assert np.isfinite(values).mean() >= 0.5


def test_generate_still_builds_the_calendar_after_generate_values() -> None:
    gen = pretraining_pool(min_length=30, max_length=30, freq="D", seed=2, members=[0])[
        0
    ]
    gen.generate_values(2, n_jobs=1)
    frame = gen.generate(1, n_jobs=1)
    assert "ds" in frame.columns and str(frame["ds"].dtype).startswith("datetime64")
