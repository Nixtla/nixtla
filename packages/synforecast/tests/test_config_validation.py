"""Validation contracts shared by all public configuration models."""

import inspect

import pytest
from pydantic import ValidationError

from synforecast.exogenous import CorrelatedExogConfig, ExogenousConfig
from synforecast.generators import RandomWalkGenerator

BASE = {"min_length": 10, "max_length": 10, "freq": "D"}


def test_generator_rejects_unknown_parameters() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RandomWalkGenerator(**BASE, volatilty=2.0)


def test_runtime_state_is_not_constructor_configuration() -> None:
    signature = inspect.signature(RandomWalkGenerator)
    assert "rng" not in signature.parameters

    with pytest.raises(ValidationError, match="extra_forbidden"):
        RandomWalkGenerator(**BASE, rng="not a generator")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"engine": "numpy"}, "engine"),
        ({"id_col": "y"}, "must be distinct"),
        ({"time_col": ""}, "must not be empty"),
        (
            {"innovation_distribution": "normal", "innovation_params": {"df": 5}},
            "unsupported innovation_params",
        ),
        (
            {"innovation_distribution": "t", "innovation_params": {"df": 2}},
            "must be > 2",
        ),
        (
            {
                "innovation_distribution": "t",
                "innovation_params": {"df": float("nan")},
            },
            "must be > 2",
        ),
        ({"changepoint_locations": [-0.1]}, r"values in \[0, 1\]"),
        ({"changepoint_variance_changes": [0.0]}, "must be positive"),
    ],
)
def test_generator_rejects_invalid_shared_configuration(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        RandomWalkGenerator(**BASE, **kwargs)


def test_exogenous_models_reject_unknown_parameters() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CorrelatedExogConfig(name="x", corelation=0.5)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExogenousConfig(datetime_feature=True)


def test_correlated_exogenous_names_are_unique_and_do_not_collide() -> None:
    duplicate = [CorrelatedExogConfig(name="x"), CorrelatedExogConfig(name="x")]
    with pytest.raises(ValidationError, match="names must be unique"):
        ExogenousConfig(correlated=duplicate)

    with pytest.raises(ValidationError, match="collide with output columns"):
        RandomWalkGenerator(
            **BASE,
            exogenous=ExogenousConfig(correlated=[CorrelatedExogConfig(name="y")]),
        )
