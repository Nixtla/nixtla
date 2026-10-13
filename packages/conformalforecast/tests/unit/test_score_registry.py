"""The ``kind`` tag, the score registry and ``ScoreField``."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ConfigDict, ValidationError

from conformalforecast._model import ArrayModel
from conformalforecast.base import ScoreFunction
from conformalforecast.nonconformity import (
    AbsoluteResidual,
    QuantileScore,
    ScoreField,
    SignedResidual,
)


class _Holder(ArrayModel):
    model_config = ConfigDict(extra="forbid")
    score: ScoreField = None


class _ScaledSigned(SignedResidual):
    """A custom score with config of its own, declared at module level."""

    kind: Literal["scaled_signed"] = "scaled_signed"
    scale: float = 1.0


@pytest.mark.parametrize("score", [None, AbsoluteResidual(), SignedResidual(), QuantileScore()])
def test_builtin_scores_round_trip_through_a_dict(score: ScoreFunction | None) -> None:
    holder = _Holder(score=score)
    reloaded = _Holder.model_validate(holder.model_dump())
    assert reloaded == holder
    assert type(reloaded.score) is type(score)


@pytest.mark.parametrize("score", [AbsoluteResidual(), SignedResidual(), QuantileScore()])
def test_builtin_scores_round_trip_through_json(score: ScoreFunction) -> None:
    holder = _Holder(score=score)
    assert _Holder.model_validate_json(holder.model_dump_json()) == holder


def test_custom_score_with_its_own_kind_round_trips_with_its_fields() -> None:
    holder = _Holder(score=_ScaledSigned(scale=3.0))
    dumped = holder.model_dump()
    assert dumped["score"] == {"kind": "scaled_signed", "scale": 3.0}
    reloaded = _Holder.model_validate(dumped)
    assert type(reloaded.score) is _ScaledSigned
    assert reloaded.score.scale == 3.0


def test_subclass_inheriting_its_parents_kind_fails_at_class_definition() -> None:
    with pytest.raises(TypeError, match="reuses the score tag 'absolute'"):

        class _Forgetful(AbsoluteResidual):
            pass


def test_concrete_subclass_without_a_kind_fails_at_class_definition() -> None:
    with pytest.raises(TypeError, match="must declare its own score tag"):

        class _Untagged(ScoreFunction):
            def score(self, prediction, truth):  # type: ignore[no-untyped-def]
                return truth - prediction

            def invert(self, prediction, score_threshold):  # type: ignore[no-untyped-def]
                return prediction


def test_redefining_the_same_class_replaces_its_registration() -> None:
    """A class defined in a function body is redefined on every call."""

    def define() -> type[ScoreFunction]:
        class _Local(AbsoluteResidual):
            kind: Literal["local_redefined"] = "local_redefined"

        return _Local

    first, second = define(), define()
    assert ScoreFunction._registry["local_redefined"] is second
    assert first is not second


def test_unknown_kind_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="unknown score kind 'nope'"):
        _Holder.model_validate({"score": {"kind": "nope"}})


def test_a_non_score_is_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        _Holder(score="absolute")  # type: ignore[arg-type]


def test_builtin_kinds_cannot_be_overridden_at_construction() -> None:
    with pytest.raises(ValidationError):
        AbsoluteResidual(kind="signed")  # type: ignore[arg-type]


def test_json_schema_advertises_the_builtin_kinds() -> None:
    schema = _Holder.model_json_schema()["properties"]["score"]
    kinds = {
        option["properties"]["kind"]["const"]
        for option in schema["anyOf"]
        if option.get("type") == "object"
    }
    assert kinds == {"absolute", "signed", "quantile"}
