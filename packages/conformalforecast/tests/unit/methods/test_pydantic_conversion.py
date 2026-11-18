"""Pinned: every method is a pure-config pydantic model that round-trips.

A method declares only config as fields -- its fitted state lives on the
separate fitted object ``fit`` / ``calibrate`` return -- so every method
forbids extras and satisfies ``model_validate(model_dump())``, before and after
it has been used to fit. These flat, table-free assertions are what stops
future changes from regressing that: ``ALL_METHODS`` is derived from the
:class:`~conformalforecast.methods.base.ConformalMethod` subclasses in
``conformalforecast.methods.__all__``, so a new method joins this file the
moment it is exported, rather than only if someone remembers to hand-list it.

The fitted classes exported alongside the methods are deliberately excluded:
they hold calibration arrays and, once calibrated, a live adapter, so they do
not round-trip (see ``conformalforecast._model``'s module docstring).

The two result types (``CalibrationResult``, ``PredictionResult``) are not in
this table either, and cannot be: ``model_json_schema()`` *raises*
``PydanticInvalidForJsonSchema`` on both. A bare ``NDArray`` field reaches
pydantic as an ``IsInstanceSchema`` (via ``arbitrary_types_allowed``), which has
no JSON schema, and a result type is *all* array payload. That they are outputs
— nothing persists and reloads a result — is why the constraint costs nothing,
not why it exists. Their actual contract, elementwise-safe equality and
``extra="forbid"``, is pinned in ``tests/unit/test_base.py``, alongside the
schema-raises assertion that keeps this asymmetry deliberate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast import methods
from conformalforecast.base import ScoreFunction
from conformalforecast.methods import (
    AdaptiveConformalInference,
    AggregatedAdaptiveConformalInference,
    ConformalizedQuantileRegression,
    NonexchangeableConformalPrediction,
    SequentialPredictiveConformalInference,
    SplitConformal,
    WeightedSplit,
)
from conformalforecast.methods.base import ConformalMethod
from conformalforecast.nonconformity import AbsoluteResidual, QuantileScore, SignedResidual

ALL_METHODS = [
    cls
    for cls in (getattr(methods, name) for name in methods.__all__)
    if isinstance(cls, type) and issubclass(cls, ConformalMethod)
]

#: A non-default but compatible score per method (SCORE_TYPE / Split's
#: symmetric rule), plus any config that score needs.
NON_DEFAULT_SCORE: dict[type, tuple[ScoreFunction, dict[str, Any]]] = {
    SplitConformal: (
        SignedResidual(),
        {"symmetric": False, "alpha_lo": 0.05, "alpha_hi": 0.95},
    ),
    ConformalizedQuantileRegression: (QuantileScore(), {}),
    AdaptiveConformalInference: (SignedResidual(), {}),
    AggregatedAdaptiveConformalInference: (SignedResidual(), {}),
    NonexchangeableConformalPrediction: (SignedResidual(), {}),
    SequentialPredictiveConformalInference: (SignedResidual(), {}),
    WeightedSplit: (AbsoluteResidual(), {}),
}


def test_every_exported_method_is_covered() -> None:
    assert set(ALL_METHODS) == set(NON_DEFAULT_SCORE)


@pytest.mark.parametrize("cls", ALL_METHODS, ids=lambda c: c.__name__)
def test_every_method_forbids_extras(cls: type) -> None:
    """A misspelled knob must fail loudly rather than be silently ignored."""
    with pytest.raises(ValidationError):
        cls(alpha=0.1, nonexistent_knob=1)


@pytest.mark.parametrize("cls", ALL_METHODS, ids=lambda c: c.__name__)
def test_class_metadata_is_never_a_field(cls: type) -> None:
    """ClassVar, not a field — otherwise it lands on the LLM-facing schema."""
    for name in ("IS_ONLINE", "CONSUMES_QUANTILES", "SCORE_TYPE", "REQUIRED_CAPABILITIES"):
        assert name not in cls.model_fields
    assert isinstance(cls.IS_ONLINE, bool)
    assert isinstance(cls.CONSUMES_QUANTILES, bool)


@pytest.mark.parametrize("cls", ALL_METHODS, ids=lambda c: c.__name__)
def test_schema_is_config_only_and_offers_the_score(cls: type) -> None:
    properties = cls.model_json_schema()["properties"]
    assert "score" in properties
    assert not [name for name in properties if name.endswith("_")]


@pytest.mark.parametrize("cls", ALL_METHODS, ids=lambda c: c.__name__)
def test_config_round_trips_through_model_dump(cls: type) -> None:
    """``model_validate(model_dump())`` reproduces the instance.

    An agent that persists a configured method and reloads it goes through
    exactly this path, so a validator that rejects its own derived output
    is a real defect, not a curiosity.
    """
    instance = cls(alpha=0.1)
    assert cls.model_validate(instance.model_dump()) == instance


@pytest.mark.parametrize("cls", ALL_METHODS, ids=lambda c: c.__name__)
def test_non_default_score_round_trips(cls: type) -> None:
    score, extra = NON_DEFAULT_SCORE[cls]
    instance = cls(alpha=0.1, score=score, **extra)
    reloaded = cls.model_validate(instance.model_dump())
    assert reloaded == instance
    assert type(reloaded.score) is type(score)
    assert cls.model_validate_json(instance.model_dump_json()) == instance


@pytest.mark.parametrize(
    "cls",
    [c for c in ALL_METHODS if c is not SequentialPredictiveConformalInference],
    ids=lambda c: c.__name__,
)
def test_a_method_used_for_fitting_still_round_trips(cls: type) -> None:
    """Fitting never writes to the method, so it stays pure config.

    SPCI is covered separately, by ``test_spci_used_for_fitting_still_round_trips``,
    because it needs the ``[spci]`` extra and a calibration panel longer than
    its ``window_size``.
    """
    rng = np.random.default_rng(0)
    preds = rng.normal(size=(2, 40, 3))
    truths = preds + rng.normal(size=(2, 40, 3))
    if cls.CONSUMES_QUANTILES:
        preds = np.stack([preds - 1.0, preds + 1.0], axis=-1)
    instance = cls(alpha=0.1)
    before = instance.model_dump()
    instance.fit(preds, truths)
    assert instance.model_dump() == before
    assert cls.model_validate(instance.model_dump()) == instance
    assert cls.model_validate_json(instance.model_dump_json()) == instance


def test_spci_used_for_fitting_still_round_trips() -> None:
    """SPCI's case of the test above, which needs the ``[spci]`` extra."""
    pytest.importorskip("quantile_forest")
    rng = np.random.default_rng(0)
    preds = rng.normal(size=(1, 40, 1))
    truths = preds + rng.normal(size=(1, 40, 1))
    instance = SequentialPredictiveConformalInference(alpha=0.1, window_size=5)
    before = instance.model_dump()
    instance.fit(preds, truths)
    assert instance.model_dump() == before
    assert SequentialPredictiveConformalInference.model_validate(before) == instance


def test_asymmetric_cqr_round_trips() -> None:
    """The asymmetric path, where the levels are genuine user config."""
    instance = ConformalizedQuantileRegression(
        alpha=0.1, symmetric=False, alpha_lo=0.02, alpha_hi=0.92
    )
    assert ConformalizedQuantileRegression.model_validate(instance.model_dump()) == instance


def _advertised_kinds(cls: type) -> set[str]:
    options = cls.model_json_schema()["properties"]["score"]["anyOf"]
    return {o["properties"]["kind"]["const"] for o in options if o.get("type") == "object"}


@pytest.mark.parametrize("cls", ALL_METHODS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kind", ["absolute", "signed", "quantile"])
def test_schema_advertises_exactly_the_score_kinds_construction_accepts(
    cls: type, kind: str
) -> None:
    """A score the schema calls valid must construct, and one it omits must not."""
    try:
        cls.model_validate({"alpha": 0.1, "score": {"kind": kind}})
    except ValidationError:
        accepted = False
    else:
        accepted = True
    assert (kind in _advertised_kinds(cls)) is accepted


@pytest.mark.parametrize(
    "cls",
    [c for c in ALL_METHODS if not c.CONSUMES_QUANTILES],
    ids=lambda c: c.__name__,
)
def test_point_methods_reject_a_quantile_score_at_construction(cls: type) -> None:
    with pytest.raises(ValidationError, match="QuantileScore"):
        cls(alpha=0.1, score=QuantileScore())
