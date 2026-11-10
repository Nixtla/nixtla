"""``ScoreField``: the annotated type of every method's ``score`` config field."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any

from pydantic import BeforeValidator, SerializeAsAny, WithJsonSchema

from ..base import ScoreFunction

#: Built-in tags advertised in the JSON schema. Custom scores still validate
#: and round-trip through the registry; the schema only lists what ships.
BUILTIN_KINDS = ("absolute", "signed", "quantile")


def _from_registry(value: Any) -> Any:
    """Rebuild a dumped score from its ``kind`` tag; pass anything else through."""
    if isinstance(value, dict):
        return ScoreFunction.from_kind(value)
    return value


def score_options(kinds: Iterable[str]) -> list[dict[str, Any]]:
    """
    The JSON-schema ``anyOf`` options of a ``score`` field accepting ``kinds``.

    Parameters
    ----------
    kinds : iterable of str
        The score tags to advertise.

    Returns
    -------
    options : list of dict
        One ``{"kind": <tag>}`` object schema per tag, then ``{"type": "null"}``
        for the method's default.
    """
    return [
        *(
            {
                "type": "object",
                "properties": {"kind": {"const": kind}},
                "required": ["kind"],
                "additionalProperties": False,
            }
            for kind in kinds
        ),
        {"type": "null"},
    ]


ScoreField = Annotated[
    SerializeAsAny[ScoreFunction] | None,
    BeforeValidator(_from_registry),
    WithJsonSchema({"anyOf": score_options(BUILTIN_KINDS), "default": None}),
]
"""A nonconformity score as method config, or ``None`` for the method's default.

Accepts a :class:`~conformalforecast.base.ScoreFunction` instance or its dump.
``SerializeAsAny`` makes a subclass dump its own fields (and its own ``kind``),
and the before-validator rebuilds a dump as the class its ``kind`` names, so a
configured score -- custom ones included, once their module is imported --
survives ``model_validate(model_dump())``.

The schema here lists every built-in kind; ``ConformalMethod`` narrows it to the
kinds each method accepts.
"""
