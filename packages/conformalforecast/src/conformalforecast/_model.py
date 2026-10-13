"""Shared pydantic base for classes holding numpy arrays or pandas/polars DataFrames.

Pydantic gives this package two things: a JSON schema an LLM can fill in, and
scalar validation with field descriptions. What it cannot give is deep
validation of the array and library objects these classes carry — numpy arrays,
``StatsForecast`` instances and the like reach pydantic through
``arbitrary_types_allowed=True``, which reduces their validation to an
``isinstance`` check. The win is descriptions and scalar validators, not
structural validation of those fields.

Nor does such a class round-trip: ``model_validate(model_dump())`` cannot
rebuild a live ``StatsForecast`` or a DataFrame from a dump. The four adapters,
the two result types and the fitted forecasters a method's ``fit`` returns are
excluded from that expectation for this reason -- a fitted forecaster holds
calibration arrays and, once calibrated, a live adapter. Every other model in
the package round-trips, methods included: a method is pure config and never
holds fitted state, so it round-trips whether or not it has been used to fit
(``tests/unit/methods/test_pydantic_conversion.py``).
"""

from __future__ import annotations

import narwhals as nw
import numpy as np
from pydantic import BaseModel, ConfigDict

# Values that are never a frame, so they skip the narwhals conversion.
_SCALARS = (type(None), bool, int, float, complex, str, bytes)


def _values_equal(a: object, b: object) -> bool:
    """Elementwise-safe equality for one field value.

    Parameters
    ----------
    a, b : object
        Field values to compare. Arrays, pandas or polars DataFrames, and dicts,
        lists or tuples of either are compared without ever calling ``bool()`` on an elementwise
        result. Only those two container shapes recurse: ``diagnostics`` is
        typed ``dict[str, object]``, and a dict of arrays and a dict of lists
        of arrays are the shapes that reach here in practice.

    Returns
    -------
    bool
        Whether the two values are equal. NaN compares equal to NaN, since
        ragged calibration arrays carry NaN by design (left-NaN padding marks
        windows that do not exist for a series).
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
    if isinstance(a, _SCALARS) and isinstance(b, _SCALARS):
        return bool(a == b)
    frame_a = nw.from_native(a, eager_only=True, pass_through=True)
    frame_b = nw.from_native(b, eager_only=True, pass_through=True)
    if isinstance(frame_a, nw.DataFrame) or isinstance(frame_b, nw.DataFrame):
        if not (isinstance(frame_a, nw.DataFrame) and isinstance(frame_b, nw.DataFrame)):
            return False
        # pandas and polars both define ``equals``, and both treat NaN as equal.
        same_backend = frame_a.implementation == frame_b.implementation
        return same_backend and bool(a.equals(b))  # type: ignore[attr-defined]
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_values_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return bool(a == b)


class ArrayModel(BaseModel):
    """Base model for classes holding numpy arrays or pandas/polars DataFrames.

    ``arbitrary_types_allowed=True`` lets array and wrapped-library fields
    through (see the module docstring for the ceiling that implies).
    ``validate_assignment=False`` keeps the online ``update()`` loop from
    re-validating an ``(n_series, horizon)`` array on every step.

    ``__eq__`` is overridden because pydantic generates equality as a
    field-dict comparison, which calls ``bool()`` on an elementwise array
    comparison and raises ``ValueError`` instead of returning ``False``.
    Instances are explicitly unhashable, like a dataclass with ``eq=True``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    def __eq__(self, other: object) -> bool:
        """Compare field-by-field, elementwise-safely."""
        if not isinstance(other, type(self)) or not isinstance(self, type(other)):
            return NotImplemented if not isinstance(other, ArrayModel) else False
        mine, theirs = self._comparable_fields(), other._comparable_fields()
        if mine.keys() != theirs.keys():
            return False
        return all(_values_equal(mine[k], theirs[k]) for k in mine)

    def _comparable_fields(self) -> dict[str, object]:
        """Declared fields plus any ``extra="allow"`` values, in one mapping.

        A subclass with ``extra="allow"`` keeps undeclared config in
        ``__pydantic_extra__``, not in ``__dict__``. ``QRFQuantileRegressor``
        is the one such class: it forwards ``**kwargs`` to the backend forest,
        so its extras are real config. Without merging them, two instances
        differing only in a passthrough kwarg would compare equal.
        """
        extra: dict[str, object] = dict(self.__pydantic_extra__ or {})
        return {**extra, **self.__dict__}

    __hash__ = None  # type: ignore[assignment]
