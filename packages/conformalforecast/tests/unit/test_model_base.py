"""Tests for the shared `ArrayModel` pydantic base."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from conformalforecast._model import ArrayModel


class _Holder(ArrayModel):
    name: str
    data: NDArray[np.floating] | None = None


class _DiagHolder(ArrayModel):
    diagnostics: dict[str, object] = {}


def test_equality_with_arrays_returns_bool_not_raises() -> None:
    a = _Holder(name="x", data=np.arange(6.0))
    b = _Holder(name="x", data=np.arange(6.0))
    c = _Holder(name="x", data=np.arange(6.0) + 1)
    d = _Holder(name="y", data=np.arange(6.0))

    assert a == b  # equal values compare equal
    assert not (a == c)  # noqa: SIM201  # different values compare unequal, no exception
    assert not (a == d)  # noqa: SIM201  # different scalar field
    assert not (a == object())  # noqa: SIM201


def test_arrays_are_accepted_without_coercion() -> None:
    arr = np.arange(6.0, dtype=np.float32)
    holder = _Holder(name="x", data=arr)
    assert holder.data is arr  # not copied
    assert holder.data.dtype == np.float32  # not upcast


def test_assignment_is_not_revalidated() -> None:
    holder = _Holder(name="x")
    holder.data = np.arange(3.0)  # would raise under validate_assignment
    assert holder.data is not None
    assert holder.data.shape == (3,)


def test_dict_of_arrays_field_compares_elementwise() -> None:
    """`CalibrationResult.diagnostics` is a dict whose values are arrays."""
    a = _DiagHolder(diagnostics={"score_quantile": np.array([1.0, np.nan]), "gamma": 0.05})
    b = _DiagHolder(diagnostics={"score_quantile": np.array([1.0, np.nan]), "gamma": 0.05})
    c = _DiagHolder(diagnostics={"score_quantile": np.array([2.0, np.nan]), "gamma": 0.05})
    d = _DiagHolder(diagnostics={"score_quantile": np.array([1.0, np.nan])})

    assert a == b  # NaN in a nested array must not break equality
    assert not (a == c)  # noqa: SIM201
    assert not (a == d)  # noqa: SIM201  # different keys
    assert not (a == _Holder(name="x"))  # noqa: SIM201  # different model type


def test_dict_of_list_of_arrays_compares_elementwise() -> None:
    """`diagnostics` is typed `dict[str, object]`, so a list of arrays is permitted.

    Without the sequence branch this raises the same ambiguous-truth-value
    ``ValueError`` the module exists to prevent.
    """
    arrs = [np.array([1.0, 2.0]), np.array([3.0, np.nan])]
    a = _DiagHolder(diagnostics={"per_expert": [arrs[0].copy(), arrs[1].copy()]})
    b = _DiagHolder(diagnostics={"per_expert": [arrs[0].copy(), arrs[1].copy()]})
    c = _DiagHolder(diagnostics={"per_expert": [arrs[0].copy(), np.array([9.0, np.nan])]})
    d = _DiagHolder(diagnostics={"per_expert": [arrs[0].copy()]})
    e = _DiagHolder(diagnostics={"per_expert": (arrs[0].copy(), arrs[1].copy())})

    assert a == b
    assert not (a == c)  # noqa: SIM201  # differing array inside the list
    assert not (a == d)  # noqa: SIM201  # different length
    assert not (a == e)  # noqa: SIM201  # tuple is not a list


class _FrameHolder(ArrayModel):
    frame: object


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_frames_compare_by_value_with_nan(lib: str) -> None:
    mod = pytest.importorskip(lib)
    a = mod.DataFrame({"x": [1.0, float("nan")], "id": ["a", "b"]})
    b = mod.DataFrame({"x": [1.0, float("nan")], "id": ["a", "b"]})
    c = mod.DataFrame({"x": [1.0, 2.0], "id": ["a", "b"]})
    assert _FrameHolder(frame=a) == _FrameHolder(frame=b)
    assert _FrameHolder(frame=a) != _FrameHolder(frame=c)


def test_frames_from_different_backends_are_unequal() -> None:
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    data = {"x": [1.0, 2.0]}
    assert _FrameHolder(frame=pd.DataFrame(data)) != _FrameHolder(frame=pl.DataFrame(data))


def test_frame_never_equals_a_non_frame() -> None:
    pd = pytest.importorskip("pandas")
    assert _FrameHolder(frame=pd.DataFrame({"x": [1.0]})) != _FrameHolder(frame=[1.0])
