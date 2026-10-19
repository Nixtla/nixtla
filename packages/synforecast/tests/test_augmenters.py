"""Tests for the augmentation specs and the SynAugment facade over them."""

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from synforecast import (
    DBA,
    FitAndSimulate,
    Mixup,
    MovingBlockBootstrap,
    SynAugment,
    analyze,
    generate_series,
)
from tests.helpers import to_pandas

src = generate_series(n_series=3, min_length=80, max_length=80, seed=9)

SPECS = [
    FitAndSimulate(seed=1, n_augment=2),
    Mixup(seed=1, n_series=4),
    MovingBlockBootstrap(seed=1, n_augment=2),
    DBA(seed=1, n_augment=2),
]


@pytest.fixture(params=SPECS, ids=lambda spec: type(spec).__name__)
def aug(request):
    return request.param


def test_apply_idempotent(aug):
    assert_frame_equal(to_pandas(aug.apply(src)), to_pandas(aug.apply(src)))


def test_seed_override_differs(aug):
    assert not to_pandas(aug.apply(src, seed=2)).equals(to_pandas(aug.apply(src)))


def test_seed_none_varies(aug):
    a = aug.model_copy(update={"seed": None})
    assert not to_pandas(a.apply(src)).equals(to_pandas(a.apply(src)))


def test_roundtrip(aug):
    again = type(aug).model_validate(aug.model_dump())
    assert_frame_equal(to_pandas(again.apply(src)), to_pandas(aug.apply(src)))


def test_include_original_false_drops_sources(aug):
    out = to_pandas(aug.apply(src, include_original=False))
    ids = set(out["unique_id"].astype(str))
    assert ids.isdisjoint(set(to_pandas(src)["unique_id"].astype(str)))


def test_frozen(aug):
    with pytest.raises(ValidationError):
        aug.seed = 5


def test_extra_fields_forbidden(aug):
    with pytest.raises(ValidationError):
        type(aug)(nonsense=1)


def test_polars_stays_polars(aug):
    assert isinstance(aug.apply(pl.from_pandas(to_pandas(src))), pl.DataFrame)


def test_custom_columns(aug):
    renamed = to_pandas(src).rename(
        columns={"unique_id": "item", "ds": "ts", "y": "value"}
    )
    out = to_pandas(
        aug.apply(renamed, id_col="item", time_col="ts", target_col="value")
    )
    assert set(out.columns) == {"item", "ts", "value"}


def test_invalid_values_rejected():
    with pytest.raises(ValidationError):
        Mixup(alpha=0)
    with pytest.raises(ValidationError):
        FitAndSimulate(n_augment=0)
    with pytest.raises(ValidationError):
        FitAndSimulate(generator_override={"0": "NotAGenerator"})
    with pytest.raises(ValidationError):
        FitAndSimulate(preserve_timestamps=True)
    with pytest.raises(ValidationError):
        FitAndSimulate(generator_override={1: "SARIMAGenerator", "1": "GARCHGenerator"})


def test_call_seed_reaches_generators():
    spec = FitAndSimulate(n_augment=2)
    assert_frame_equal(
        to_pandas(spec.apply(src, seed=5)), to_pandas(spec.apply(src, seed=5))
    )


def test_unknown_override_key_warns(caplog):
    FitAndSimulate(seed=1, generator_override={"nope": "SARIMAGenerator"}).apply(src)
    assert sum("nope" in r.getMessage() for r in caplog.records) == 1


def test_override_int_keys_match(caplog):
    first = to_pandas(src)["unique_id"].iloc[0]  # integer id
    out = FitAndSimulate(seed=1, generator_override={first: "SARIMAGenerator"}).apply(
        src
    )
    assert not any("generator_override" in r.getMessage() for r in caplog.records)
    assert not to_pandas(out).equals(to_pandas(FitAndSimulate(seed=1).apply(src)))


@pytest.mark.parametrize(
    ("method", "kwargs", "spec"),
    [
        (
            "augment",
            {"n_augment": 2, "preserve_timestamps": False},
            FitAndSimulate(seed=1, n_augment=2),
        ),
        ("mixup", {"n_series": 4}, Mixup(seed=1, n_series=4)),
        ("mbb", {"n_augment": 2}, MovingBlockBootstrap(seed=1, n_augment=2)),
        ("dba", {"n_augment": 2}, DBA(seed=1, n_augment=2)),
    ],
)
def test_facade_matches_spec(method, kwargs, spec):
    assert_frame_equal(
        to_pandas(getattr(SynAugment(seed=1), method)(src, **kwargs)),
        to_pandas(spec.apply(src)),
    )


def test_facade_analyze_and_single_series_match():
    a, b = SynAugment(seed=1).analyze(src), analyze(src)
    assert a.keys() == b.keys()
    for k in a:  # values may hold arrays
        assert repr(a[k]) == repr(b[k])
    s = to_pandas(src)
    sid = s["unique_id"].iloc[0]
    one = s[s["unique_id"] == sid]
    args = (sid, one["y"].to_numpy(), one["ds"].to_numpy())
    old = SynAugment(seed=1).augment_single_series(*args, n_augment=2)
    new = FitAndSimulate(seed=1, n_augment=2).apply_series(*args)
    assert [(i, v.tolist()) for i, v, _ in old] == [(i, v.tolist()) for i, v, _ in new]


def test_facade_repeat_calls_identical():
    s = SynAugment(seed=1)
    assert_frame_equal(
        to_pandas(s.mbb(src, n_augment=2)), to_pandas(s.mbb(src, n_augment=2))
    )
    assert not hasattr(s, "rng")


def _sorted(df):
    return (
        to_pandas(df)
        .astype({"unique_id": str})
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def test_fit_simulate_polars_deterministic():
    panel = pl.from_pandas(
        to_pandas(
            generate_series(n_series=50, min_length=40, max_length=40, seed=2)
        ).astype({"unique_id": str})
    )
    assert_frame_equal(
        _sorted(FitAndSimulate(seed=1).apply(panel)),
        _sorted(FitAndSimulate(seed=1).apply(panel)),
    )


def test_aug_ids_never_collide():
    a = to_pandas(
        generate_series(n_series=1, min_length=30, max_length=30, seed=1)
    ).astype({"unique_id": str})
    d = pd.concat([a.assign(unique_id="A"), a.assign(unique_id="A_aug_0")])
    out = to_pandas(FitAndSimulate(seed=1).apply(d))
    assert not out.duplicated(["unique_id", "ds"]).any()
    assert out["unique_id"].astype(str).nunique() == 4
