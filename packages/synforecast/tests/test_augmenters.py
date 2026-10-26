"""Tests for the augmentation specs and the SynAugment facade over them."""

import copy
import hashlib
import pickle
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from pydantic import Field, ValidationError

from synforecast import (
    DBA,
    FitAndSimulate,
    Mixup,
    MovingBlockBootstrap,
    SynAugment,
    augment,
    generate_series,
)
from synforecast.generators import RandomWalkGenerator
from tests.helpers import to_pandas

src = generate_series(n_series=3, min_length=80, max_length=80, seed=9)

SPECS = [
    FitAndSimulate(seed=1, n_augment=2),
    Mixup(seed=1, n_series=4),
    MovingBlockBootstrap(seed=1, n_augment=2),
    DBA(seed=1, n_augment=2),
]
SPEC_CLASSES = [type(spec) for spec in SPECS]

# Facade method, its kwargs, and the spec those build.
FACADE = [
    ("augment", {"n_augment": 2}, FitAndSimulate(n_augment=2)),
    ("mixup", {"n_series": 4}, Mixup(n_series=4)),
    ("mbb", {"n_augment": 2}, MovingBlockBootstrap(n_augment=2)),
    ("dba", {"n_augment": 2}, DBA(n_augment=2)),
]

# Series start 5 days apart and have different lengths, so timestamps and
# the row counter ``x`` tell the series apart.
_v = to_pandas(generate_series(n_series=3, min_length=60, max_length=100, seed=9))
src_x = _v.assign(
    ds=_v["ds"] + pd.to_timedelta(_v["unique_id"].cat.codes * 5, unit="D"),
    x=np.arange(len(_v), dtype=float),
    label=lambda d: "L" + d["unique_id"].astype(str),
).reset_index(drop=True)


@pytest.fixture(params=SPECS, ids=lambda spec: type(spec).__name__)
def aug(request):
    return request.param


def _sorted(df):
    return (
        to_pandas(df)
        .astype({"unique_id": str})
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def _assert_rows_from_source(out, d, blended, cols=("ds", "x", "label")):
    """Check each synthetic series' extra columns against its source rows.

    FitAndSimulate, MBB and DBA ids name their source (DBA: the reference),
    which must match ``src`` and supply every row. Mixup has no source in its
    id; its rows must be a contiguous window of the ``src`` anchor.
    """
    cols = list(cols)
    for gid, g in out.groupby("unique_id", observed=True):
        g = g.sort_values("ds").reset_index(drop=True)
        sid = str(g["src"].iloc[0])
        assert (g["src"].astype(str) == sid).all()
        if not blended:
            assert sid == str(gid).rsplit("_", 2)[0]
        rows = d[d["unique_id"].astype(str) == sid].sort_values("ds")
        if blended:
            start = rows["ds"].tolist().index(g["ds"].iloc[0])
            rows = rows.iloc[start : start + len(g)]
        assert_frame_equal(g[cols], rows[cols].reset_index(drop=True))


def test_apply_idempotent(aug):
    assert_frame_equal(to_pandas(aug.apply(src)), to_pandas(aug.apply(src)))


def test_seed_override_differs(aug):
    assert not to_pandas(aug.apply(src, seed=2)).equals(to_pandas(aug.apply(src)))


def test_seed_none_varies(aug):
    a = aug.model_copy(update={"seed": None})
    assert not to_pandas(a.apply(src)).equals(to_pandas(a.apply(src)))


def test_include_original_false_drops_sources(aug):
    out = to_pandas(aug.apply(src, include_original=False))
    ids = set(out["unique_id"].astype(str))
    assert ids.isdisjoint(set(to_pandas(src)["unique_id"].astype(str)))


def test_include_original_keeps_input_rows(aug):
    out = _sorted(aug.apply(src_x))
    real = out[out["unique_id"].isin(src_x["unique_id"].astype(str))]
    assert_frame_equal(real.reset_index(drop=True), _sorted(src_x))


def test_frozen_and_closed(aug):
    with pytest.raises(ValidationError):
        aug.seed = 5
    with pytest.raises(ValidationError):
        type(aug)(nonsense=1)


def test_polars_matches_pandas(aug):
    out = aug.apply(pl.from_pandas(to_pandas(src)))
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(_sorted(out), _sorted(aug.apply(src)), check_dtype=False)


def test_custom_columns(aug):
    names = {"unique_id": "item", "ds": "ts", "y": "value"}
    out = to_pandas(
        aug.apply(
            to_pandas(src).rename(columns=names),
            id_col="item",
            time_col="ts",
            target_col="value",
        )
    )
    back = out.rename(columns={v: k for k, v in names.items()})
    assert_frame_equal(back, to_pandas(aug.apply(src)))


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (Mixup, {"alpha": 0}),
        (Mixup, {"alpha": float("nan")}),
        (Mixup, {"alpha": float("inf")}),
        (FitAndSimulate, {"n_augment": 0}),
        (FitAndSimulate, {"generator_override": {"0": "NotAGenerator"}}),
        (FitAndSimulate, {"preserve_timestamps": True}),
        (
            FitAndSimulate,
            {"generator_override": {1: "SARIMAGenerator", "1": "GARCHGenerator"}},
        ),
    ],
)
def test_invalid_values_rejected(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs)


def test_unknown_override_key_warns(caplog):
    FitAndSimulate(seed=1, generator_override={"nope": "SARIMAGenerator"}).apply(src)
    assert sum("nope" in r.getMessage() for r in caplog.records) == 1


def test_override_partial_match_warns_and_applies(caplog):
    sid = str(to_pandas(src)["unique_id"].iloc[0])
    partial = FitAndSimulate(
        seed=1, generator_override={sid: "SARIMAGenerator", "nope": "SARIMAGenerator"}
    )
    full = FitAndSimulate(seed=1, generator_override={sid: "SARIMAGenerator"})
    assert_frame_equal(to_pandas(partial.apply(src)), to_pandas(full.apply(src)))
    warned = [
        r.getMessage() for r in caplog.records if "generator_override" in r.getMessage()
    ]
    assert len(warned) == 1 and warned[0].endswith("['nope']")


@pytest.mark.parametrize(
    "key", [1.0, ("a", 1), pd.Timestamp("2020-01-01")], ids=["float", "tuple", "ts"]
)
def test_override_stores_str_keys(key):
    spec = FitAndSimulate(generator_override={key: "SARIMAGenerator"})
    assert list(spec.generator_override) == [str(key)]
    dumped = spec.model_dump()["generator_override"]
    assert type(dumped) is dict and list(dumped) == [str(key)]


def _roundtrips_equal(d, key, caplog):
    spec = FitAndSimulate(seed=1, generator_override={key: "SARIMAGenerator"})
    expected = to_pandas(spec.apply(d))
    for again in (
        FitAndSimulate.model_validate_json(spec.model_dump_json()),
        FitAndSimulate.model_validate(spec.model_dump()),
    ):
        assert again == spec and hash(again) == hash(spec)
        assert_frame_equal(to_pandas(again.apply(d)), expected)
    assert not any("generator_override" in r.getMessage() for r in caplog.records)
    assert not expected.equals(to_pandas(FitAndSimulate(seed=1).apply(d)))


def test_override_int_key_roundtrips_equal(caplog):
    _roundtrips_equal(src, int(to_pandas(src)["unique_id"].iloc[0]), caplog)


def test_override_numpy_int_key_roundtrips_equal(caplog):
    _roundtrips_equal(src, np.int64(to_pandas(src)["unique_id"].iloc[0]), caplog)


def test_override_str_key_matches_int_ids(caplog):
    _roundtrips_equal(src, str(to_pandas(src)["unique_id"].iloc[0]), caplog)


def test_override_float_key_roundtrips_equal(caplog):
    d = to_pandas(src).astype({"unique_id": int}).astype({"unique_id": float})
    _roundtrips_equal(d, d["unique_id"].iloc[0], caplog)


def test_override_timestamp_key_roundtrips_equal(caplog):
    d = to_pandas(src)
    stamps = pd.date_range("2020-01-01", periods=d["unique_id"].nunique())
    d = d.assign(unique_id=stamps[d["unique_id"].cat.codes])
    _roundtrips_equal(d, stamps[0], caplog)


# SynAugment(seed=1)'s first call on ``src_x``'s id, ds and y columns,
# computed on main before the specs existed: sha256 of the ds bytes and
# joined ids (exact), and the last three, two middle and the sum of ``y``
# (compared with a tolerance, since float bits can differ across platforms).
GOLDEN = {
    "augment": (
        "32f716db47e980ee8ee9b963f9048fc3a29977455f81f28161dec632b7119032",
        [
            0.7316991640261288,
            0.6489450921251431,
            0.5909513686043033,
            0.2616735681940462,
            -1.4837095328096754,
            20293.766092168662,
        ],
    ),
    "mixup": (
        "36114171336795b5a7ca5770e06e1b2f073dd44c3ea30387426b798cd7fd0bdb",
        [
            0.03649951424267811,
            0.6031654081144386,
            1.6422259137383846,
            0.15554131702575183,
            -0.2876706932408688,
            6940.133699545716,
        ],
    ),
    "mbb": (
        "4cd4cae104a5346161dc7a3815c6ae28344ffa622deef7b78bbec74edaa0f614",
        [
            4.0223959826921245,
            4.665257515536804,
            4.0026787513784505,
            0.05149224748015857,
            0.4947421515213011,
            20301.912827493394,
        ],
    ),
    "dba": (
        "717d4ebc3a6dfc8a224121ab365c411d8b61d99c968085ab029d17eecaa27a74",
        [
            3.689748618081804,
            3.3427286175012396,
            1.7436521724794394,
            0.8817238039316795,
            0.20803186360411352,
            20303.342518382724,
        ],
    ),
}


def _assert_golden(out, expected):
    out = to_pandas(out)
    digest = hashlib.sha256(out["ds"].to_numpy().astype("datetime64[ns]").tobytes())
    digest.update("|".join(out["unique_id"].astype(str)).encode())
    expected_digest, expected_y = expected
    assert digest.hexdigest() == expected_digest
    y = out["y"].to_numpy()
    mid = len(y) // 2
    np.testing.assert_allclose(
        [*y[-3:], *y[mid : mid + 2], y.sum()], expected_y, rtol=1e-9
    )


@pytest.mark.parametrize(("method", "kwargs", "spec"), FACADE, ids=lambda p: p)
def test_facade_first_call_matches_main(method, kwargs, spec):  # noqa: ARG001
    out = getattr(SynAugment(seed=1), method)(src_x[["unique_id", "ds", "y"]], **kwargs)
    _assert_golden(out, GOLDEN[method])


# The second call on one SynAugment(seed=1), after a call to ``first``,
# computed on main the same way as GOLDEN.
SECOND_GOLDEN = {
    ("augment", "augment"): (
        "32f716db47e980ee8ee9b963f9048fc3a29977455f81f28161dec632b7119032",
        [
            5.767406915832556,
            5.876895983776149,
            5.923765931033473,
            0.8492348825561358,
            1.5498051651511353,
            20293.820081232734,
        ],
    ),
    ("augment", "mixup"): (
        "6d2330665aa1c8290982f8aa00f054c6ec016995b6b79f06fd127eb7c51fe479",
        [
            2.5174187405135093,
            1.9351502323383796,
            1.017568841763396,
            -0.2419904885305778,
            0.8299666871389947,
            6987.491982008804,
        ],
    ),
    ("mixup", "mbb"): (
        "4cd4cae104a5346161dc7a3815c6ae28344ffa622deef7b78bbec74edaa0f614",
        [
            4.225864092613832,
            3.380735763822192,
            4.039192022184966,
            -0.8955966790033115,
            0.7959898457586755,
            20290.495049633355,
        ],
    ),
    ("mbb", "dba"): (
        "717d4ebc3a6dfc8a224121ab365c411d8b61d99c968085ab029d17eecaa27a74",
        [
            3.6653079858075333,
            3.279016579035148,
            1.7111209769003994,
            0.6331002295050598,
            0.2313345892217783,
            20305.09297130868,
        ],
    ),
}


@pytest.mark.parametrize(
    ("first", "method"), SECOND_GOLDEN, ids=[f"{a}-then-{b}" for a, b in SECOND_GOLDEN]
)
def test_facade_second_call_matches_main(first, method):
    kwargs = {name: kw for name, kw, _ in FACADE}
    d = src_x[["unique_id", "ds", "y"]]
    s = SynAugment(seed=1)
    getattr(s, first)(d, **kwargs[first])
    _assert_golden(
        getattr(s, method)(d, **kwargs[method]), SECOND_GOLDEN[first, method]
    )


def test_facade_single_series_second_call_matches_main():
    values = src_x["y"].to_numpy()[:80]
    ts = np.arange(80)
    s = SynAugment(seed=1)
    s.augment_single_series("a", values, ts, n_augment=2)
    second = s.augment_single_series("a", values, ts, n_augment=2)
    np.testing.assert_allclose(
        second[1][1][:3],
        [-0.196758716939728, 0.37593534132528766, -0.8555001954691858],
        rtol=1e-9,
    )


FACADE_CASES = [(*case, True) for case in FACADE] + [
    (*case, False)
    for case in FACADE
    if case[0] != "augment"  # augment has no flag
]


@pytest.mark.parametrize(("method", "kwargs", "spec", "include_original"), FACADE_CASES)
def test_facade_matches_spec(method, kwargs, spec, include_original):
    if not include_original:
        kwargs = {**kwargs, "include_original": False}
    assert_frame_equal(
        to_pandas(
            getattr(SynAugment(seed=1), method)(src_x, source_col="src", **kwargs)
        ),
        to_pandas(
            spec.apply(
                src_x, seed=1, include_original=include_original, source_col="src"
            )
        ),
    )


@pytest.mark.parametrize(("method", "kwargs", "spec"), FACADE, ids=lambda p: p)
def test_facade_keeps_one_stream_across_calls(method, kwargs, spec):  # noqa: ARG001
    s = SynAugment(seed=1)
    first, second = (to_pandas(getattr(s, method)(src, **kwargs)) for _ in range(2))
    assert not first.equals(second)
    replay = SynAugment(seed=1)
    for expected in (first, second):
        assert_frame_equal(to_pandas(getattr(replay, method)(src, **kwargs)), expected)


def test_facade_single_series_keeps_one_stream():
    values = to_pandas(src)["y"].to_numpy()[:80]
    ts = np.arange(80)
    s = SynAugment(seed=1)
    first = s.augment_single_series("a", values, ts, n_augment=2)
    second = s.augment_single_series("a", values, ts, n_augment=2)
    assert not np.array_equal(first[0][1], second[0][1])
    expected = FitAndSimulate(n_augment=2).apply_series("a", values, ts, seed=1)
    assert [(i, v.tolist()) for i, v, _ in first] == [
        (i, v.tolist()) for i, v, _ in expected
    ]
    replay = SynAugment(seed=1)
    replay.augment_single_series("a", values, ts, n_augment=2)
    again = replay.augment_single_series("a", values, ts, n_augment=2)
    assert [v.tolist() for _, v, _ in again] == [v.tolist() for _, v, _ in second]


def test_facade_exposes_rng():
    assert isinstance(SynAugment(seed=1).rng, np.random.Generator)


def test_fit_simulate_polars_matches_pandas():
    panel = to_pandas(
        generate_series(n_series=50, min_length=40, max_length=40, seed=2)
    ).astype({"unique_id": str})
    assert_frame_equal(
        _sorted(FitAndSimulate(seed=1).apply(pl.from_pandas(panel))),
        _sorted(FitAndSimulate(seed=1).apply(panel)),
        check_dtype=False,
    )


def test_aug_ids_never_collide():
    a = to_pandas(
        generate_series(n_series=1, min_length=30, max_length=30, seed=1)
    ).astype({"unique_id": str})
    d = pd.concat([a.assign(unique_id="A"), a.assign(unique_id="A_aug_0")])
    out = to_pandas(FitAndSimulate(seed=1).apply(d))
    ids = out["unique_id"].astype(str)
    assert not out.duplicated(["unique_id", "ds"]).any()
    assert set(ids) == {"A", "A_aug_0", "A_aug_0_1", "A_aug_0_aug_0"}
    kept = out[ids == "A_aug_0"][["ds", "y"]].reset_index(drop=True)
    assert_frame_equal(kept, d[d["unique_id"] == "A_aug_0"][["ds", "y"]])


def test_input_row_order(aug):
    expected = _sorted(aug.apply(src_x, source_col="src"))
    # Rows interleaved by time keep the ids' first-appearance order.
    interleaved = src_x.sort_values("ds", kind="stable")
    assert_frame_equal(_sorted(aug.apply(interleaved, source_col="src")), expected)
    if not isinstance(aug, FitAndSimulate):
        # The other strategies sort ids, so any row order replays the output.
        shuffled = src_x.sample(frac=1, random_state=0)
        assert_frame_equal(_sorted(aug.apply(shuffled, source_col="src")), expected)
    out = to_pandas(aug.apply(interleaved, include_original=False))
    by_id = out.groupby("unique_id", observed=True)["ds"]
    assert by_id.apply(lambda ds: ds.is_monotonic_increasing).all()


@pytest.mark.parametrize("nan_target", [False, True])
def test_rows_come_from_source(aug, nan_target):
    d = src_x.assign(y=src_x["y"].where(src_x.index % 5 != 0)) if nan_target else src_x
    out = to_pandas(aug.apply(d, include_original=False, source_col="src"))
    _assert_rows_from_source(out, d, blended=isinstance(aug, Mixup))
    assert np.isfinite(out["y"]).all()


def test_extra_columns_keep_values_and_dtypes(aug):
    d = src_x.assign(
        x=src_x["x"].where(src_x.index % 7 != 0),
        cat=pd.Categorical(src_x["label"]),
        when=src_x["ds"] - pd.Timedelta(days=1),
    )
    out = to_pandas(aug.apply(d, include_original=False, source_col="src"))
    assert list(out.columns) == [*d.columns, "src"]
    assert out["x"].isna().any()
    _assert_rows_from_source(
        out, d, blended=isinstance(aug, Mixup), cols=("ds", "x", "label", "cat", "when")
    )


def test_extras_and_source_col_add_no_draws(aug):
    full = to_pandas(aug.apply(src_x, source_col="src"))
    plain = to_pandas(aug.apply(src_x[["unique_id", "ds", "y"]]))
    assert_frame_equal(full[["unique_id", "ds", "y"]], plain)


def test_source_col_names_origin(aug):
    out = to_pandas(aug.apply(src_x, source_col="src"))
    assert list(out.columns) == [*src_x.columns, "src"]
    real = set(src_x["unique_id"].astype(str))
    assert set(out["src"].astype(str)) <= real
    orig = out[out["unique_id"].astype(str).isin(real)]
    assert (orig["src"].astype(str) == orig["unique_id"].astype(str)).all()


@pytest.mark.parametrize("include_original", [True, False])
@pytest.mark.parametrize("engine", [None, "pandas", "polars"])
@pytest.mark.parametrize("polars_input", [False, True], ids=["pd_in", "pl_in"])
def test_source_col_dtype_across_engines(aug, polars_input, engine, include_original):
    # "note" is entirely missing in series 0, so per-series inference would differ
    null_series = src_x["unique_id"].cat.codes == 0
    d = src_x.assign(note=np.where(null_series, None, "a"))
    if polars_input:
        d = pl.from_pandas(d)  # polars ids are Int64
    out = aug.apply(
        d, engine=engine, include_original=include_original, source_col="src"
    )
    to_polars = engine == "polars" or (engine is None and polars_input)
    assert isinstance(out, pl.DataFrame) == to_polars
    assert out["src"].dtype == out["unique_id"].dtype
    p = to_pandas(out)
    assert p["src"].notna().all()
    from_null = p["src"].astype(str) == str(src_x["unique_id"][null_series].iloc[0])
    assert p["note"].isna().tolist() == from_null.tolist()
    assert (p.loc[~from_null, "note"] == "a").all()


@pytest.mark.parametrize("name", ["unique_id", "ds", "y", "x"])
def test_source_col_clash_raises(aug, name):
    with pytest.raises(ValueError, match="source_col"):
        aug.apply(src_x, source_col=name)


class _Boom(RandomWalkGenerator):
    def generate_single_series(self, length):  # noqa: ARG002
        raise RuntimeError("boom")


@pytest.mark.allow_ar1_fallback
def test_ar1_fallback_rows_carry_extras(monkeypatch, caplog):
    monkeypatch.setitem(augment._generator_map(), "RandomWalkGenerator", _Boom)
    # integer ids
    override = dict.fromkeys(src_x["unique_id"].unique(), "RandomWalkGenerator")
    spec = FitAndSimulate(seed=1, on_error="ar1", generator_override=override)
    out = to_pandas(spec.apply(src_x, include_original=False, source_col="src"))
    _assert_rows_from_source(out, src_x, blended=False)
    assert sum("substituted AR(1)" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("include_original", [True, False])
@pytest.mark.parametrize("engine", [None, "pandas"])
def test_pandas_output_has_fresh_range_index(aug, engine, include_original):
    d = src_x.set_axis(src_x.index + 1000)  # a non-default input index
    out = aug.apply(d, include_original=include_original, engine=engine)
    assert out.index.equals(pd.RangeIndex(len(out)))
    expected = aug.apply(src_x, include_original=include_original)
    assert_frame_equal(out, expected, check_exact=True)


def test_mbb_block_size_is_capped_per_series():
    d = generate_series(n_series=2, min_length=10, max_length=10, seed=3)
    capped = to_pandas(MovingBlockBootstrap(seed=1, block_size=50).apply(d))
    # the cap is max(2, len // 3) == 3
    assert_frame_equal(
        capped, to_pandas(MovingBlockBootstrap(seed=1, block_size=3).apply(d))
    )


def test_mbb_accepts_four_row_series():
    d = pd.DataFrame(
        {"unique_id": "a", "ds": pd.date_range("2020", periods=4), "y": [1.0, 3, 2, 4]}
    )
    out = to_pandas(MovingBlockBootstrap(seed=1).apply(d, include_original=False))
    assert len(out) == 4 and np.isfinite(out["y"]).all()


def test_negative_seed_rejected_at_construction():
    with pytest.raises(ValidationError):
        Mixup(seed=-1)
    with pytest.raises(ValueError, match="seed"):
        SynAugment(seed=-1)


def test_syn_augment_rejects_bool_seed():
    with pytest.raises(ValidationError):
        SynAugment(seed=True)


def test_analyze_keeps_first_appearance_order_on_polars():
    ids = [f"s{i}" for i in range(200, 0, -1)]
    panel = pl.DataFrame(
        {
            "unique_id": np.repeat(ids, 10),
            "ds": np.tile(np.arange(10), len(ids)),
            "y": np.random.default_rng(0).normal(size=10 * len(ids)),
        }
    )
    assert list(augment.analyze(panel)) == ids


def _override_spec():
    return FitAndSimulate(seed=1, generator_override={0: "RandomWalkGenerator"})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.__setitem__(0, "SARIMAGenerator"),
        lambda m: m.__delitem__(0),
        lambda m: m.update({1: "SARIMAGenerator"}),
        lambda m: m.pop(0),
        lambda m: m.popitem(),
        lambda m: m.clear(),
        lambda m: m.setdefault(1, "SARIMAGenerator"),
        lambda m: m.__ior__({1: "SARIMAGenerator"}),
    ],
    ids=["setitem", "del", "update", "pop", "popitem", "clear", "setdefault", "ior"],
)
def test_generator_override_is_immutable(mutate):
    spec = _override_spec()
    with pytest.raises(TypeError):
        mutate(spec.generator_override)
    assert spec.generator_override == {"0": "RandomWalkGenerator"}


def test_generator_override_hash_and_eq():
    swapped = FitAndSimulate(
        generator_override={"a": "SARIMAGenerator", 0: "RandomWalkGenerator"}
    )
    ordered = FitAndSimulate(
        generator_override={0: "RandomWalkGenerator", "a": "SARIMAGenerator"}
    )
    assert swapped == ordered and hash(swapped) == hash(ordered)


def test_model_copy_update_is_validated():
    spec = _override_spec()
    copied = spec.model_copy(update={"generator_override": {"b": "SARIMAGenerator"}})
    assert copied.generator_override == {"b": "SARIMAGenerator"}
    assert hash(copied) != hash(spec)
    with pytest.raises(TypeError):
        copied.generator_override["c"] = "SARIMAGenerator"
    for bad in ({"n_augment": 0}, {"nonsense": 1}):
        with pytest.raises(ValidationError):
            spec.model_copy(update=bad)


def test_generator_override_spec_pickles_and_deep_copies():
    spec = _override_spec()
    expected = to_pandas(spec.apply(src))
    for clone in (
        pickle.loads(pickle.dumps(spec)),
        copy.deepcopy(spec),
        spec.model_copy(deep=True),
    ):
        assert clone == spec and hash(clone) == hash(spec)
        assert_frame_equal(to_pandas(clone.apply(src)), expected)


# One int and one float field per spec (FitAndSimulate has no float field).
NUMERIC_FIELDS = [
    (FitAndSimulate, "n_augment", int),
    (Mixup, "max_mix", int),
    (Mixup, "alpha", float),
    (MovingBlockBootstrap, "block_size", int),
    (DBA, "n_neighbors", int),
    (DBA, "window_fraction", float),
]


@pytest.mark.parametrize("value", ["3", True, np.True_], ids=["str", "bool", "np_bool"])
@pytest.mark.parametrize(("cls", "field", "base"), NUMERIC_FIELDS)
def test_numeric_fields_reject_str_and_bool(cls, field, base, value):  # noqa: ARG001
    with pytest.raises(ValidationError):
        cls(**{field: value})


@pytest.mark.parametrize(("cls", "field", "base"), NUMERIC_FIELDS)
def test_numeric_fields_accept_numpy_scalars(cls, field, base):
    scalars = (
        [np.int64(4), np.int32(4)]
        if base is int
        else [np.float64(0.5), np.float32(0.5), 1]
    )
    for value in scalars:
        stored = getattr(cls(**{field: value}), field)
        assert stored == value and type(stored) is base


@pytest.mark.parametrize("seed", [True, np.True_, "3", -1, 1.5])
def test_call_seed_validated(aug, seed):
    with pytest.raises(ValidationError):
        aug.apply(src, seed=seed)


def test_call_seed_accepts_numpy_int(aug):
    assert_frame_equal(
        to_pandas(aug.apply(src, seed=np.int64(3))), to_pandas(aug.apply(src, seed=3))
    )


def _schema(cls, field):
    prop = cls.model_json_schema()["properties"][field]
    return next((s for s in prop.get("anyOf", []) if s["type"] != "null"), prop)


@pytest.mark.parametrize(
    ("cls", "field", "expected"),
    [
        (DBA, "seed", {"minimum": 0}),
        (DBA, "n_augment", {"minimum": 1, "maximum": 1000}),
        (FitAndSimulate, "n_augment", {"minimum": 1, "maximum": 1000}),
        (Mixup, "alpha", {"exclusiveMinimum": 0}),
        (Mixup, "n_series", {"minimum": 1, "maximum": 1_000_000}),
        (MovingBlockBootstrap, "block_size", {"minimum": 2}),
    ],
)
def test_json_schema_uses_standard_bounds(cls, field, expected):
    schema = _schema(cls, field)
    assert {k: schema.get(k) for k in expected} == expected
    assert not {"ge", "le", "gt", "lt"} & schema.keys()


def test_subclass_restating_field_with_int_keeps_validation():
    class MyFit(FitAndSimulate):
        n_augment: augment.Int = Field(default=3, ge=1, le=1000)

    assert MyFit().n_augment == 3
    for bad in (True, "3", 0):
        with pytest.raises(ValidationError):
            MyFit(n_augment=bad)


@pytest.mark.parametrize(
    "y",
    [
        np.array([1.0, 2.0]),
        np.array([1.0, 3.0, 2.0]),
        np.array([1.0, 3.0, 2.0, 5.0, 4.0]),
        np.ones(5),
        np.arange(20.0),
    ],
    ids=["len2", "len3", "len5", "const5", "linear20"],
)
def test_short_series_emit_no_runtime_warning(y):
    df = pd.DataFrame(
        {"unique_id": "a", "ds": pd.date_range("2020", periods=len(y)), "y": y}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FitAndSimulate(seed=0, on_error="ar1").apply(df)
