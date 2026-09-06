"""G3 guards for the bucket-key encoding (PR 3).

``factorize`` maps key columns to dense bucket ids and a sorted vocabulary of
encoded key strings. It builds one string per distinct *combination* rather
than one per row -- hash-factorize each column, combine the codes as a mixed
radix, encode and join only the survivors -- which is what keeps a partitioned
fit off an ``O(n_rows)`` Python string path. The reference it must reproduce is
the straightforward version, ``np.unique(_join_keys(arrays))``, so these guards
assert equivalence to it rather than re-deriving the expected values:

* **G3.1 reference equivalence** -- ``(ids, uniques)`` are byte-identical to the
  per-row string path across key dtypes: str, int, float (with NaN), bool,
  datetime (with NaT), object columns mixing int/float/None, and multi-column
  keys. Includes the empty panel and the single-row panel, where the codes are
  degenerate.
* **G3.2 joined-string ordering** -- the vocabulary is sorted as *joined
  strings*, not as tuples. The two differ: ``"a"`` precedes ``"a\\nb"`` as a
  tuple, but ``"a\\nb\\x1f..."`` precedes ``"a\\x1f..."`` once joined, because
  ``\\n`` (0x0a) sorts below the ``\\x1f`` separator. Ordering by code would be
  tuple ordering, so the survivors must be joined first and sorted second --
  ``lookup`` binary-searches this vocabulary and would silently miss otherwise.
* **G3.3 radix safety** -- combining many high-cardinality columns neither
  overflows int64 nor collides. The codes are re-compressed after every column,
  so the radix product stays bounded by ``n_rows`` instead of growing as the
  product of the cardinalities.
* **G3.4 the null contract** -- ``None``, ``NaN``, ``NaT`` and ``pd.NA`` all
  collapse to the one sentinel bucket and match nothing else, which is SQL
  ``PARTITION BY`` semantics. ``pd.NA`` is a deliberate behaviour change: the
  per-row path raised ``TypeError`` on it, because ``_encode_column``'s object
  branch evaluates ``v != v``, which is ambiguous for ``pd.NA``.
"""

import numpy as np
import pandas as pd
import pytest

from mlforecast.pooled import _NULL_KEY, _join_keys, factorize


def _reference(arrays):
    """The per-row string path ``factorize`` replaced."""
    keys = _join_keys(arrays)
    uniques, ids = np.unique(keys, return_inverse=True)
    return ids.ravel().astype(np.int64, copy=False), uniques


def _assert_matches_reference(arrays, ctx=""):
    got_ids, got_uniques = factorize(arrays)
    ref_ids, ref_uniques = _reference(arrays)
    np.testing.assert_array_equal(got_uniques, ref_uniques, err_msg=f"{ctx}:uniques")
    np.testing.assert_array_equal(got_ids, ref_ids, err_msg=f"{ctx}:ids")
    # ids must index the vocabulary they are returned with
    assert got_ids.dtype == np.int64, ctx
    if len(got_ids):
        assert got_ids.min() >= 0 and got_ids.max() < len(got_uniques), ctx


_N_SERIES, _LENGTH = 40, 30
_N = _N_SERIES * _LENGTH


def _uid():
    return np.repeat(
        np.array([f"s{i}" for i in range(_N_SERIES)], dtype=object), _LENGTH
    )


def _key_columns():
    """One entry per key dtype worth pinning, as ``(id, arrays)``."""
    rng = np.random.default_rng(0)
    uid = _uid()
    floats = rng.integers(0, 7, _N).astype(float)
    floats[::53] = np.nan
    dates = np.tile(
        pd.date_range("2020-01-01", periods=_LENGTH).to_numpy(), _N_SERIES
    ).copy()
    dates[::97] = np.datetime64("NaT")
    mixed = np.array(
        [
            None if i % 61 == 0 else (float(i % 5) if i % 3 else i % 5)
            for i in range(_N)
        ],
        dtype=object,
    )
    return [
        (
            "str",
            [uid, np.array([str(x) for x in rng.integers(0, 9, _N)], dtype=object)],
        ),
        ("int", [uid, rng.integers(0, 9, _N)]),
        ("float_nan", [uid, floats]),
        ("bool", [uid, rng.integers(0, 2, _N).astype(bool)]),
        ("datetime_nat", [uid, dates]),
        ("object_mixed", [uid, mixed]),
        ("single_column", [uid]),
        (
            "three_columns",
            [
                uid,
                np.array([str(x) for x in rng.integers(0, 4, _N)], dtype=object),
                rng.integers(0, 3, _N),
            ],
        ),
    ]


# --------------------------------------------------------------------------- #
# G3.1 -- byte-identical to the per-row string path.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "arrays", [c[1] for c in _key_columns()], ids=[c[0] for c in _key_columns()]
)
def test_g3_1_matches_the_per_row_string_path(arrays):
    """Every key dtype the engine accepts must encode to the same vocabulary.

    Cross-dtype canonicalisation lives in ``_encode_column`` and is reused
    unchanged; what this pins is that factorizing first does not perturb it.
    """
    _assert_matches_reference(arrays)


@pytest.mark.parametrize(
    "arrays",
    [
        [np.array([], dtype=object), np.array([], dtype=object)],
        [np.array(["x"], dtype=object), np.array([1])],
        [np.full(20, np.nan)],
        [np.array([1.0, 2.0, 3.0]), np.array(["a", "a", "b"], dtype=object)],
    ],
    ids=["empty", "single_row", "all_null", "float_integral_keys"],
)
def test_g3_1_degenerate_inputs_match(arrays):
    """The shapes where the code path has no rows, one row, or one bucket.

    ``first[combo_ids[::-1]] = ...`` picks a representative row per combination
    and is the step that would break on an empty panel.
    """
    _assert_matches_reference(arrays)


def test_g3_1_ids_round_trip_through_the_vocabulary():
    """``uniques[ids]`` must reconstruct the per-row keys.

    ``lookup`` resolves rows against this vocabulary at predict, so an id that
    does not index back to its own key would misbucket a series.
    """
    rng = np.random.default_rng(1)
    arrays = [
        _uid(),
        np.array([str(x) for x in rng.integers(0, 6, _N)], dtype=object),
    ]
    ids, uniques = factorize(arrays)
    np.testing.assert_array_equal(uniques[ids], _join_keys(arrays))


# --------------------------------------------------------------------------- #
# G3.2 -- the vocabulary is sorted as joined strings, not as tuples.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sep_byte", ["\n", "\t", "\x01"], ids=["nl", "tab", "x01"])
def test_g3_2_orders_below_separator_bytes_like_the_string_path(sep_byte):
    """A key value holding a byte below ``\\x1f`` inverts tuple vs joined order.

    ``"a"`` before ``"a\\nb"`` as a tuple; the other way once joined. Sorting the
    combinations by code would give tuple order, and ``lookup``'s
    ``searchsorted`` would then miss on a vocabulary it believes is sorted.
    """
    first = np.array(["a", f"a{sep_byte}b"] * 5, dtype=object)
    second = np.array(["Z"] * 10, dtype=object)
    _assert_matches_reference([first, second], ctx=repr(sep_byte))


def test_g3_2_vocabulary_is_sorted():
    """The vocabulary must be sorted for ``lookup``'s binary search."""
    rng = np.random.default_rng(2)
    ids, uniques = factorize(
        [
            np.array([f"k{x}\n{x}" for x in rng.integers(0, 30, 400)], dtype=object),
            np.array([str(x) for x in rng.integers(0, 5, 400)], dtype=object),
        ]
    )
    assert list(uniques) == sorted(uniques)


# --------------------------------------------------------------------------- #
# G3.3 -- the mixed radix neither overflows nor collides.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cols,cardinality", [(4, 4000), (6, 500), (8, 60)])
def test_g3_3_many_high_cardinality_columns(n_cols, cardinality):
    """Naively the radix is the product of the cardinalities, which overflows.

    4 columns of 4000 is 2.56e14 and 8 of 60 is 1.68e14; both fit int64, but
    the *intermediate* products would not survive more columns. Codes are
    re-compressed after each column so the range stays bounded by ``n_rows``.
    """
    rng = np.random.default_rng(3)
    arrays = [rng.integers(0, cardinality, 5000) for _ in range(n_cols)]
    _assert_matches_reference(arrays, ctx=f"{n_cols}x{cardinality}")


def test_g3_3_distinct_combinations_get_distinct_ids():
    """Every distinct key combination must land in its own bucket.

    A radix collision would silently merge two buckets, which reads as a wrong
    feature value rather than an error.
    """
    a, b = np.meshgrid(np.arange(60), np.arange(70))
    arrays = [a.ravel(), b.ravel()]
    ids, uniques = factorize(arrays)
    assert len(uniques) == 60 * 70
    assert len(np.unique(ids)) == 60 * 70


# --------------------------------------------------------------------------- #
# G3.4 -- missing values collapse to one sentinel bucket.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "column",
    [
        np.array([None, "a", None, "b"], dtype=object),
        np.array([np.nan, 1.0, np.nan, 2.0]),
        np.array(["NaT", "2020-01-01", "NaT", "2020-01-02"], dtype="datetime64[ns]"),
    ],
    ids=["none", "nan", "nat"],
)
def test_g3_4_missing_collapses_to_the_sentinel(column):
    """Missing matches missing and nothing else -- SQL ``PARTITION BY``."""
    ids, uniques = factorize([column])
    assert _NULL_KEY in set(uniques)
    assert ids[0] == ids[2], "the two missing values must share a bucket"
    assert len({ids[0], ids[1], ids[3]}) == 3, "missing must not match a real key"
    _assert_matches_reference([column])


def test_g3_4_mixed_none_and_nan_share_one_bucket():
    """``None`` and ``NaN`` in one object column encode to the same sentinel.

    They are distinct raw values, so they factorize apart and only merge at the
    dedupe after joining -- the step that keeps the vocabulary a *set*.
    """
    column = np.array([None, np.nan, "a"], dtype=object)
    ids, uniques = factorize([column])
    assert ids[0] == ids[1]
    assert len(uniques) == 2
    _assert_matches_reference([column])


def test_g3_4_pd_na_lands_in_the_sentinel_bucket():
    """Deliberate behaviour change, pinned so it cannot regress silently.

    The per-row path raises on ``pd.NA``: ``_encode_column``'s object branch
    tests ``v != v``, which returns ``pd.NA`` rather than a bool. Factorizing
    first hands the check to pandas, which treats it as missing like any other
    NA -- consistent with the ``_NULL_KEY`` contract, so it is kept.
    """
    column = np.array(["a", pd.NA, "b"], dtype=object)
    with pytest.raises(TypeError, match="ambiguous"):
        _reference([column])
    ids, uniques = factorize([column])
    assert uniques[ids[1]] == _NULL_KEY
    assert len(uniques) == 3
