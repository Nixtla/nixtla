"""Tests for the NaN-aware exact-order-statistic quantile helper."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from conformalforecast._quantiles import (
    conformal_quantile,
    weighted_conformal_quantile,
    weighted_quantile_from_weights,
    weighted_quantile_per_test_point,
)
from conformalforecast.base import InsufficientCalibrationSamples


def _scores(values: list[float]) -> np.ndarray:
    """Wrap a flat list into shape (1, n, 1)."""
    return np.asarray(values, dtype=np.float64).reshape(1, -1, 1)


class TestExactOrderStatistic:
    def test_hand_computed_order_statistic(self) -> None:
        # n=9, alpha=0.2 -> k = ceil(0.8 * 10) = 8 -> 8th smallest = 8.0
        result = conformal_quantile(_scores([1, 2, 3, 4, 5, 6, 7, 8, 9]), alpha=0.2)
        assert result.shape == (1, 1)
        assert result[0, 0] == 8.0

    def test_no_interpolation(self) -> None:
        # n=4, alpha=0.5 -> k = ceil(0.5 * 5) = 3 -> exactly the 3rd smallest.
        # Linear interpolation (np.quantile default) would give a non-sample value.
        result = conformal_quantile(_scores([1.0, 2.0, 10.0, 20.0]), alpha=0.5)
        assert result[0, 0] == 10.0

    def test_quantile_level_one_edge(self) -> None:
        # n = ceil(1/alpha) = 5 with alpha=0.2 -> k = ceil(0.8 * 6) = 5 = n -> max.
        result = conformal_quantile(_scores([3.0, 1.0, 4.0, 1.0, 5.0]), alpha=0.2)
        assert result[0, 0] == 5.0

    def test_per_series_per_horizon(self) -> None:
        # (2 series, 9 windows, 2 horizons), horizon 1 = horizon 0 scaled by 10.
        base = np.arange(1.0, 10.0)
        scores = np.empty((2, 9, 2))
        scores[0, :, 0] = base
        scores[0, :, 1] = base * 10
        scores[1, :, 0] = base + 100
        scores[1, :, 1] = (base + 100) * 10
        result = conformal_quantile(scores, alpha=0.2)  # k = 8
        expected = np.array([[8.0, 80.0], [108.0, 1080.0]])
        np.testing.assert_array_equal(result, expected)


class TestNaNAwareness:
    def test_nan_windows_ignored(self) -> None:
        # NaN-padded series must give the same answer as its dense sub-array.
        dense = _scores([1, 2, 3, 4, 5, 6, 7, 8, 9])
        padded = np.concatenate([np.full((1, 3, 1), np.nan), dense], axis=1)
        np.testing.assert_array_equal(
            conformal_quantile(padded, alpha=0.2),
            conformal_quantile(dense, alpha=0.2),
        )

    def test_insufficient_samples_raises_with_indices(self) -> None:
        # Series 0 has 9 effective windows; series 1 has only 4 (< ceil(1/0.2) = 5).
        scores = np.empty((2, 9, 1))
        scores[0, :, 0] = np.arange(1.0, 10.0)
        scores[1, :, 0] = np.arange(1.0, 10.0)
        scores[1, :5, 0] = np.nan
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            conformal_quantile(scores, alpha=0.2)
        assert excinfo.value.series_indices == [1]
        assert "1" in str(excinfo.value)

    def test_all_series_insufficient(self) -> None:
        scores = np.ones((3, 2, 1))
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            conformal_quantile(scores, alpha=0.1)  # needs ceil(1/0.1) = 10
        assert excinfo.value.series_indices == [0, 1, 2]


class TestDtypeAndValidation:
    def test_float32_preserved(self) -> None:
        scores = np.arange(1.0, 10.0, dtype=np.float32).reshape(1, 9, 1)
        assert conformal_quantile(scores, alpha=0.2).dtype == np.float32

    def test_rejects_non_3d(self) -> None:
        with pytest.raises(ValueError, match="3-D"):
            conformal_quantile(np.ones((5, 2)), alpha=0.2)

    def test_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            conformal_quantile(np.ones((1, 10, 1)), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            conformal_quantile(np.ones((1, 10, 1)), alpha=1.0)


class TestMaskedQuantile:
    def test_matches_np_quantile_dense(self) -> None:
        from conformalforecast._quantiles import masked_quantile

        rng = np.random.default_rng(0)
        scores = rng.normal(size=(2, 40, 3))
        levels = np.full((2, 3), 0.9)
        out = masked_quantile(scores, levels)
        assert out.shape == (2, 3)
        for s in range(2):
            for h in range(3):
                assert np.isclose(out[s, h], np.quantile(scores[s, :, h], 0.9))

    def test_ignores_nan_windows(self) -> None:
        from conformalforecast._quantiles import masked_quantile

        dense = np.arange(1.0, 10.0).reshape(1, 9, 1)
        padded = np.concatenate([np.full((1, 4, 1), np.nan), dense], axis=1)
        levels = np.full((1, 1), 0.8)
        np.testing.assert_allclose(masked_quantile(padded, levels), masked_quantile(dense, levels))

    def test_saturation_and_all_nan(self) -> None:
        from conformalforecast._quantiles import _SATURATION_SENTINEL, masked_quantile

        scores = np.arange(1.0, 6.0).reshape(1, 5, 1)
        assert masked_quantile(scores, np.full((1, 1), 1.5))[0, 0] == _SATURATION_SENTINEL
        assert masked_quantile(scores, np.full((1, 1), -0.5))[0, 0] == -_SATURATION_SENTINEL
        all_nan = np.full((1, 5, 1), np.nan)
        assert masked_quantile(all_nan, np.full((1, 1), 0.5))[0, 0] == _SATURATION_SENTINEL

    def test_float32_preserved(self) -> None:
        from conformalforecast._quantiles import masked_quantile

        scores = np.arange(1.0, 10.0, dtype=np.float32).reshape(1, 9, 1)
        assert masked_quantile(scores, np.full((1, 1), 0.5)).dtype == np.float32


#: numpy computed the quantile virtual index and gamma in the *array* dtype
#: through 2.4 and switched to float64 in 2.5. The helper follows 2.5, so
#: float32 bit-identity holds from 2.5 onward and lands within float32
#: epsilon below it. float64 is unaffected either way.
_NUMPY_GE_2_5 = tuple(int(p) for p in np.__version__.split(".")[:2]) >= (2, 5)

#: Tolerance for the float32 comparison on numpy < 2.5. The inputs below are
#: standard normal, so a handful of float32 epsilons is the width of the
#: arithmetic difference; anything larger is a real bug, not rounding.
_F32_ATOL = 8 * float(np.finfo(np.float32).eps) * 4.0


def _np_quantile(values: np.ndarray, level: object) -> np.floating:
    """``np.quantile`` with the level as a *Python* float.

    Deliberate, not incidental: from numpy 2.5 a weak Python scalar keeps the
    interpolation in the array dtype, while a numpy scalar promotes the whole
    computation -- and the result -- to float64. The helper's contract is to
    preserve the input dtype, so the Python-float call is the like-for-like
    comparison.
    """
    return np.quantile(values, float(level))  # type: ignore[no-any-return]


def _assert_matches_numpy(got: float, expected: float, dtype: type, context: object) -> None:
    """Exact equality, except float32 on numpy < 2.5 (see ``_NUMPY_GE_2_5``)."""
    if dtype is np.float64 or _NUMPY_GE_2_5:
        assert got == expected, context
    else:
        assert abs(float(got) - float(expected)) <= _F32_ATOL, context


class TestLinearQuantileFromSorted:
    """Agreement with ``np.quantile(method="linear")``.

    The batched gather reproduces numpy's arithmetic: the virtual index and
    gamma in float64, the interpolation in the array dtype (numpy hands gamma
    to ``_lerp`` as a weak Python float), and numpy's two-branch ``_lerp``.
    Getting any of those wrong shifts ``float32`` results, which is why this is
    brute-forced rather than spot-checked.
    """

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_matches_np_quantile_bruteforce(self, dtype: type) -> None:
        from conformalforecast._quantiles import _linear_quantile_from_sorted

        rng = np.random.default_rng(0)
        for _ in range(40):
            n = int(rng.integers(1, 90))
            values = rng.normal(size=n).astype(dtype)
            sorted_scores = np.sort(values).reshape(1, n, 1)
            n_valid = np.array([[n]])
            for level in rng.uniform(1e-9, 1 - 1e-9, 150):
                got = _linear_quantile_from_sorted(
                    sorted_scores, n_valid, np.array([[level]], dtype=np.float64)
                )
                expected = _np_quantile(values, level)
                assert got.dtype == np.dtype(dtype)
                _assert_matches_numpy(
                    got[0, 0], expected, dtype, (n, float(level), np.dtype(dtype).name)
                )

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_ragged_cells_match_dense_per_cell(self, dtype: type) -> None:
        from conformalforecast._quantiles import masked_quantile

        rng = np.random.default_rng(1)
        n_series, n_windows, horizon = 4, 30, 3
        scores = rng.normal(size=(n_series, n_windows, horizon)).astype(dtype)
        for i in range(n_series):
            scores[i, : 3 * i, :] = np.nan
        levels = rng.uniform(0.05, 0.95, size=(n_series, horizon))
        out = masked_quantile(scores, levels)
        for s in range(n_series):
            for h in range(horizon):
                cell = scores[s, :, h]
                dense = cell[~np.isnan(cell)]
                _assert_matches_numpy(out[s, h], _np_quantile(dense, levels[s, h]), dtype, (s, h))

    def test_saturation_and_empty_cells(self) -> None:
        from conformalforecast._quantiles import (
            _SATURATION_SENTINEL,
            _linear_quantile_from_sorted,
        )

        sorted_scores = np.array([[[1.0], [2.0], [3.0], [np.nan]]])  # (1, 4, 1)
        n_valid = np.array([[3]])
        assert (
            _linear_quantile_from_sorted(sorted_scores, n_valid, np.array([[1.0]]))[0, 0]
            == _SATURATION_SENTINEL
        )
        assert (
            _linear_quantile_from_sorted(sorted_scores, n_valid, np.array([[1.5]]))[0, 0]
            == _SATURATION_SENTINEL
        )
        assert (
            _linear_quantile_from_sorted(sorted_scores, n_valid, np.array([[0.0]]))[0, 0]
            == -_SATURATION_SENTINEL
        )
        empty = np.full((1, 4, 1), np.nan)
        assert (
            _linear_quantile_from_sorted(empty, np.array([[0]]), np.array([[0.5]]))[0, 0]
            == _SATURATION_SENTINEL
        )

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_leading_batch_axis_matches_looped_calls(self, dtype: type) -> None:
        from conformalforecast._quantiles import masked_quantile

        rng = np.random.default_rng(2)
        n_experts, n_series, horizon = 6, 5, 4
        scores = rng.normal(size=(n_series, 25, horizon)).astype(dtype)
        levels = rng.uniform(0.01, 0.99, size=(n_experts, n_series, horizon))
        batched = masked_quantile(scores, levels)
        assert batched.shape == (n_experts, n_series, horizon)
        assert batched.dtype == np.dtype(dtype)
        for k in range(n_experts):
            np.testing.assert_array_equal(batched[k], masked_quantile(scores, levels[k]))

    def test_single_valid_value_cell(self) -> None:
        from conformalforecast._quantiles import masked_quantile

        scores = np.concatenate([np.full((1, 5, 1), np.nan), np.full((1, 1, 1), 7.5)], axis=1)
        for level in (0.01, 0.5, 0.99):
            assert masked_quantile(scores, np.full((1, 1), level))[0, 0] == 7.5


class TestExpandingSortedScores:
    """The prefix state must equal ``masked_quantile`` on the same prefix slice."""

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize(("n_series", "horizon"), [(1, 1), (5, 12)])
    def test_matches_masked_quantile_at_every_prefix(
        self, dtype: type, n_series: int, horizon: int
    ) -> None:
        from conformalforecast._quantiles import ExpandingSortedScores, masked_quantile

        rng = np.random.default_rng(3)
        n_windows = 15
        scores = rng.normal(size=(n_series, n_windows, horizon)).astype(dtype)
        prefix = ExpandingSortedScores(n_series, horizon, np.dtype(dtype), capacity=n_windows)
        for t in range(n_windows):
            levels = rng.uniform(0.02, 0.98, size=(n_series, horizon))
            got = prefix.quantile(levels)
            assert got.dtype == np.dtype(dtype)
            assert prefix.n_pushed == t
            if t > 0:
                np.testing.assert_array_equal(got, masked_quantile(scores[:, :t, :], levels))
            prefix.push(scores[:, t, :])

    def test_ragged_prefix_matches(self) -> None:
        from conformalforecast._quantiles import ExpandingSortedScores, masked_quantile

        rng = np.random.default_rng(4)
        n_series, n_windows, horizon = 4, 12, 3
        scores = rng.normal(size=(n_series, n_windows, horizon))
        for i in range(n_series):
            scores[i, : 2 * i, :] = np.nan
        prefix = ExpandingSortedScores(n_series, horizon, scores.dtype, capacity=n_windows)
        for t in range(n_windows):
            levels = rng.uniform(0.02, 0.98, size=(n_series, horizon))
            if t > 0:
                np.testing.assert_array_equal(
                    prefix.quantile(levels), masked_quantile(scores[:, :t, :], levels)
                )
            prefix.push(scores[:, t, :])

    def test_batched_levels_match_per_expert_calls(self) -> None:
        from conformalforecast._quantiles import ExpandingSortedScores

        rng = np.random.default_rng(5)
        n_experts, n_series, horizon = 6, 3, 4
        scores = rng.normal(size=(n_series, 10, horizon))
        prefix = ExpandingSortedScores(n_series, horizon, scores.dtype, capacity=10)
        for t in range(10):
            prefix.push(scores[:, t, :])
        levels = rng.uniform(0.01, 0.99, size=(n_experts, n_series, horizon))
        batched = prefix.quantile(levels)
        assert batched.shape == (n_experts, n_series, horizon)
        for k in range(n_experts):
            np.testing.assert_array_equal(batched[k], prefix.quantile(levels[k]))

    def test_empty_prefix_returns_sentinel(self) -> None:
        from conformalforecast._quantiles import (
            _SATURATION_SENTINEL,
            ExpandingSortedScores,
        )

        prefix = ExpandingSortedScores(2, 3, np.dtype(np.float64))
        out = prefix.quantile(np.full((2, 3), 0.5))
        assert out.shape == (2, 3)
        assert np.all(out == _SATURATION_SENTINEL)
        prefix.push(np.full((2, 3), np.nan))
        assert np.all(prefix.quantile(np.full((2, 3), 0.5)) == _SATURATION_SENTINEL)

    def test_grows_without_capacity_hint(self) -> None:
        from conformalforecast._quantiles import ExpandingSortedScores, masked_quantile

        rng = np.random.default_rng(6)
        scores = rng.normal(size=(2, 9, 2))
        prefix = ExpandingSortedScores(2, 2, scores.dtype)  # no capacity
        for t in range(9):
            prefix.push(scores[:, t, :])
        levels = np.full((2, 2), 0.75)
        np.testing.assert_array_equal(prefix.quantile(levels), masked_quantile(scores, levels))


class TestWeightedConformalQuantile:
    def test_rho_one_matches_exact_order_statistic(self) -> None:
        from conformalforecast._quantiles import (
            conformal_quantile,
            weighted_conformal_quantile,
        )

        rng = np.random.default_rng(1)
        scores = np.abs(rng.normal(size=(3, 60, 2)))
        np.testing.assert_allclose(
            weighted_conformal_quantile(scores, alpha=0.1, rho=1.0),
            conformal_quantile(scores, alpha=0.1),
        )

    @pytest.mark.parametrize(("alpha", "m"), [(0.2, 9), (0.25, 7), (0.5, 5), (1 - 0.04, 24)])
    def test_target_landing_exactly_on_a_cumulative_weight(self, alpha: float, m: int) -> None:
        """The ``>=`` in the crossing test is ``searchsorted(side="left")``.

        With ``rho == 1`` the cumulative weights are exactly ``(j + 1) / m`` and
        the target is ``(1 - alpha)(m + 1) / m``, so whenever
        ``(1 - alpha)(m + 1)`` is an integer the target lands *exactly* on a
        cumulative weight — the one case where ``>=`` and ``>`` disagree. The
        ``>=`` branch is the one that matches the exact ``ceil((1 - alpha)(n + 1))``
        order statistic.
        """
        from conformalforecast._quantiles import (
            conformal_quantile,
            weighted_conformal_quantile,
        )

        scores = np.arange(1.0, m + 1.0).reshape(1, m, 1)
        np.testing.assert_array_equal(
            weighted_conformal_quantile(scores, alpha=alpha, rho=1.0),
            conformal_quantile(scores, alpha=alpha),
        )

    def test_recency_weighting_shifts_quantile(self) -> None:
        from conformalforecast._quantiles import weighted_conformal_quantile

        scores = np.concatenate([np.full((1, 50, 1), 10.0), np.full((1, 50, 1), 1.0)], axis=1)
        q_recent = weighted_conformal_quantile(scores, alpha=0.1, rho=0.9)
        q_uniform = weighted_conformal_quantile(scores, alpha=0.1, rho=1.0)
        assert q_recent[0, 0] < q_uniform[0, 0]

    def test_ragged_equals_dense(self) -> None:
        from conformalforecast._quantiles import weighted_conformal_quantile

        rng = np.random.default_rng(2)
        dense = np.abs(rng.normal(size=(1, 30, 1)))
        padded = np.concatenate([np.full((1, 7, 1), np.nan), dense], axis=1)
        np.testing.assert_allclose(
            weighted_conformal_quantile(padded, alpha=0.1, rho=0.95),
            weighted_conformal_quantile(dense, alpha=0.1, rho=0.95),
        )

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_per_cell_valid_counts_match_dense_sub_arrays(self, dtype: type) -> None:
        """Every (series, horizon) cell has its own effective count ``m``.

        The weights are gathered as ``pow_table[m - 1 - rank]``, so a panel where
        ``m`` differs per *cell* (not just per series) is what pins that index
        expression. Each cell is checked against the same function called on that
        cell's dense sub-array, exactly — the ragged-equals-dense contract.
        """
        from conformalforecast._quantiles import weighted_conformal_quantile

        rng = np.random.default_rng(31)
        n_series, n_windows, horizon = 4, 30, 3
        alpha, rho = 0.1, 0.98
        scores = np.abs(rng.normal(size=(n_series, n_windows, horizon))).astype(dtype)
        for s in range(n_series):
            for h in range(horizon):
                scores[s, : 3 * s + h, h] = np.nan  # leading prefix, distinct per cell

        out = weighted_conformal_quantile(scores, alpha, rho)
        assert out.dtype == np.dtype(dtype)
        for s in range(n_series):
            for h in range(horizon):
                cell = scores[s, :, h]
                dense = cell[~np.isnan(cell)].reshape(1, -1, 1)
                expected = weighted_conformal_quantile(dense, alpha, rho)
                assert out[s, h] == expected[0, 0], (s, h, dense.shape[1])

    @pytest.mark.parametrize("rho", [0.9, 0.99])
    def test_duplicate_scores_match_dense_sub_arrays(self, rho: float) -> None:
        """Heavy ties: the padded panel and the dense sub-array must agree.

        Both paths use an unstable argsort, and the padded one sorts ``+inf``
        keys alongside the real values, so tie order within a duplicate group can
        differ between them and perturb intermediate cumulative weights. The
        selected order statistic must not depend on that.
        """
        from conformalforecast._quantiles import weighted_conformal_quantile

        alpha = 0.1
        # Three distinct values, 12 repeats each, chronologically interleaved.
        block = np.tile(np.array([1.0, 2.0, 3.0]), 12)
        scores = np.stack([block, block[::-1]])[:, :, None]  # (2, 36, 1)
        scores = np.concatenate([scores, scores * 2.0], axis=2)  # (2, 36, 2)
        scores[0, :5, :] = np.nan  # ragged, so the padded path has +inf keys

        out = weighted_conformal_quantile(scores, alpha, rho)
        for s in range(scores.shape[0]):
            for h in range(scores.shape[2]):
                cell = scores[s, :, h]
                dense = cell[~np.isnan(cell)]
                expected = weighted_conformal_quantile(dense.reshape(1, -1, 1), alpha, rho)
                assert out[s, h] == expected[0, 0], (s, h)
                # A weighted order statistic is always an observed score.
                assert out[s, h] in set(dense.tolist()), (s, h, out[s, h])

    def test_ess_below_required_raises_with_indices(self) -> None:
        from conformalforecast._quantiles import weighted_conformal_quantile
        from conformalforecast.base import InsufficientCalibrationSamples

        scores = np.abs(np.random.default_rng(3).normal(size=(1, 20, 1)))
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            weighted_conformal_quantile(scores, alpha=0.05, rho=0.5)
        assert excinfo.value.series_indices == [0]

    def test_rejects_bad_rho(self) -> None:
        from conformalforecast._quantiles import weighted_conformal_quantile

        with pytest.raises(ValueError, match="rho"):
            weighted_conformal_quantile(np.ones((1, 20, 1)), alpha=0.1, rho=0.0)
        with pytest.raises(ValueError, match="rho"):
            weighted_conformal_quantile(np.ones((1, 20, 1)), alpha=0.1, rho=1.5)


def _reference_weighted(scores, alpha, rho):
    """Pre-refactor algorithm, kept here as the behavioural reference."""
    arr = np.asarray(scores, dtype=np.float64)
    n_series, n_windows, horizon = arr.shape
    out = np.empty((n_series, horizon), dtype=np.float64)
    for s in range(n_series):
        for h in range(horizon):
            cell = arr[s, :, h]
            valid = cell[~np.isnan(cell)]
            m = valid.size
            if m == 0:
                out[s, h] = 1e300
                continue
            w = rho ** (m - 1 - np.arange(m, dtype=np.float64))
            big_w = w.sum()
            target = min(max((1.0 - alpha) * (big_w + 1.0) / big_w, 0.0), 1.0)
            order = np.argsort(valid)
            cumw = np.cumsum(w[order]) / big_w
            hit = np.flatnonzero(cumw >= target)
            idx = hit[0] if hit.size else m - 1
            out[s, h] = valid[order][idx]
    return out


def test_weighted_from_weights_matches_rho_path():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(4, 30, 3))
    rho = 0.97
    m = scores.shape[1]
    w = rho ** (m - 1 - np.arange(m, dtype=np.float64))
    weights = np.broadcast_to(w[None, :, None], scores.shape).copy()
    np.testing.assert_array_equal(
        weighted_quantile_from_weights(scores, 0.1, weights),
        weighted_conformal_quantile(scores, 0.1, rho),
    )


def test_weighted_conformal_quantile_matches_reference():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(3, 40, 2))
    scores[0, :5, :] = np.nan  # ragged prefix
    # rho=0.8 is deliberately excluded here: its asymptotic ESS, (1+rho)/(1-rho)
    # = 9, is below ceil(1/alpha) = 10 for any window count, so
    # weighted_conformal_quantile(scores, 0.1, 0.8) always raises
    # InsufficientCalibrationSamples before the comparison below can run (this
    # is pre-existing behavior, unchanged by this refactor). rho=0.9 keeps a
    # comparable decay while its ESS (~18) safely clears the threshold.
    for rho in (1.0, 0.99, 0.9):
        np.testing.assert_allclose(
            weighted_conformal_quantile(scores, 0.1, rho),
            _reference_weighted(scores, 0.1, rho),
            rtol=0,
            atol=0,
        )


def test_uniform_weights_recover_unweighted_quantile():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(2, 25, 4))
    weights = np.ones_like(scores)
    np.testing.assert_allclose(
        weighted_quantile_from_weights(scores, 0.1, weights),
        weighted_conformal_quantile(scores, 0.1, 1.0),
    )


def test_weighted_from_weights_preserves_float32():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(2, 30, 2)).astype(np.float32)
    out = weighted_quantile_from_weights(scores, 0.1, np.ones_like(scores))
    assert out.dtype == np.float32


def test_weighted_from_weights_rejects_bad_input():
    scores = np.zeros((2, 30, 2))
    with pytest.raises(ValueError, match="same shape"):
        weighted_quantile_from_weights(scores, 0.1, np.ones((2, 29, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        w = np.ones_like(scores)
        w[0, 0, 0] = -1.0
        weighted_quantile_from_weights(scores, 0.1, w)


def test_weighted_from_weights_rejects_all_zero_weights():
    """Zero total weight makes the ESS NaN, and `nan < min_samples` is False,
    so without its own check this returned the largest score from no
    information at all."""
    scores = np.zeros((3, 40, 2))
    weights = np.ones_like(scores)
    weights[1] = 0.0
    with pytest.raises(InsufficientCalibrationSamples, match="summing to zero") as exc:
        weighted_quantile_from_weights(scores, 0.1, weights)
    assert exc.value.series_indices == [1]


def test_weighted_from_weights_raises_insufficient_with_indices():
    scores = np.full((3, 4, 2), 1.0)
    with pytest.raises(InsufficientCalibrationSamples) as exc:
        weighted_quantile_from_weights(scores, 0.1, np.ones_like(scores))
    assert exc.value.series_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# weighted_quantile_per_test_point
# ---------------------------------------------------------------------------


def _reference_per_test_point(
    scores: np.ndarray, weights: np.ndarray, test_weight: float, alpha: float
) -> float:
    """
    Per-cell Tibshirani et al. (2019) weighted quantile, written the slow way.

    The vectorized implementation folds the test weight into the target rather
    than renormalizing by ``sum(w) + w_test``; this reference does it the way
    the paper states it, so agreement checks the algebra as well as the code.

    Parameters
    ----------
    scores : np.ndarray, shape (n_windows,)
        Calibration scores for one (series, horizon) cell. May contain NaN.
    weights : np.ndarray, shape (n_windows,)
        Matching non-negative calibration weights.
    test_weight : float
        The test point's own weight.
    alpha : float
        Miscoverage level in (0, 1).

    Returns
    -------
    float
    """
    valid = ~np.isnan(scores)
    s = scores[valid]
    w = weights[valid]
    order = np.argsort(s)
    s = s[order]
    w = w[order]
    normalizer = w.sum() + test_weight
    cum = np.cumsum(w) / normalizer
    reached = cum >= (1.0 - alpha)
    idx = int(np.argmax(reached)) if reached.any() else len(s) - 1
    return float(s[idx])


def test_per_test_point_matches_per_cell_reference_on_a_ragged_panel():
    """The centrepiece math, checked against a loop-written Tibshirani reference."""
    rng = np.random.default_rng(0)
    n_series, n_windows, horizon, n_samples = 3, 40, 4, 5
    scores = rng.normal(size=(n_series, n_windows, horizon))
    scores[0, :6, :] = np.nan  # ragged: series 0 starts late
    weights = rng.uniform(0.1, 3.0, size=(n_series, n_windows, horizon))
    test_weights = rng.uniform(0.1, 3.0, size=(n_series, n_samples, horizon))
    alpha = 0.1

    got = weighted_quantile_per_test_point(scores, alpha, weights, test_weights)
    expected = np.array(
        [
            [
                [
                    _reference_per_test_point(
                        scores[a, :, c], weights[a, :, c], test_weights[a, b, c], alpha
                    )
                    for c in range(horizon)
                ]
                for b in range(n_samples)
            ]
            for a in range(n_series)
        ]
    )
    assert got.shape == (n_series, n_samples, horizon)
    np.testing.assert_array_equal(got, expected)


def test_per_test_point_with_unit_test_weight_matches_from_weights():
    """A test weight of 1 is exactly the (W + 1) / W normalization already shipped."""
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(2, 30, 3))
    weights = rng.uniform(0.5, 2.0, size=(2, 30, 3))
    got = weighted_quantile_per_test_point(scores, 0.2, weights, np.ones((2, 1, 3)))
    np.testing.assert_array_equal(
        got[:, 0, :], weighted_quantile_from_weights(scores, 0.2, weights)
    )


def test_per_test_point_preserves_float32():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(2, 30, 3)).astype(np.float32)
    weights = rng.uniform(0.5, 2.0, size=(2, 30, 3))
    out = weighted_quantile_per_test_point(scores, 0.1, weights, np.ones((2, 4, 3)))
    assert out.dtype == np.float32


def test_per_test_point_rejects_bad_alpha():
    scores = np.zeros((1, 10, 1))
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        weighted_quantile_per_test_point(scores, 1.0, np.ones((1, 10, 1)), np.ones((1, 2, 1)))


def test_per_test_point_rejects_mismatched_shapes():
    scores = np.zeros((1, 10, 1))
    with pytest.raises(ValueError, match="3-D and matching"):
        weighted_quantile_per_test_point(scores, 0.1, np.ones((1, 9, 1)), np.ones((1, 2, 1)))
    with pytest.raises(ValueError, match="test_weights must have shape"):
        weighted_quantile_per_test_point(scores, 0.1, np.ones((1, 10, 1)), np.ones((2, 2, 1)))
    with pytest.raises(ValueError, match="test_weights must have shape"):
        weighted_quantile_per_test_point(scores, 0.1, np.ones((1, 10, 1)), np.ones((1, 2)))


def test_weighted_quantiles_reject_non_finite_weights():
    """NaN passes ``w < 0`` and the zero-sum guard, and used to silently
    return the largest calibration score instead of the quantile."""
    scores = np.arange(30.0).reshape(1, 30, 1)
    for bad_value in (np.nan, np.inf):
        w = np.ones_like(scores)
        w[0, 0, 0] = bad_value
        with pytest.raises(ValueError, match="finite"):
            weighted_quantile_from_weights(scores, 0.1, w)
        tw = np.ones((1, 2, 1))
        tw[0, 0, 0] = bad_value
        with pytest.raises(ValueError, match="finite"):
            weighted_quantile_per_test_point(scores, 0.1, np.ones_like(scores), tw)
        with pytest.raises(ValueError, match="finite"):
            weighted_quantile_per_test_point(scores, 0.1, w, np.ones((1, 2, 1)))


def test_per_test_point_rejects_negative_weights():
    scores = np.zeros((1, 10, 1))
    bad = np.ones((1, 10, 1))
    bad[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        weighted_quantile_per_test_point(scores, 0.1, bad, np.ones((1, 2, 1)))
    bad_test = np.ones((1, 2, 1))
    bad_test[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        weighted_quantile_per_test_point(scores, 0.1, np.ones((1, 10, 1)), bad_test)


class TestFloatNoiseInAlpha:
    """An alpha derived from a coverage level carries float noise: ``1 - 0.8`` is
    ``0.19999999999999996`` and its reciprocal ``5.000000000000001``. The floor
    and the order statistic must still be the ones for alpha = 0.2."""

    ALPHA = 1 - 0.8

    @pytest.mark.parametrize(
        "quantile",
        [
            lambda s, a: conformal_quantile(s, a),
            lambda s, a: weighted_conformal_quantile(s, a, rho=1.0),
            lambda s, a: weighted_quantile_from_weights(s, a, np.ones_like(s)),
            lambda s, a: weighted_quantile_per_test_point(
                s, a, np.ones_like(s), np.ones((1, 1, s.shape[2]))
            )[:, 0, :],
        ],
        ids=["conformal", "weighted_rho", "from_weights", "per_test_point"],
    )
    def test_accepts_the_exact_floor(
        self, quantile: Callable[[np.ndarray, float], np.ndarray]
    ) -> None:
        scores = np.random.default_rng(0).normal(size=(1, 5, 2))
        np.testing.assert_array_equal(quantile(scores, self.ALPHA), scores.max(axis=1))

    @pytest.mark.parametrize("weight", [1.0, 0.1])
    @pytest.mark.parametrize(
        "weighted",
        [
            lambda s, a, c: weighted_quantile_from_weights(s, a, np.full_like(s, c)),
            lambda s, a, c: weighted_quantile_per_test_point(
                s, a, np.full_like(s, c), np.full((1, 1, s.shape[2]), c)
            )[:, 0, :],
        ],
        ids=["from_weights", "per_test_point"],
    )
    def test_flat_weights_reach_the_exact_floor(
        self, weighted: Callable[[np.ndarray, float, float], np.ndarray], weight: float
    ) -> None:
        # Five flat weights of 0.1 give an effective sample size (Σw)² / Σw² of
        # 4.999999999999999, not 5: the noise sits on the ESS side of the check.
        scores = np.random.default_rng(0).normal(size=(1, 5, 2))
        np.testing.assert_array_equal(weighted(scores, self.ALPHA, weight), scores.max(axis=1))

    def test_order_statistic_ignores_noise(self) -> None:
        # alpha = 1 - 0.01 is 0.010000000000000009 short of 0.99, so
        # (1 - alpha) * (n + 1) = 1.0000000000000009 for n = 99; k must be 1.
        scores = np.random.default_rng(4).normal(size=(1, 99, 1))
        result = conformal_quantile(scores, 1 - 0.01)
        assert result[0, 0] == scores.min()

    @pytest.mark.parametrize(
        "weighted",
        [
            lambda s, a: weighted_conformal_quantile(s, a, rho=1.0),
            lambda s, a: weighted_quantile_from_weights(s, a, np.ones_like(s)),
            lambda s, a: weighted_quantile_per_test_point(
                s, a, np.ones_like(s), np.ones((1, 1, s.shape[2]))
            )[:, 0, :],
        ],
        ids=["rho_one", "from_weights", "per_test_point"],
    )
    def test_uniform_weights_match_the_order_statistic(
        self, weighted: Callable[[np.ndarray, float], np.ndarray]
    ) -> None:
        # alpha = 1 - 0.04 leaves (1 - alpha) * (n + 1) = 1.0000000000000009 for
        # n = 24: the exact rank is 1, and the cumulative-weight crossing must
        # agree with it rather than land one rank higher.
        alpha = 1 - 0.04
        scores = np.random.default_rng(5).normal(size=(1, 24, 2))
        np.testing.assert_array_equal(weighted(scores, alpha), conformal_quantile(scores, alpha))

    @pytest.mark.parametrize(
        ("rho", "n", "alpha"), [(0.9, 250, 1 / 18), (0.95, 600, 0.03), (0.99, 2500, 0.008)]
    )
    def test_saturated_target_keeps_the_largest_score(
        self, rho: float, n: int, alpha: float
    ) -> None:
        # (1 - alpha)(W + 1)/W >= 1 clips the target to 1: the answer is the
        # largest valid score, even an old one whose weight is below the tolerance.
        scores = np.random.default_rng(5).normal(size=(1, n, 1))
        scores[0, 0, 0] = 100.0  # oldest window
        weights = (rho ** np.arange(n - 1, -1, -1.0))[None, :, None]
        assert weighted_conformal_quantile(scores, alpha, rho=rho)[0, 0] == 100.0
        assert weighted_quantile_from_weights(scores, alpha, weights)[0, 0] == 100.0
        per_point = weighted_quantile_per_test_point(scores, alpha, weights, np.ones((1, 1, 1)))
        assert per_point[0, 0, 0] == 100.0
