"""Tests for the private DTW and DBA numeric utilities."""

from functools import lru_cache

import numpy as np
import pytest

from synforecast._dtw import (
    dba_barycenter,
    dtw_alignment,
    dtw_distance,
    nearest_dtw_neighbors,
)


@lru_cache(None)
def _all_paths(n: int, m: int) -> tuple:
    """Enumerate paths, without using the production dynamic program.

    Enumeration breaks equal-cost ties from the endpoint backwards in
    diagonal/up/left order, matching the documented native convention.
    """
    if n == m == 1:
        return (((0, 0),),)
    return tuple(
        (*path, (n - 1, m - 1))
        for dn, dm in ((1, 1), (1, 0), (0, 1))
        if n > dn and m > dm
        for path in _all_paths(n - dn, m - dm)
    )


def _exhaustive_alignment(a, b, band):
    width = max(len(a), len(b)) if band is None else max(band, abs(len(a) - len(b)))
    paths = (
        p for p in _all_paths(len(a), len(b)) if all(abs(i - j) <= width for i, j in p)
    )
    return min(
        ((sum((a[i] - b[j]) ** 2 for i, j in p), p) for p in paths),
        key=lambda result: result[0],
    )


def _pairwise_oracle(series, window_fraction):
    matrix = np.zeros((len(series), len(series)))
    for i, a in enumerate(series):
        for j in range(i + 1, len(series)):
            b = series[j]
            band = max(
                int(np.ceil(window_fraction * max(len(a), len(b)))),
                abs(len(a) - len(b)) + 1,
            )
            matrix[i, j] = matrix[j, i] = dtw_distance(a, b, band)
    return matrix


class TestDtwAlignment:
    def test_matches_exhaustive_paths(self) -> None:
        rng = np.random.default_rng(405)
        for n in range(1, 6):
            for m in range(1, 6):
                for band in range(6):
                    for _ in range(10):
                        a = rng.integers(-3, 4, n).astype(float)
                        b = rng.integers(-3, 4, m).astype(float)
                        cost, expected_path = _exhaustive_alignment(a, b, band)
                        distance, path = dtw_alignment(a, b, band)
                        assert distance**2 == pytest.approx(cost)
                        np.testing.assert_array_equal(path, expected_path)
                        assert sum((a[i] - b[j]) ** 2 for i, j in path) == cost
                        assert dtw_distance(a, b, band) ** 2 == pytest.approx(cost)

    @pytest.mark.parametrize("n_neighbors", [1, 3, 200])
    def test_nearest_neighbors_match_full_matrix(self, n_neighbors: int) -> None:
        # Over 4096 pairs exercises chunk boundaries; duplicates exercise ties.
        rng = np.random.default_rng(123)
        series = [rng.normal(size=5 + i % 7) for i in range(94)]
        series.extend([series[0].copy(), series[0].copy()])
        matrix = _pairwise_oracle(series, 0.1)
        nearest = nearest_dtw_neighbors(series, 0.1, n_neighbors)
        for i, neighbors in enumerate(nearest):
            expected = sorted(
                ((j, matrix[i, j]) for j in range(len(series)) if i != j),
                key=lambda item: (item[1], item[0]),
            )[:n_neighbors]
            assert [j for j, _ in neighbors] == [j for j, _ in expected]
            np.testing.assert_allclose(
                [d for _, d in neighbors], [d for _, d in expected], rtol=1e-12
            )

    def test_nearest_neighbors_reject_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="n_neighbors"):
            nearest_dtw_neighbors([np.ones(3)], 0.1, 0)
        with pytest.raises(ValueError, match="window_fraction"):
            nearest_dtw_neighbors([np.ones(3)], float("nan"), 1)
        with pytest.raises(ValueError, match="finite"):
            nearest_dtw_neighbors([np.array([1.0, np.nan])], 0.1, 1)
        assert nearest_dtw_neighbors([], 0.1, 1) == []
        assert nearest_dtw_neighbors([np.ones(3)], 0.1, 1) == [[]]

    def test_identical_series_use_zero_cost_diagonal(self) -> None:
        values = np.array([0.0, 1.0, 2.0])

        distance, path = dtw_alignment(values, values, band=0)

        assert distance == 0.0
        np.testing.assert_array_equal(path, [[0, 0], [1, 1], [2, 2]])

    def test_known_warped_distance_and_path(self) -> None:
        distance, path = dtw_alignment(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0]), band=None
        )

        assert distance == pytest.approx(1.0)
        assert tuple(path[0]) == (0, 0)
        assert tuple(path[-1]) == (2, 1)

    def test_distance_kernel_matches_alignment_distance(self) -> None:
        rng = np.random.default_rng(3)
        first = rng.normal(size=40).cumsum()
        second = rng.normal(size=37).cumsum()
        for band in (None, 3, 10):
            assert dtw_distance(first, second, band) == pytest.approx(
                dtw_alignment(first, second, band)[0], rel=1e-12
            )

    def test_band_constrains_alignment(self) -> None:
        first = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        second = np.roll(first, 2)
        assert dtw_distance(first, second, None) == pytest.approx(0.0)
        assert dtw_distance(first, second, 2) == pytest.approx(0.0)
        assert dtw_distance(first, second, 1) > 1.0

    @pytest.mark.parametrize("window_fraction", [0.0, 1.5])
    def test_nearest_rejects_bad_window_fraction(self, window_fraction: float) -> None:
        with pytest.raises(ValueError, match="window_fraction"):
            nearest_dtw_neighbors([np.ones(3), np.ones(3)], window_fraction, 1)

    def test_distance_is_symmetric_for_unequal_lengths(self) -> None:
        first = np.array([0.0, 1.0, 1.5, 2.0])
        second = np.array([0.0, 0.5, 2.0])

        assert dtw_distance(first, second, 1) == pytest.approx(
            dtw_distance(second, first, 1)
        )

    @pytest.mark.parametrize(
        ("first", "second", "band", "message"),
        [
            (np.array([]), np.ones(2), None, "non-empty"),
            (np.ones((2, 2)), np.ones(2), None, "one-dimensional"),
            (np.array([0.0, np.nan]), np.ones(2), None, "finite"),
            (np.ones(2), np.ones(2), -1, "non-negative"),
        ],
    )
    def test_invalid_inputs_rejected(
        self,
        first: np.ndarray,
        second: np.ndarray,
        band: int | None,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            dtw_alignment(first, second, band)


class TestDbaBarycenter:
    @pytest.mark.parametrize("iterations", [1, 2, 4])
    @pytest.mark.parametrize("band", [1, None])
    def test_warped_updates_match_exhaustive_alignment(
        self, iterations: int, band: int | None
    ) -> None:
        series = [
            np.array(v, dtype=float)
            for v in ([3, 0, -2, 5], [-3, 1, -2], [4, -3, 0, 0, 2])
        ]
        weights = [0.2, 0.3, 0.5]
        expected = series[0].copy()
        first = None
        for _ in range(iterations):
            # Explicit association lists keep this independent of the native
            # sums/totals update and its banded path reconstruction.
            associations = [[] for _ in expected]
            for weight, values in zip(weights, series, strict=True):
                _, path = _exhaustive_alignment(expected, values, band)
                for i, j in path:
                    associations[i].append((values[j], weight))
            expected = np.array(
                [
                    sum(v * w for v, w in group) / sum(w for _, w in group)
                    for group in associations
                ]
            )
            if first is None:
                first = expected.copy()
        if iterations > 1:
            assert not np.allclose(expected, first)
        actual = dba_barycenter(
            series[0], series[1:], np.array(weights), iterations, band
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_weighted_diagonal_average(self) -> None:
        result = dba_barycenter(
            reference=np.array([0.0, 2.0]),
            neighbors=[np.array([2.0, 4.0])],
            weights=np.array([0.25, 0.75]),
            n_iterations=1,
            band=0,
        )

        np.testing.assert_allclose(result, [1.5, 3.5])

    @pytest.mark.parametrize(
        "weights", [np.array([1.0]), np.array([1.0, -0.5]), np.array([0.0, 0.0])]
    )
    def test_invalid_weights_rejected(self, weights: np.ndarray) -> None:
        with pytest.raises(ValueError, match="weights"):
            dba_barycenter(np.ones(3), [np.ones(3)], weights, n_iterations=1, band=None)

    def test_invalid_iteration_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_iterations"):
            dba_barycenter(
                np.ones(3),
                [np.ones(3)],
                np.ones(2),
                n_iterations=0,
                band=None,
            )


@pytest.mark.parametrize("iterations", [1, 4])
def test_batched_dba_matches_individually_verified_updates(iterations):
    from synforecast._lib import augmentation

    reference = np.array([3.0, 0.0, -2.0, 5.0])
    neighbors = [np.array([-3.0, 1.0, -2.0]), np.array([4.0, -3.0, 0.0, 0.0, 2.0])]
    weights = [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.0, 1.0, 0.0]]
    actual = augmentation.dba_barycenters(reference, neighbors, weights, iterations, 1)
    for got, row in zip(actual, weights, strict=True):
        expected = dba_barycenter(reference, neighbors, np.array(row), iterations, 1)
        np.testing.assert_array_equal(got, expected)


def test_alignment_budget_rejects_before_quadratic_work():
    values = np.ones(10000)
    with pytest.raises(ValueError, match="64 MiB"):
        dtw_alignment(values, values, None)
    # An enormous explicit band is equivalent to unbanded, not an overflow.
    np.testing.assert_array_equal(
        dtw_alignment(np.arange(5.0), np.arange(5.0), 2**63 - 1)[1],
        np.column_stack([np.arange(5), np.arange(5)]),
    )


@pytest.mark.parametrize("iterations", [0, 1001])
def test_native_dba_iteration_limit(iterations):
    from synforecast._lib import augmentation

    with pytest.raises(ValueError, match="n_iterations"):
        augmentation.dba_barycenters(np.ones(4), [np.ones(4)], [[1.0, 1.0]], iterations)


def test_dba_delivers_keyboard_interrupt():
    import os
    import subprocess
    import sys

    if sys.platform == "win32":
        pytest.skip("POSIX signal delivery")
    code = """
import os, signal, threading
import numpy as np
from synforecast._lib import augmentation
values = np.random.default_rng(1).normal(size=512)
timer = threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGINT))
timer.start()
try:
    augmentation.dba_barycenters(values, [values[::-1].copy()], [[0.6, 0.4]]*16, 1000, 50)
except KeyboardInterrupt:
    print("interrupted")
finally:
    timer.cancel()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "RAYON_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "interrupted" in result.stdout
