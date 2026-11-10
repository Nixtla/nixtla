"""Tests for the DataFrame-free panel conversion layer."""

from __future__ import annotations

import numpy as np
import pytest

from conformalforecast.panel import (
    PanelIndex,
    cv_frames_to_arrays,
    exog_to_long,
    level_to_quantiles,
    long_to_panel,
    panel_to_long,
    quantiles_to_level,
)


class TestLongToPanel:
    def test_rectangular(self) -> None:
        # two series, three timestamps each.
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        sids = ["a", "a", "a", "b", "b", "b"]
        times = [0, 1, 2, 0, 1, 2]
        panel, index = long_to_panel(values, sids, times)
        assert panel.shape == (2, 3)
        np.testing.assert_array_equal(panel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert index.series_ids == ("a", "b")
        np.testing.assert_array_equal(index.start_offsets, [0, 0])
        np.testing.assert_array_equal(index.lengths, [3, 3])
        assert index.timestamps == (0, 1, 2)

    def test_ragged_left_padded(self) -> None:
        # series "b" starts one step late -> a leading NaN.
        values = [1.0, 2.0, 3.0, 5.0, 6.0]
        sids = ["a", "a", "a", "b", "b"]
        times = [0, 1, 2, 1, 2]
        panel, index = long_to_panel(values, sids, times)
        assert panel.shape == (2, 3)
        assert np.isnan(panel[1, 0])
        np.testing.assert_array_equal(panel[1, 1:], [5.0, 6.0])
        np.testing.assert_array_equal(index.start_offsets, [0, 1])
        np.testing.assert_array_equal(index.lengths, [3, 2])

    def test_interior_gap_raises(self) -> None:
        # series "a" skips timestamp 1 -> interior NaN, illegal.
        values = [1.0, 3.0, 4.0, 5.0, 6.0]
        sids = ["a", "a", "b", "b", "b"]
        times = [0, 2, 0, 1, 2]
        with pytest.raises(ValueError, match="interior"):
            long_to_panel(values, sids, times)

    def test_trailing_gap_raises(self) -> None:
        # series "b" is missing its final timestamp (2) -> trailing NaN, illegal.
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        sids = ["a", "a", "a", "b", "b"]
        times = [0, 1, 2, 0, 1]
        with pytest.raises(ValueError, match="trailing"):
            long_to_panel(values, sids, times)

    def test_float32_dtype(self) -> None:
        panel, _ = long_to_panel([1.0, 2.0], ["a", "a"], [0, 1], dtype=np.float32)
        assert panel.dtype == np.float32

    def test_series_order_is_sorted(self) -> None:
        _, index = long_to_panel([1.0, 2.0], ["z", "a"], [0, 0])
        assert index.series_ids == ("a", "z")


class TestRoundTrip:
    def test_panel_to_long_inverts_long_to_panel(self) -> None:
        values = [1.0, 2.0, 3.0, 5.0, 6.0]
        sids = ["a", "a", "a", "b", "b"]
        times = [0, 1, 2, 1, 2]
        panel, index = long_to_panel(values, sids, times)
        r_sids, r_times, r_vals = panel_to_long(panel, index)
        # NaN cells are dropped; order is (series, time).
        assert r_sids == ["a", "a", "a", "b", "b"]
        assert r_times == [0, 1, 2, 1, 2]
        np.testing.assert_array_equal(r_vals, [1.0, 2.0, 3.0, 5.0, 6.0])

    def test_round_trip_no_timestamps(self) -> None:
        panel = np.array([[1.0, 2.0], [3.0, 4.0]])
        index = PanelIndex(
            series_ids=("a", "b"),
            start_offsets=np.array([0, 0]),
            lengths=np.array([2, 2]),
            timestamps=None,
        )
        r_sids, r_times, r_vals = panel_to_long(panel, index)
        assert r_times == [0, 1, 0, 1]  # positional fallback


class TestCvFramesToArrays:
    def test_aligned_preserves_nan(self) -> None:
        preds = np.array([[[1.0], [2.0], [3.0]], [[np.nan], [5.0], [6.0]]])  # (2,3,1)
        truths = preds + 0.5
        out = cv_frames_to_arrays([preds, truths], aligned=True)
        assert len(out) == 2
        np.testing.assert_array_equal(out[0], preds)
        assert np.isnan(out[0][1, 0, 0])

    def test_per_series_densifies_and_truncates(self) -> None:
        # series 0 has 3 valid windows, series 1 has 2 (leading NaN window).
        preds = np.array([[[1.0], [2.0], [3.0]], [[np.nan], [5.0], [6.0]]])  # (2,3,1)
        truths = preds + 0.5
        out = cv_frames_to_arrays([preds, truths], aligned=False)
        # w_min = 2; each series' last 2 valid windows, dense (no NaN).
        assert out[0].shape == (2, 2, 1)
        np.testing.assert_array_equal(out[0][:, :, 0], [[2.0, 3.0], [5.0, 6.0]])
        np.testing.assert_array_equal(out[1][:, :, 0], [[2.5, 3.5], [5.5, 6.5]])
        assert not np.isnan(out[0]).any()

    def test_equal_length_is_noop_for_both_modes(self) -> None:
        preds = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
        truths = preds + 1.0
        aligned = cv_frames_to_arrays([preds, truths], aligned=True)
        dense = cv_frames_to_arrays([preds, truths], aligned=False)
        np.testing.assert_array_equal(aligned[0], dense[0])
        np.testing.assert_array_equal(aligned[0], preds)

    def test_zero_valid_windows_raises(self) -> None:
        preds = np.full((2, 2, 1), np.nan)
        preds[0] = 1.0
        with pytest.raises(ValueError, match="zero valid"):
            cv_frames_to_arrays([preds], aligned=False)


class TestExogToLong:
    def test_flatten_order(self) -> None:
        # (n_series=2, n_samples=1, horizon=2, n_features=3)
        exog = np.arange(2 * 1 * 2 * 3, dtype=np.float64).reshape(2, 1, 2, 3)
        r_sids, r_samp, r_step, feats = exog_to_long(exog, ("a", "b"))
        assert r_sids == ["a", "a", "b", "b"]
        assert r_samp == [0, 0, 0, 0]
        assert r_step == [0, 1, 0, 1]
        assert feats.shape == (4, 3)
        np.testing.assert_array_equal(feats[0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(feats[3], [9.0, 10.0, 11.0])

    def test_rejects_non_4d(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            exog_to_long(np.zeros((2, 3, 4)), ("a", "b"))


class TestLevelQuantiles:
    def test_level_to_quantiles(self) -> None:
        assert level_to_quantiles(80) == (0.1, 0.9)
        assert level_to_quantiles(90) == (0.05, 0.95)

    def test_quantiles_to_level(self) -> None:
        assert quantiles_to_level(0.1, 0.9) == 80
        assert quantiles_to_level(0.05, 0.95) == 90

    def test_quantiles_to_level_rejects_asymmetric(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            quantiles_to_level(0.1, 0.8)

    def test_round_trip(self) -> None:
        for lvl in (50, 80, 90, 95):
            lo, hi = level_to_quantiles(lvl)
            assert quantiles_to_level(lo, hi) == lvl
