"""Unit tests for the Nixtla helpers that don't need a fitted model."""

from __future__ import annotations

import tracemalloc
from typing import Any

import numpy as np
import pytest

pytest.importorskip("utilsforecast")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
import narwhals as nw  # noqa: E402

from conformalforecast.adapters import _nixtla_common as _nx  # noqa: E402
from conformalforecast.base import InsufficientCalibrationSamples  # noqa: E402
from conformalforecast.methods import SplitConformal  # noqa: E402

FREQ = {"pandas": "D", "polars": "1d"}


@pytest.fixture(params=["pandas", "polars"])
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _to(backend: str, df: pd.DataFrame) -> nw.DataFrame[Any]:
    """The same frame in ``backend``, built column by column so NaN stays NaN."""
    if backend == "polars":
        df = pl.DataFrame(
            {
                c: df[c].to_numpy().astype("datetime64[ns]")
                if df[c].dtype.kind == "M"
                else df[c].to_numpy()
                for c in df.columns
            },
            nan_to_null=False,
        )
    return nw.from_native(df, eager_only=True)


def _split_conformal(alpha: float) -> SplitConformal:
    """SplitConformal: only its array surface is exercised here, so no adapter."""
    return SplitConformal(alpha=alpha)


def test_exog_to_futr_df_layout(backend: str) -> None:
    # (n_series=2, n_samples=1, horizon=2, n_features=2)
    exog = np.arange(2 * 1 * 2 * 2, dtype=np.float64).reshape(2, 1, 2, 2)
    like = _to(backend, _long_df({"a": [1.0, 2.0], "b": [3.0, 4.0]}, end="2020-01-10"))
    df = _nx.exog_to_futr_df(
        exog,
        series_ids=("a", "b"),
        shift=0,
        feature_names=["f0", "f1"],
        id_col="unique_id",
        time_col="ds",
        freq=FREQ[backend],
        like=like,
    )
    assert df.implementation == like.implementation
    assert df.columns == ["unique_id", "ds", "f0", "f1"]
    a_rows = df.filter(nw.col("unique_id") == "a").sort("ds")
    # future timestamps are one step past the series end.
    assert _stamps(a_rows["ds"]) == ["2020-01-11", "2020-01-12"]
    np.testing.assert_array_equal(a_rows.select("f0", "f1").to_numpy(), [[0.0, 1.0], [2.0, 3.0]])


def test_exog_to_futr_df_feature_count_mismatch(backend: str) -> None:
    exog = np.zeros((1, 1, 2, 3))
    like = _to(backend, _long_df({"a": [1.0, 2.0]}, end="2020-01-10"))
    with pytest.raises(ValueError, match="feature_names"):
        _nx.exog_to_futr_df(
            exog,
            series_ids=("a",),
            shift=0,
            feature_names=["only_one"],
            id_col="unique_id",
            time_col="ds",
            freq=FREQ[backend],
            like=like,
        )


# ----------------------------------------------------------------------
# Forecast anchoring
# ----------------------------------------------------------------------


def _long_df(values: dict[str, list[float]], end: str = "2020-04-29") -> pd.DataFrame:
    """Long frame where every series ends on ``end``, daily."""
    frames = []
    for sid, vals in values.items():
        stamps = pd.date_range(end=end, periods=len(vals), freq="D")
        frames.append(pd.DataFrame({"unique_id": sid, "ds": stamps, "y": vals}))
    return pd.concat(frames, ignore_index=True)


def _panel_and_ends(backend: str, df: pd.DataFrame) -> tuple[np.ndarray, nw.Series[Any]]:
    frame = _to(backend, df)
    ids, ends = _nx.build_series_bounds(frame, 0, id_col="unique_id", time_col="ds")
    panel = _nx.df_to_ragged_panel(
        frame, series_ids=ids, id_col="unique_id", time_col="ds", target_col="y"
    )
    return panel, ends


def _stamps(series: nw.Series[Any]) -> list[str]:
    return [str(t)[:10] for t in series.to_list()]


class TestDfToRaggedPanel:
    def test_right_aligns_and_left_pads_ragged_series(self, backend: str) -> None:
        df = _long_df({"a": [1.0, 2.0, 3.0, 4.0], "b": [7.0, 8.0]})
        panel, _ = _panel_and_ends(backend, df)
        assert panel.shape == (2, 4)
        np.testing.assert_array_equal(panel[0], [1.0, 2.0, 3.0, 4.0])
        # `b` is shorter, so its NaN is a contiguous *leading* prefix.
        assert np.isnan(panel[1, :2]).all()
        np.testing.assert_array_equal(panel[1, 2:], [7.0, 8.0])


class TestAnchorShiftForPanel:
    """The panel carries no dates, so `predict` locates it in the training data."""

    def _anchor(self, backend, panel, train_panel, ends):
        shift = _nx.anchor_shift_for_panel(panel, train_panel=train_panel)
        return _nx.offset_times(ends, FREQ[backend], shift)

    def test_full_panel_anchors_at_the_training_end(self, backend: str) -> None:
        df = _long_df({"a": list(np.arange(10.0))})
        train, ends = _panel_and_ends(backend, df)
        assert _stamps(self._anchor(backend, train, train, ends)) == _stamps(ends)

    def test_prefix_anchors_back_by_the_dropped_columns(self, backend: str) -> None:
        df = _long_df({"a": list(np.arange(10.0))})
        train, ends = _panel_and_ends(backend, df)
        got = self._anchor(backend, train[:, :6], train, ends)
        assert _stamps(got) == _stamps(_nx.offset_times(ends, FREQ[backend], -4))

    def test_recent_window_anchors_at_the_training_end(self, backend: str) -> None:
        """The regression: a fixed-width sliding window is not a prefix.

        Width-derived dating stamped these rows with the *first* 6 observations'
        dates, silently shifting every calendar feature by 4 days.
        """
        df = _long_df({"a": list(np.arange(10.0))})
        train, ends = _panel_and_ends(backend, df)
        assert _stamps(self._anchor(backend, train[:, -6:], train, ends)) == _stamps(ends)

    def test_extension_anchors_forward(self, backend: str) -> None:
        df = _long_df({"a": list(np.arange(10.0))})
        train, ends = _panel_and_ends(backend, df)
        extended = np.concatenate([train, np.array([[10.0, 11.0]])], axis=1)
        got = self._anchor(backend, extended, train, ends)
        assert _stamps(got) == _stamps(_nx.offset_times(ends, FREQ[backend], 2))

    def test_unlocatable_history_raises(self, backend: str) -> None:
        df = _long_df({"a": list(np.arange(10.0))})
        train, ends = _panel_and_ends(backend, df)
        interior = train[:, 2:8]  # neither a prefix nor a suffix
        with pytest.raises(ValueError, match="at either edge"):
            self._anchor(backend, interior, train, ends)

    def test_wider_but_not_an_extension_raises(self, backend: str) -> None:
        df = _long_df({"a": list(np.arange(10.0))})
        train, ends = _panel_and_ends(backend, df)
        bogus = np.concatenate([train + 100.0, np.array([[10.0, 11.0]])], axis=1)
        with pytest.raises(ValueError, match="at either edge"):
            self._anchor(backend, bogus, train, ends)

    def test_ragged_panel_matches_through_its_nan_padding(self, backend: str) -> None:
        df = _long_df({"a": list(np.arange(10.0)), "b": list(np.arange(6.0))})
        train, ends = _panel_and_ends(backend, df)
        assert np.isnan(train[1, :4]).all()
        # A prefix keeps the leading NaN, so matching has to treat NaN as equal.
        got = self._anchor(backend, train[:, :7], train, ends)
        assert _stamps(got) == _stamps(_nx.offset_times(ends, FREQ[backend], -3))

    def test_fully_padded_panel_is_located_not_mis_dated(self, backend: str) -> None:
        """Every series shorter than the panel width: the other reported failure.

        Width-only arithmetic compared the panel width against the *max
        non-NaN count*, producing a positive delta and anchoring the forecast
        after the real end of the data.
        """
        df = _long_df({"a": list(np.arange(10.0)), "b": list(np.arange(4.0))})
        train, ends = _panel_and_ends(backend, df)
        padded = np.full((2, 12), np.nan)
        padded[:, 2:] = train
        got = self._anchor(backend, padded, train, ends)
        assert _stamps(got) == _stamps(ends)


# ----------------------------------------------------------------------
# Ragged end dates: step-indexed pivot and CV reshaping
# ----------------------------------------------------------------------


def _cv_frame(
    spec: dict[str, list[str]],
    horizon: int,
    *,
    quantiles: bool = False,
) -> pd.DataFrame:
    """Nixtla-format CV frame from ``{series_id: [cutoff dates]}``.

    Values are ``series_index * 1000 + window_index * 10 + step``, so every cell
    identifies itself and a mis-placed value is visible in an assertion.
    """
    frames = []
    for s_idx, (sid, cutoffs) in enumerate(spec.items()):
        for w_idx, cutoff in enumerate(cutoffs):
            cut = pd.Timestamp(cutoff)
            base = float(s_idx * 1000 + w_idx * 10)
            values = [base + step for step in range(horizon)]
            frame = pd.DataFrame(
                {
                    "unique_id": sid,
                    "ds": pd.date_range(cut + pd.Timedelta(days=1), periods=horizon, freq="D"),
                    "cutoff": cut,
                    "y": values,
                    "Model": [v + 0.5 for v in values],
                }
            )
            if quantiles:
                frame["Model-lo-80"] = [v - 1.0 for v in values]
                frame["Model-hi-80"] = [v + 2.0 for v in values]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _calendar_pivot(df: pd.DataFrame) -> np.ndarray:
    """What pivoting onto the union of timestamps would produce.

    Not a shipped helper -- the package has no calendar-axis pivot any more.
    Kept here so the contrast `df_to_step_panel` exists to draw stays visible.
    """
    pivot = df.pivot(index="unique_id", columns="ds", values="y").reindex(["a", "b"])
    return np.asarray(pivot.sort_index(axis=1).to_numpy(dtype=np.float64))


class TestDfToStepPanel:
    """The horizon axis is a per-series step offset, not a shared calendar."""

    def _step_panel(
        self, backend: str, df: pd.DataFrame, n_steps: int, ids=("a", "b")
    ) -> np.ndarray:
        return _nx.df_to_step_panel(
            _to(backend, df),
            "y",
            series_ids=ids,
            id_col="unique_id",
            time_col="ds",
            n_steps=n_steps,
        )

    def test_matches_the_union_pivot_when_ends_coincide(self, backend: str) -> None:
        df = _cv_frame({"a": ["2020-01-10"], "b": ["2020-01-10"]}, horizon=3)
        step = self._step_panel(backend, df, 3)
        union = _calendar_pivot(df)
        np.testing.assert_array_equal(step, union)

    def test_different_origins_stay_on_the_step_axis(self, backend: str) -> None:
        """The union pivot would widen this to 6 columns and split the series."""
        df = _cv_frame({"a": ["2020-01-10"], "b": ["2020-01-04"]}, horizon=3)
        step = self._step_panel(backend, df, 3)
        assert step.shape == (2, 3)
        np.testing.assert_array_equal(step[0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(step[1], [1000.0, 1001.0, 1002.0])

        assert _calendar_pivot(df).shape == (2, 6)  # the defect this helper exists to avoid

    def test_absent_series_raises_rather_than_yielding_a_nan_row(self, backend: str) -> None:
        """Every caller is one forecast origin, where a missing series is a defect.

        NaN-padding it would put NaN straight into the returned intervals with
        nothing raised. "This window does not exist" is a cross-validation
        concept, and no cross-validation path reaches this function any more.
        """
        df = _cv_frame({"a": ["2020-01-10"]}, horizon=2)
        with pytest.raises(ValueError, match=r"'b' is missing"):
            self._step_panel(backend, df, 2)

    def test_partial_horizon_raises_naming_the_series_and_its_row_count(self, backend: str) -> None:
        df = _cv_frame({"a": ["2020-01-10"], "b": ["2020-01-10"]}, horizon=3)
        truncated = df.drop(df.index[-1])  # 'b' now has 2 of 3 steps
        with pytest.raises(ValueError, match=r"'b' has 2"):
            self._step_panel(backend, truncated, 3)


class TestReshapeCvQuantilesRagged:
    def _reshape(self, backend: str, df: pd.DataFrame, horizon: int = 2):
        return _nx.reshape_cv_quantiles(
            _to(backend, df),
            model_name="Model",
            target_col="y",
            q_arr=np.array([0.1, 0.9]),
            resolver=_nx.sf_resolve_quantile_column,
            series_ids=("a", "b"),
            id_col="unique_id",
            time_col="ds",
            horizon=horizon,
            library_name="StatsForecast",
        )

    def test_ragged_ends_densify_instead_of_raising(self, backend: str) -> None:
        df = _cv_frame(TestReshapeCvMany.SPEC, horizon=2, quantiles=True)
        qp, truths = self._reshape(backend, df)
        assert qp.shape == (2, 6, 2, 2)
        assert truths.shape == (2, 6, 2)
        assert not np.isnan(qp).any()
        # Quantile order follows q_arr: lo then hi, bracketing the truth values.
        assert np.all(qp[..., 0] < truths)
        assert np.all(truths < qp[..., 1])

    def test_nan_in_a_returned_cell_is_still_an_upstream_error(self, backend: str) -> None:
        df = _cv_frame(TestReshapeCvMany.SPEC, horizon=2, quantiles=True)
        df.loc[df.index[0], "Model-lo-80"] = np.nan
        with pytest.raises(RuntimeError, match="did return rows for"):
            self._reshape(backend, df)


# ----------------------------------------------------------------------
# Single-pass CV reshaping
# ----------------------------------------------------------------------


class TestReshapeCvMany:
    """One pass over the CV frame, sized to the output rather than the calendar."""

    # 'a' runs six cutoffs to 2020-12-30; 'b' stopped reporting eleven days
    # earlier, so the two cutoff sets are disjoint.
    SPEC = {
        "a": [f"2020-12-{d}" for d in range(25, 31)],
        "b": [f"2020-12-{d}" for d in range(14, 20)],
    }
    SAME_END = {
        "a": [f"2020-12-{d}" for d in range(25, 31)],
        "b": [f"2020-12-{d}" for d in range(25, 31)],
    }

    def _many(
        self,
        backend: str,
        df: pd.DataFrame,
        value_cols: list[str] | None = None,
        *,
        aligned: bool = False,
        horizon: int = 2,
        ids: tuple[str, ...] = ("a", "b"),
        **kwargs: object,
    ) -> list[np.ndarray]:
        return _nx.reshape_cv_many(
            _to(backend, df),
            value_cols if value_cols is not None else ["y", "Model"],
            series_ids=ids,
            id_col="unique_id",
            time_col="ds",
            horizon=horizon,
            aligned=aligned,
            **kwargs,
        )

    def test_dense_gives_each_series_its_own_windows(self, backend: str) -> None:
        truths, preds = self._many(backend, _cv_frame(self.SPEC, 2))
        assert truths.shape == (2, 6, 2) == preds.shape
        assert not np.isnan(truths).any()
        # Each series' own windows, chronological, on a shared step axis.
        np.testing.assert_array_equal(truths[0, 0], [0.0, 1.0])
        np.testing.assert_array_equal(truths[1, 0], [1000.0, 1001.0])
        np.testing.assert_array_equal(preds, truths + 0.5)

    def test_aligned_keeps_the_union_of_cutoffs_with_nan_holes(self, backend: str) -> None:
        (truths,) = self._many(backend, _cv_frame(self.SPEC, 2), ["y"], aligned=True)
        assert truths.shape == (2, 12, 2)
        # Cutoffs sort ascending, so 'b' occupies the first six, 'a' the last six.
        assert np.isnan(truths[0, :6]).all()
        assert not np.isnan(truths[0, 6:]).any()
        assert not np.isnan(truths[1, :6]).any()
        assert np.isnan(truths[1, 6:]).all()
        np.testing.assert_array_equal(truths[0, 6], [0.0, 1.0])
        np.testing.assert_array_equal(truths[1, 0], [1000.0, 1001.0])

    def test_dense_and_aligned_agree_when_every_series_shares_its_cutoffs(
        self, backend: str
    ) -> None:
        df = _cv_frame(self.SAME_END, 2)
        dense = self._many(backend, df, aligned=False)
        aligned = self._many(backend, df, aligned=True)
        for d, a in zip(dense, aligned, strict=True):
            np.testing.assert_array_equal(d, a)

    def test_dense_does_not_materialise_the_union_of_cutoffs(self, backend: str) -> None:
        """The point of the single pass: allocation tracks the output, not the calendar.

        200 series with disjoint cutoff sets have a 2400-column union, so a cube
        on the calendar axis costs ~15 MB per value column while the dense
        result is 77 KB. Anything that builds the union first blows the bound.
        """
        spec = {
            f"s{i}": [
                str((pd.Timestamp("2020-12-31") - pd.Timedelta(days=40 * i + w)).date())
                for w in range(12)
            ]
            for i in range(200)
        }
        df = _cv_frame(spec, 4)
        ids = tuple(spec)

        tracemalloc.start()
        try:
            out = self._many(backend, df, horizon=4, ids=ids)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        assert out[0].shape == (200, 12, 4)
        assert peak < 4_000_000, f"peak allocation {peak / 1e6:.1f} MB suggests a union cube"

    def test_uneven_window_counts_truncate_to_the_most_recent_shared_count(
        self, backend: str
    ) -> None:
        spec = {"a": self.SPEC["a"], "b": self.SPEC["b"][:4]}
        (truths,) = self._many(backend, _cv_frame(spec, 2), ["y"])
        assert truths.shape == (2, 4, 2)
        # 'a' keeps its four most recent windows (2, 3, 4, 5), not its first four.
        np.testing.assert_array_equal(truths[0, 0], [20.0, 21.0])
        np.testing.assert_array_equal(truths[1, 0], [1000.0, 1001.0])

    def test_partial_horizon_raises_naming_the_series_and_its_real_cutoff(
        self, backend: str
    ) -> None:
        df = _cv_frame(self.SPEC, 2)
        truncated = df.drop(df.index[-1])  # 'b' loses a step at its last cutoff
        with pytest.raises(ValueError, match=r"'b' at cutoff 2020-12-19.*has 1"):
            self._many(backend, truncated)

    def test_nan_in_a_returned_cell_raises_when_strict(self, backend: str) -> None:
        df = _cv_frame(self.SPEC, 2, quantiles=True)
        df.loc[df.index[0], "Model-lo-80"] = np.nan
        with pytest.raises(RuntimeError, match="did return rows for"):
            self._many(
                backend, df, ["y", "Model-lo-80"], strict_nan=True, library_name="StatsForecast"
            )

    def test_nan_in_a_returned_cell_drops_that_window_when_not_strict(self, backend: str) -> None:
        df = _cv_frame(self.SPEC, 2)
        df.loc[df.index[0], "Model"] = np.nan  # kills 'a' window 0 for both cubes
        truths, preds = self._many(backend, df)
        assert truths.shape == (2, 5, 2) == preds.shape
        assert not np.isnan(truths).any() and not np.isnan(preds).any()
        np.testing.assert_array_equal(truths[0, 0], [10.0, 11.0])  # window 0 gone

    def test_series_with_no_usable_window_is_named(self, backend: str) -> None:
        df = _cv_frame(self.SPEC, 2)
        df.loc[df["unique_id"] == "b", "Model"] = np.nan
        with pytest.raises(InsufficientCalibrationSamples, match=r"Series \['b'\]") as caught:
            self._many(backend, df)
        # Ids in the message, indices on the attribute: the method only has the latter.
        assert caught.value.series_indices == [1]

    def test_every_value_column_shares_one_window_selection(self, backend: str) -> None:
        """Truths and quantiles must be densified together, cell for cell."""
        df = _cv_frame(self.SPEC, 2, quantiles=True)
        truths, lo, hi = self._many(backend, df, ["y", "Model-lo-80", "Model-hi-80"])
        assert truths.shape == (2, 6, 2) == lo.shape == hi.shape
        np.testing.assert_array_equal(lo, truths - 1.0)
        np.testing.assert_array_equal(hi, truths + 2.0)

    def test_series_absent_from_the_frame_is_named_not_silently_padded(self, backend: str) -> None:
        """A series with no rows at all is the same failure as one with no usable window."""
        with pytest.raises(InsufficientCalibrationSamples, match=r"Series \['b'\]") as caught:
            self._many(backend, _cv_frame({"a": self.SPEC["a"]}, 2), ["y"])
        assert caught.value.series_indices == [1]

    def test_aligned_leaves_a_starved_series_to_the_method(self, backend: str) -> None:
        """Aligned mode represents "no window here" as NaN, so it must not raise.

        The dense check is structural -- there is no rectangular array to return.
        Aligned has one, and only the method's array surface knows the
        ``alpha`` that decides whether the non-NaN windows suffice.
        """
        df = _cv_frame(self.SPEC, 2)
        df.loc[df["unique_id"] == "b", "Model"] = np.nan

        truths, preds = self._many(backend, df, aligned=True)
        assert np.isnan(preds[1]).all()  # 'b' survives as an all-NaN row
        assert not np.isnan(preds[0, 6:]).any()

        with pytest.raises(InsufficientCalibrationSamples) as caught:
            _split_conformal(alpha=0.2).fit(preds, truths)
        assert caught.value.series_indices == [1]

    def test_null_in_a_polars_cell_is_a_hole(self) -> None:
        """polars stores a missing forecast as null, not NaN; it must still be a hole."""
        raw = _cv_frame(self.SPEC, 2)
        native = _to("polars", raw).to_native()
        b_last = native.filter(pl.col("unique_id") == "b")["cutoff"].max()
        holed = nw.from_native(
            native.with_columns(
                pl.when((pl.col("unique_id") == "b") & (pl.col("cutoff") == b_last))
                .then(None)
                .otherwise(pl.col("Model"))
                .alias("Model")
            )
        )
        kwargs: dict[str, Any] = {
            "series_ids": ("a", "b"),
            "id_col": "unique_id",
            "time_col": "ds",
            "horizon": 2,
        }
        # Dense: 'b' loses its newest window, so both series truncate to 5.
        truths, preds = _nx.reshape_cv_many(holed, ["y", "Model"], aligned=False, **kwargs)
        assert preds.shape == (2, 5, 2)
        assert not np.isnan(preds).any()
        with pytest.raises(RuntimeError, match="did return rows for"):
            _nx.reshape_cv_many(holed, ["y", "Model"], aligned=False, strict_nan=True, **kwargs)

    def test_handles_a_frame_whose_index_is_not_unique(self, backend: str) -> None:
        """A frame concatenated without ``ignore_index`` repeats index labels."""
        df = _cv_frame(self.SPEC, 2)
        df.index = np.zeros(len(df), dtype=int)  # every row shares a label
        truths, preds = self._many(backend, df)
        assert truths.shape == (2, 6, 2)
        np.testing.assert_array_equal(truths[0, 0], [0.0, 1.0])
        np.testing.assert_array_equal(truths[1, 0], [1000.0, 1001.0])
        np.testing.assert_array_equal(preds, truths + 0.5)


class TestBoundary:
    def test_train_frame_accepts_pandas_and_polars(self, backend: str) -> None:
        native = _to(backend, pd.DataFrame({"a": [1.0]})).to_native()
        assert _nx._pandas_or_polars(native) is native

    @pytest.mark.parametrize("bad", [3, "frame", [1, 2]])
    def test_train_frame_rejects_non_frames(self, bad: object) -> None:
        with pytest.raises(ValueError, match="pandas or polars DataFrame"):
            _nx._pandas_or_polars(bad)

    def test_train_frame_rejects_other_backends(self) -> None:
        pa = pytest.importorskip("pyarrow")
        with pytest.raises(ValueError, match="got Table"):
            _nx._pandas_or_polars(pa.table({"a": [1]}))

    def test_offset_and_ranges_follow_the_calendar(self, backend: str) -> None:
        month = {"pandas": "MS", "polars": "1mo"}[backend]
        ends = _to(backend, pd.DataFrame({"ds": pd.to_datetime(["2024-03-01"])}))["ds"]
        back = _nx.offset_times(ends, month, -2)
        grid = _nx.time_ranges(back, month, 3)
        assert [str(t)[:10] for t in grid.to_list()] == ["2024-01-01", "2024-02-01", "2024-03-01"]

    @pytest.mark.parametrize(
        ("starts", "freq"),
        [
            (pd.to_datetime(["2024-01-01", "2023-06-15"]), "D"),
            (pd.to_datetime(["2024-03-10 22:00", "2024-01-01 05:00"]).as_unit("s"), "h"),
            (pd.to_datetime(["2024-01-01", "2023-11-01"]), "MS"),
            (pd.to_datetime(["2024-01-31", "2023-11-30"]).as_unit("ms"), "ME"),
            (pd.to_datetime(["2024-03-30", "2024-10-26"]).tz_localize("Europe/Madrid"), "D"),
            (pd.to_datetime(["2024-01-01", "2024-02-01"]).tz_localize("Europe/Madrid"), "MS"),
            (pd.Index([5, 100]), 1),
        ],
    )
    def test_pandas_time_ranges_match_utilsforecast(self, starts: Any, freq: str | int) -> None:
        from utilsforecast import processing as ufp

        native = pd.Series(starts)
        got = _nx.time_ranges(nw.from_native(native, series_only=True), freq, 40).to_native()
        pd.testing.assert_series_equal(
            got, ufp.time_ranges(native, freq, 40), check_names=False, check_index=False
        )

    def test_wrappers_keep_a_pyarrow_backed_pandas_time_column(self) -> None:
        pytest.importorskip("pyarrow")
        raw = _long_df({"a": [1.0, 2.0, 3.0], "b": [4.0]}).convert_dtypes(dtype_backend="pyarrow")
        df = nw.from_native(raw, eager_only=True)
        _nx.validate_freq(df, time_col="ds", freq="D")
        _nx.validate_contiguity(df, id_col="unique_id", time_col="ds", freq="D")
        grid = _nx.time_ranges(_nx.offset_times(df["ds"].head(1), "D", -1), "D", 2)
        assert grid.to_native().dtype == raw["ds"].dtype
        assert _stamps(grid) == ["2020-04-26", "2020-04-27"]


class TestValidation:
    def test_validate_freq_accepts_the_frames_own_dialect(self, backend: str) -> None:
        df = _to(backend, _long_df({"a": [1.0, 2.0]}))
        _nx.validate_freq(df, time_col="ds", freq=FREQ[backend])

    def test_validate_freq_rejects_the_other_dialect(self, backend: str) -> None:
        other = {"pandas": "1mo", "polars": "D"}[backend]
        df = _to(backend, _long_df({"a": [1.0, 2.0]}))
        with pytest.raises(ValueError, match=r"Invalid frequency.*polars interval"):
            _nx.validate_freq(df, time_col="ds", freq=other)

    def test_validate_contiguity_names_the_gap(self, backend: str) -> None:
        raw = _long_df({"a": [1.0, 2.0, 3.0, 4.0]}).drop(index=2)  # 2020-04-28 removed
        with pytest.raises(
            ValueError, match=r"Non-contiguous timestamps in series 'a'.*position 2"
        ):
            _nx.validate_contiguity(
                _to(backend, raw), id_col="unique_id", time_col="ds", freq=FREQ[backend]
            )

    def test_validate_contiguity_passes_ragged_but_regular_series(self, backend: str) -> None:
        df = _to(backend, _long_df({"a": [1.0, 2.0, 3.0], "b": [4.0]}))
        _nx.validate_contiguity(df, id_col="unique_id", time_col="ds", freq=FREQ[backend])

    def test_validate_contiguity_steps_month_ends(self, backend: str) -> None:
        month = {"pandas": "ME", "polars": "1mo"}[backend]
        stamps = pd.date_range("2024-01-31", periods=5, freq="ME")  # through Feb 29
        raw = pd.DataFrame({"unique_id": "a", "ds": stamps, "y": 1.0})
        _nx.validate_contiguity(_to(backend, raw), id_col="unique_id", time_col="ds", freq=month)
        with pytest.raises(
            ValueError,
            match=r"series 'a': unexpected 2024-04-30.*position 2.*expected: 2024-03-31",
        ):
            _nx.validate_contiguity(
                _to(backend, raw.drop(index=2)), id_col="unique_id", time_col="ds", freq=month
            )

    @pytest.mark.parametrize(
        ("freq", "first", "rest"),
        [
            ("MS", "2020-01-15", pd.date_range("2020-02-01", periods=3, freq="MS")),
            ("W-SUN", "2020-01-01", pd.date_range("2020-01-05", periods=3, freq="W-SUN")),
        ],
    )
    def test_validate_contiguity_rejects_an_off_anchor_first_stamp(
        self, freq: str, first: str, rest: pd.DatetimeIndex
    ) -> None:
        # Only rows after the first are compared with a predecessor, so an
        # off-grid first stamp needs its own check.
        stamps = pd.DatetimeIndex([pd.Timestamp(first), *rest])
        raw = pd.DataFrame({"unique_id": "a", "ds": stamps, "y": 1.0})
        with pytest.raises(ValueError, match=rf"series 'a'.*{first}"):
            _nx.validate_contiguity(
                _to("pandas", raw), id_col="unique_id", time_col="ds", freq=freq
            )

    def test_validate_contiguity_accepts_a_polars_month_anchored_on_day_30(self) -> None:
        # Mar 30 -> Apr 30 is a month end, so stepping back snaps to Mar 31;
        # polars intervals are not anchored, so any first stamp is on the grid.
        df = nw.from_native(_day_30_monthly())
        _nx.validate_contiguity(df, id_col="unique_id", time_col="ds", freq="1mo")

    def _clipped_months(self, *, mar: int = 30) -> pl.DataFrame:
        """A day-30 series through Feb 29, plus a month-end series, both on '1mo'."""
        day_30 = pl.datetime_range(
            pl.datetime(2024, 1, 30), pl.datetime(2024, 6, 30), "1mo", eager=True
        ).to_list()
        day_30[2] = day_30[2].replace(day=mar)
        month_end = pl.datetime_range(
            pl.datetime(2024, 1, 31), pl.datetime(2024, 6, 30), "1mo", eager=True
        ).to_list()
        return pl.DataFrame(
            {"unique_id": ["a"] * 6 + ["b"] * 6, "ds": day_30 + month_end, "y": 1.0}
        )

    def test_validate_contiguity_follows_a_polars_start_anchored_month(self) -> None:
        # polars dates a '1mo' range from its start: Jan 30, Feb 29, Mar 30. Stepping
        # Feb 29 alone gives Mar 29, so a row-to-row check would reject it.
        df = nw.from_native(self._clipped_months())
        _nx.validate_contiguity(df, id_col="unique_id", time_col="ds", freq="1mo")

    def test_validate_contiguity_tells_month_ends_from_day_30_starts(self) -> None:
        # Both series start on a month end, so the start alone cannot say whether
        # the grid is month ends (b) or polars' own day-30 range (a).
        day_30 = pl.datetime_range(
            pl.datetime(2023, 9, 30), pl.datetime(2024, 4, 30), "1mo", eager=True
        )
        month_ends = pl.datetime_range(
            pl.datetime(2023, 4, 1), pl.datetime(2023, 11, 1), "1mo", eager=True
        ).dt.month_end()
        df = pl.DataFrame(
            {"unique_id": ["a"] * 8 + ["b"] * 8, "ds": pl.concat([day_30, month_ends]), "y": 1.0}
        )
        _nx.validate_contiguity(nw.from_native(df), id_col="unique_id", time_col="ds", freq="1mo")

    def test_validate_contiguity_rejects_an_off_grid_polars_month(self) -> None:
        df = nw.from_native(self._clipped_months(mar=29))
        with pytest.raises(
            ValueError,
            match=r"series 'a': unexpected 2024-03-29.*position 2.*expected: 2024-03-30",
        ):
            _nx.validate_contiguity(df, id_col="unique_id", time_col="ds", freq="1mo")

    def test_validate_no_nan_catches_null_and_nan(self) -> None:
        df = nw.from_native(
            pl.DataFrame({"unique_id": ["a", "b", "c"], "y": [1.0, None, float("nan")]})
        )
        with pytest.raises(ValueError, match=r"for series: \['b', 'c'\]"):
            _nx.validate_no_nan(df, id_col="unique_id", target_col="y")

    def test_build_series_bounds_returns_sorted_ids_and_ends(self, backend: str) -> None:
        df = _to(backend, _long_df({"b": [1.0] * 7, "a": [1.0] * 9}))
        ids, ends = _nx.build_series_bounds(df, 3, id_col="unique_id", time_col="ds")
        assert ids == ("a", "b")
        assert [str(t)[:10] for t in ends.to_list()] == ["2020-04-29", "2020-04-29"]

    def test_build_series_bounds_sorts_ids_by_value_not_category_order(self, backend: str) -> None:
        raw = pd.concat(
            [_long_df({"a": [1.0] * 7}), _long_df({"b": [1.0] * 9}, end="2020-04-20")],
            ignore_index=True,
        )
        if backend == "pandas":
            native = raw.astype({"unique_id": pd.CategoricalDtype(["b", "a"])})
        else:
            native = pl.from_pandas(raw).with_columns(pl.col("unique_id").cast(pl.Enum(["b", "a"])))
        df = nw.from_native(native, eager_only=True)
        ids, ends = _nx.build_series_bounds(df, 3, id_col="unique_id", time_col="ds")
        assert ids == ("a", "b")
        assert _stamps(ends) == ["2020-04-29", "2020-04-20"]

    def test_build_series_bounds_names_short_series(self, backend: str) -> None:
        df = _to(backend, _long_df({"a": [1.0] * 9, "b": [1.0] * 4}))
        with pytest.raises(ValueError, match=r"\['b'\] span fewer than"):
            _nx.build_series_bounds(df, 3, id_col="unique_id", time_col="ds")


class TestPanelToDfRagged:
    def test_round_trips_a_ragged_panel(self, backend: str) -> None:
        raw = _long_df({"a": [1.0, 2.0, 3.0, 4.0], "b": [7.0, 8.0]}, end="2020-01-10")
        panel, ends = _panel_and_ends(backend, raw)
        like = _to(backend, raw)
        out = _nx.panel_to_df_ragged(
            panel,
            series_ids=("a", "b"),
            shift=0,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            freq=FREQ[backend],
            like=like,
        )
        assert out.implementation == like.implementation
        assert out.columns == ["unique_id", "ds", "y"]
        np.testing.assert_array_equal(
            _nx.df_to_ragged_panel(
                out, series_ids=("a", "b"), id_col="unique_id", time_col="ds", target_col="y"
            ),
            panel,
        )
        b = out.filter(nw.col("unique_id") == "b")
        assert _stamps(b["ds"]) == ["2020-01-09", "2020-01-10"]

    def test_panel_to_df_ragged_keeps_the_training_id_dtype(self) -> None:
        train = pl.DataFrame(
            {"unique_id": ["a", "a", "a"], "ds": [1, 2, 3], "y": [1.0, 2.0, 3.0]},
            schema_overrides={"unique_id": pl.Categorical},
        )
        like = nw.from_native(train)
        out = _nx.panel_to_df_ragged(
            np.array([[1.0, 2.0, 3.0]]),
            series_ids=("a",),
            shift=0,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            freq=1,
            like=like,
        )
        assert out.schema["unique_id"] == like.schema["unique_id"]
        assert out["ds"].to_list() == [1, 2, 3]

    def test_pandas_categorical_ids_keep_their_categories(self) -> None:
        dtype = pd.CategoricalDtype(["z", "a", "q"])
        raw = _long_df({"a": [1.0, 2.0], "z": [3.0, 4.0]}).astype({"unique_id": dtype})
        panel, ends = _panel_and_ends("pandas", raw)
        out = _nx.panel_to_df_ragged(
            panel,
            series_ids=("a", "z"),
            shift=0,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            freq="D",
            like=_to("pandas", raw),
        )
        assert out["unique_id"].to_native().dtype == dtype

    def test_monthly_history_is_dated_on_the_calendar(self, backend: str) -> None:
        month = {"pandas": "MS", "polars": "1mo"}[backend]
        raw = pd.DataFrame(
            {
                "unique_id": ["a"] * 3,
                "ds": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "y": [1.0, 2.0, 3.0],
            }
        )
        like = _to(backend, raw)
        out = _nx.panel_to_df_ragged(
            np.array([[1.0, 2.0, 3.0]]),
            series_ids=("a",),
            shift=0,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            freq=month,
            like=like,
        )
        assert _stamps(out["ds"]) == ["2024-01-01", "2024-02-01", "2024-03-01"]


def test_attach_exog_history_joins_and_flags_unknown_dates(backend: str) -> None:
    raw = _long_df({"a": [1.0, 2.0, 3.0]}, end="2020-01-10").assign(x=[10.0, 20.0, 30.0])
    train = _to(backend, raw)
    hist = _to(backend, raw[["unique_id", "ds", "y"]].iloc[:2])
    out = _nx.attach_exog_history(
        hist, train_df=train, exog_features=["x"], id_col="unique_id", time_col="ds"
    )
    assert out["x"].to_list() == [10.0, 20.0]
    future = raw[["unique_id", "ds", "y"]].assign(ds=raw["ds"] + pd.Timedelta(days=5))
    with pytest.raises(ValueError, match="No values for exogenous features"):
        _nx.attach_exog_history(
            _to(backend, future),
            train_df=train,
            exog_features=["x"],
            id_col="unique_id",
            time_col="ds",
        )


def test_attach_exog_history_shows_sub_daily_dates_in_full() -> None:
    stamps = pd.date_range("2020-01-10", periods=3, freq="h")
    train = nw.from_native(pd.DataFrame({"unique_id": "a", "ds": stamps, "y": 1.0, "x": 2.0}))
    future = nw.from_native(
        pd.DataFrame({"unique_id": "a", "ds": stamps + pd.Timedelta(hours=5), "y": 1.0})
    )
    with pytest.raises(ValueError, match=r"dated 2020-01-10 05:00:00, 2020-01-10 06:00:00"):
        _nx.attach_exog_history(
            future, train_df=train, exog_features=["x"], id_col="unique_id", time_col="ds"
        )


def _day_30_monthly() -> pl.DataFrame:
    """One polars '1mo' series anchored on day 30; Apr 30 and Jun 30 are month ends."""
    stamps = pl.datetime_range(
        pl.datetime(2024, 3, 30), pl.datetime(2024, 6, 30), "1mo", eager=True
    )
    return pl.DataFrame({"unique_id": "a", "ds": stamps, "y": [1.0, 2.0, 3.0, 4.0], "x": 9.0})


class TestPolarsMonthAnchor:
    """Dates a polars '1mo' history from the training stamps, not a month-end snap."""

    def test_history_keeps_the_training_stamps(self) -> None:
        like = nw.from_native(_day_30_monthly())
        out = _nx.panel_to_df_ragged(
            np.array([[1.0, 2.0, 3.0]]),
            series_ids=("a",),
            shift=-1,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            freq="1mo",
            like=like,
        )
        # Stepping back from Jun 30 snapped to month ends: May 31, Apr 30, Mar 31.
        assert _stamps(out["ds"]) == ["2024-03-30", "2024-04-30", "2024-05-30"]
        joined = _nx.attach_exog_history(
            out, train_df=like, exog_features=["x"], id_col="unique_id", time_col="ds"
        )
        assert joined["x"].to_list() == [9.0, 9.0, 9.0]


class TestExogShape:
    def _futr(self, backend: str, exog: np.ndarray, ids: tuple[str, ...]) -> Any:
        like = _to(backend, _long_df({"a": [1.0, 2.0]}, end="2020-01-10"))
        return _nx.exog_to_futr_df(
            exog,
            series_ids=ids,
            shift=0,
            feature_names=["f0"],
            id_col="unique_id",
            time_col="ds",
            freq=FREQ[backend],
            like=like,
        )

    def test_rejects_a_non_4d_exog(self, backend: str) -> None:
        with pytest.raises(ValueError, match="exog must be 4-D"):
            self._futr(backend, np.zeros((1, 2, 1)), ("a",))

    def test_rejects_a_series_count_mismatch(self, backend: str) -> None:
        with pytest.raises(ValueError, match="series_ids length 1 != exog leading axis 2"):
            self._futr(backend, np.zeros((2, 1, 2, 1)), ("a",))


def test_attach_exog_history_passes_a_nan_regressor_through(backend: str) -> None:
    # A NaN regressor value is data, not a missing date: both backends keep it.
    raw = _long_df({"a": [1.0, 2.0, 3.0]}, end="2020-01-10").assign(x=[10.0, np.nan, 30.0])
    train = _to(backend, raw)
    out = _nx.attach_exog_history(
        _to(backend, raw[["unique_id", "ds", "y"]]),
        train_df=train,
        exog_features=["x"],
        id_col="unique_id",
        time_col="ds",
    )
    x = out["x"].to_numpy()
    assert x[0] == 10.0 and np.isnan(x[1]) and x[2] == 30.0
