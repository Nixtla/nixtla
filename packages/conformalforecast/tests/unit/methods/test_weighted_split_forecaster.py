"""Orchestration-layer plumbing tests for WeightedSplit and the transfer surface.

Statistical correctness (coverage under shift, scale alignment, NaN masking)
is covered by the array-surface tests in
``tests/unit/methods/test_weighted_split.py``. These tests check that the
orchestration surface wires an adapter to the calibration math: calibrate
dispatch, recalibrate on a target panel, adapt reporting its path, and one
end-to-end coverage smoke. Every entry point returns a new fitted object.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.base import CalibrationError
from conformalforecast.methods.split import SplitConformal
from conformalforecast.methods.transfer import SupportsTransfer
from conformalforecast.methods.weighted_split import WeightedSplit, WeightedSplitForecaster
from conformalforecast.nonconformity.absolute import AbsoluteResidual
from conformalforecast.weighting import PrecomputedWeights, UniformWeights


def _adapter(n_series: int = 2, horizon: int = 1) -> CallableAdapter:
    """A zero-forecast CallableAdapter.

    tests/helpers.py ships only quantile-capable adapters, so point-forecast
    tests build their own. predict_fn returns (n_series, horizon) — the adapter
    inserts the window axis.
    """
    return CallableAdapter(
        predict_fn=lambda history: np.zeros((history.shape[0], horizon)),
        horizon=horizon,
        n_series=n_series,
    )


def _panel(n_series, n_windows, horizon, seed):  # type: ignore[no-untyped-def]
    """Calibration windows. Histories must be NaN-free: CallableAdapter
    validates them and rejects NaN."""
    rng = np.random.default_rng(seed)
    histories = [rng.normal(size=(n_series, 50)) for _ in range(n_windows)]
    truths = rng.normal(size=(n_series, n_windows, horizon))
    return histories, truths


class TestConfig:
    def test_no_capabilities_required(self) -> None:
        assert WeightedSplit.REQUIRED_CAPABILITIES == ()
        assert WeightedSplit.IS_ONLINE is False

    def test_config_lives_on_the_method(self) -> None:
        m = WeightedSplit(alpha=0.1, pooling="global", scale_estimator="mad")
        assert m.pooling == "global"
        assert m.scale_estimator == "mad"
        assert "forecaster_" not in type(m).model_fields  # bound on the fitted object

    def test_transfer_support_is_an_isinstance_check_on_the_fitted_object(self) -> None:
        h, t = _panel(2, 40, 1, 0)
        assert isinstance(WeightedSplit(alpha=0.1).calibrate(_adapter(), h, t), SupportsTransfer)
        assert not isinstance(
            SplitConformal(alpha=0.1).calibrate(_adapter(), h, t), SupportsTransfer
        )

    def test_a_non_transfer_fitted_object_has_no_adapt(self) -> None:
        h, t = _panel(2, 40, 1, 0)
        assert not hasattr(SplitConformal(alpha=0.1).calibrate(_adapter(), h, t), "adapt")


class TestCalibrateDispatch:
    def test_calibrate_then_predict_smoke(self) -> None:
        h, t = _panel(2, 60, 3, 0)
        f = WeightedSplit(alpha=0.1).calibrate(_adapter(n_series=2, horizon=3), h, t)
        rng = np.random.default_rng(1)
        out = f.predict(rng.normal(size=(2, 50)))
        assert out.interval.shape == (2, 1, 3, 2)

    def test_calibrate_marks_the_loop_path(self) -> None:
        h, t = _panel(2, 40, 1, 0)
        f = WeightedSplit(alpha=0.1).calibrate(_adapter(), h, t)
        assert isinstance(f, WeightedSplitForecaster)
        assert f.calibration_.diagnostics["path"] == "loop"

    def test_coverage_smoke(self) -> None:
        h, t = _panel(1, 400, 1, 2)
        f = WeightedSplit(alpha=0.1).calibrate(_adapter(n_series=1, horizon=1), h, t)
        rng = np.random.default_rng(3)
        covered = []
        for _ in range(400):
            hist = rng.normal(size=(1, 50))
            truth = rng.normal(size=(1, 1, 1))
            iv = f.predict(hist).interval
            covered.append(bool((truth >= iv[..., 0]).all() and (truth <= iv[..., 1]).all()))
        assert abs(np.mean(covered) - 0.9) < 0.06


class TestRecalibrate:
    def test_marks_its_path_in_diagnostics(self) -> None:
        h, t = _panel(2, 40, 1, 0)
        f = WeightedSplit(alpha=0.1).calibrate(_adapter(), h, t)
        th, tt = _panel(2, 40, 1, 1)
        g = f.recalibrate(th, tt)
        assert isinstance(g, WeightedSplitForecaster)
        assert g.calibration_.diagnostics["path"] == "recalibrate"

    def test_returns_a_target_fit_and_leaves_the_source_fit_alone(self) -> None:
        h, t = _panel(2, 60, 1, 0)
        f = WeightedSplit(alpha=0.1).calibrate(_adapter(), h, t)
        before = f.score_quantile_.copy()
        rng = np.random.default_rng(2)
        th = [rng.normal(size=(2, 50)) for _ in range(60)]
        tt = rng.normal(scale=8.0, size=(2, 60, 1))  # much noisier target
        g = f.recalibrate(th, tt)
        assert g is not f
        assert np.all(g.score_quantile_ > before)
        np.testing.assert_array_equal(f.score_quantile_, before)

    def test_reuses_the_bound_forecaster(self) -> None:
        """No forecaster argument: recalibrate is calibrate on a new panel."""
        adapter = _adapter()
        h, t = _panel(2, 40, 1, 0)
        f = WeightedSplit(alpha=0.1).calibrate(adapter, h, t)
        th, tt = _panel(2, 40, 1, 1)
        assert f.recalibrate(th, tt).forecaster_ is adapter

    def test_without_a_bound_forecaster_raises(self) -> None:
        """An object from a bare fit has no forecaster to re-run."""
        rng = np.random.default_rng(5)
        f = WeightedSplit(alpha=0.1).fit(rng.normal(size=(2, 40, 1)), rng.normal(size=(2, 40, 1)))
        th, tt = _panel(2, 40, 1, 1)
        with pytest.raises(CalibrationError, match="recalibrate"):
            f.recalibrate(th, tt)

    def test_rejects_mismatched_series_count(self) -> None:
        h, t = _panel(2, 40, 1, 0)
        f = WeightedSplit(alpha=0.1).calibrate(_adapter(n_series=2), h, t)
        th, tt = _panel(5, 40, 1, 1)
        with pytest.raises(ValueError, match="n_series"):
            f.recalibrate(th, tt)

    def test_requires_both_arguments(self) -> None:
        h, t = _panel(2, 40, 1, 0)
        f = WeightedSplit(alpha=0.1).calibrate(_adapter(), h, t)
        with pytest.raises(ValueError, match="both"):
            f.recalibrate(None, None)

    def test_preserves_a_non_default_score(self) -> None:
        """The score is config on method_, so recalibrate must not silently
        fall back to the method default."""

        class _Louder(AbsoluteResidual):
            kind: Literal["louder"] = "louder"

        h, t = _panel(2, 40, 1, 0)
        f = WeightedSplit(alpha=0.1, score=_Louder()).calibrate(_adapter(), h, t)
        th, tt = _panel(2, 40, 1, 1)
        assert type(f.recalibrate(th, tt).score_fn_) is _Louder

    def test_discards_adaptation_state(self) -> None:
        """Documented behaviour: target labels supersede the unlabelled proxy,
        so the re-fit starts from uniform weights and no scale ratio."""
        h, t = _panel(2, 60, 1, 0)
        f = WeightedSplit(alpha=0.1, scale_estimator="std").calibrate(_adapter(), h, t)
        rng = np.random.default_rng(7)
        a = f.adapt(target_series=np.cumsum(rng.normal(scale=3.0, size=(2, 400)), axis=1))
        assert a.scale_ratio_ is not None
        assert a.target_scales_ is not None

        th, tt = _panel(2, 60, 1, 1)
        g = a.recalibrate(th, tt)
        assert isinstance(g.weighting_, UniformWeights)
        assert g.scale_ratio_ is None
        assert g.target_scales_ is None
        # The adapted receiver keeps its adaptation.
        assert a.scale_ratio_ is not None

    def test_is_available_on_every_method(self) -> None:
        """recalibrate is concrete on the fitted base, not a WeightedSplit extra."""
        h, t = _panel(2, 40, 1, 0)
        f = SplitConformal(alpha=0.1).calibrate(_adapter(), h, t)
        th, tt = _panel(2, 40, 1, 1)
        assert f.recalibrate(th, tt).calibration_.diagnostics["path"] == "recalibrate"


class TestAdapt:
    def test_reports_its_path(self) -> None:
        h, t = _panel(2, 60, 1, 0)
        f = WeightedSplit(alpha=0.1, scale_estimator="std").calibrate(_adapter(), h, t)
        rng = np.random.default_rng(3)
        a = f.adapt(target_series=np.cumsum(rng.normal(size=(2, 400)), axis=1))
        assert a.calibration_.diagnostics["path"] == "adapt"
        assert a.calibration_.diagnostics["weight_summary"]["median"] == 1.0

    def test_keeps_the_bound_forecaster(self) -> None:
        """An adapted object still predicts: it carries the receiver's adapter."""
        adapter = _adapter()
        h, t = _panel(2, 60, 1, 0)
        f = WeightedSplit(alpha=0.1, scale_estimator="std").calibrate(adapter, h, t)
        rng = np.random.default_rng(3)
        a = f.adapt(target_series=np.cumsum(rng.normal(size=(2, 400)), axis=1))
        assert a.forecaster_ is adapter
        assert a.predict(rng.normal(size=(2, 50))).interval.shape == (2, 1, 1, 2)

    def test_precomputed_weighting_binds_through_adapt(self) -> None:
        h, t = _panel(1, 60, 1, 4)
        f = WeightedSplit(alpha=0.1).calibrate(_adapter(n_series=1), h, t)
        a = f.adapt(weighting=PrecomputedWeights(weights=np.ones((1, 60, 1))))
        assert a.weighting_.is_adapted_ is True
