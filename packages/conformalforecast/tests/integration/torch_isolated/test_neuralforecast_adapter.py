"""Integration tests for NeuralForecastAdapter.

Note: NeuralForecast pulls in PyTorch and PyTorch Lightning, so the test
suite is significantly heavier than the StatsForecast/MLForecast equivalents.
Models are configured for cheap, deterministic CI runs (``max_steps`` small,
``random_seed=0``, ``accelerator='cpu'``, progress bar / logger / summary
disabled).
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

# torch and scikit-learn each vendor their own libomp. Loading both into one
# process segfaults on macOS arm64 (KMP_DUPLICATE_LIB_OK does not help), so the
# torch-backed integration tests run in their own pytest session.
pytestmark = pytest.mark.torch

nf_mod = pytest.importorskip("neuralforecast")
NeuralForecast = nf_mod.NeuralForecast

from neuralforecast.losses.pytorch import (  # noqa: E402
    MAE,
    DistributionLoss,
    MQLoss,
)
from neuralforecast.models import MLP, MLPMultivariate  # noqa: E402
from neuralforecast.utils import PredictionIntervals  # noqa: E402

from conformalforecast.adapters.neuralforecast import NeuralForecastAdapter  # noqa: E402
from conformalforecast.base import UnsupportedCapability  # noqa: E402
from conformalforecast.capabilities import (  # noqa: E402
    SupportsBootstrap,
    SupportsCrossValidation,
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
    SupportsRefit,
)
from conformalforecast.methods.split import SplitConformal  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Shared kwargs that pin determinism + silence Lightning's noisy progress output.
_TRAINER_QUIET: dict[str, Any] = {
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
    "accelerator": "cpu",
}


def _make_df(
    n_series: int = 1,
    t_len: int = 100,
    freq: str = "D",
    start: str = "2020-01-01",
    seed: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Create a synthetic long-format DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=t_len, freq=freq)
    rows: list[pd.DataFrame] = []
    for i in range(n_series):
        sid = f"series_{i}"
        rows.append(
            pd.DataFrame(
                {
                    id_col: sid,
                    time_col: dates,
                    target_col: rng.standard_normal(t_len).cumsum(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _make_model(kind: str, *, horizon: int, n_series: int, **overrides: Any) -> Any:
    """Build a univariate or multivariate MLP with deterministic, cheap settings."""
    common: dict[str, Any] = {
        "h": horizon,
        "input_size": 12,
        "max_steps": 10,
        "loss": MAE(),
        "random_seed": 0,
        **_TRAINER_QUIET,
    }
    common.update(overrides)
    if kind == "univariate":
        return MLP(**common)
    if kind == "multivariate":
        return MLPMultivariate(n_series=n_series, **common)
    raise ValueError(f"Unknown model kind: {kind}")


def _exog_df(t_len: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    """Long frame whose target is a strong positive function of ``price``."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=t_len, freq="D")
    price = rng.standard_normal(t_len).cumsum()
    df = pd.DataFrame(
        {
            "unique_id": "series_0",
            "ds": dates,
            "price": price,
            "y": 10 + 3 * price + rng.standard_normal(t_len) * 0.5,
        }
    )
    return df, price


def _exog_adapter(horizon: int = 3) -> tuple[NeuralForecastAdapter, pd.DataFrame]:
    df, _ = _exog_df()
    model = _make_model("univariate", horizon=horizon, n_series=1, futr_exog_list=["price"])
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)
    return NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP"), df


def _make_adapter(
    kind: str = "univariate",
    n_series: int = 1,
    t_len: int = 100,
    horizon: int = 3,
    freq: str = "D",
    **model_overrides: Any,
) -> tuple[NeuralForecastAdapter, pd.DataFrame]:
    """Fit a NeuralForecast and return an adapter plus its training DataFrame."""
    df = _make_df(n_series=n_series, t_len=t_len, freq=freq)
    model = _make_model(kind, horizon=horizon, n_series=n_series, **model_overrides)
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df)
    model_name = type(model).__name__ if model.alias is None else model.alias
    adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq=freq, model_name=model_name)
    return adapter, df


# Most parametrized tests run against both univariate and multivariate models.
# MLPMultivariate jointly models all series and requires ``n_series >= 2``;
# the parametrization picks ``n_series=2`` for it and ``n_series=1`` for MLP.
@pytest.fixture(
    params=[
        pytest.param(("univariate", 1), id="univariate-n1"),
        pytest.param(("univariate", 3), id="univariate-n3"),
        pytest.param(("multivariate", 3), id="multivariate-n3"),
    ]
)
def adapter_and_df(request: pytest.FixtureRequest) -> tuple[NeuralForecastAdapter, pd.DataFrame]:
    kind, n_series = request.param
    return _make_adapter(kind=kind, n_series=n_series, horizon=3, t_len=100)


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Successful construction and validation failures."""

    def test_univariate_construction(self) -> None:
        adapter, _ = _make_adapter(kind="univariate", n_series=1, horizon=3)
        assert adapter.n_series == 1
        assert adapter.horizon == 3
        assert adapter.model_name == "MLP"

    def test_multivariate_construction(self) -> None:
        adapter, _ = _make_adapter(kind="multivariate", n_series=3, horizon=3)
        assert adapter.n_series == 3
        assert adapter.horizon == 3
        assert adapter.model_name == "MLPMultivariate"

    def test_alias_override(self) -> None:
        df = _make_df(n_series=1, t_len=80)
        model = _make_model("univariate", horizon=3, n_series=1, alias="custom_mlp")
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df)
        adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="custom_mlp")
        assert adapter.model_name == "custom_mlp"

    # --- validation failure tests ---

    def test_construction_is_keyword_only_and_shape_is_derived(self) -> None:
        df = _make_df(n_series=2, t_len=80)
        model = _make_model("univariate", horizon=4, n_series=2)
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df)
        with pytest.raises(TypeError):
            NeuralForecastAdapter(nf, df, "D", "MLP")  # type: ignore[misc]

        adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")
        assert adapter.horizon == 4  # read off the model, never passed
        assert adapter.n_series == 2

    @pytest.mark.parametrize(
        "derived", [{"horizon": 4}, {"n_series": 99}, {"exog_features": ["price"]}]
    )
    def test_passing_a_derived_field_raises(self, derived: dict[str, Any]) -> None:
        df = _make_df(n_series=2, t_len=80)
        nf = NeuralForecast(models=[_make_model("univariate", horizon=4, n_series=2)], freq="D")
        nf.fit(df)
        with pytest.raises(ValidationError, match="must not be passed"):
            NeuralForecastAdapter(
                nf=nf,
                train_df=df,
                freq="D",
                model_name="MLP",
                **derived,  # type: ignore[arg-type]
            )

    def test_not_fitted_raises(self) -> None:
        df = _make_df()
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=1)],
            freq="D",
        )
        with pytest.raises(ValueError, match="not fitted"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")

    def test_bad_model_name_raises(self) -> None:
        df = _make_df()
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=1)],
            freq="D",
        )
        nf.fit(df)
        with pytest.raises(ValueError, match="model_name 'NoSuchModel'"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="NoSuchModel")

    def test_missing_columns_raises(self) -> None:
        df = _make_df()
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=1)],
            freq="D",
        )
        nf.fit(df)
        bad_df = df.rename(columns={"y": "target"})
        with pytest.raises(ValueError, match="missing required columns"):
            NeuralForecastAdapter(nf=nf, train_df=bad_df, freq="D", model_name="MLP")

    def test_invalid_freq_raises(self) -> None:
        df = _make_df()
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=1)],
            freq="D",
        )
        nf.fit(df)
        with pytest.raises(ValueError, match="Invalid frequency"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="BOGUS", model_name="MLP")

    def test_non_contiguous_raises(self) -> None:
        df = _make_df(t_len=100)
        df = df.drop(index=50).reset_index(drop=True)
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=1)],
            freq="D",
        )
        # NF rejects gaps internally; refit on a clean panel and pass the bad df
        # to the adapter constructor instead.
        clean = _make_df(t_len=100)
        nf.fit(clean)
        with pytest.raises(ValueError, match="Non-contiguous"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")

    def test_nan_target_raises(self) -> None:
        clean_df = _make_df()
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=1)],
            freq="D",
        )
        nf.fit(clean_df)
        bad_df = clean_df.copy()
        bad_df.loc[10, "y"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            NeuralForecastAdapter(nf=nf, train_df=bad_df, freq="D", model_name="MLP")

    def test_non_overlapping_series_both_survive(self) -> None:
        """Series with non-overlapping ranges no longer get intersected away.

        Each series individually has plenty of span for horizon=3, so both
        survive construction — unlike the removed common-range intersection
        that would have discarded data (or raised) here.
        """
        dates1 = pd.date_range("2020-01-01", periods=50, freq="D")
        dates2 = pd.date_range("2020-02-15", periods=50, freq="D")
        rng = np.random.default_rng(0)
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": "a",
                        "ds": dates1,
                        "y": rng.standard_normal(50).cumsum(),
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": "b",
                        "ds": dates2,
                        "y": rng.standard_normal(50).cumsum(),
                    }
                ),
            ],
            ignore_index=True,
        )
        # Fit on a clean panel to satisfy NF, then pass the bad panel to the adapter.
        clean = _make_df(n_series=2, t_len=100)
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=2)],
            freq="D",
        )
        nf.fit(clean)
        adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")
        assert set(adapter._series_ids) == {"a", "b"}

    def test_short_series_raises(self) -> None:
        """A series that individually spans fewer than 2*horizon steps raises,
        naming the offending series, even though the other series has plenty
        of data (no common-range truncation masks this)."""
        dates_a = pd.date_range("2020-01-01", periods=50, freq="D")
        dates_b = pd.date_range("2020-01-01", periods=5, freq="D")
        rng = np.random.default_rng(0)
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": "a",
                        "ds": dates_a,
                        "y": rng.standard_normal(50).cumsum(),
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": "b",
                        "ds": dates_b,
                        "y": rng.standard_normal(5).cumsum(),
                    }
                ),
            ],
            ignore_index=True,
        )
        # Fit on a clean panel to satisfy NF, then pass the bad panel to the adapter.
        clean = _make_df(n_series=2, t_len=100)
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=2)],
            freq="D",
        )
        nf.fit(clean)
        with pytest.raises(ValueError, match=r"\['b'\] span fewer than"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")

    def test_not_neuralforecast_instance_raises(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="NeuralForecast instance"):
            NeuralForecastAdapter(
                nf="not_nf",  # type: ignore[arg-type]
                train_df=df,
                freq="D",
                model_name="MLP",
            )

    def test_prediction_intervals_rejected(self) -> None:
        """A fit with prediction_intervals must be rejected at construction."""
        df = _make_df(n_series=1, t_len=100)
        nf = NeuralForecast(
            models=[_make_model("univariate", horizon=3, n_series=1)],
            freq="D",
        )
        nf.fit(df, prediction_intervals=PredictionIntervals(n_windows=2))
        # Confirm the test setup actually exercises the rejection path.
        assert nf._cs_df is not None
        with pytest.raises(ValueError, match="produces its own intervals"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")

    def test_horizon_mismatch_rejected(self) -> None:
        """Adapter rejects an NF with mismatched horizons across models.

        NeuralForecast itself asserts equal ``h`` at fit time, so the only way
        to construct an instance carrying mismatched horizons is to mutate a
        model's ``h`` attribute after fit. The adapter is the second line of
        defence; this test exercises that line directly.
        """
        df = _make_df(n_series=1, t_len=120)
        nf = NeuralForecast(
            models=[
                _make_model("univariate", horizon=10, n_series=1, alias="m10"),
                _make_model("univariate", horizon=10, n_series=1, alias="m12"),
            ],
            freq="D",
        )
        nf.fit(df)
        nf.models[1].h = 12  # induce a mismatch
        with pytest.raises(ValueError, match="must share the same horizon"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="m10")

    def test_future_exog_accepted_and_read_off_the_model(self) -> None:
        """The adapter takes no exog argument: the model already declares them."""
        adapter, _ = _exog_adapter()
        assert adapter.exog_features == ["price"]

    def test_historical_exog_still_rejected(self) -> None:
        """Past-only covariates have nowhere to travel: a panel is target-only."""
        df = _make_df(n_series=1, t_len=100)
        rng = np.random.default_rng(0)
        df = df.assign(feat_x=rng.standard_normal(len(df)))
        model = _make_model("univariate", horizon=3, n_series=1, hist_exog_list=["feat_x"])
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df)
        with pytest.raises(ValueError, match="historical exogenous"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")

    def test_train_df_must_carry_the_declared_regressors(self) -> None:
        df, _ = _exog_df()
        model = _make_model("univariate", horizon=3, n_series=1, futr_exog_list=["price"])
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df)
        with pytest.raises(ValueError, match=r"exog_features \['price'\] are not columns"):
            NeuralForecastAdapter(
                nf=nf, train_df=df[["unique_id", "ds", "y"]], freq="D", model_name="MLP"
            )


# ===========================================================================
# predict tests
# ===========================================================================


class TestPredict:
    """predict() shape, determinism, and anchor semantics."""

    def test_output_shape(self, adapter_and_df: tuple[NeuralForecastAdapter, pd.DataFrame]) -> None:
        adapter, df = adapter_and_df
        history = adapter._df_to_panel(nw.from_native(df), "y")
        result = adapter.predict(history)
        assert result.shape == (adapter.n_series, 1, adapter.horizon)

    @pytest.mark.parametrize("horizon", [1, 6])
    def test_horizon_extraction(self, horizon: int) -> None:
        """Adapter horizon equals the model's h, regardless of value."""
        adapter, _ = _make_adapter(kind="univariate", n_series=1, horizon=horizon, t_len=100)
        assert adapter.horizon == horizon

    def test_determinism(self, adapter_and_df: tuple[NeuralForecastAdapter, pd.DataFrame]) -> None:
        adapter, df = adapter_and_df
        history = adapter._df_to_panel(nw.from_native(df), "y")
        r1 = adapter.predict(history)
        r2 = adapter.predict(history)
        np.testing.assert_array_equal(r1, r2)


# ===========================================================================
# refit tests
# ===========================================================================


class TestRefit:
    """refit() updates the model and anchoring."""

    def test_refit_with_longer_panel_shifts_end(
        self, adapter_and_df: tuple[NeuralForecastAdapter, pd.DataFrame]
    ) -> None:
        adapter, _ = adapter_and_df
        old_ends = dict(zip(adapter._series_ids, adapter._series_ends.to_list(), strict=True))

        new_df = _make_df(n_series=adapter.n_series, t_len=120, seed=1)
        new_panel = adapter._df_to_panel(nw.from_native(new_df), "y")

        adapter.refit(new_panel)

        for sid, old_end in old_ends.items():
            new_end = adapter._series_ends[adapter._series_ids.index(sid)]
            assert new_end == old_end + pd.Timedelta(days=20)
        result = adapter.predict(new_panel)
        assert result.shape == (adapter.n_series, 1, adapter.horizon)

    def test_refit_with_different_series_raises(self) -> None:
        adapter, _ = _make_adapter(kind="univariate", n_series=2, horizon=3)
        bad_panel = np.random.default_rng(0).standard_normal((3, 100))
        with pytest.raises(ValueError, match="leading axis"):
            adapter.refit(bad_panel)


# ===========================================================================
# cross_validate tests
# ===========================================================================


class TestCrossValidate:
    """cross_validate() shapes and validation."""

    def test_output_shape(self, adapter_and_df: tuple[NeuralForecastAdapter, pd.DataFrame]) -> None:
        adapter, _ = adapter_and_df
        n_windows = 3
        step_size = 1
        preds, truths = adapter.cross_validate(n_windows=n_windows, step_size=step_size)
        assert preds.shape == (adapter.n_series, n_windows, adapter.horizon)
        assert truths.shape == (adapter.n_series, n_windows, adapter.horizon)

    def test_invalid_n_windows_raises(self) -> None:
        adapter, _ = _make_adapter()
        with pytest.raises(ValueError, match="n_windows"):
            adapter.cross_validate(n_windows=0, step_size=1)

    def test_invalid_step_size_raises(self) -> None:
        adapter, _ = _make_adapter()
        with pytest.raises(ValueError, match="step_size"):
            adapter.cross_validate(n_windows=2, step_size=0)


# ===========================================================================
# Capability declarations
# ===========================================================================


class TestCapabilities:
    """isinstance checks for capability mixins."""

    def test_supports_refit(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsRefit)

    def test_supports_cross_validation(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsCrossValidation)

    def test_supports_quantiles_class_level(self) -> None:
        """SupportsQuantiles is always declared, regardless of the loss."""
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsQuantiles)

    def test_supports_cross_validation_quantiles_class_level(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsCrossValidationQuantiles)

    def test_runtime_quantile_flag_false_for_point_loss(self) -> None:
        adapter, _ = _make_adapter()
        # Default _make_adapter uses MAE — a point loss.
        assert adapter._supports_quantiles_runtime is False

    def test_not_supports_bootstrap(self) -> None:
        adapter, _ = _make_adapter()
        assert not isinstance(adapter, SupportsBootstrap)


# ===========================================================================
# SplitConformal calibration via cross_validate
# ===========================================================================


class TestSplitConformalCV:
    """SplitConformal.calibrate_cv(n_windows=...) dispatches to forecaster.cross_validate."""

    def test_cv_calibration_then_predict(self) -> None:
        """End-to-end: calibrate via CV, then predict produces a valid interval."""
        adapter, df = _make_adapter(kind="univariate", n_series=1, horizon=3, t_len=100)
        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=10, step_size=1)

        history = adapter._df_to_panel(nw.from_native(df), "y")
        result = fitted.predict(history)

        assert result.point.shape == (1, 1, 3)
        assert result.interval.shape == (1, 1, 3, 2)
        assert np.all(result.interval[..., 0] <= result.point)
        assert np.all(result.point <= result.interval[..., 1])


# ===========================================================================
# Quantile-loss detection and runtime gating
# ===========================================================================


# Probabilistic-loss factories. Run only against the univariate MLP since
# quantile capability is loss-driven, not architecture-driven, and
# MLPMultivariate is already covered by the existing adapter tests.
def _mqloss(quantiles: list[float]) -> MQLoss:
    return MQLoss(quantiles=quantiles)


def _normal_distribution_loss() -> DistributionLoss:
    return DistributionLoss(distribution="Normal", level=[80, 90])


class TestQuantileLossDetection:
    """Constructor-time detection of quantile/distribution losses."""

    def test_point_loss_runtime_flag_false(self) -> None:
        adapter, _ = _make_adapter(kind="univariate")  # default MAE
        assert adapter._supports_quantiles_runtime is False
        assert adapter._loss_class_name == "MAE"

    def test_quantile_loss_runtime_flag_true(self) -> None:
        adapter, _ = _make_adapter(kind="univariate", loss=_mqloss([0.1, 0.5, 0.9]))
        assert adapter._supports_quantiles_runtime is True
        assert adapter._loss_class_name == "MQLoss"

    def test_distribution_loss_runtime_flag_true(self) -> None:
        adapter, _ = _make_adapter(kind="univariate", loss=_normal_distribution_loss())
        assert adapter._supports_quantiles_runtime is True
        assert adapter._loss_class_name == "DistributionLoss"


class TestPredictQuantilesGated:
    """predict_quantiles raises for point losses, returns shapes for probabilistic ones."""

    def test_point_loss_raises_unsupported(self) -> None:
        adapter, df = _make_adapter(kind="univariate", n_series=1, t_len=100)
        history = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(UnsupportedCapability, match="non-probabilistic loss"):
            adapter.predict_quantiles(history, np.array([0.1, 0.9]))

    def test_quantile_loss_returns_shape(self) -> None:
        adapter, df = _make_adapter(
            kind="univariate",
            n_series=1,
            horizon=3,
            t_len=100,
            loss=_mqloss([0.1, 0.5, 0.9]),
        )
        history = adapter._df_to_panel(nw.from_native(df), "y")
        result = adapter.predict_quantiles(history, np.array([0.1, 0.9]))
        assert result.shape == (1, 2, 3)

    def test_distribution_loss_returns_shape(self) -> None:
        adapter, df = _make_adapter(
            kind="univariate",
            n_series=1,
            horizon=3,
            t_len=100,
            loss=_normal_distribution_loss(),
        )
        history = adapter._df_to_panel(nw.from_native(df), "y")
        result = adapter.predict_quantiles(history, np.array([0.1, 0.9]))
        assert result.shape == (1, 2, 3)

    def test_quantile_not_in_loss_raises(self) -> None:
        """MQLoss outputs only the quantiles it was trained with — requesting
        a different symmetric pair must raise.

        NeuralForecast's ``level=`` argument is honoured by ``DistributionLoss``
        but ignored by ``MQLoss`` (which always emits the trained quantiles).
        We train with a level-90 pair (0.05/0.95) and ask for a level-80 pair
        (0.1/0.9); the output columns are level-90 only, so the level-80
        lookup fails.
        """
        adapter, df = _make_adapter(
            kind="univariate",
            n_series=1,
            horizon=3,
            t_len=100,
            loss=_mqloss([0.05, 0.5, 0.95]),
        )
        history = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(ValueError, match="did not return quantile column"):
            adapter.predict_quantiles(history, np.array([0.1, 0.9]))


class TestCrossValidateQuantilesGated:
    """cross_validate_quantiles raises for point losses, returns shapes for probabilistic ones."""

    def test_point_loss_raises_unsupported(self) -> None:
        adapter, _ = _make_adapter(kind="univariate", n_series=1, t_len=100)
        with pytest.raises(UnsupportedCapability, match="non-probabilistic loss"):
            adapter.cross_validate_quantiles(
                n_windows=2, step_size=1, quantiles=np.array([0.1, 0.9])
            )

    def test_quantile_loss_returns_shape(self) -> None:
        adapter, _ = _make_adapter(
            kind="univariate",
            n_series=1,
            horizon=3,
            t_len=100,
            loss=_mqloss([0.1, 0.5, 0.9]),
        )
        preds, truths = adapter.cross_validate_quantiles(
            n_windows=3, step_size=1, quantiles=np.array([0.1, 0.9])
        )
        assert preds.shape == (1, 3, 3, 2)
        assert truths.shape == (1, 3, 3)

    def test_distribution_loss_returns_shape(self) -> None:
        adapter, _ = _make_adapter(
            kind="univariate",
            n_series=1,
            horizon=3,
            t_len=100,
            loss=_normal_distribution_loss(),
        )
        preds, truths = adapter.cross_validate_quantiles(
            n_windows=3, step_size=1, quantiles=np.array([0.1, 0.9])
        )
        assert preds.shape == (1, 3, 3, 2)
        assert truths.shape == (1, 3, 3)


# ===========================================================================
# supports() runtime quantile gate at construction
# ===========================================================================


def test_supports_reflects_loss() -> None:
    """supports() defers to isinstance for class-level capabilities but is
    runtime-gated by loss for the two quantile capabilities."""
    adapter, _ = _make_adapter(kind="univariate")  # default MAE, a point loss
    assert adapter.supports(SupportsCrossValidation) is True  # class-level, always
    assert adapter.supports(SupportsQuantiles) is False  # runtime-gated by loss


def test_cqr_rejects_point_loss_when_the_forecaster_is_bound() -> None:
    """ConformalizedQuantileRegression must reject a point-loss adapter as soon as
    it is bound at calibrate time, not defer the failure to predict()."""
    from conformalforecast.methods import ConformalizedQuantileRegression as CQR

    adapter, _ = _make_adapter(kind="univariate")  # default MAE, a point loss
    with pytest.raises(UnsupportedCapability):
        CQR(alpha=0.1).calibrate_cv(adapter, n_windows=2)


# ===========================================================================
# Future exogenous features
# ===========================================================================


class TestFutureExogenous:
    """NeuralForecast rejects a target-only history for a `futr_exog_list` model.

    Unlike the other two adapters this one takes no `exog_features` argument:
    the model records its own list, and `predict` on such a model *requires*
    those values, so the feature is not opt-in.
    """

    def test_predict_with_future_exogenous(self) -> None:
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        out = adapter.predict(panel, X=np.ones((adapter.n_series, 1, 3, 1)))
        assert out.shape == (adapter.n_series, 1, 3)
        assert np.isfinite(out).all()

    def test_predict_actually_uses_the_future_regressors(self) -> None:
        """Shape alone would pass if the columns were silently ignored."""
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        low = adapter.predict(panel, X=np.full((adapter.n_series, 1, 3, 1), -5.0))
        high = adapter.predict(panel, X=np.full((adapter.n_series, 1, 3, 1), 5.0))
        assert not np.allclose(low, high)

    def test_predict_on_a_truncated_history(self) -> None:
        """The calibrate loop's path: each window is dated earlier than the last."""
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        X = np.ones((adapter.n_series, 1, 3, 1))
        full = adapter.predict(panel, X=X)
        short = adapter.predict(panel[:, :-10], X=X)
        assert short.shape == full.shape
        assert np.isfinite(short).all()
        assert not np.allclose(full, short)

    def test_calibrate_and_predict_round_trip(self) -> None:
        """The workflow the construction-time refusal blocked, end to end."""
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        n_windows, horizon = 6, 3
        histories = [panel[:, : panel.shape[1] - horizon - k] for k in range(n_windows)][::-1]
        truths = np.stack([panel[:, h.shape[1] : h.shape[1] + horizon] for h in histories], axis=1)

        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate(
            adapter, histories, truths, X=np.ones((adapter.n_series, n_windows, horizon, 1))
        )
        result = fitted.calibration_
        assert result.n_calibration_samples.shape == (adapter.n_series,)

        out = fitted.predict(panel, X=np.ones((adapter.n_series, 1, horizon, 1)))
        assert out.point.shape == (adapter.n_series, 1, horizon)
        assert np.all(out.interval[..., 0] <= out.point)
        assert np.all(out.point <= out.interval[..., 1])

    def test_cross_validate_needs_no_x(self) -> None:
        """CV runs against the stored frame; NeuralForecast splits the exog itself."""
        adapter, _ = _exog_adapter()
        preds, truths = adapter.cross_validate(n_windows=4, step_size=1)
        assert preds.shape == (adapter.n_series, 4, 3)
        assert not np.isnan(preds).any()
        assert not np.isnan(truths).any()

    def test_predict_without_x_raises_naming_the_features(self) -> None:
        """NeuralForecast raises on its own, but names neither adapter nor fix."""
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(ValueError, match=r"requires future exogenous features \['price'\]"):
            adapter.predict(panel)

    def test_history_beyond_the_training_frame_raises(self) -> None:
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        extended = np.concatenate([panel, np.zeros((adapter.n_series, 5))], axis=1)
        with pytest.raises(ValueError, match="No values for exogenous features"):
            adapter.predict(extended, X=np.ones((adapter.n_series, 1, 3, 1)))

    def test_x_on_a_non_exog_model_raises(self) -> None:
        adapter, df = _make_adapter(horizon=3)
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(ValueError, match="declares no future exogenous"):
            adapter.predict(panel, X=np.ones((adapter.n_series, 1, 3, 1)))

    def test_refit_is_refused_and_gated_in_supports(self) -> None:
        adapter, df = _exog_adapter()
        assert adapter.supports(SupportsRefit) is False
        assert adapter.supports(SupportsCrossValidation) is True
        with pytest.raises(ValueError, match="refit is not available"):
            adapter.refit(adapter._df_to_panel(nw.from_native(df), "y"))

    def test_quantiles_are_gated_off(self) -> None:
        """predict_quantiles has no X, so it cannot feed the model its regressors.

        Gated in `supports` so CQR fails at construction rather than after a
        full calibration run.
        """
        df, _ = _exog_df()
        model = _make_model(
            "univariate", horizon=3, n_series=1, futr_exog_list=["price"], loss=MQLoss(level=[80])
        )
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df)
        adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")
        assert adapter.supports(SupportsQuantiles) is False
        assert adapter.supports(SupportsCrossValidationQuantiles) is False
        with pytest.raises(UnsupportedCapability, match="no X parameter"):
            adapter.predict_quantiles(
                adapter._df_to_panel(nw.from_native(df), "y"), np.array([0.1, 0.9])
            )


# ===========================================================================
# Ragged end dates
# ===========================================================================


def _ragged_end_df(freq: str = "D") -> pd.DataFrame:
    """Both series start together, but 'b' stopped reporting ten steps early."""
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {
                "unique_id": sid,
                "ds": pd.date_range("2020-01-01", periods=n, freq=freq),
                "y": rng.standard_normal(n).cumsum(),
            }
        )
        for sid, n in (("a", 100), ("b", 90))
    ]
    return pd.concat(frames, ignore_index=True)


class TestRaggedEndDates:
    """A series that stopped reporting early is an ordinary panel, not an error.

    Regression coverage for the reported ``KeyError: "['a'] not in index"``,
    which NeuralForecast reached through the same pivot as the other adapters.
    """

    def _adapter(self, horizon: int = 3) -> tuple[NeuralForecastAdapter, pd.DataFrame]:
        df = _ragged_end_df()
        model = _make_model("univariate", horizon=horizon, n_series=1)
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df)
        model_name = type(model).__name__ if model.alias is None else model.alias
        adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name=model_name)
        return adapter, df

    @pytest.mark.parametrize("aligned", [False, True])
    def test_calibrate_cv_succeeds(self, aligned: bool) -> None:
        adapter, _ = self._adapter()
        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate_cv(adapter, n_windows=6, step_size=1, aligned=aligned)
        result = fitted.calibration_

        assert result.n_calibration_samples.shape == (2,)
        np.testing.assert_array_equal(result.n_calibration_samples, [6, 6])

    def test_predict_keeps_the_horizon_axis_and_the_right_values(self) -> None:
        """Shape alone is not enough: the old pivot also mis-slotted the values."""
        adapter, _ = self._adapter()
        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate_cv(adapter, n_windows=6, step_size=1)

        result = fitted.predict(adapter._train_panel)
        assert result.point.shape == (2, 1, 3)
        assert result.interval.shape == (2, 1, 3, 2)
        assert not np.isnan(result.point).any()

        # Independent path: NeuralForecast forecasting from its own stored
        # history. Agreement proves the panel round trip keeps each series on
        # its own step axis rather than spreading it across a shared calendar.
        expected = adapter.nf.predict()
        for row, sid in enumerate(adapter._series_ids):
            want = expected.loc[expected["unique_id"] == sid, adapter.model_name].to_numpy()
            np.testing.assert_allclose(result.point[row, 0], want, rtol=1e-5)


# ===========================================================================
# Custom column names
# ===========================================================================

_CUSTOM = {"id_col": "item", "time_col": "time", "target_col": "value"}


def _custom_name_adapter(
    n_series: int = 2,
    t_len: int = 100,
    horizon: int = 3,
    freq: str = "D",
    **model_overrides: Any,
) -> tuple[NeuralForecastAdapter, pd.DataFrame]:
    """Adapter over an ``(item, time, value)`` frame, fit with those names."""
    df = _make_df(n_series=n_series, t_len=t_len, freq=freq, **_CUSTOM)
    model = _make_model("univariate", horizon=horizon, n_series=n_series, **model_overrides)
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df, **_CUSTOM)
    adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq=freq, model_name="MLP", **_CUSTOM)
    return adapter, df


class TestCustomColumnNames:
    """``id_col``/``time_col``/``target_col`` must reach the NeuralForecast calls.

    ``NeuralForecast.predict`` reuses the names ``fit`` recorded, so it was
    already fine; ``cross_validation`` and the ``fit`` behind ``refit`` take them
    as arguments and defaulted to ``unique_id``/``ds``/``y``, which raised
    ``KeyError: 'ds'``.
    """

    def test_predict(self) -> None:
        adapter, _ = _custom_name_adapter()
        forecast = adapter.predict(adapter._train_panel)
        assert forecast.shape == (2, 1, 3)
        assert np.isfinite(forecast).all()

    def test_predict_values_land_on_the_right_series_and_step(self) -> None:
        """Independent path: the network forecasting from its own stored history."""
        adapter, _ = _custom_name_adapter()
        forecast = adapter.predict(adapter._train_panel)
        expected = adapter.nf.predict()
        for row, sid in enumerate(adapter._series_ids):
            want = expected.loc[expected["item"] == sid, adapter.model_name].to_numpy()
            np.testing.assert_allclose(forecast[row, 0], want, rtol=1e-5)

    def test_cross_validate(self) -> None:
        adapter, _ = _custom_name_adapter()
        predictions, truths = adapter.cross_validate(n_windows=4, step_size=1)
        assert predictions.shape == truths.shape == (2, 4, 3)
        assert np.isfinite(predictions).all()
        assert np.isfinite(truths).all()

    def test_cross_validate_quantiles(self) -> None:
        adapter, _ = _custom_name_adapter(n_series=1, loss=_normal_distribution_loss())
        preds, truths = adapter.cross_validate_quantiles(
            n_windows=3, step_size=1, quantiles=np.array([0.1, 0.9])
        )
        assert preds.shape == (1, 3, 3, 2)
        assert truths.shape == (1, 3, 3)
        assert np.isfinite(preds).all()

    def test_predict_quantiles(self) -> None:
        adapter, _ = _custom_name_adapter(n_series=1, loss=_normal_distribution_loss())
        out = adapter.predict_quantiles(adapter._train_panel, np.array([0.1, 0.9]))
        assert out.shape == (1, 2, 3)
        assert np.isfinite(out).all()

    def test_refit(self) -> None:
        adapter, _ = _custom_name_adapter()
        adapter.refit(adapter._train_panel)
        assert np.isfinite(adapter.predict(adapter._train_panel)).all()

    def test_end_to_end_split_conformal(self) -> None:
        adapter, _ = _custom_name_adapter(t_len=120)
        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate_cv(adapter, n_windows=6, step_size=1)
        result = fitted.predict(adapter._train_panel)
        assert result.interval.shape == (2, 1, 3, 2)
        assert np.isfinite(result.interval).all()

    def test_names_disagreeing_with_the_fitted_model_are_refused(self) -> None:
        """``NeuralForecast.predict`` takes no names, so a mismatch cannot be fixed later.

        Reported at construction, where the user can act on it, rather than as a
        ``KeyError`` from the reshaping code.
        """
        df = _make_df(n_series=2, t_len=100, **_CUSTOM)
        default_df = df.rename(columns={"item": "unique_id", "time": "ds", "value": "y"})
        model = _make_model("univariate", horizon=3, n_series=2)
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(default_df)
        with pytest.raises(ValueError, match="Column names disagree"):
            NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP", **_CUSTOM)
