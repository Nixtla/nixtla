"""Integration: the mlforecast shim against real mlforecast internals.

Skipped entirely without ``conformalforecast[nixtla]``. Verified against
mlforecast 1.1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

mlf_mod = pytest.importorskip("mlforecast", minversion="1.1")
pytest.importorskip("sklearn")

from mlforecast import MLForecast  # noqa: E402
from mlforecast.conformal_prediction import (  # noqa: E402
    _INTERVAL_METHODS,
    PredictionIntervals,
    _add_conformal_error_intervals,
    get_conformal_method,
)
from sklearn.linear_model import LinearRegression  # noqa: E402

from conformalforecast.compat.mlforecast_shim import (  # noqa: E402
    make_interval_method,
    register_interval_method,
)
from conformalforecast.methods import SplitConformal  # noqa: E402

N_SERIES = 2
HORIZON = 2
# Methods require ceil(1/alpha) = 11 calibration windows for level=90.
N_WINDOWS = 12
LEVEL = [90]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    ids = np.repeat([f"id{i}" for i in range(N_SERIES)], HORIZON)
    fcst_df = pd.DataFrame(
        {"unique_id": ids, "ds": np.tile(np.arange(HORIZON), N_SERIES), "model": 50.0}
    )
    scores = np.abs(rng.normal(size=(N_WINDOWS, N_SERIES, HORIZON))) + 0.5
    cs_df = pd.DataFrame({"model": scores.ravel()})
    return fcst_df, cs_df


def _long_series(t_len: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=t_len, freq="D")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"id{i}",
                    "ds": dates,
                    "y": 10.0 + np.arange(t_len) * 0.1 + rng.normal(scale=0.5, size=t_len),
                }
            )
            for i in range(N_SERIES)
        ],
        ignore_index=True,
    )


def test_shim_is_column_compatible_and_wider_than_mlforecast() -> None:
    """Same columns as mlforecast; exact order statistic >= linear interpolation."""
    fcst_df, cs_df = _frames()
    kwargs = {
        "model_names": ["model"],
        "level": LEVEL,
        "cs_n_windows": N_WINDOWS,
        "cs_h": HORIZON,
        "n_series": N_SERIES,
        "horizon": HORIZON,
    }
    native = _add_conformal_error_intervals(fcst_df.copy(), cs_df, **kwargs)
    shimmed = make_interval_method(SplitConformal)(fcst_df.copy(), cs_df, **kwargs)

    assert list(native.columns) == list(shimmed.columns)
    native_w = (native["model-hi-90"] - native["model-lo-90"]).to_numpy()
    shim_w = (shimmed["model-hi-90"] - shimmed["model-lo-90"]).to_numpy()
    # Exact order statistic is never narrower than the interpolated quantile.
    assert (shim_w >= native_w - 1e-12).all()
    # ...and not absurdly wider: within the score range.
    assert (shim_w - native_w <= cs_df["model"].max() * 2).all()


def test_registered_name_is_resolvable_but_not_yet_selectable() -> None:
    """Pins the upstream gap: the registry accepts a new name, PredictionIntervals does not.

    When mlforecast derives ``allowed_methods`` from ``_INTERVAL_METHODS`` this
    test will fail on the second half — that is the signal to delete the
    override dance in the end-to-end test below and register a real new name.
    """
    name = "conformal_method_split"
    try:
        hook = register_interval_method(name, SplitConformal)
        assert get_conformal_method(name) is hook
        with pytest.raises(ValueError, match="method must be one of"):
            PredictionIntervals(n_windows=N_WINDOWS, h=HORIZON, method=name)
    finally:
        _INTERVAL_METHODS.pop(name, None)


def test_end_to_end_via_registry_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The migration path: a method serves mlforecast's own registered method name.

    Overrides the ``"conformal_error"`` entry rather than adding a new key,
    because ``PredictionIntervals`` will not accept an unknown method name yet
    (see the test above). ``monkeypatch.setitem`` restores the native function.
    """
    monkeypatch.setitem(_INTERVAL_METHODS, "conformal_error", make_interval_method(SplitConformal))

    df = _long_series()
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1, 2, 3])
    fcst.fit(
        df,
        prediction_intervals=PredictionIntervals(
            n_windows=N_WINDOWS, h=HORIZON, method="conformal_error"
        ),
    )
    out = fcst.predict(h=HORIZON, level=LEVEL)

    lo, hi = "LinearRegression-lo-90", "LinearRegression-hi-90"
    assert {lo, hi} <= set(out.columns)
    assert np.isfinite(out[[lo, hi]].to_numpy()).all()
    assert (out[lo] <= out["LinearRegression"]).all()
    assert (out["LinearRegression"] <= out[hi]).all()
