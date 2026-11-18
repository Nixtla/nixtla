import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast


def _panel(n_series=3, n_times=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "unique_id": np.repeat([f"s{i}" for i in range(n_series)], n_times),
            "ds": np.tile(np.arange(n_times), n_series),
            "y": rng.normal(10, 1, n_series * n_times),
            "brand": np.repeat(np.arange(n_series) % 2, n_times),
            "price": rng.normal(5, 1, n_series * n_times),
            "w": 1.0,
        }
    )


def _future(df, h, cols=("price",)):
    ids = np.repeat(sorted(df["unique_id"].unique()), h)
    last = df["ds"].max()
    out = pd.DataFrame(
        {
            "unique_id": ids,
            "ds": np.tile(np.arange(last + 1, last + 1 + h), len(ids) // h),
        }
    )
    for c in cols:
        out[c] = 1.0
    return out


def _fcst():
    return MLForecast(models=[LinearRegression()], freq=1, lags=[1])


@pytest.mark.parametrize("static_features", [None, []])
def test_default_is_no_static_features(static_features):
    df = _panel().drop(columns="w")
    fcst = _fcst().fit(df, static_features=static_features)
    assert list(fcst.ts.static_features_.columns) == ["unique_id"]
    with pytest.raises(ValueError, match=r"\['brand', 'price'\].*static_features"):
        fcst.predict(2)
    preds = fcst.predict(2, X_df=_future(df, 2, cols=("brand", "price")))
    assert preds.shape[0] == 6


def test_listed_statics_are_repeated():
    df = _panel().drop(columns="w")
    fcst = _fcst().fit(df, static_features=["brand"])
    assert fcst.ts.features_order_ == ["brand", "price", "lag1"]
    fcst.predict(2, X_df=_future(df, 2))


def test_unknown_static_raises():
    with pytest.raises(ValueError, match=r"\['foo'\] are not in the data"):
        _fcst().fit(_panel(), static_features=["foo"])


def test_weight_col_static_raises():
    with pytest.raises(ValueError, match="weight column 'w'"):
        _fcst().fit(_panel(), static_features=["w"], weight_col="w")


def test_changing_static_raises():
    with pytest.raises(ValueError, match="price is declared as a static feature"):
        _fcst().fit(_panel(), static_features=["price"])


def test_id_as_static_feature():
    df = _panel().drop(columns=["w", "price"])
    df["unique_id"] = df["unique_id"].str[1:].astype(int)
    fcst = _fcst().fit(df, static_features=["unique_id", "brand"])
    assert fcst.ts.features_order_ == ["unique_id", "brand", "lag1"]


def test_partial_x_df_raises():
    df = _panel().drop(columns="w")
    fcst = _fcst().fit(df, static_features=["brand"])
    X_df = _future(df, 3)
    with pytest.raises(ValueError, match="Found missing inputs in X_df"):
        fcst.predict(3, X_df=X_df.iloc[1:])


def test_cross_validation_uses_its_own_statics():
    df = _panel().drop(columns="w")
    fcst = _fcst()
    fcst.fit(df)
    cv = fcst.cross_validation(df, n_windows=2, h=2, static_features=["brand"])
    assert cv.shape[0] == 12
    assert list(fcst.ts.static_features_.columns) == ["unique_id", "brand"]


def test_transfer_keeps_fit_split_and_rejects_unseen_columns():
    df = _panel().drop(columns="w")
    fcst = _fcst().fit(df, static_features=["brand"])
    new = _panel(seed=1).drop(columns="w")
    new["unique_id"] = new["unique_id"].str.replace("s", "n")
    out = fcst.transfer(new)
    assert list(out.ts.static_features_.columns) == ["unique_id", "brand"]
    out.predict(2, X_df=_future(new, 2))
    with pytest.raises(ValueError, match=r"weren't seen during training: \['extra'\]"):
        fcst.transfer(new.assign(extra=1.0))
