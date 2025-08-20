import httpx
import pytest

import numpy as np
import pandas as pd
import zstandard as zstd

from contextlib import contextmanager
from copy import deepcopy
from nixtla_tests.conftest import HYPER_PARAMS_TEST
from nixtla_tests.helpers.checks import check_num_partitions_same_results
from nixtla_tests.helpers.checks import check_equal_fcsts_add_history


CAPTURED_REQUEST = None


class CapturingClient(httpx.Client):
    def post(self, *args, **kwargs):
        request = self.build_request("POST", *args, **kwargs)
        global CAPTURED_REQUEST
        CAPTURED_REQUEST = {
            "headers": dict(request.headers),
            "content": request.content,
            "method": request.method,
            "url": str(request.url),
        }
        return super().post(*args, **kwargs)


@contextmanager
def capture_request():
    original_client = httpx.Client
    httpx.Client = CapturingClient
    try:
        yield
    finally:
        httpx.Client = original_client


@pytest.mark.parametrize(
    "df_converter, freq",
    [
        pytest.param(lambda series, with_gaps: with_gaps, "5min", id="gaps"),
        pytest.param(
            lambda series, with_gaps: pd.concat([series, series]),
            "5min",
            id="duplicates",
        ),
        pytest.param(lambda series, with_gaps: series, "1min", id="wrong_freq"),
    ],
)
def test_forecast_with_error(series_with_gaps, nixtla_test_client, df_converter, freq):
    series, with_gaps = series_with_gaps

    with pytest.raises(
        ValueError,
        match="missing or duplicate timestamps, or the timestamps do not match",
    ):
        nixtla_test_client.forecast(df=df_converter(series, with_gaps), h=1, freq=freq)


def test_cv_forecast_consistency(nixtla_test_client, cv_series_with_features):
    series_with_features, train, valid, x_cols, h, freq = cv_series_with_features
    for hist_exog_list in [None, [], [x_cols[2], x_cols[1]], x_cols]:
        cv_res = nixtla_test_client.cross_validation(
            series_with_features,
            n_windows=1,
            h=h,
            freq=freq,
            hist_exog_list=hist_exog_list,
        )
        fcst_res = nixtla_test_client.forecast(
            train,
            h=h,
            freq=freq,
            hist_exog_list=hist_exog_list,
            X_df=valid,
        )
        np.testing.assert_allclose(
            cv_res["TimeGPT"], fcst_res["TimeGPT"], atol=1e-4, rtol=1e-3
        )


def test_forecast_different_hist_exog_gives_different_results(
    nixtla_test_client, cv_series_with_features
):
    _, train, valid, x_cols, h, freq = cv_series_with_features
    for X_df in (None, valid):
        res1 = nixtla_test_client.forecast(
            train, h=h, X_df=X_df, freq=freq, hist_exog_list=x_cols[:2]
        )
        res2 = nixtla_test_client.forecast(
            train, h=h, X_df=X_df, freq=freq, hist_exog_list=x_cols[2:]
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                res1["TimeGPT"],
                res2["TimeGPT"],
                atol=1e-4,
                rtol=1e-3,
            )


def test_forecast_date_features_multiple_series_and_different_ends(
    nixtla_test_client, two_short_series
):
    h = 12
    fcst_test_series = nixtla_test_client.forecast(
        two_short_series, h=h, date_features=["dayofweek"]
    )
    uids = two_short_series["unique_id"]
    for uid in uids:
        expected = pd.date_range(
            periods=h + 1, start=two_short_series.query("unique_id == @uid")["ds"].max()
        )[1:].tolist()
        actual = fcst_test_series.query("unique_id == @uid")["ds"].tolist()
        assert actual == expected


def test_compression(nixtla_test_client, series_1MB_payload):
    with capture_request():
        nixtla_test_client.forecast(
            df=series_1MB_payload,
            freq="D",
            h=1,
            hist_exog_list=["static_0", "static_1"],
        )
        assert CAPTURED_REQUEST["headers"]["content-encoding"] == "zstd"
        content = CAPTURED_REQUEST["content"]
        assert len(content) < 2**20
        assert len(zstd.ZstdDecompressor().decompress(content)) > 2**20


def test_cv_refit_equivalence(nixtla_test_client, air_passengers_df):
    cv_kwargs = dict(
        df=air_passengers_df,
        n_windows=2,
        h=12,
        freq="MS",
        time_col="timestamp",
        target_col="value",
        finetune_steps=2,
    )
    res_refit = nixtla_test_client.cross_validation(refit=True, **cv_kwargs)
    res_no_refit = nixtla_test_client.cross_validation(refit=False, **cv_kwargs)
    np.testing.assert_allclose(res_refit["value"], res_no_refit["value"])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            res_refit["TimeGPT"],
            res_no_refit["TimeGPT"],
            atol=1e-4,
            rtol=1e-3,
        )


def test_forecast_quantiles_error(nixtla_test_client, air_passengers_df):
    with pytest.raises(Exception) as excinfo:
        nixtla_test_client.forecast(
            df=air_passengers_df,
            h=12,
            time_col="timestamp",
            target_col="value",
            level=[80],
            quantiles=[0.2, 0.3],
        )
    assert "not both" in str(excinfo.value)


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("forecast", {}),
        ("forecast", {"add_history": True}),
        ("cross_validation", {}),
    ],
)
def test_forecast_quantiles_output(
    nixtla_test_client, air_passengers_df, method, kwargs
):
    test_qls = list(np.arange(0.1, 1, 0.1))
    exp_q_cols = [f"TimeGPT-q-{int(100 * q)}" for q in test_qls]

    args = {
        "df": air_passengers_df,
        "h": 12,
        "time_col": "timestamp",
        "target_col": "value",
        "quantiles": test_qls,
        **kwargs,
    }
    if method == "cross_validation":
        func = nixtla_test_client.cross_validation
    elif method == "forecast":
        func = nixtla_test_client.forecast

    df_qls = func(**args)

    assert all(col in df_qls.columns for col in exp_q_cols)
    assert not any("-lo-" in col for col in df_qls.columns)
    # test monotonicity of quantiles
    for c1, c2 in zip(exp_q_cols[:-1], exp_q_cols[1:]):
        assert df_qls[c1].lt(df_qls[c2]).all()


@pytest.mark.parametrize("freq", ["D", "W-THU", "Q-DEC", "15T"])
@pytest.mark.parametrize(
    "method_name,method_kwargs,exog",
    [
        ("detect_anomalies", {"level": 98}, False),
        ("cross_validation", {"h": 7, "n_windows": 2}, False),
        ("forecast", {"h": 7, "add_history": True}, False),
        ("detect_anomalies", {"level": 98}, True),
        ("cross_validation", {"h": 7, "n_windows": 2}, False),
        ("forecast", {"h": 7, "add_history": True}, False),
    ],
)
def test_num_partitions_same_results_parametrized(
    nixtla_test_client, df_freq_generator, method_name, method_kwargs, freq, exog
):
    mathod_mapper = {
        "detect_anomalies": nixtla_test_client.detect_anomalies,
        "cross_validation": nixtla_test_client.cross_validation,
        "forecast": nixtla_test_client.forecast,
    }
    method = mathod_mapper[method_name]

    df_freq = df_freq_generator(n_series=10, min_length=500, max_length=550, freq=freq)
    df_freq["ds"] = df_freq.groupby("unique_id", observed=True)["ds"].transform(
        lambda x: pd.date_range(periods=len(x), freq=freq, end="2023-01-01")
    )
    if exog:
        df_freq["exog_1"] = 1

    kwargs = {
        "method": method,
        "num_partitions": 2,
        "df": df_freq,
        **method_kwargs,
    }

    check_num_partitions_same_results(**kwargs)


@pytest.mark.parametrize(
    "freq,h",
    [
        ("D", 7),
        ("W-THU", 52),
        ("Q-DEC", 8),
        ("15T", 4 * 24 * 7),
    ],
)
def test_forecast_models_different_results(
    nixtla_test_client, df_freq_generator, freq, h
):
    df_freq = df_freq_generator(n_series=10, min_length=500, max_length=550, freq=freq)
    df_freq["ds"] = df_freq.groupby("unique_id", observed=True)["ds"].transform(
        lambda x: pd.date_range(periods=len(x), freq=freq, end="2023-01-01")
    )
    kwargs = dict(df=df_freq, h=h)
    fcst_1_df = check_equal_fcsts_add_history(
        nixtla_test_client, **{**kwargs, "model": "timegpt-1"}
    )
    fcst_2_df = check_equal_fcsts_add_history(
        nixtla_test_client, **{**kwargs, "model": "timegpt-1-long-horizon"}
    )
    with pytest.raises(
        AssertionError, match=r'\(column name="TimeGPT"\) are different'
    ):
        pd.testing.assert_frame_equal(fcst_1_df, fcst_2_df)


@pytest.mark.parametrize(
    "method, method_kwargs",
    [
        (
            "forecast",
            dict(
                h=12,
                level=[90, 95],
                add_history=True,
                time_col="timestamp",
                target_col="value",
            ),
        ),
        (
            "cross_validation",
            dict(h=12, level=[90, 95], time_col="timestamp", target_col="value"),
        ),
        ("detect_anomalies", dict(level=99, time_col="timestamp", target_col="value")),
    ],
)
def test_different_models_give_different_results(
    air_passengers_df, nixtla_test_client, method, method_kwargs
):
    mathod_mapper = {
        "detect_anomalies": nixtla_test_client.detect_anomalies,
        "cross_validation": nixtla_test_client.cross_validation,
        "forecast": nixtla_test_client.forecast,
    }
    execute = mathod_mapper[method]

    # Run with first model
    out1 = execute(df=air_passengers_df, model="timegpt-1", **method_kwargs)
    # Run with second model
    out2 = execute(
        df=air_passengers_df, model="timegpt-1-long-horizon", **method_kwargs
    )
    # Compare only the TimeGPT column
    with pytest.raises(
        AssertionError, match=r'\(column name="TimeGPT"\) are different'
    ):
        pd.testing.assert_frame_equal(out1[["TimeGPT"]], out2[["TimeGPT"]])

    # test unsupported model
    method_kwargs["model"] = "my-awesome-model"
    with pytest.raises(ValueError, match="unsupported model"):
        execute(df=air_passengers_df, **method_kwargs)


def test_shap_features(nixtla_test_client, date_features_result):
    # Test shap values are returned and sum to predictions
    df_date_features, future_df, _ = date_features_result
    h = 12
    fcst_df = nixtla_test_client.forecast(
        df=df_date_features, h=h, X_df=future_df, feature_contributions=True
    )
    shap_values = nixtla_test_client.feature_contributions
    assert len(shap_values) == len(fcst_df)
    np.testing.assert_allclose(
        fcst_df["TimeGPT"].values, shap_values.iloc[:, 3:].sum(axis=1).values, rtol=1e-3
    )

    fcst_hist_df = nixtla_test_client.forecast(
        df=df_date_features,
        h=h,
        X_df=future_df,
        add_history=True,
        feature_contributions=True,
    )
    shap_values_hist = nixtla_test_client.feature_contributions
    assert len(shap_values_hist) == len(fcst_hist_df)
    np.testing.assert_allclose(
        fcst_hist_df["TimeGPT"].values,
        shap_values_hist.iloc[:, 3:].sum(axis=1).values,
        atol=1e-4,
    )

    # test num partitions
    _ = nixtla_test_client.feature_contributions
    pd.testing.assert_frame_equal(
        nixtla_test_client.feature_contributions, shap_values_hist, atol=1e-4, rtol=1e-3
    )


@pytest.mark.parametrize("hyp", HYPER_PARAMS_TEST)
def test_exogenous_variables_cv(nixtla_test_client, exog_data, hyp):
    df_ex_, df_train, df_test, x_df_test = exog_data
    fcst_test = nixtla_test_client.forecast(
        df_train.merge(df_ex_.drop(columns="y")), h=12, X_df=x_df_test, **hyp
    )
    fcst_test = df_test[["unique_id", "ds", "y"]].merge(fcst_test)
    fcst_test = fcst_test.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    fcst_cv = nixtla_test_client.cross_validation(df_ex_, h=12, **hyp)
    fcst_cv = fcst_cv.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        fcst_test,
        fcst_cv.drop(columns="cutoff"),
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.parametrize("hyp", HYPER_PARAMS_TEST)
def test_forecast_vs_cv_no_exog(
    nixtla_test_client, train_test_split, air_passengers_renamed_df, hyp
):
    df_train, df_test = train_test_split
    fcst_test = nixtla_test_client.forecast(df_train, h=12, **hyp)
    fcst_test = df_test[["unique_id", "ds", "y"]].merge(fcst_test)
    fcst_test = fcst_test.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    fcst_cv = nixtla_test_client.cross_validation(
        air_passengers_renamed_df, h=12, **hyp
    )
    fcst_cv = fcst_cv.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        fcst_test,
        fcst_cv.drop(columns="cutoff"),
        rtol=1e-2,
    )


@pytest.mark.parametrize("hyp", HYPER_PARAMS_TEST)
def test_forecast_vs_cv_insert_y(
    nixtla_test_client, train_test_split, air_passengers_renamed_df, hyp
):
    df_train, df_test = train_test_split
    fcst_test = nixtla_test_client.forecast(df_train, h=12, **hyp)
    fcst_test.insert(2, "y", df_test["y"].values)
    fcst_test = fcst_test.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    fcst_cv = nixtla_test_client.cross_validation(
        air_passengers_renamed_df, h=12, **hyp
    )
    fcst_cv = fcst_cv.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        fcst_test,
        fcst_cv.drop(columns="cutoff"),
        rtol=1e-2,
    )


def test_forecast_and_anomalies_index_vs_columns(
    nixtla_test_client, air_passengers_renamed_df, air_passengers_renamed_df_with_index
):
    fcst_inferred_df_index = nixtla_test_client.forecast(
        air_passengers_renamed_df_with_index, h=10
    )
    anom_inferred_df_index = nixtla_test_client.detect_anomalies(
        air_passengers_renamed_df_with_index
    )
    fcst_inferred_df = nixtla_test_client.forecast(
        air_passengers_renamed_df[["ds", "unique_id", "y"]], h=10
    )
    anom_inferred_df = nixtla_test_client.detect_anomalies(
        air_passengers_renamed_df[["ds", "unique_id", "y"]]
    )
    pd.testing.assert_frame_equal(
        fcst_inferred_df_index, fcst_inferred_df, atol=1e-4, rtol=1e-3
    )
    pd.testing.assert_frame_equal(
        anom_inferred_df_index, anom_inferred_df, atol=1e-4, rtol=1e-3
    )


@pytest.mark.parametrize("freq", ["Y", "W-MON", "Q-DEC", "H"])
def test_forecast_index_vs_columns_various_freq(
    nixtla_test_client, air_passengers_renamed_df_with_index, freq
):
    df_ds_index = air_passengers_renamed_df_with_index.groupby("unique_id").tail(80)
    df_ds_index.index = np.concatenate(
        df_ds_index["unique_id"].nunique()
        * [pd.date_range(end="2023-01-01", periods=80, freq=freq)]
    )
    df_ds_index.index.name = "ds"
    fcst_inferred_df_index = nixtla_test_client.forecast(df_ds_index, h=10)
    df_test = df_ds_index.reset_index()
    fcst_inferred_df = nixtla_test_client.forecast(df_test, h=10)
    pd.testing.assert_frame_equal(
        fcst_inferred_df_index, fcst_inferred_df, atol=1e-4, rtol=1e-3
    )


def test_index_as_time_col(nixtla_test_client, air_passengers_df):
    df_test = deepcopy(air_passengers_df)
    df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
    df_test.set_index(df_test["timestamp"], inplace=True)
    df_test.drop(columns="timestamp", inplace=True)
    # Using user_provided time_col and freq
    timegpt_anomalies_df_1 = nixtla_test_client.detect_anomalies(
        air_passengers_df, time_col="timestamp", target_col="value", freq="M"
    )
    # Infer time_col and freq from index
    timegpt_anomalies_df_2 = nixtla_test_client.detect_anomalies(
        df_test, time_col="timestamp", target_col="value"
    )
    pd.testing.assert_frame_equal(
        timegpt_anomalies_df_1,
        timegpt_anomalies_df_2,
        atol=1e-4,
        rtol=1e-3,
    )
