from functools import partial, reduce

import numpy as np
import pandas as pd
import polars as pl
import polars.testing
import pytest

import utilsforecast.date_features as dtf
import utilsforecast.processing as ufp
from utilsforecast.data import generate_series
from utilsforecast.feature_engineering import (
    fourier,
    future_exog_to_historic,
    pipeline,
    time_features,
    trend,
)


@pytest.fixture
def setup_series():
    series = generate_series(5, equal_ends=True)
    series_pl = generate_series(5, equal_ends=True, engine="polars")
    return series, series_pl


def _to_polars(series):
    # polars 1.0 can't convert the integer categorical ids
    return pl.from_pandas(series.astype({"unique_id": "int64"}))


def _calendar_series(case):
    if case == "hourly_tz":
        series = generate_series(5, freq="h", equal_ends=True)
        # crosses the daylight saving time change of 2021-03-14
        ds = series["ds"] + (pd.Timestamp("2021-03-01") - pd.Timestamp("2000-01-01"))
        series["ds"] = ds.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        return series, _to_polars(series), "h", "1h"
    series = generate_series(5, equal_ends=True)
    if case == "shuffled":
        series = series.sample(frac=1.0, random_state=0).reset_index(drop=True)
        series.index = series.index + 10
    series_pl = _to_polars(series)
    if case == "polars_date":
        series_pl = series_pl.with_columns(pl.col("ds").cast(pl.Date))
    return series, series_pl, "D", "1d"


@pytest.mark.parametrize("case", ["daily", "hourly_tz", "shuffled", "polars_date"])
def test_time_features_calendar_features(case):
    series, series_pl, freq, freq_pl = _calendar_series(case)
    features = dtf.available()
    names = [f.name for f in features]
    transformed_df, future_df = time_features(series, freq=freq, features=features, h=2)
    transformed_pl, future_pl = time_features(
        series_pl, freq=freq_pl, features=features, h=2
    )
    for pd_df, pl_df in [(transformed_df, transformed_pl), (future_df, future_pl)]:
        for feature in features:
            pd_vals = pd_df[feature.name].to_numpy()
            pl_vals = pl_df[feature.name].to_numpy()
            assert pd_vals.dtype == pl_vals.dtype == feature.dtype
            np.testing.assert_array_equal(
                pd_vals, feature.compute(pd_df["ds"]), err_msg=feature.name
            )
            np.testing.assert_array_equal(pd_vals, pl_vals, err_msg=feature.name)
    assert future_df.columns.tolist() == ["unique_id", "ds", *names]


def test_time_features_mixed_features(setup_series):
    series, series_pl = setup_series

    def is_weekend(times):
        return times.dayofweek >= 5

    def month_and_day(times):
        return [
            times.dt.month().name.suffix("_month"),
            times.dt.day().name.suffix("_day"),
        ]

    _, future = time_features(
        series, freq="D", features=["month", is_weekend, dtf.week], h=1
    )
    assert future.columns.tolist() == [
        "unique_id",
        "ds",
        "month",
        "is_weekend",
        "week",
    ]
    _, future_pl = time_features(
        series_pl, freq="1d", features=["month", month_and_day, dtf.week], h=1
    )
    assert future_pl.columns == [
        "unique_id",
        "ds",
        "month",
        "ds_month",
        "ds_day",
        "week",
    ]


@pytest.mark.parametrize(
    "features",
    [["month", dtf.month], [dtf.day, dtf.day], ["day", "day"]],
    ids=["string_and_calendar", "calendar", "string"],
)
def test_time_features_duplicate_names_raise(setup_series, features):
    series, series_pl = setup_series
    for df, freq in [(series, "D"), (series_pl, "1d")]:
        with pytest.raises(ValueError, match="duplicate time feature names"):
            time_features(df, freq=freq, features=features, h=1)


@pytest.mark.parametrize("rename", [{"unique_id": "day"}, {"y": "day"}], ids=str)
def test_time_features_named_like_existing_columns_raise(setup_series, rename):
    series, series_pl = setup_series
    id_col = rename.get("unique_id", "unique_id")
    for df, freq in [(series, "D"), (series_pl, "1d")]:
        df = ufp.rename(df, rename)
        for day in [dtf.day, "day"]:
            with pytest.raises(ValueError, match=r"existing columns: \['day'\]"):
                time_features(
                    df, freq=freq, features=[day, dtf.month], h=1, id_col=id_col
                )


def test_time_features_callable_series_ignores_index(setup_series):
    series, _ = setup_series
    series = series.set_index(series.index + 10)

    def month(times):
        return pd.Series(times.month)

    transformed, _ = time_features(series, freq="D", features=[month])
    np.testing.assert_array_equal(transformed["month"], series["ds"].dt.month)


def test_time_features_callable_keeps_extension_dtypes(setup_series):
    series, _ = setup_series

    def nullable_day(times):
        return pd.Index(pd.array(times.day, dtype="Int64"))

    def day_name(times):
        return pd.Series(pd.Categorical(times.day_name()))

    transformed, _ = time_features(series, freq="D", features=[nullable_day, day_name])
    assert transformed["nullable_day"].dtype == "Int64"
    assert isinstance(transformed["day_name"].dtype, pd.CategoricalDtype)


def test_time_features_named_like_time_column_raise(setup_series):
    series, series_pl = setup_series
    for df, freq in [(series, "D"), (series_pl, "1d")]:
        df = ufp.rename(df, {"ds": "day"})
        with pytest.raises(ValueError, match="name of the time column"):
            time_features(df, freq=freq, features=[dtf.day], time_col="day")


def test_time_features_polars_runtime_names_validated(setup_series):
    _, series_pl = setup_series

    def renamed_to_ds(times):
        return [times.dt.day().name.map(lambda _: "ds")]

    with pytest.raises(ValueError, match="name of the time column"):
        time_features(series_pl, freq="1d", features=[renamed_to_ds], h=1)

    def regex_month(_times):
        return [pl.col("^d.*$").dt.month().name.suffix("_month")]

    def regex_day(_times):
        return [pl.col("^d.*$").dt.day().name.suffix("_day")]

    _, future = time_features(
        series_pl, freq="1d", features=[regex_month, regex_day], h=1
    )
    assert future.columns == ["unique_id", "ds", "ds_month", "ds_day"]


@pytest.mark.filterwarnings("ignore")
def test_time_features_polars_udfs_not_run_for_names(setup_series):
    _, series_pl = setup_series

    def month_udf(times):
        return times.map_elements(lambda t: t.month)

    def first_day_udf(times):
        return times.map_batches(lambda s: pl.repeat(s[0].day, s.len(), eager=True))

    transformed, _ = time_features(
        series_pl, freq="1d", features=[month_udf, first_day_udf]
    )
    np.testing.assert_array_equal(
        transformed["month_udf"], transformed["ds"].dt.month()
    )
    assert transformed["first_day_udf"].n_unique() == 1


def test_time_features_null_dates_raise(setup_series):
    series, series_pl = setup_series
    series = series.assign(ds=series["ds"].where(series.index % 2 == 0))
    series_pl = series_pl.with_columns(
        pl.when(pl.int_range(pl.len()) % 2 == 0).then(pl.col("ds")).alias("ds")
    )
    for df in (series, series_pl):
        for feature in [dtf.month, "month", lambda times: times]:
            with pytest.raises(ValueError, match="found null dates"):
                time_features(df, freq="D", features=[feature])


def test_time_features_object_times_with_none_and_nan_raise():
    times = [pd.Timestamp("2020-01-01"), None, pd.Timestamp("2020-03-01"), np.nan]
    df = pd.DataFrame(
        {"unique_id": 0, "ds": pd.Series(times, dtype=object), "y": range(4)}
    )
    with pytest.raises(ValueError, match="found null dates"):
        time_features(df, freq="D", features=["month"])


@pytest.mark.skipif(
    not dtf._PANDAS_GE_2, reason="pandas<2 doesn't support pyarrow dates"
)
def test_time_features_pyarrow_backed(setup_series):
    series, _ = setup_series
    series = series.assign(
        ds=series["ds"]
        .dt.tz_localize("America/New_York")
        .astype("timestamp[ns, tz=America/New_York][pyarrow]")
    )
    transformed, future = time_features(series, freq="D", features=dtf.available(), h=2)
    for df in [transformed, future]:
        for feature in dtf.available():
            np.testing.assert_array_equal(
                df[feature.name].to_numpy(),
                feature.compute(df["ds"]),
                err_msg=feature.name,
            )


@pytest.mark.skipif(
    not dtf._PANDAS_GE_2, reason="pandas<2 doesn't support pyarrow dates"
)
@pytest.mark.parametrize(
    "arrow_dtype", ["timestamp[ns][pyarrow]", "timestamp[us, tz=UTC][pyarrow]"]
)
def test_time_features_pyarrow_backed_strings_and_callables(setup_series, arrow_dtype):
    series, _ = setup_series
    if "tz=" in arrow_dtype:
        series = series.assign(ds=series["ds"].dt.tz_localize("UTC"))
    arrow_series = series.assign(ds=series["ds"].astype(arrow_dtype))

    def month_end(times):
        return times.is_month_end

    features = ["month", "dayofweek", "week", month_end, dtf.day]
    expected, expected_future = time_features(series, freq="D", features=features, h=2)
    transformed, future = time_features(
        arrow_series, freq="D", features=features, h=2
    )
    names = ["month", "dayofweek", "week", "month_end", "day"]
    pd.testing.assert_frame_equal(transformed[names], expected[names])
    pd.testing.assert_frame_equal(future[names], expected_future[names])


def test_fourier_transform(setup_series):
    series, series_pl = setup_series

    transformed_df, future_df = fourier(series, freq="D", season_length=7, k=2, h=1)
    transformed_df2, future_df2 = fourier(
        series.sample(frac=1.0), freq="D", season_length=7, k=2, h=1
    )
    pd.testing.assert_frame_equal(
        transformed_df,
        transformed_df2.sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(future_df, future_df2)

    transformed_pl, future_pl = fourier(series_pl, freq="1d", season_length=7, k=2, h=1)
    transformed_pl2, future_pl2 = fourier(
        series_pl.sample(fraction=1.0), freq="1d", season_length=7, k=2, h=1
    )
    pl.testing.assert_frame_equal(transformed_pl, transformed_pl2)
    pl.testing.assert_frame_equal(future_pl, future_pl2)
    pd.testing.assert_frame_equal(
        transformed_df.drop(columns=["unique_id"]),
        transformed_pl.drop("unique_id").to_pandas(),
    )
    pd.testing.assert_frame_equal(
        future_df.drop(columns=["unique_id"]), future_pl.drop("unique_id").to_pandas()
    )
    series = generate_series(5, equal_ends=True)
    transformed_df, future_df = trend(series, freq="D", h=1)
    transformed_df
    future_df
    transformed_df, future_df = time_features(
        series, freq="D", features=["month", "day", "week"], h=1
    )
    series_with_prices = series.assign(price=np.random.rand(len(series))).sample(
        frac=1.0
    )
    series_with_prices
    transformed_df, future_df = future_exog_to_historic(
        df=series_with_prices,
        freq="D",
        features=["price"],
        h=2,
    )
    pd.testing.assert_frame_equal(
        (
            series_with_prices.sort_values(["unique_id", "ds"])
            .groupby("unique_id", observed=True)
            .tail(2)[["unique_id", "price"]]
            .reset_index(drop=True)
        ),
        future_df[["unique_id", "price"]],
    )
    series_with_prices_pl = _to_polars(series_with_prices)
    transformed_pl, future_pl = future_exog_to_historic(
        df=series_with_prices_pl,
        freq="1d",
        features=["price"],
        h=2,
    )
    pd.testing.assert_frame_equal(
        future_pl.to_pandas(), future_df.astype({"unique_id": "int64"})
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_time_features_weekofyear(setup_series, engine):
    # weekofyear is an alias of the ISO week
    series, series_pl = setup_series
    features = ["week", "weekofyear"]
    if engine == "pandas":
        transformed, future = time_features(series, freq="D", features=features, h=2)
    else:
        transformed, future = time_features(
            series_pl, freq="1d", features=features, h=2
        )
        transformed, future = transformed.to_pandas(), future.to_pandas()
    expected = transformed["ds"].dt.isocalendar().week.to_numpy()
    np.testing.assert_array_equal(transformed["weekofyear"].to_numpy(), expected)
    for df in (transformed, future):
        pd.testing.assert_series_equal(df["weekofyear"], df["week"], check_names=False)


def is_weekend(times):
    if isinstance(times, pd.Index):
        dow = times.weekday + 1  # monday=0 in pandas and 1 in polars
    else:
        dow = times.dt.weekday()
    return dow >= 6


def even_days_and_months(times):
    if isinstance(times, pd.Index):
        out = pd.DataFrame(
            {
                "even_day": (times.weekday + 1) % 2 == 0,
                "even_month": times.month % 2 == 0,
            }
        )
    else:
        # for polars you can return a list of expressions
        out = [
            (times.dt.weekday() % 2 == 0).alias("even_day"),
            (times.dt.month() % 2 == 0).alias("even_month"),
        ]
    return out


@pytest.fixture
def setup_features():
    features = [
        trend,
        partial(fourier, season_length=7, k=1),
        partial(fourier, season_length=28, k=1),
        partial(time_features, features=["day", is_weekend, even_days_and_months]),
    ]
    return features


@pytest.fixture
def setup_pipeline(setup_series, setup_features):
    def _inner(freq):
        series, _ = setup_series
        transformed_df, future_df = pipeline(
            series,
            features=setup_features,
            freq=freq,
            h=1,
        )
        return transformed_df, future_df

    return _inner


def reduce_join(dfs, on):
    return reduce(
        lambda left, right: left.merge(right, on=on, how="left"),
        dfs,
    )


@pytest.mark.parametrize("freq", ["D", "1d"])
def test_pipeline(setup_series, freq, setup_pipeline, setup_features):
    series, series_pl = setup_series
    transformed_df, future_df = setup_pipeline(freq)

    individual_results = [f(series, freq=freq, h=1) for f in setup_features]
    expected_transformed = reduce_join(
        [r[0] for r in individual_results], on=["unique_id", "ds", "y"]
    )
    expected_future = reduce_join(
        [r[1] for r in individual_results], on=["unique_id", "ds"]
    )

    pd.testing.assert_frame_equal(transformed_df, expected_transformed)
    pd.testing.assert_frame_equal(future_df, expected_future)

    transformed_pl, future_pl = pipeline(
        series_pl,
        features=setup_features,
        freq="1d",
        h=1,
    )
    pd.testing.assert_frame_equal(
        transformed_pl.drop("unique_id").to_pandas(),
        transformed_df.drop(columns="unique_id"),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        future_pl.drop("unique_id").to_pandas(),
        future_df.drop(columns="unique_id"),
        check_dtype=False,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pipeline_no_horizon(setup_series, setup_features, engine):
    # with h=0 there are no future values, but every feature must be computed
    series, series_pl = setup_series
    individual_results = [f(series, freq="D", h=0) for f in setup_features]
    expected = reduce_join(
        [r[0] for r in individual_results], on=["unique_id", "ds", "y"]
    )
    if engine == "pandas":
        transformed, future = pipeline(series, features=setup_features, freq="D", h=0)
    else:
        transformed, future = pipeline(
            series_pl, features=setup_features, freq="1d", h=0
        )
        transformed = transformed.to_pandas()
    pd.testing.assert_frame_equal(
        transformed.drop(columns="unique_id"),
        expected.drop(columns="unique_id"),
        check_dtype=False,
    )
    assert future.shape == (0, 0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fourier_drops_redundant_sine(engine):
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2000-01-01", periods=120, freq="MS"),
            "y": np.arange(120.0),
        }
    )
    freq = "MS"
    if engine == "polars":
        df = pl.from_pandas(df)
        freq = "1mo"
    train, future = fourier(df, freq=freq, season_length=12, k=6, h=12)
    for out in (train, future):
        assert "sin6_12" not in out.columns
        assert "cos6_12" in out.columns
        assert [f"sin{i}_12" for i in range(1, 6)] == [
            c for c in out.columns if c.startswith("sin")
        ]
