import pytest

from utilsforecast.data import generate_series


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_generate_series_with_exogenous_features(engine):
    df, futr_df = generate_series(
        n_series=2,
        min_length=3,
        max_length=3,
        n_hist_exog=2,
        n_futr_exog=3,
        h=4,
        engine=engine,
    )

    assert df.shape[0] == 6
    assert futr_df.shape[0] == 8
    assert {"hist_exog_0", "hist_exog_1"}.issubset(df.columns)
    assert {"futr_exog_0", "futr_exog_1", "futr_exog_2"}.issubset(df.columns)
    assert "hist_exog_0" not in futr_df.columns
    assert {"futr_exog_0", "futr_exog_1", "futr_exog_2"}.issubset(futr_df.columns)
    if engine == "polars":
        last_dates = df.group_by("unique_id").max()
        first_futr_dates = futr_df.group_by("unique_id").min()
    else:
        last_dates = df.groupby("unique_id", observed=True)["ds"].max()
        first_futr_dates = futr_df.groupby("unique_id", observed=True)["ds"].min()
    if engine == "polars":
        assert (first_futr_dates["ds"] > last_dates["ds"]).all()
    else:
        assert (first_futr_dates > last_dates).all()


def test_generate_series_requires_positive_horizon_for_future_exogenous_features():
    with pytest.raises(ValueError, match="h must be at least 1"):
        generate_series(n_series=1, n_futr_exog=1, h=0)


def test_data():
    synthetic_panel = generate_series(n_series=2)
    synthetic_panel.groupby("unique_id", observed=True).head(4)
    level = [40, 80, 95]
    series = generate_series(100, n_models=2, level=level)
    for model in ["model0", "model1"]:
        for lv in level:
            assert (
                series[model]
                .between(series[f"{model}-lo-{lv}"], series[f"{model}-hi-{lv}"])
                .all()
            )
        for lv_lo, lv_hi in zip(level[:-1], level[1:]):
            assert series[f"{model}-lo-{lv_lo}"].ge(series[f"{model}-lo-{lv_hi}"]).all()
            assert series[f"{model}-hi-{lv_lo}"].le(series[f"{model}-hi-{lv_hi}"]).all()
