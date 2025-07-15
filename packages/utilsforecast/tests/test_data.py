from utilsforecast.data import generate_series


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
