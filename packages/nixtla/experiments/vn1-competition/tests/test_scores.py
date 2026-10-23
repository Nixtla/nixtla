import pandas as pd

from src.main import vn1_competition_evaluation, get_competition_forecasts


def test_vn1_competition_evaluation():
    forecasts = get_competition_forecasts()
    eval_df = vn1_competition_evaluation(forecasts)
    assert len(eval_df) == 5
    pd.testing.assert_series_equal(
        eval_df["score"], 
        pd.Series([0.4637, 0.4657, 0.4758, 0.4774, 0.4808]),
        check_names=False,
    )
