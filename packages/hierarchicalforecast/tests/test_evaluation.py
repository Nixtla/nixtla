from functools import partial

import numpy as np
import pandas as pd
import polars as pl
import pytest
import utilsforecast.losses as ufl
from statsforecast.core import StatsForecast
from statsforecast.models import AutoETS

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation, evaluate, mse
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate


def rmse(y, y_hat):
    return np.mean(np.sqrt(np.mean((y - y_hat) ** 2, axis=1)))


def mase(y, y_hat, y_insample, seasonality=4):
    errors = np.mean(np.abs(y - y_hat), axis=1)
    scale = np.mean(
        np.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1
    )
    return np.mean(errors / scale)



@pytest.fixture
def hierarchy_specs(hiers_grouped, hiers_strictly):
    """Fixture to provide hierarchy specifications."""
    return {"grouped": hiers_grouped, "strictly": hiers_strictly}


@pytest.fixture
def grouped_data(tourism_df, hierarchy_specs):
    """Fixture to provide grouped hierarchical data."""
    # getting df
    hier_grouped_df, S_grouped, tags_grouped = aggregate(
        tourism_df, hierarchy_specs["grouped"]
    )

    # split train/test
    hier_grouped_df["y_model"] = hier_grouped_df["y"]
    # we should be able to recover y using the methods
    hier_grouped_df_h = (
        hier_grouped_df.groupby("unique_id").tail(12).reset_index(drop=True)
    )
    ds_h = hier_grouped_df_h["ds"].unique()
    hier_grouped_df = hier_grouped_df.query("~(ds in @ds_h)")
    # adding noise to `y_model` to avoid perfect fited values
    rng = np.random.default_rng(0)
    hier_grouped_df["y_model"] += rng.uniform(-1, 1, len(hier_grouped_df))

    return {
        "hier_grouped_df": hier_grouped_df,
        "hier_grouped_df_h": hier_grouped_df_h,
        "S_grouped": S_grouped,
        "tags_grouped": tags_grouped,
    }


@pytest.fixture
def reconciled_data(grouped_data):
    """Fixture to provide reconciled hierarchical data."""
    # hierachical reconciliation
    hrec = HierarchicalReconciliation(
        reconcilers=[
            # these methods should reconstruct the original y
            BottomUp(),
            MinTrace(method="ols"),
            MinTrace(method="wls_struct"),
            MinTrace(method="wls_var"),
            MinTrace(method="mint_shrink"),
        ]
    )
    reconciled = hrec.reconcile(
        Y_hat_df=grouped_data["hier_grouped_df_h"],
        Y_df=grouped_data["hier_grouped_df"],
        S=grouped_data["S_grouped"],
        tags=grouped_data["tags_grouped"],
    )
    return reconciled


def test_mse_rmse_evaluation(reconciled_data, grouped_data):
    """Test MSE and RMSE evaluation."""
    # Test mse, rmse
    evaluator = HierarchicalEvaluation([mse, rmse])
    evaluation_old = evaluator.evaluate(
        Y_hat_df=reconciled_data.drop(columns="y"),
        Y_test_df=reconciled_data[["unique_id", "ds", "y"]],
        tags=grouped_data["tags_grouped"],
    )
    # Test mse, rmse
    evaluation = evaluate(
        reconciled_data, metrics=[ufl.mse, ufl.rmse], tags=grouped_data["tags_grouped"]
    )

    pd.testing.assert_frame_equal(evaluation_old, evaluation)


def test_mse_rmse_evaluation_polars(reconciled_data, grouped_data):
    """Test MSE and RMSE evaluation with polars."""
    evaluator = HierarchicalEvaluation([mse, rmse])
    # polars
    # Test mse, rmse
    reconciled_pl = pl.from_pandas(reconciled_data)
    evaluation_pl = evaluator.evaluate(
        Y_hat_df=reconciled_pl.drop("y"),
        Y_test_df=reconciled_pl[["unique_id", "ds", "y"]],
        tags=grouped_data["tags_grouped"],
    )

    evaluation = evaluate(
        reconciled_data, metrics=[ufl.mse, ufl.rmse], tags=grouped_data["tags_grouped"]
    )

    pd.testing.assert_frame_equal(evaluation, evaluation_pl.to_pandas())


def test_mase_evaluation(reconciled_data, grouped_data):
    """Test MASE evaluation."""
    # Test mase
    evaluator = HierarchicalEvaluation([mase])
    evaluation_old = evaluator.evaluate(
        Y_hat_df=reconciled_data.drop(columns="y"),
        Y_test_df=reconciled_data[["unique_id", "ds", "y"]],
        tags=grouped_data["tags_grouped"],
        Y_df=grouped_data["hier_grouped_df"],
    )
    evaluation = evaluate(
        reconciled_data,
        metrics=[partial(ufl.mase, seasonality=4)],
        train_df=grouped_data["hier_grouped_df"],
        tags=grouped_data["tags_grouped"],
    )

    pd.testing.assert_frame_equal(evaluation_old, evaluation)


def test_mase_evaluation_polars(reconciled_data, grouped_data):
    """Test MASE evaluation with polars."""
    evaluator = HierarchicalEvaluation([mase])
    # polars
    reconciled_pl = pl.from_pandas(reconciled_data)
    hier_grouped_df_pl = pl.from_pandas(grouped_data["hier_grouped_df"])

    evaluation_pl = evaluator.evaluate(
        Y_hat_df=reconciled_pl.drop("y"),
        Y_test_df=reconciled_pl[["unique_id", "ds", "y"]],
        tags=grouped_data["tags_grouped"],
        Y_df=hier_grouped_df_pl,
    )

    evaluation = evaluate(
        reconciled_data,
        metrics=[partial(ufl.mase, seasonality=4)],
        train_df=grouped_data["hier_grouped_df"],
        tags=grouped_data["tags_grouped"],
    )

    pd.testing.assert_frame_equal(evaluation, evaluation_pl.to_pandas())


def test_evaluation_h1(reconciled_data, grouped_data):
    """Test that evaluation works for h=1."""
    evaluator = HierarchicalEvaluation([mase])
    evaluation_old = evaluator.evaluate(
        Y_hat_df=reconciled_data.groupby("unique_id").tail(1).drop(columns="y"),
        Y_test_df=reconciled_data.groupby("unique_id").tail(1)[
            ["unique_id", "ds", "y"]
        ],
        tags=grouped_data["tags_grouped"],
        Y_df=grouped_data["hier_grouped_df"],
    )
    # test work for h=1
    evaluation = evaluate(
        reconciled_data.groupby("unique_id").tail(1),
        metrics=[partial(ufl.mase, seasonality=4)],
        train_df=grouped_data["hier_grouped_df"],
        tags=grouped_data["tags_grouped"],
    )

    pd.testing.assert_frame_equal(evaluation_old, evaluation)


def test_evaluation_h1_polars(reconciled_data, grouped_data):
    """Test that evaluation works for h=1 with polars."""
    evaluator = HierarchicalEvaluation([mase])
    evaluation_old = evaluator.evaluate(
        Y_hat_df=reconciled_data.groupby("unique_id").tail(1).drop(columns="y"),
        Y_test_df=reconciled_data.groupby("unique_id").tail(1)[
            ["unique_id", "ds", "y"]
        ],
        tags=grouped_data["tags_grouped"],
        Y_df=grouped_data["hier_grouped_df"],
    )

    # polars
    # test work for h=1
    reconciled_pl = pl.from_pandas(reconciled_data)
    evaluation_pl = evaluate(
        reconciled_pl.group_by("unique_id").tail(1),
        metrics=[partial(ufl.mase, seasonality=4)],
        train_df=pl.from_pandas(grouped_data["hier_grouped_df"]),
        tags=grouped_data["tags_grouped"],
    )

    pd.testing.assert_frame_equal(evaluation_old, evaluation_pl.to_pandas())


@pytest.fixture
def statsforecast_data(tourism_df, hiers_grouped):
    """Fixture to provide StatsForecast processed data."""

    # Load TourismSmall dataset
    df = tourism_df.copy()
    qs = df["ds"].str.replace(r"(\d+) (Q\d)", r"\1-\2", regex=True)
    df["ds"] = pd.PeriodIndex(qs, freq="Q").to_timestamp()

    # Create hierarchical series based on geographic levels and purpose
    # And Convert quarterly ds string to pd.datetime format

    Y_df, S_df, tags = aggregate(df=df, spec=hiers_grouped)

    # Split train/test sets
    Y_test_df = Y_df.groupby("unique_id").tail(8)
    Y_train_df = Y_df.drop(Y_test_df.index)

    # Compute base auto-ETS predictions
    # Careful identifying correct data freq, this data quarterly 'Q'
    fcst = StatsForecast(
        models=[AutoETS(season_length=4, model="ZZA")], freq="QS", n_jobs=-1
    )
    Y_hat_df = fcst.forecast(df=Y_train_df, h=8, fitted=True)
    Y_fitted_df = fcst.forecast_fitted_values()

    reconcilers = [
        BottomUp(),
        MinTrace(method="ols"),
        MinTrace(method="mint_shrink"),
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)

    # Evaluate
    eval_tags = {}
    eval_tags["Total"] = tags["Country"]
    eval_tags["Purpose"] = tags["Country/Purpose"]
    eval_tags["State"] = tags["Country/State"]
    eval_tags["Regions"] = tags["Country/State/Region"]
    eval_tags["Bottom"] = tags["Country/State/Region/Purpose"]

    return {
        "Y_rec_df": Y_rec_df,
        "Y_test_df": Y_test_df,
        "Y_train_df": Y_train_df,
        "eval_tags": eval_tags,
        "Y_hat_df": Y_hat_df,
        "Y_fitted_df": Y_fitted_df,
        "S_df": S_df,
        "tags": tags,
        "hierarchy_levels": hiers_grouped,
        "df": df,
    }


def test_statsforecast_evaluation_mase(statsforecast_data):
    """Test StatsForecast evaluation with MASE metric."""
    evaluator = HierarchicalEvaluation(evaluators=[mase, rmse])
    evaluation_old = evaluator.evaluate(
        Y_hat_df=statsforecast_data["Y_rec_df"],
        Y_test_df=statsforecast_data["Y_test_df"],
        tags=statsforecast_data["eval_tags"],
        Y_df=statsforecast_data["Y_train_df"],
    )
    numeric_cols = evaluation_old.select_dtypes(include="number").columns
    evaluation_old[numeric_cols] = (
        evaluation_old[numeric_cols].map("{:.2f}".format).astype(np.float64)
    )

    evaluation_check = pd.DataFrame(
        {
            "level": ["Total", "Purpose", "State", "Regions", "Bottom", "Overall"],
            "metric": 6 * ["mase"],
            "AutoETS": [1.59, 1.32, 1.39, 1.12, 0.98, 1.02],
            "AutoETS/BottomUp": [3.16, 2.28, 1.90, 1.19, 0.98, 1.06],
        }
    )

    pd.testing.assert_frame_equal(
        evaluation_old.query("metric == 'mase'")[
            ["level", "metric", "AutoETS", "AutoETS/BottomUp"]
        ].reset_index(drop=True),
        evaluation_check,
    )


def test_statsforecast_evaluate_function(statsforecast_data):
    """Test the evaluate function with StatsForecast data."""
    Y_rec_df_with_y = statsforecast_data["Y_rec_df"].merge(
        statsforecast_data["Y_test_df"], on=["unique_id", "ds"], how="left"
    )
    mase_p = partial(ufl.mase, seasonality=4)

    evaluation = evaluate(
        Y_rec_df_with_y,
        metrics=[mase_p, ufl.rmse],
        tags=statsforecast_data["eval_tags"],
        train_df=statsforecast_data["Y_train_df"],
    )

    numeric_cols = evaluation.select_dtypes(include="number").columns
    evaluation[numeric_cols] = (
        evaluation[numeric_cols].map("{:.2f}".format).astype(np.float64)
    )

    evaluation_check = pd.DataFrame(
        {
            "level": ["Total", "Purpose", "State", "Regions", "Bottom", "Overall"],
            "metric": 6 * ["mase"],
            "AutoETS": [1.59, 1.32, 1.39, 1.12, 0.98, 1.02],
            "AutoETS/BottomUp": [3.16, 2.28, 1.90, 1.19, 0.98, 1.06],
        }
    )

    pd.testing.assert_frame_equal(
        evaluation.query("metric == 'mase'")[
            ["level", "metric", "AutoETS", "AutoETS/BottomUp"]
        ].reset_index(drop=True),
        evaluation_check,
    )

    # Compare with old evaluation method
    evaluator = HierarchicalEvaluation(evaluators=[mase, rmse])
    evaluation_old = evaluator.evaluate(
        Y_hat_df=statsforecast_data["Y_rec_df"],
        Y_test_df=statsforecast_data["Y_test_df"],
        tags=statsforecast_data["eval_tags"],
        Y_df=statsforecast_data["Y_train_df"],
    )
    evaluation_old[numeric_cols] = (
        evaluation_old[numeric_cols].map("{:.2f}".format).astype(np.float64)
    )

    pd.testing.assert_frame_equal(evaluation_old, evaluation)


def test_polars_statsforecast_evaluation(statsforecast_data):
    """Test if polars gives equal results to pandas for StatsForecast evaluation."""

    # Test if polars gives equal results to pandas
    df_pl = pl.from_pandas(statsforecast_data["df"])
    Y_df, S_df, tags = aggregate(df=df_pl, spec=statsforecast_data["hierarchy_levels"])

    # Split train/test sets
    Y_test_df = Y_df.group_by("unique_id").tail(8)
    Y_train_df = Y_df.group_by("unique_id").head(72)

    # Compute base auto-ETS predictions
    # Careful identifying correct data freq, this data quarterly 'Q'
    fcst = StatsForecast(
        models=[AutoETS(season_length=4, model="ZZA")], freq="1q", n_jobs=-1
    )
    Y_hat_df_pl = fcst.forecast(df=Y_train_df, h=8, fitted=True)
    Y_fitted_df = fcst.forecast_fitted_values()

    # Reconcile the base predictions
    reconcilers = [BottomUp(), MinTrace(method="ols"), MinTrace(method="mint_shrink")]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df_pl = hrec.reconcile(
        Y_hat_df=Y_hat_df_pl, Y_df=Y_fitted_df, S=S_df, tags=tags
    )

    # Evaluate
    Y_rec_df_pl_with_y = Y_rec_df_pl.join(Y_test_df, on=["unique_id", "ds"], how="left")
    mase_p = partial(ufl.mase, seasonality=4)
    evaluation_pl = evaluate(
        Y_rec_df_pl_with_y,
        metrics=[mase_p, ufl.rmse],
        tags=statsforecast_data["eval_tags"],
        train_df=Y_train_df,
    )

    # Handle numeric columns properly for polars
    if hasattr(evaluation_pl, "select_dtypes"):
        # pandas DataFrame
        numeric_cols = evaluation_pl.select_dtypes(include="number").columns
        evaluation_pl[numeric_cols] = (
            evaluation_pl[numeric_cols].map("{:.2f}".format).astype(np.float64)
        )
    else:
        # polars DataFrame
        numeric_cols = [
            col
            for col in evaluation_pl.columns
            if evaluation_pl[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        evaluation_pl = evaluation_pl.with_columns(
            pl.col(numeric_cols).round(2).cast(pl.Float64)
        )

    # Check if polars gives identical results as pandas
    pd.testing.assert_frame_equal(
        Y_hat_df_pl.to_pandas(), statsforecast_data["Y_hat_df"]
    )
    pd.testing.assert_frame_equal(
        Y_rec_df_pl.to_pandas(), statsforecast_data["Y_rec_df"]
    )

    # Compare evaluation results
    Y_rec_df_with_y = statsforecast_data["Y_rec_df"].merge(
        statsforecast_data["Y_test_df"], on=["unique_id", "ds"], how="left"
    )
    evaluation = evaluate(
        Y_rec_df_with_y,
        metrics=[mase_p, ufl.rmse],
        tags=statsforecast_data["eval_tags"],
        train_df=statsforecast_data["Y_train_df"],
    )

    # Format numeric columns for comparison
    if hasattr(evaluation, "select_dtypes"):
        numeric_cols_pandas = evaluation.select_dtypes(include="number").columns
        evaluation[numeric_cols_pandas] = (
            evaluation[numeric_cols_pandas].map("{:.2f}".format).astype(np.float64)
        )

    # Convert polars to pandas for comparison
    if hasattr(evaluation_pl, "to_pandas"):
        evaluation_pl_pandas = evaluation_pl.to_pandas()
    else:
        evaluation_pl_pandas = evaluation_pl

    pd.testing.assert_frame_equal(evaluation_pl_pandas, evaluation)
