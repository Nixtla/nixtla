import sys
from functools import partial
from itertools import product

import dask.dataframe as dd
import datasetsforecast.losses as ds_losses
import fugue.api as fa
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import polars.testing
import pytest
from datasetsforecast.evaluation import accuracy as ds_evaluate
from pyspark.sql import SparkSession

import utilsforecast.processing as ufp
from utilsforecast.data import generate_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import (
    bias,
    calibration,
    cfe,
    coverage,
    linex,
    mae,
    mape,
    mase,
    mqloss,
    mse,
    msse,
    nd,
    pis,
    quantile_loss,
    rmae,
    rmse,
    rmsse,
    scaled_crps,
    scaled_mqloss,
    scaled_quantile_loss,
    smape,
    spis,
    tweedie_deviance,
)


@pytest.fixture
def setup_series():
    series = generate_series(10, n_models=2, level=[80, 95])
    series["unique_id"] = series["unique_id"].astype("int")
    return series


@pytest.fixture
def setup_models():
    return ["model0", "model1"]


@pytest.fixture
def setup_metrics():
    return [
        mae,
        mse,
        rmse,
        mape,
        smape,
        partial(mase, seasonality=7),
        quantile_loss,
        mqloss,
        coverage,
        calibration,
        scaled_crps,
    ]


def generate_cv_series(n_series=5, n_models=2, level=None, n_cutoffs=3, engine="pandas", seed=0):
    """Generate cross-validation data with multiple cutoff dates.

    Uses utilsforecast.data.generate_series and utilsforecast.processing.backtest_splits
    to create realistic cross-validation scenarios. Uses utilsforecast.processing for
    engine-agnostic operations.

    Args:
        n_series: Number of time series
        n_models: Number of model predictions to generate
        level: List of confidence levels for prediction intervals
        n_cutoffs: Number of cross-validation folds
        engine: 'pandas' or 'polars'
        seed: Random seed

    Returns:
        tuple: (cv_df, train_df) where both dataframes include a cutoff column.
            cv_df contains predictions with cutoff information.
            train_df contains training data for each cutoff (for per-fold scale computation).
    """
    from utilsforecast.data import generate_series
    from utilsforecast.processing import backtest_splits
    import utilsforecast.processing as ufp

    rng = np.random.RandomState(seed)

    # Parameters for cross-validation
    h = 10  # forecast horizon
    # Generate enough data for n_cutoffs windows with step_size = h
    # Need: training data + h * n_cutoffs
    min_length = 50 + h * n_cutoffs
    max_length = min_length + 20

    # Frequency format differs between pandas ("D") and polars ("1d")
    freq = "D" if engine == "pandas" else "1d"

    # Generate base time series (without model predictions)
    df = generate_series(
        n_series=n_series,
        freq=freq,
        min_length=min_length,
        max_length=max_length,
        equal_ends=True,  # All series end at the same time for CV
        engine=engine,
        seed=seed
    )

    # Perform cross-validation splits
    cv_list = []
    train_list = []

    for cutoffs_df, train, valid in backtest_splits(
        df=df,
        n_windows=n_cutoffs,
        h=h,
        id_col="unique_id",
        time_col="ds",
        freq=freq,
    ):
        # Join cutoffs with validation data using ufp.join
        valid_with_cutoff = ufp.join(valid, cutoffs_df, on="unique_id", how="left")

        # Get y values as numpy array using narwhals
        y_values = nw.from_native(valid_with_cutoff)["y"].to_numpy()

        # Add model predictions using ufp.assign_columns
        for model_idx in range(n_models):
            rand_factors = rng.rand(len(y_values)) * 0.2 + 0.9
            predictions = y_values * rand_factors
            valid_with_cutoff = ufp.assign_columns(
                valid_with_cutoff, f"model{model_idx}", predictions
            )

        # Add prediction intervals if level is specified
        if level:
            for model_idx in range(n_models):
                # Get model predictions as numpy array
                model_preds = nw.from_native(valid_with_cutoff)[f"model{model_idx}"].to_numpy()
                for lv in level:
                    lv_rands = 0.5 * rng.rand(len(model_preds)) * lv / 100
                    lo_preds = model_preds * (1 - lv_rands)
                    hi_preds = model_preds * (1 + lv_rands)
                    valid_with_cutoff = ufp.assign_columns(
                        valid_with_cutoff, f"model{model_idx}-lo-{lv}", lo_preds
                    )
                    valid_with_cutoff = ufp.assign_columns(
                        valid_with_cutoff, f"model{model_idx}-hi-{lv}", hi_preds
                    )

        cv_list.append(valid_with_cutoff)

        # Add cutoff information to training data for per-fold scale computation
        train_with_cutoff = ufp.join(train, cutoffs_df, on="unique_id", how="left")
        train_list.append(train_with_cutoff)

    # Concatenate all validation folds using ufp.vertical_concat
    # Set match_categories=False since we have more than 2 dataframes
    cv_df = ufp.vertical_concat(cv_list, match_categories=False)

    # Concatenate all training folds with cutoff information
    train_df = ufp.vertical_concat(train_list, match_categories=False)

    return cv_df, train_df


def test_evaluate(setup_series, setup_models, setup_metrics):
    evaluation = evaluate(
        setup_series,
        metrics=setup_metrics,
        models=setup_models,
        train_df=setup_series,
        level=[80, 95],
    )
    summary = (
        evaluation.drop(columns="unique_id").groupby("metric").mean().reset_index()
    )

    series_pl = generate_series(10, n_models=2, level=[80, 95], engine="polars")
    pl_evaluation = evaluate(
        series_pl,
        metrics=setup_metrics,
        train_df=series_pl,
        level=[80, 95],
    ).drop("unique_id")
    pl_summary = ufp.group_by(pl_evaluation, "metric").mean()
    pd.testing.assert_frame_equal(
        summary.sort_values("metric"),
        pl_summary.sort("metric").to_pandas(),
    )
    pl.testing.assert_frame_equal(
        evaluate(
            series_pl,
            metrics=setup_metrics,
            train_df=series_pl,
            level=[80, 95],
            agg_fn="mean",
        ).sort("metric"),
        pl_summary.sort("metric"),
    )


def daily_mase(y, y_hat, y_train):
    return ds_losses.mase(y, y_hat, y_train, seasonality=7)


def test_datasets_evaluate(setup_series, setup_models, setup_metrics):
    level = [80, 95]
    for agg_fn in [None, "mean"]:
        uf_res = evaluate(
            setup_series,
            metrics=setup_metrics,
            models=setup_models,
            train_df=setup_series,
            level=level,
            agg_fn=agg_fn,
        )
        agg_by = None if agg_fn == "mean" else ["unique_id"]
        ds_res = ds_evaluate(
            setup_series,
            metrics=[
                ds_losses.mae,
                ds_losses.mse,
                ds_losses.rmse,
                ds_losses.mape,
                daily_mase,
                ds_losses.smape,
                ds_losses.quantile_loss,
                ds_losses.mqloss,
                ds_losses.coverage,
                ds_losses.calibration,
                ds_losses.scaled_crps,
            ],
            level=level,
            Y_df=setup_series,
            agg_by=agg_by,
        )
        ds_res["metric"] = ds_res["metric"].str.replace("-", "_")
        ds_res["metric"] = ds_res["metric"].str.replace("q_", "q")
        ds_res["metric"] = ds_res["metric"].str.replace("lv_", "level")
        ds_res["metric"] = ds_res["metric"].str.replace("daily_mase", "mase")
        # utils doesn't multiply pct metrics by 100
        ds_res.loc[
            ds_res["metric"].str.startswith("coverage"), ["model0", "model1"]
        ] /= 100
        ds_res.loc[ds_res["metric"].eq("mape"), ["model0", "model1"]] /= 100
        # we report smape between 0 and 1 instead of 0-200
        ds_res.loc[ds_res["metric"].eq("smape"), ["model0", "model1"]] /= 200

        ds_res = ds_res[uf_res.columns]
        if agg_fn is None:
            ds_res = ds_res.sort_values(["unique_id", "metric"])
            uf_res = uf_res.sort_values(["unique_id", "metric"])
        else:
            ds_res = ds_res.sort_values("metric")
            uf_res = uf_res.sort_values("metric")

        pd.testing.assert_frame_equal(
            uf_res.reset_index(drop=True),
            ds_res.reset_index(drop=True),
        )

@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_distributed_evaluate(setup_series):
    level = [80, 95]
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("FATAL")
    dask_df = dd.from_pandas(setup_series, npartitions=2)
    spark_df = spark.createDataFrame(setup_series).repartition(2)
    for distributed_df, use_train in product([dask_df, spark_df], [True, False]):
        distr_metrics = [rmse, mae]
        if use_train:
            distr_metrics.append(partial(mase, seasonality=7))
            local_train = setup_series
            distr_train = distributed_df
        else:
            local_train = None
            distr_train = None
        local_res = evaluate(
            setup_series, metrics=distr_metrics, level=level, train_df=local_train
        )
        distr_res = fa.as_fugue_df(
            evaluate(
                distributed_df,
                metrics=distr_metrics,
                level=level,
                train_df=distr_train,
            )
        ).as_pandas()
        pd.testing.assert_frame_equal(
            local_res.sort_values(["unique_id", "metric"]).reset_index(drop=True),
            distr_res.sort_values(["unique_id", "metric"]).reset_index(drop=True),
            check_dtype=False,
        )


# ========================================
# Cross-Validation Tests with Cutoff Column
# ========================================


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_losses_with_cutoff(engine):
    """Test basic losses (no train_df required) with cross-validation cutoff column."""
    cv_df, train_df = generate_cv_series(n_series=5, n_models=2, n_cutoffs=3, engine=engine, seed=42)
    models = ["model0", "model1"]

    # Test each basic loss function directly
    losses = [
        ("mae", mae),
        ("mse", mse),
        ("rmse", rmse),
        ("bias", bias),
        ("cfe", cfe),
        ("pis", pis),
        ("mape", mape),
        ("smape", smape),
        ("nd", nd),
        ("tweedie_deviance", partial(tweedie_deviance, power=1.5)),
        ("linex", partial(linex, a=1.0)),
        ("spis", partial(spis, train_df=train_df)),
        ("mase", partial(mase, seasonality=7, train_df=train_df)),
        ("msse", partial(msse, seasonality=7, train_df=train_df)),
        ("rmsse", partial(rmsse, seasonality=7, train_df=train_df)),
    ]

    losses_eval = [
        ("mae", mae),
        ("mse", mse),
        ("rmse", rmse),
        ("bias", bias),
        ("cfe", cfe),
        ("pis", pis),
        ("mape", mape),
        ("smape", smape),
        ("nd", nd),
        ("tweedie_deviance", partial(tweedie_deviance, power=1.5)),
        ("linex", partial(linex, a=1.0)),
        ("spis", spis),
        ("mase", partial(mase, seasonality=7)),
        ("msse", partial(msse, seasonality=7)),
        ("rmsse", partial(rmsse, seasonality=7)),
    ]

    cv_nw = nw.from_native(cv_df)
    n_cutoffs = cv_nw["cutoff"].n_unique()
    n_series = cv_nw["unique_id"].n_unique()
    expected_rows = n_cutoffs * n_series

    # Store individual loss results for later comparison with evaluate()
    loss_results = {}

    for loss_name, loss_fn in losses:

        test_models = ["model1"] if loss_name == "rmae" else models
        result = loss_fn(
            df=cv_df,
            models=test_models,
            id_col="unique_id",
            target_col="y",
            cutoff_col="cutoff"
        )

        # Store result for later comparison
        loss_results[loss_name] = result

        result_nw = nw.from_native(result)

        # Verify result has correct shape
        assert result_nw.shape[0] == expected_rows, (
            f"{loss_name}: Expected {expected_rows} rows (cutoff × series), got {result_nw.shape[0]}"
        )

        # Verify cutoff column is present
        assert "cutoff" in result_nw.columns, f"{loss_name}: cutoff column missing from result"

        # Verify models columns are present
        for model in test_models:
            assert model in result_nw.columns, f"{loss_name}: {model} column missing from result"

        # Verify no NaN values
        for model in test_models:
            assert result_nw[model].null_count() == 0, f"{loss_name}: NaN values in {model}"
    # Test via evaluate() function with all basic losses
    all_metrics = [loss_fn for _, loss_fn in losses_eval]
    evaluation = evaluate(
        df=cv_df,
        metrics=all_metrics,
        models=models,
        cutoff_col="cutoff",
        train_df = train_df
    )

    eval_nw = nw.from_native(evaluation)

    # Verify evaluation has correct structure
    assert "cutoff" in eval_nw.columns, "evaluate(): cutoff column missing"
    assert "metric" in eval_nw.columns, "evaluate(): metric column missing"

    # Should have one row per (cutoff, unique_id, metric) combination
    n_metrics = len(losses)
    expected_rows_eval = n_cutoffs * n_series * n_metrics
    assert eval_nw.shape[0] == expected_rows_eval, (
        f"evaluate(): Expected {expected_rows_eval} rows, got {eval_nw.shape[0]}"
    )

    # Verify that evaluate() produces the same results as calling loss functions directly
    for loss_name, loss_fn in losses:
        # Get the direct result from loss_results
        direct_result = loss_results[loss_name]
        direct_nw = nw.from_native(direct_result).sort("unique_id", "cutoff")

        # Get the corresponding rows from evaluation
        eval_subset = nw.from_native(evaluation)
        eval_subset = eval_subset.filter(nw.col("metric") == loss_name)
        eval_subset = eval_subset.drop("metric").sort("unique_id", "cutoff")

        # Compare each model's values
        for model in models:
            direct_values = direct_nw[model].to_numpy()
            eval_values = eval_subset[model].to_numpy()

            assert np.allclose(direct_values, eval_values, rtol=1e-10), (
                f"evaluate() result for {loss_name} doesn't match direct {loss_name}() call for {model}"
            )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_quantile_losses_with_cutoff(engine):
    """Test quantile/probabilistic losses with cross-validation cutoff column."""
    level = [80, 95]
    cv_df, train_df = generate_cv_series(
        n_series=5, n_models=2, level=level, n_cutoffs=3, engine=engine, seed=42
    )
    models = ["model0", "model1"]

    cv_nw = nw.from_native(cv_df)
    n_cutoffs = cv_nw["cutoff"].n_unique()
    n_series = cv_nw["unique_id"].n_unique()
    expected_rows = n_cutoffs * n_series

    # Test quantile_loss (requires dict of models)
    for lv in level:
        quantiles = [lv / 200, 1 - lv / 200]
        for q, side in zip(quantiles, ["lo", "hi"]):
            model_dict = {m: f"{m}-{side}-{lv}" for m in models}
            result = quantile_loss(
                df=cv_df,
                models=model_dict,
                q=q,
                id_col="unique_id",
                target_col="y",
                cutoff_col="cutoff"
            )

            result_nw = nw.from_native(result)
            assert result_nw.shape[0] == expected_rows, f"quantile_loss: Expected {expected_rows} rows"
            assert "cutoff" in result_nw.columns, "quantile_loss: cutoff column missing"

    # Test scaled_quantile_loss
    for lv in level:
        quantiles = [lv / 200, 1 - lv / 200]
        for q, side in zip(quantiles, ["lo", "hi"]):
            model_dict = {m: f"{m}-{side}-{lv}" for m in models}
            result = scaled_quantile_loss(
                df=cv_df,
                models=model_dict,
                seasonality=7,
                train_df=train_df,
                q=q,
                id_col="unique_id",
                target_col="y",
                cutoff_col="cutoff"
            )

            result_nw = nw.from_native(result)
            assert result_nw.shape[0] == expected_rows, f"scaled_quantile_loss: Expected {expected_rows} rows"
            assert "cutoff" in result_nw.columns, "scaled_quantile_loss: cutoff column missing"

    # Test coverage
    for lv in level:
        result = coverage(
            df=cv_df,
            models=models,
            level=lv,
            id_col="unique_id",
            target_col="y",
            cutoff_col="cutoff"
        )

        result_nw = nw.from_native(result)
        assert result_nw.shape[0] == expected_rows, f"coverage: Expected {expected_rows} rows"
        assert "cutoff" in result_nw.columns, "coverage: cutoff column missing"

    # Test calibration
    for lv in level:
        quantiles = [lv / 200, 1 - lv / 200]
        for q, side in zip(quantiles, ["lo", "hi"]):
            model_dict = {m: f"{m}-{side}-{lv}" for m in models}
            result = calibration(
                df=cv_df,
                models=model_dict,
                id_col="unique_id",
                target_col="y",
                cutoff_col="cutoff"
            )

            result_nw = nw.from_native(result)
            assert result_nw.shape[0] == expected_rows, f"calibration: Expected {expected_rows} rows"
            assert "cutoff" in result_nw.columns, "calibration: cutoff column missing"

    # Test mqloss with sorted quantiles first to establish baseline
    quantiles_sorted = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # Create model dict for mqloss (maps model names to lists of quantile predictions)
    # Generate predictions based on quantile VALUE (lower quantiles = lower predictions)
    rng = np.random.RandomState(42)
    cv_nw = nw.from_native(cv_df)
    model_dict_sorted = {}
    for m in models:
        quantile_cols = []
        model_preds = cv_nw[m].to_numpy()
        for i, q in enumerate(quantiles_sorted):
            col_name = f"{m}_sorted_q{i}"
            # Predictions increase with quantile value
            quantile_factor = 0.7 + 0.6 * q + rng.rand(len(model_preds)) * 0.1
            quantile_preds = model_preds * quantile_factor
            cv_df = ufp.assign_columns(cv_df, col_name, quantile_preds)
            quantile_cols.append(col_name)
        model_dict_sorted[m] = quantile_cols

    result_sorted = mqloss(
        df=cv_df,
        models=model_dict_sorted,
        quantiles=quantiles_sorted,
        id_col="unique_id",
        target_col="y",
        cutoff_col="cutoff"
    )

    # Now test with UNSORTED quantiles but same predictions in shuffled order
    # This tests that mqloss correctly sorts quantiles internally
    quantiles_unsorted = np.array([0.9, 0.8, 0.7, 0.4, 0.5, 0.6, 0.3, 0.2, 0.1])
    # Map unsorted quantiles to their corresponding sorted columns
    model_dict_unsorted = {}
    for m in models:
        unsorted_cols = []
        for q_unsorted in quantiles_unsorted:
            # Find index of this quantile in sorted array
            sorted_idx = np.where(quantiles_sorted == q_unsorted)[0][0]
            unsorted_cols.append(f"{m}_sorted_q{sorted_idx}")
        model_dict_unsorted[m] = unsorted_cols

    result_unsorted = mqloss(
        df=cv_df,
        models=model_dict_unsorted,
        quantiles=quantiles_unsorted,
        id_col="unique_id",
        target_col="y",
        cutoff_col="cutoff"
    )

    # Results should be identical regardless of quantile order
    result_sorted_nw = nw.from_native(result_sorted)
    result_unsorted_nw = nw.from_native(result_unsorted)

    assert result_sorted_nw.shape[0] == expected_rows, f"mqloss: Expected {expected_rows} rows"
    assert "cutoff" in result_sorted_nw.columns, "mqloss: cutoff column missing"

    # Sort both results by id and cutoff to ensure same row order for comparison
    result_sorted_nw = result_sorted_nw.sort(["unique_id", "cutoff"])
    result_unsorted_nw = result_unsorted_nw.sort(["unique_id", "cutoff"])

    # Verify both results are identical (sorted internally)
    for model in models:
        sorted_vals = result_sorted_nw[model].to_numpy()
        unsorted_vals = result_unsorted_nw[model].to_numpy()
        np.testing.assert_allclose(
            sorted_vals, unsorted_vals,
            err_msg=f"mqloss results differ for {model} with sorted vs unsorted quantiles"
        )

    # Test scaled_mqloss (uses sorted quantiles)
    result = scaled_mqloss(
        df=cv_df,
        models=model_dict_sorted,
        quantiles=quantiles_sorted,
        seasonality=7,
        train_df=train_df,
        id_col="unique_id",
        target_col="y",
        cutoff_col="cutoff"
    )

    result_nw = nw.from_native(result)
    assert result_nw.shape[0] == expected_rows, f"scaled_mqloss: Expected {expected_rows} rows"
    assert "cutoff" in result_nw.columns, "scaled_mqloss: cutoff column missing"

    # Test scaled_crps (uses sorted quantiles)
    result = scaled_crps(
        df=cv_df,
        models=model_dict_sorted,
        quantiles=quantiles_sorted,
        id_col="unique_id",
        target_col="y",
        cutoff_col="cutoff"
    )

    result_nw = nw.from_native(result)
    assert result_nw.shape[0] == expected_rows, f"scaled_crps: Expected {expected_rows} rows"
    assert "cutoff" in result_nw.columns, "scaled_crps: cutoff column missing"

    # Test via evaluate() function
    evaluation = evaluate(
        df=cv_df,
        metrics=[
            quantile_loss,
            mqloss,
            coverage,
            calibration,
            scaled_crps,
        ],
        models=models,
        train_df=train_df,
        level=level,
        cutoff_col="cutoff"
    )

    eval_nw = nw.from_native(evaluation)

    # Verify evaluation has correct structure
    assert "cutoff" in eval_nw.columns, "evaluate(): cutoff column missing"
    assert "metric" in eval_nw.columns, "evaluate(): metric column missing"

    # Should have rows for each metric-quantile combination
    assert eval_nw.shape[0] > 0, "evaluate(): No results returned"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_multiple_cutoffs_consistency(engine):
    """Test that results are consistent and correctly grouped with multiple cutoffs."""
    models = ["model0", "model1"]

    # Test with different numbers of cutoffs
    for n_cutoffs in [1, 2, 3, 5]:
        cv_df, train_df = generate_cv_series(
            n_series=5, n_models=2, n_cutoffs=n_cutoffs, engine=engine, seed=42
        )

        # Test mae as representative
        result = mae(
            df=cv_df,
            models=models,
            id_col="unique_id",
            target_col="y",
            cutoff_col="cutoff"
        )

        # Verify correct number of rows
        cv_nw = nw.from_native(cv_df)
        result_nw = nw.from_native(result)
        n_series = cv_nw["unique_id"].n_unique()
        expected_rows = n_cutoffs * n_series
        assert result_nw.shape[0] == expected_rows, (
            f"With {n_cutoffs} cutoffs: Expected {expected_rows} rows, got {result_nw.shape[0]}"
        )

        # Verify each cutoff has results for all series
        for (cutoff,), cutoff_group in result_nw.group_by("cutoff"):
            assert cutoff_group.shape[0] == n_series, (
                f"Cutoff {cutoff} should have {n_series} series, got {cutoff_group.shape[0]}"
            )

        # Verify each series has results for all cutoffs
        for (uid,), uid_group in result_nw.group_by("unique_id"):
            assert uid_group.shape[0] == n_cutoffs, (
                f"Series {uid} should have {n_cutoffs} cutoffs, got {uid_group.shape[0]}"
            )

    # Test that different cutoffs produce different results
    cv_df, train_df = generate_cv_series(n_series=5, n_models=2, n_cutoffs=3, engine=engine, seed=42)
    result = mae(
        df=cv_df,
        models=models,
        id_col="unique_id",
        target_col="y",
        cutoff_col="cutoff"
    )

    # Group by cutoff and compute mean using narwhals
    result_nw = nw.from_native(result)
    cutoff_means = result_nw.group_by("cutoff").agg([nw.col(m).mean() for m in models])

    # Verify that at least some cutoffs have different mean values
    # (they should differ because we're forecasting different time periods)
    for model in models:
        unique_means = cutoff_means[model].n_unique()
        # At least 2 different mean values across cutoffs
        assert unique_means >= 2, (
            f"Expected different results across cutoffs for {model}, "
            f"but got {unique_means} unique mean values"
        )

    # Test via evaluate() with aggregation
    evaluation = evaluate(
        df=cv_df,
        metrics=[mae, mse, rmse],
        models=models,
        cutoff_col="cutoff"
    )

    eval_nw = nw.from_native(evaluation)
    # Should have cutoff in results when not aggregated
    assert "cutoff" in eval_nw.columns, "cutoff column should be in evaluation results"

    # Test with aggregation - cutoff should still be preserved
    evaluation_agg = evaluate(
        df=cv_df,
        metrics=[mae],
        models=models,
        cutoff_col="cutoff",
        agg_fn="mean"
    )

    eval_agg_nw = nw.from_native(evaluation_agg)
    # With agg_fn, we aggregate across unique_id but not across cutoff
    # So we should have one row per metric (no unique_id column)
    assert "unique_id" not in eval_agg_nw.columns, "unique_id should be aggregated out"
    assert "metric" in eval_agg_nw.columns, "metric column should be present"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_cutoff_aggregation_equivalence(engine):
    """Test that dropping cutoff vs keeping cutoff then averaging produces different results.

    For linear non-scaled metrics (mae, mse):
        - Dropping cutoff then evaluating should equal evaluating with cutoff then averaging
        - This property holds because these metrics use linear averaging

    For scaled metrics (mase, msse, rmsse, spis):
        - These two approaches should produce DIFFERENT results
        - When train_df has cutoff column, each cutoff uses its own training window
        - This produces different scales per cutoff, leading to different aggregated values
        - Scenario A (drop cutoff): All data pooled → one scale per series
        - Scenario B (keep cutoff then average): Different scales per cutoff → different result

    Note: RMSE does not have the equivalence property because sqrt is non-linear.
    """
    # Generate CV data with multiple cutoffs
    cv_df, train_df = generate_cv_series(n_series=5, n_models=2, n_cutoffs=3, engine=engine)
    models = ["model0", "model1"]

    # ========== Test Non-Scaled Metrics ==========
    # These should be equivalent when dropping cutoff vs keeping cutoff then aggregating
    # Format: (metric_name, metric_function, aggregation_type)
    # - "mean" aggregation: drop-then-eval == eval-then-mean
    # - "sum" aggregation: drop-then-eval == eval-then-sum

    equivalent_metrics = [
        # Mean-aggregated metrics
        ("mae", mae, "mean"),
        ("mse", mse, "mean"),
        ("bias", bias, "mean"),
        ("mape", mape, "mean"),
        ("smape", smape, "mean"),
        ("tweedie_deviance", partial(tweedie_deviance, power=1.5), "mean"),
        ("linex", partial(linex, a=1.0), "mean"),
        # Sum-aggregated metrics
        ("cfe", cfe, "sum"),
        ("pis", pis, "sum"),
    ]

    for metric_name, metric, agg_type in equivalent_metrics:
        # Scenario A: Drop cutoff column, then evaluate
        # This aggregates over all forecasts as if it's one big evaluation
        cv_df_no_cutoff = ufp.drop_columns(cv_df, "cutoff")
        eval_no_cutoff = metric(
            df=cv_df_no_cutoff,
            models=models,
            id_col="unique_id",
            target_col="y"
        )
        eval_no_cutoff_nw = nw.from_native(eval_no_cutoff)

        # Should have one row per series (no cutoff dimension)
        actual_n_series = nw.from_native(cv_df)["unique_id"].n_unique()
        assert eval_no_cutoff_nw.shape[0] == actual_n_series, (
            f"{metric_name}: Expected {actual_n_series} rows (one per series), "
            f"got {eval_no_cutoff_nw.shape[0]}"
        )

        # Scenario B: Keep cutoff, evaluate, then manually aggregate across cutoffs
        eval_with_cutoff = metric(
            df=cv_df,
            models=models,
            id_col="unique_id",
            target_col="y",
            cutoff_col="cutoff"
        )
        eval_with_cutoff_nw = nw.from_native(eval_with_cutoff)

        # Manually compute aggregation across cutoffs for each series
        if agg_type == "mean":
            eval_aggregated = (
                eval_with_cutoff_nw
                .group_by("unique_id")
                .agg([nw.col(m).mean().alias(m) for m in models])
                .sort("unique_id")
            )
        else:  # agg_type == "sum"
            eval_aggregated = (
                eval_with_cutoff_nw
                .group_by("unique_id")
                .agg([nw.col(m).sum().alias(m) for m in models])
                .sort("unique_id")
            )

        # Sort both for comparison
        eval_no_cutoff_sorted = eval_no_cutoff_nw.sort("unique_id")

        # For non-scaled metrics, these should be approximately equal
        for model in models:
            no_cutoff_values = eval_no_cutoff_sorted[model].to_numpy()
            aggregated_values = eval_aggregated[model].to_numpy()

            assert np.allclose(no_cutoff_values, aggregated_values, rtol=1e-10), (
                f"{metric_name} ({model}): {agg_type.capitalize()}-aggregated metric should be equivalent when "
                f"dropping cutoff vs keeping cutoff then {agg_type}ming.\n"
                f"Drop-then-eval: {no_cutoff_values}\n"
                f"Eval-then-{agg_type}: {aggregated_values}"
            )

    # ========== Test Scaled Metrics & Non-Linear Metrics ==========
    # These should be DIFFERENT when dropping cutoff vs keeping cutoff then averaging

    # Verify train_df has cutoff column for per-fold scale computation
    train_df_nw = nw.from_native(train_df)
    assert "cutoff" in train_df_nw.columns, (
        "train_df must have cutoff column for proper per-fold scale computation"
    )

    non_equivalent_metrics = [
        # Scaled metrics
        ("mase", partial(mase, seasonality=7)),
        ("msse", partial(msse, seasonality=7)),
        ("rmsse", partial(rmsse, seasonality=7)),
        ("spis", spis),
        # Other non equivalent metrics
        ("nd", nd),  # nd uses ratio-of-sums: sum(|errors|) / sum(|y|), not mean aggregation
    ]

    for metric_name, metric_fn in non_equivalent_metrics:
        # Scenario A: Drop cutoff column, then evaluate
        # This pools all data together, computing one scale per series
        cv_df_no_cutoff = ufp.drop_columns(cv_df, "cutoff")
        train_df_no_cutoff = ufp.drop_columns(train_df, "cutoff")

        # nd doesn't require train_df, other metrics do
        if metric_name == "nd":
            eval_no_cutoff = metric_fn(
                df=cv_df_no_cutoff,
                models=models,
                id_col="unique_id",
                target_col="y"
            )
        else:
            eval_no_cutoff = metric_fn(
                df=cv_df_no_cutoff,
                models=models,
                id_col="unique_id",
                target_col="y",
                train_df=train_df_no_cutoff
            )
        eval_no_cutoff_nw = nw.from_native(eval_no_cutoff)

        # Scenario B: Keep cutoff, evaluate, then manually average across cutoffs
        if metric_name == "nd":
            eval_with_cutoff = metric_fn(
                df=cv_df,
                models=models,
                id_col="unique_id",
                target_col="y",
                cutoff_col="cutoff"
            )
        else:
            eval_with_cutoff = metric_fn(
                df=cv_df,
                models=models,
                id_col="unique_id",
                target_col="y",
                train_df=train_df,
                cutoff_col="cutoff"
            )
        eval_with_cutoff_nw = nw.from_native(eval_with_cutoff)

        # Manually compute mean across cutoffs for each series
        eval_averaged = (
            eval_with_cutoff_nw
            .group_by("unique_id")
            .agg([nw.col(m).mean().alias(m) for m in models])
            .sort("unique_id")
        )

        # Sort both for comparison
        eval_no_cutoff_sorted = eval_no_cutoff_nw.sort("unique_id")

        # For scaled metrics, these should be DIFFERENT
        for model in models:
            no_cutoff_values = eval_no_cutoff_sorted[model].to_numpy()
            averaged_values = eval_averaged[model].to_numpy()

            # Check that they are NOT approximately equal (should differ significantly)
            are_different = not np.allclose(no_cutoff_values, averaged_values, rtol=1e-5)
            assert are_different, (
                f"{metric_name} ({model}): Scaled metric should produce DIFFERENT results "
                f"when dropping cutoff vs keeping cutoff then averaging.\n"
                f"Drop-then-eval: {no_cutoff_values}\n"
                f"Eval-then-average: {averaged_values}"
            )
