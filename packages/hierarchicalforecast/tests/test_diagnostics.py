import copy

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate


@pytest.fixture(scope="module")
def diagnostic_data(tourism_df, hiers_grouped):
    """Prepares data specifically for diagnostics testing with added noise."""
    df = copy.deepcopy(tourism_df)
    df["ds"] = pd.to_datetime(
        df["ds"].str.replace(r"(\d+) (Q\d)", r"\1-\2", regex=True)
    )

    # Pandas
    Y_df, S_df, tags = aggregate(df, hiers_grouped)
    Y_df["y_model"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(4).copy()
    ds_h = Y_hat_df["ds"].unique() # noqa: F841
    Y_train_df = Y_df.query("~(ds in @ds_h)").copy()

    # Add noise to base forecasts to make them incoherent
    rng = np.random.default_rng(42)
    Y_hat_df["y_model"] = Y_hat_df["y_model"] + rng.uniform(-50, 50, len(Y_hat_df))
    Y_train_df["y_model"] = Y_train_df["y_model"] + rng.uniform(
        -10, 10, len(Y_train_df)
    )

    # Polars
    Y_hat_df_pl = pl.from_pandas(Y_hat_df)
    Y_train_df_pl = pl.from_pandas(Y_train_df)
    S_df_pl = pl.from_pandas(S_df)

    return {
        "pandas": {
            "Y_hat_df": Y_hat_df,
            "Y_train_df": Y_train_df,
            "S_df": S_df,
            "tags": tags,
        },
        "polars": {
            "Y_hat_df": Y_hat_df_pl,
            "Y_train_df": Y_train_df_pl,
            "S_df": S_df_pl,
            "tags": tags,
        },
    }


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_basic_structure(diagnostic_data, lib):
    """Test that diagnostics DataFrame has correct structure."""
    data = diagnostic_data[lib]

    hrec = HierarchicalReconciliation([BottomUp(), MinTrace(method="ols")])
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        Y_df=data["Y_train_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
    )

    assert hasattr(hrec, "diagnostics")
    assert hrec.diagnostics is not None

    diag = nw.from_native(hrec.diagnostics)

    # Check required columns exist
    assert "level" in diag.columns
    assert "metric" in diag.columns
    assert "y_model/BottomUp" in diag.columns
    assert "y_model/MinTrace_method-ols" in diag.columns

    # Check expected metrics exist
    metrics = diag["metric"].unique().to_list()
    expected_metrics = [
        "coherence_residual_mae_before",
        "coherence_residual_mae_after",
        "adjustment_mae",
        "negative_count_before",
        "negative_count_after",
        "is_coherent",
        "coherence_max_violation",
    ]
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    # Check expected levels exist (including Overall)
    levels = diag["level"].unique().to_list()
    assert "Overall" in levels


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_coherence_after_reconciliation(diagnostic_data, lib):
    """Test that reconciled forecasts are coherent (residuals near zero)."""
    data = diagnostic_data[lib]

    hrec = HierarchicalReconciliation([BottomUp()])
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
    )

    diag = nw.from_native(hrec.diagnostics)

    # Filter for coherence_residual_mae_after
    after_mae = diag.filter(nw.col("metric") == "coherence_residual_mae_after")

    # All values should be near zero (within numerical tolerance)
    for val in after_mae["y_model/BottomUp"].to_list():
        assert val < 1e-6, f"Coherence residual after reconciliation too large: {val}"

    # is_coherent should be 1.0 for Overall level
    is_coherent = diag.filter(
        (nw.col("metric") == "is_coherent") & (nw.col("level") == "Overall")
    )
    assert is_coherent["y_model/BottomUp"].to_list()[0] == 1.0


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_incoherent_base_forecasts(diagnostic_data, lib):
    """Test that base forecasts show incoherence (non-zero residuals)."""
    data = diagnostic_data[lib]

    hrec = HierarchicalReconciliation([BottomUp()])
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
    )

    diag = nw.from_native(hrec.diagnostics)

    # Filter for coherence_residual_mae_before at Overall level
    before_mae = diag.filter(
        (nw.col("metric") == "coherence_residual_mae_before")
        & (nw.col("level") == "Overall")
    )

    # Should be non-zero due to added noise
    val = before_mae["y_model/BottomUp"].to_list()[0]
    assert val > 0, "Base forecasts should show incoherence"


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_adjustment_metrics(diagnostic_data, lib):
    """Test that adjustment metrics are computed correctly."""
    data = diagnostic_data[lib]

    hrec = HierarchicalReconciliation([BottomUp()])
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
    )

    diag = nw.from_native(hrec.diagnostics)

    # Get adjustment metrics for Overall level
    adj_mae = diag.filter(
        (nw.col("metric") == "adjustment_mae") & (nw.col("level") == "Overall")
    )
    adj_max = diag.filter(
        (nw.col("metric") == "adjustment_max") & (nw.col("level") == "Overall")
    )

    mae_val = adj_mae["y_model/BottomUp"].to_list()[0]
    max_val = adj_max["y_model/BottomUp"].to_list()[0]

    # Adjustments should be non-negative
    assert mae_val >= 0
    assert max_val >= 0
    # Max should be >= MAE
    assert max_val >= mae_val


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_disabled_by_default(diagnostic_data, lib):
    """Test that diagnostics are not computed when disabled."""
    data = diagnostic_data[lib]

    hrec = HierarchicalReconciliation([BottomUp()])
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=False,
    )

    assert hrec.diagnostics is None


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_negative_counting(diagnostic_data, lib):
    """Test negative value counting with deliberately negative forecasts."""
    data = diagnostic_data[lib]

    # Create a copy with some negative values
    if lib == "pandas":
        Y_hat_df = data["Y_hat_df"].copy()
        rng = np.random.default_rng(0)
        mask = rng.random(len(Y_hat_df)) < 0.1
        Y_hat_df.loc[mask, "y_model"] = -100
    else:
        # For polars, convert to pandas, modify, then back to polars
        Y_hat_pd = data["Y_hat_df"].to_pandas()
        rng = np.random.default_rng(0)
        mask = rng.random(len(Y_hat_pd)) < 0.1
        Y_hat_pd.loc[mask, "y_model"] = -100
        Y_hat_df = pl.from_pandas(Y_hat_pd)

    hrec = HierarchicalReconciliation([BottomUp()])
    _ = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
    )

    diag = nw.from_native(hrec.diagnostics)

    # Check negative_count_before is > 0 at Overall level
    neg_before = diag.filter(
        (nw.col("metric") == "negative_count_before") & (nw.col("level") == "Overall")
    )
    assert neg_before["y_model/BottomUp"].to_list()[0] > 0


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_multiple_reconcilers(diagnostic_data, lib):
    """Test diagnostics with multiple reconciliation methods."""
    data = diagnostic_data[lib]

    reconcilers = [
        BottomUp(),
        MinTrace(method="ols"),
        MinTrace(method="wls_struct"),
    ]
    hrec = HierarchicalReconciliation(reconcilers)
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        Y_df=data["Y_train_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
    )

    diag = nw.from_native(hrec.diagnostics)

    # Check all reconcilers have diagnostics
    assert "y_model/BottomUp" in diag.columns
    assert "y_model/MinTrace_method-ols" in diag.columns
    assert "y_model/MinTrace_method-wls_struct" in diag.columns

    # All should show coherent reconciled forecasts
    is_coherent = diag.filter(
        (nw.col("metric") == "is_coherent") & (nw.col("level") == "Overall")
    )
    for col in ["y_model/BottomUp", "y_model/MinTrace_method-ols", "y_model/MinTrace_method-wls_struct"]:
        assert is_coherent[col].to_list()[0] == 1.0


def test_diagnostics_csv_export(diagnostic_data, tmp_path):
    """Test that diagnostics can be exported to CSV."""
    data = diagnostic_data["pandas"]

    hrec = HierarchicalReconciliation([BottomUp()])
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
    )

    # Export to CSV
    csv_path = tmp_path / "diagnostics.csv"
    hrec.diagnostics.to_csv(csv_path, index=False)

    # Verify CSV can be read back
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == len(hrec.diagnostics)
    assert "level" in loaded.columns
    assert "metric" in loaded.columns


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_custom_tolerance(diagnostic_data, lib):
    """Test that custom tolerance affects is_coherent metric."""
    data = diagnostic_data[lib]

    hrec = HierarchicalReconciliation([BottomUp()])

    # With very tight tolerance
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
        diagnostics_atol=1e-15,  # Very tight
    )

    diag_tight = nw.from_native(hrec.diagnostics)
    max_violation_tight = diag_tight.filter(
        (nw.col("metric") == "coherence_max_violation") & (nw.col("level") == "Overall")
    )["y_model/BottomUp"].to_list()[0]

    # With loose tolerance
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
        diagnostics_atol=1.0,  # Very loose
    )

    diag_loose = nw.from_native(hrec.diagnostics)
    is_coherent_loose = diag_loose.filter(
        (nw.col("metric") == "is_coherent") & (nw.col("level") == "Overall")
    )["y_model/BottomUp"].to_list()[0]

    # Loose tolerance should always show coherent
    assert is_coherent_loose == 1.0

    # Max violation should be the same regardless of tolerance
    max_violation_loose = diag_loose.filter(
        (nw.col("metric") == "coherence_max_violation") & (nw.col("level") == "Overall")
    )["y_model/BottomUp"].to_list()[0]
    assert abs(max_violation_tight - max_violation_loose) < 1e-10


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_diagnostics_level_wise_metrics(diagnostic_data, lib):
    """Test that metrics are computed for each level separately."""
    data = diagnostic_data[lib]

    hrec = HierarchicalReconciliation([BottomUp()])
    _ = hrec.reconcile(
        Y_hat_df=data["Y_hat_df"],
        S_df=data["S_df"],
        tags=data["tags"],
        diagnostics=True,
    )

    diag = nw.from_native(hrec.diagnostics)

    # Get all levels
    levels = diag["level"].unique().to_list()

    # Each level should have its own set of metrics
    for level in levels:
        level_metrics = diag.filter(nw.col("level") == level)
        metrics = level_metrics["metric"].to_list()
        assert "adjustment_mae" in metrics
        assert "coherence_residual_mae_before" in metrics
