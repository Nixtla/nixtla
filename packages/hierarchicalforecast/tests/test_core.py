import copy

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from hierarchicalforecast.core import HierarchicalReconciliation, _build_fn_name
from hierarchicalforecast.methods import (
    BottomUp,
    MiddleOut,
    MinTrace,
    TopDown,
)
from hierarchicalforecast.utils import aggregate


@pytest.fixture(scope="module")
def common_data(tourism_df, hiers_grouped, hiers_strictly):
    """Prepares the base tourism dataframe and hierarchy specifications."""
    df = copy.deepcopy(tourism_df)
    df['ds'] = pd.to_datetime(df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True))

    return {
        "df": df,
        "grouped_spec": hiers_grouped,
        "strict_spec": hiers_strictly
    }
@pytest.fixture(scope="module")
def middle_out_level():
    return 'Country/State'

@pytest.fixture(scope="module")
def grouped_data(common_data):
    """Prepares data for a grouped (non-strictly hierarchical) structure."""
    df = common_data['df']
    spec = common_data['grouped_spec']

    # Pandas
    Y_df, S_df, tags = aggregate(df, spec)
    Y_df['y_model'] = Y_df['y']
    Y_hat_df = Y_df.groupby('unique_id').tail(12).copy()
    ds_h = Y_hat_df['ds'].unique()
    Y_train_df = Y_df.query('~(ds in @ds_h)').copy()
    Y_train_df['y_model'] += np.random.uniform(-1, 1, len(Y_train_df))

    # Polars
    df_pl = pl.from_pandas(df)
    Y_df_pl, S_df_pl, tags_pl = aggregate(df_pl, spec)
    Y_hat_df_pl = pl.from_pandas(Y_hat_df)
    Y_train_df_pl = pl.from_pandas(Y_train_df)

    return {
        "pandas": {
            "Y_hat_df": Y_hat_df,
            "Y_train_df": Y_train_df,
            "S_df": S_df,
            "tags": tags
        },
        "polars": {
            "Y_hat_df": Y_hat_df_pl,
            "Y_train_df": Y_train_df_pl,
            "S_df": S_df_pl,
            "tags": tags_pl
        }
    }

@pytest.fixture(scope="module")
def strict_data(common_data):
    """Prepares data for a strictly hierarchical structure."""
    df = common_data['df']
    spec = common_data['strict_spec']

    # Pandas
    Y_df, S_df, tags = aggregate(df, spec)
    Y_df['y_model'] = Y_df['y']
    Y_hat_df = Y_df.groupby('unique_id').tail(12).copy()
    ds_h = Y_hat_df['ds'].unique()
    Y_train_df = Y_df.query('~(ds in @ds_h)').copy()
    Y_train_df['y_model'] += np.random.uniform(-1, 1, len(Y_train_df))

    # Polars
    df_pl = pl.from_pandas(df)
    Y_df_pl, S_df_pl, tags_pl = aggregate(df_pl, spec)
    Y_hat_df_pl = pl.from_pandas(Y_hat_df)
    Y_train_df_pl = pl.from_pandas(Y_train_df)

    return {
        "pandas": {
            "Y_hat_df": Y_hat_df,
            "Y_train_df": Y_train_df,
            "S_df": S_df,
            "tags": tags
        },
        "polars": {
            "Y_hat_df": Y_hat_df_pl,
            "Y_train_df": Y_train_df_pl,
            "S_df": S_df_pl,
            "tags": tags_pl
        }
    }

#endregion

#region: Tests

def test_fn_name():
    """Tests the _build_fn_name utility function."""
    assert _build_fn_name(BottomUp()) == 'BottomUp'
    assert _build_fn_name(MinTrace(method='ols')) == 'MinTrace_method-ols'
    assert _build_fn_name(MinTrace(method='ols', nonnegative=True)) == 'MinTrace_method-ols_nonnegative-True'
    assert _build_fn_name(MinTrace(method='mint_shrink')) == 'MinTrace_method-mint_shrink'

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_reconciliation_recovers_original_y(grouped_data, lib):
    """Tests if various reconciliation methods can reconstruct the original `y` values."""
    data = grouped_data[lib]
    Y_hat, S, Y_train, tags = data["Y_hat_df"], data["S_df"], data["Y_train_df"], data["tags"]

    reconcilers = [
        BottomUp(), MinTrace(method='ols'), MinTrace(method='wls_struct'),
        MinTrace(method='wls_var'), MinTrace(method='mint_shrink'),
        MinTrace(method='ols', nonnegative=True), MinTrace(method='wls_struct', nonnegative=True),
        MinTrace(method='wls_var', nonnegative=True), MinTrace(method='mint_shrink', nonnegative=True),
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    reconciled_df = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_train, S_df=S, tags=tags)

    reconciled_native = nw.from_native(reconciled_df)

    drop_cols = ["unique_id", "ds", "y"]

    models = reconciled_native.drop(*drop_cols).columns

    for model in models:
        np.testing.assert_allclose(reconciled_native['y'], reconciled_native[model], atol=1e-1)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_reconcile_raises_on_invalid_y_hat_dtype(grouped_data, lib):
    """Tests for failures when Y_hat_df contains non-numeric or null data."""
    data = grouped_data[lib]
    Y_hat, S, Y_train, tags = data["Y_hat_df"], data["S_df"], data["Y_train_df"], data["tags"]

    hrec = HierarchicalReconciliation(reconcilers=[BottomUp()])

    # Test string values (should always raise non-numeric error)
    Y_hat_str = nw.from_native(Y_hat).clone()
    Y_hat_str = Y_hat_str.with_columns(nw.col('y_model').cast(nw.String))

    with pytest.raises(ValueError, match="non-numeric values"):
        hrec.reconcile(Y_hat_df=Y_hat_str, S_df=S, tags=tags, Y_df=Y_train)
    # Test null values
    Y_hat_null = nw.from_native(Y_hat).clone()
    Y_hat_null = Y_hat_null.with_columns(
        nw.when(nw.col('unique_id') == 'Australia')
        .then(None)
        .otherwise(nw.col('y_model'))
        .alias('y_model')
    )

    with pytest.raises(ValueError, match="null values|non-numeric values"):
        hrec.reconcile(Y_hat_df=Y_hat_null, S_df=S, tags=tags, Y_df=Y_train)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_topdown_raises_on_non_strict_hierarchy(grouped_data, lib):
    """Tests that TopDown fails for non-strictly hierarchical (grouped) data."""
    data = grouped_data[lib]
    Y_hat, Y_train, S, tags = data["Y_hat_df"], data["Y_train_df"], data["S_df"], data["tags"]
    hrec = HierarchicalReconciliation([TopDown(method='average_proportions')])

    with pytest.raises(ValueError, match='Top-down reconciliation requires strictly hierarchical structures'):
        hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_train, S_df=S, tags=tags)

@pytest.mark.parametrize("lib", ["polars"])
def test_reconciliation_with_strict_hierarchy(strict_data, lib, middle_out_level):
    """Tests reconciliation methods with a strictly hierarchical structure."""
    data = strict_data[lib]
    Y_hat, S, Y_train, tags = data["Y_hat_df"], data["S_df"], data["Y_train_df"], data["tags"]

    reconcilers = [
        BottomUp(), MinTrace(method='ols'), TopDown(method='forecast_proportions'),
        TopDown(method='average_proportions'), MiddleOut(middle_level=middle_out_level, top_down_method='forecast_proportions')
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    reconciled = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_train, S_df=S, tags=tags)

    reconciled = nw.from_native(reconciled)
    for model_name in ['y_model/BottomUp', 'y_model/MinTrace_method-ols', 'y_model/TopDown_method-forecast_proportions']:
        np.testing.assert_allclose(reconciled['y'], reconciled[model_name], atol=1e-1)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(reconciled['y'], reconciled['y_model/TopDown_method-average_proportions'], atol=1e-1)

    total_tag = tags['Country'][0]
    total_y_df = reconciled.filter(nw.col('unique_id') == total_tag)
    np.testing.assert_allclose(total_y_df['y'], total_y_df['y_model/TopDown_method-average_proportions'], atol=1e-2)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_mintrace_raises_on_zero_residuals(grouped_data, lib):
    """Tests that MinTrace fails gracefully when insample residuals are all zero."""
    data = grouped_data[lib]
    Y_hat, S, Y_train, tags = data["Y_hat_df"], data["S_df"], data["Y_train_df"], data["tags"]

    zero_df = nw.from_native(Y_train).clone()
    zero_df = zero_df.with_columns([nw.lit(0, dtype=nw.Float64).alias('y'), nw.lit(0, dtype=nw.Float64).alias('y_model')])

    hrec = HierarchicalReconciliation([MinTrace(method='mint_shrink')])
    with pytest.raises(Exception, match='Insample residuals close to 0'):
        hrec.reconcile(Y_hat_df=Y_hat, S_df=S, tags=tags, Y_df=zero_df)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_permbu_raises_on_non_strict_hierarchy(grouped_data, lib):
    """Ensures permbu intervals fail with non-strictly hierarchical data."""
    data = grouped_data[lib]
    Y_hat, S, Y_train, tags = data["Y_hat_df"], data["S_df"], data["Y_train_df"], data["tags"]

    Y_hat_native = nw.from_native(Y_hat)
    Y_hat_with_intervals = Y_hat_native.with_columns([
            (nw.col('y_model') * 0.9).alias('y_model-lo-80'),
            (nw.col('y_model') * 1.1).alias('y_model-hi-80')
        ])

    hrec = HierarchicalReconciliation([BottomUp()])
    with pytest.raises(ValueError, match='PERMBU probabilistic reconciliation requires strictly hierarchical structures'):
        hrec.reconcile(
            Y_hat_df=Y_hat_with_intervals, Y_df=Y_train, S_df=S, tags=tags,
            level=[80], intervals_method='permbu'
        )

@pytest.mark.parametrize("lib", ["pandas", "polars"])
@pytest.mark.parametrize("level, method", [([-1, 80], 'permbu'), ([80, 101], 'normality')])
def test_reconcile_raises_on_invalid_level(grouped_data, lib, level, method):
    """Tests failure when prediction interval levels are outside the [0, 100] range."""
    data = grouped_data[lib]
    Y_hat, S, Y_train, tags = data["Y_hat_df"], data["S_df"], data["Y_train_df"], data["tags"]
    hrec = HierarchicalReconciliation([BottomUp()])
    with pytest.raises(ValueError, match=r"Level must be a list containing floating values in the interval \[0, 100\)"):
         hrec.reconcile(
            Y_hat_df=Y_hat, Y_df=Y_train, S_df=S, tags=tags,
            level=level, intervals_method=method
        )
