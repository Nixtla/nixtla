import numpy as np
import pandas as pd
import pytest
import warnings

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp


def create_test_data():
    """Create test data for deprecation tests."""
    # Create a small test dataset
    n_series = 5
    h = 3
    
    # Create unique_ids for bottom series (b0, b1, ...)
    bottom_ids = [f"b{i}" for i in range(n_series)]
    
    # Create unique_ids for aggregated series (total)
    agg_ids = ["total"]
    
    # All unique_ids
    all_ids = agg_ids + bottom_ids
    
    # Create a forecast dataframe
    Y_hat_df = pd.DataFrame({
        "unique_id": np.repeat(all_ids, h),
        "ds": np.tile(np.arange(h), len(all_ids)),
        "model1": np.random.normal(0, 1, len(all_ids) * h)
    })
    
    # Create a summing matrix
    S_data = {"unique_id": all_ids}
    for i in range(n_series):
        S_data[f"b{i}"] = np.zeros(len(all_ids))
        # Set identity for bottom part
        S_data[f"b{i}"][i+1] = 1  # +1 because the first row is "total"
    # Set all ones for the total row
    for i in range(n_series):
        S_data[f"b{i}"][0] = 1
        
    S = pd.DataFrame(S_data)
    
    # Create tags
    tags = {
        "total": ["total"],
        "bottom": bottom_ids
    }
    
    return Y_hat_df, S, tags


def test_S_parameter_deprecation_warning():
    """Test that using S instead of S_df raises a deprecation warning."""
    
    Y_hat_df, S, tags = create_test_data()
    
    # Create HierarchicalReconciliation object with a simple reconciliation method
    hrec = HierarchicalReconciliation(reconcilers=[BottomUp()])
    
    # Test that using S parameter raises a deprecation warning
    with pytest.warns(DeprecationWarning, match="The 'S' parameter is deprecated"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S=S,  # Use S instead of S_df
            tags=tags
        )
    
    # Test that using both S and S_df raises a ValueError
    with pytest.raises(ValueError, match="Both 'S' and 'S_df' parameters were provided"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S=S,
            S_df=S,
            tags=tags
        )


def test_S_parameter_functionality():
    """Test that using S instead of S_df still works correctly."""
    
    Y_hat_df, S, tags = create_test_data()
    
    # Create HierarchicalReconciliation object
    hrec = HierarchicalReconciliation(reconcilers=[BottomUp()])
    
    # Get results using S_df (new parameter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_with_S_df = hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S_df=S,
            tags=tags
        )
    
    # Get results using S (deprecated parameter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        result_with_S = hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S=S,
            tags=tags
        )
    
    # Verify the results are identical (same column names and values)
    pd.testing.assert_frame_equal(result_with_S, result_with_S_df)
