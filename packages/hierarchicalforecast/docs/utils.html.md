---
output-file: utils.html
title: Aggregation/Visualization Utils
---


The `HierarchicalForecast` package contains utility functions to wrangle
and visualize hierarchical series datasets. The
[`aggregate`](https://Nixtla.github.io/hierarchicalforecast/src/utils.html#aggregate)
function of the module allows you to create a hierarchy from categorical
variables representing the structure levels, returning also the
aggregation contraints matrix $\mathbf{S}$.

In addition, `HierarchicalForecast` ensures compatibility of its
reconciliation methods with other popular machine-learning libraries via
its external forecast adapters that transform output base forecasts from
external libraries into a compatible data frame format.

## Aggregate Function

::: hierarchicalforecast.utils.aggregate

::: hierarchicalforecast.utils.aggregate_temporal

::: hierarchicalforecast.utils.make_future_dataframe

::: hierarchicalforecast.utils.get_cross_temporal_tags

## Hierarchical Visualization

::: hierarchicalforecast.utils.HierarchicalPlot
    options:
      members:
        - plot_summing_matrix
        - plot_series
        - plot_hierarchically_linked_series
        - plot_hierarchical_predictions_gap

### Example

```python
from statsforecast.core import StatsForecast
from statsforecast.models import AutoETS
from datasetsforecast.hierarchical import HierarchicalData

Y_df, S, tags = HierarchicalData.load('./data', 'Labour')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])
S = S.reset_index(names="unique_id")

Y_test_df  = Y_df.groupby('unique_id').tail(24)
Y_train_df = Y_df.drop(Y_test_df.index)

fcst = StatsForecast(
    models=[AutoETS(season_length=12, model='AAZ')],
    freq='MS',
    n_jobs=-1
)
Y_hat_df = fcst.forecast(df=Y_train_df, h=24).reset_index()

# Plot prediction difference of different aggregation
# Levels Country, Country/Region, Country/Gender/Region ...
hplots = HierarchicalPlot(S=S, tags=tags)

hplots.plot_hierarchical_predictions_gap(
    Y_df=Y_hat_df, models='AutoETS',
    xlabel='Month', ylabel='Predictions',
)
```
