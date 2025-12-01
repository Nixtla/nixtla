---
title: Core
description: Core
---

##

HierarchicalForecast contains pure Python implementations of
hierarchical reconciliation methods as well as a
`core.HierarchicalReconciliation` wrapper class that enables easy
interaction with these methods through pandas DataFrames containing the
hierarchical time series and the base predictions.

The `core.HierarchicalReconciliation` reconciliation class operates with
the hierarchical time series pd.DataFrame `Y_df`, the base predictions
pd.DataFrame `Y_hat_df`, the aggregation constraints matrix `S_df`. For
more information on the creation of aggregation constraints matrix see
the utils [aggregation
method](https://nixtlaverse.nixtla.io/hierarchicalforecast/src/utils.html#aggregate)

::: hierarchicalforecast.core.HierarchicalReconciliation
    options:
      members:
        - reconcile
        - bootstrap_reconcile

### Example

```python
import pandas as pd

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.evaluation import evaluate
from statsforecast.core import StatsForecast
from statsforecast.models import AutoETS
from utilsforecast.losses import mase, rmse
from functools import partial

# Load TourismSmall dataset
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
df.insert(0, 'Country', 'Australia')
qs = df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
df['ds'] = pd.PeriodIndex(qs, freq='Q').to_timestamp()

# Create hierarchical seires based on geographic levels and purpose
# And Convert quarterly ds string to pd.datetime format
hierarchy_levels = [['Country'],
                    ['Country', 'State'],
                    ['Country', 'Purpose'],
                    ['Country', 'State', 'Region'],
                    ['Country', 'State', 'Purpose'],
                    ['Country', 'State', 'Region', 'Purpose']]

Y_df, S_df, tags = aggregate(df=df, spec=hierarchy_levels)

# Split train/test sets
Y_test_df  = Y_df.groupby('unique_id').tail(8)
Y_train_df = Y_df.drop(Y_test_df.index)

# Compute base auto-ETS predictions
# Careful identifying correct data freq, this data quarterly 'Q'
fcst = StatsForecast(models=[AutoETS(season_length=4, model='ZZA')], freq='QS', n_jobs=-1)
Y_hat_df = fcst.forecast(df=Y_train_df, h=8, fitted=True)
Y_fitted_df = fcst.forecast_fitted_values()

reconcilers = [
                BottomUp(),
                MinTrace(method='ols'),
                MinTrace(method='mint_shrink'),
               ]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                          Y_df=Y_fitted_df,
                          S_df=S_df, tags=tags)

# Evaluate
eval_tags = {}
eval_tags['Total'] = tags['Country']
eval_tags['Purpose'] = tags['Country/Purpose']
eval_tags['State'] = tags['Country/State']
eval_tags['Regions'] = tags['Country/State/Region']
eval_tags['Bottom'] = tags['Country/State/Region/Purpose']

Y_rec_df_with_y = Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'], how='left')
mase_p = partial(mase, seasonality=4)

evaluation = evaluate(Y_rec_df_with_y,
         metrics=[mase_p, rmse],
         tags=eval_tags,
         train_df=Y_train_df)

numeric_cols = evaluation.select_dtypes(include="number").columns
evaluation[numeric_cols] = evaluation[numeric_cols].map('{:.2f}'.format)
```