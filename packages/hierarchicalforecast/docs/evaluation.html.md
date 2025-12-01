---
output-file: evaluation.html
title: Hierarchical Evaluation
---


To assist the evaluation of hierarchical forecasting systems, we make
available an
[`evaluate`](https://Nixtla.github.io/hierarchicalforecast/src/evaluation.html#evaluate)
function that can be used in combination with loss functions from
`utilsforecast.losses`.

------------------------------------------------------------------------

::: hierarchicalforecast.evaluation.evaluate

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

### References

- [Gneiting, Tilmann, and Adrian E. Raftery. (2007). "Strictly proper
  scoring rules, prediction and estimation". Journal of the American Statistical
  Association.](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
- [Gneiting, Tilmann. (2011). "Quantiles as optimal point forecasts".
  International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207010000063)
- [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos,
    Zhi Chen, Anil Gaba, Ilia Tsetlin, Robert L. Winkler. (2022). "The
    M5 uncertainty competition: Results, findings and conclusions".
    International Journal of
    Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207021001722)
- [Anastasios Panagiotelis, Puwasala Gamakumara, George
    Athanasopoulos, Rob J. Hyndman. (2022). "Probabilistic forecast
    reconciliation: Properties, evaluation and score optimisation".
    European Journal of Operational
    Research.](https://www.sciencedirect.com/science/article/pii/S0377221722006087)
- [Syama Sundar Rangapuram, Lucien D Werner, Konstantinos Benidis,
    Pedro Mercado, Jan Gasthaus, Tim Januschowski. (2021). "End-to-End
    Learning of Coherent Probabilistic Forecasts for Hierarchical Time
    Series". Proceedings of the 38th International Conference on Machine
    Learning
    (ICML).](https://proceedings.mlr.press/v139/rangapuram21a.html)
- [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei
    Cao, Lee Dicker (2022). “Probabilistic Hierarchical Forecasting with
    Deep Poisson Mixtures”. Submitted to the International Journal
    Forecasting, Working paper available at
    arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
- [Makridakis, S., Spiliotis E., and Assimakopoulos V. (2022). “M5
    Accuracy Competition: Results, Findings, and Conclusions.”,
    International Journal of Forecasting, Volume 38, Issue
    4.](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
