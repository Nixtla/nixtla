# Nixtla &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/branding/logo_mid.png">
<h1 align="center">Hierarchical 👑 Forecast</h1>
<h3 align="center">Hierarchical forecasting with statistical and econometric methods</h3>
    
[![CI](https://github.com/Nixtla/hierarchicalforecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Nixtla/hierarchicalforecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/hierarchicalforecast)](https://pypi.org/project/hierarchicalforecast/)
[![PyPi](https://img.shields.io/pypi/v/hierarchicalforecast?color=blue)](https://pypi.org/project/hierarchicalforecast/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Nixtla/hierarchicalforecast/blob/main/LICENSE)
    
**HierarchicalForecast** offers a collection of reconciliation methods, including `BottomUp`, `TopDown`, `MiddleOut`, `MinTrace` and `ERM`. 
</div>

## 🎊 Features 

* Classic reconciliation methods:
    - `BottomUp`: Simple addition to the upper levels.
    - `TopDown`: Distributes the top levels forecasts trough the hierarchies.
* Alternative reconciliation methods:
    - `MiddleOut`: It anchors the base predictions in a middle level. The levels above the base predictions use the bottom-up approach, while the levels below use a top-down.
    - `MinTrace`: Minimizes the total forecast variance of the space of coherent forecasts, with the Minimum Trace reconciliation.
    - `ERM`: Optimizes the reconciliation matrix minimizing an L1 regularized objective.

Missing something? Please open an issue here or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## 📖 Why? 

**Short**: We want to contribute to the ML field by providing reliable baselines and benchmarks for hierarchical forecasting task in industry and academia. Here's the complete [paper]().

**Verbose**: `HierarchicalForecast` integrates publicly available processed datasets, evaluation metrics, and a curated set of statistical baselines. In this library we provide usage examples and references to extensive experiments where we showcase the baseline's use and evaluate the accuracy of their predictions. With this work, we hope to contribute to Machine Learning forecasting by bridging the gap to statistical and econometric modeling, as well as providing tools for the development of novel hierarchical forecasting algorithms rooted in a thorough comparison of these well-established models. We intend to continue maintaining and increasing the repository, promoting collaboration across the forecasting community.

## 💻 Installation

### PyPI

You can install the *released version* of `HierarchicalForecast` from the [Python package index](https://pypi.org) with:

```python
pip install hierarchicalforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)

### Conda

Also you can install the *released version* of `HierarchicalForecast` from [conda](https://anaconda.org) with:

```python
conda install -c conda-forge hierarchicalforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)

### Dev Mode

If you want to make some modifications to the code and see the effects in real time (without reinstalling), follow the steps below:

```bash
git clone https://github.com/Nixtla/hierarchicalforecast.git
cd hierarchicalforecast
pip install -e .
```

## 🧬 How to use

The following example needs `statsforecast` and `datasetsforecast` as additional packages. If not installed, install it via your preferred method, e.g. `pip install statsforecast datasetsforecast`.
The `datasetsforecast` library allows us to download hierarhical datasets and we will use `statsforecast` to compute base forecasts to be reconciled.

```python
import numpy as np
import pandas as pd

#obtain hierarchical dataset
from datasetsforecast.hierarchical import HierarchicalData
#obtain hierarchical reconciliation methods and evaluation
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut
# compute base forecast no coherent
from statsforecast.core import StatsForecast
from statsforecast.models import auto_arima, naive

# Load TourismSmall dataset
Y_df, S, tags = HierarchicalData.load('./data', 'TourismSmall')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])

#split train/test sets
Y_df_test = Y_df.groupby('unique_id').tail(12)
Y_df_train = Y_df.drop(Y_df_test.index)
Y_df_test = Y_df_test.set_index('unique_id')
Y_df_train = Y_df_train.set_index('unique_id')

# Compute base auto-ARIMA predictions
fcst = StatsForecast(df=Y_df_train, models=[(auto_arima,12), naive], freq='M', n_jobs=-1)
Y_hat_df = fcst.forecast(h=12)

# Reconcile the base predictions
reconcilers = [
    BottomUp(),
    TopDown(method='forecast_proportions'),
    MiddleOut(level='Country/Purpose/State', top_down_method='forecast_proportions')
]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(Y_hat_df, Y_df_train, S, tags)
```

### Evaluation

```python
def mse(y, y_hat):
    return np.mean((y-y_hat)**2)

evaluator = HierarchicalEvaluation(evaluators=[mse])
evaluator.evaluate(Y_h=Y_rec_df, Y_test=Y_df_test, 
                   tags=tags, benchmark='naive')
```

## 📃 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Nixtla/neuralforecast/blob/main/LICENSE) file for details.

##  How to cite

```bibtex
@article{olivares2022hierarchicalforecast,
    author    = {Kin G. Olivares and
                 Federico Garza and 
                 David Luo and 
                 Cristian Challú and
                 Max Mergenthaler},
    title     = {HierarchicalForecast: A Python Benchmarking Framework for Hierarchical Forecasting},
    journal   = {Computing Research Repository},
    volume    = {abs/2207.03517},
    year      = {2022},
    url       = {https://arxiv.org/abs/2207.03517},
    archivePrefix = {arXiv}
}
```
