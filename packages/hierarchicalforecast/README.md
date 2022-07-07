# Nixtla &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/branding/logo_mid.png">
<h1 align="center">Hierarchical 🪜  Forecast</h1>
<h3 align="center">Hierarchical forecasting with statistical and econometric methods</h3>
    
[![CI](https://github.com/Nixtla/hierarchical/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/hierarchicalforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/hierarchicalforecast)](https://pypi.org/project/hierarchicalforecast/)
[![PyPi](https://img.shields.io/pypi/v/hierarchicalforecast?color=blue)](https://pypi.org/project/hierarchicalforecast/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Nixtla/hierarchicalforecast/blob/main/LICENSE)
    
**HierarchicalForecast** offers a collection of widely used reconciliation methods, including `BottomUp`, `TopDown`, `MiddleOut`, `MinTrace` and `ERM`. 
</div>

## 🎊 Features 

* Classic reconciliation methods:
    - `BottomUp`: Simple addition to the upper levels.
    - `TopDown`: Distributes the top levels forecasts trough the hierarchies.
* Alternative reconciliation methods:
    - `MiddleOut`: It anchors the base predictions in a middle level. The levels above the base predictions use the bottom-up approach, while the levels below use a top-down.
    - `MinTrace`: Minimizes the total forecast variance of the space of coherent forecasts, with the Minimum Trace reconciliation.
    - `ERM`: Optimizes the reconciliation matrix minimizing an L1 regularized objective.

Missing something? Please open an issue or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## 📖 Why? 

Given the widespread adoption of Python in the ML field, we identified that the community struggled with finding reliable baselines and benchmarks for their own hierarchical forecasting research.

## 💻 Installation
<details>
<summary>PyPI</summary>

You can install the *released version* of `HierarchicalForecast` from the [Python package index](https://pypi.org) with:

```python
pip install hierarchicalforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)
</details>

<details>
<summary>Conda</summary>
  
Also you can install the *released version* of `HierarchicalForecast` from [conda](https://anaconda.org) with:

```python
conda install -c conda-forge hierarchicalforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)
</details>

<details>
<summary>Dev Mode</summary>
If you want to make some modifications to the code and see the effects in real time (without reinstalling), follow the steps below:

```bash
git clone https://github.com/Nixtla/hierarchicalforecast.git
cd hierarchicalforecast
pip install -e .
```
</details>

## 🧬 How to use

The following example needs `statsforecast` and `datasetsforecast` as additional packages. If not installed, install it via your preferred method, e.g. `pip install statsforecast datasetsforecast`.
The `datasetsforecast` library allows us to download hierarhical datasets and we will use `statsforecast` to compute base forecasts to be reconciled.

```python
from datasetsforecast.hierarchical import HierarchicalData
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace, ERM
from statsforecast.core import StatsForecast
from statsforecast.models import auto_arima

# Load TourismL dataset
Y_df, S, tags = HierarchicalData.load('./data', 'TourismLarge')
Y_df = Y_df.set_index('unique_id')

# Compute base auto-ARIMA predictions
fcst = StatsForecast(df=Y_df, models=[(auto_arima,12)], freq='M', n_jobs=-1)
Y_hat_df = fcst.forecast(h=12)

# Reconcile the base predictions
reconcilers = [
    BottomUp(),
    MinTrace(method='ols'),
    MinTrace(method='wls_struct'),
    MinTrace(method='wls_var'),
    MinTrace(method='mint_shrink'),
    ERM(method='lasso'),
]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(Y_hat_df, Y_df, S, tags)
```

### Evaluation

```python
from hierarchicalforecast.core import HierarchicalEvaluation, mase

evaluator = HierarchicalEvaluation(evaluators=[mase])
evaluator.evaluate(Y_h=Y_hat_df, Y_test=Y_test, 
                   tags=tags, benchmark='naive')
```

##  📃 How to cite

```bibtex
@article{hierarchicalforecast_arxiv,
  author  = {XXXX},
  title   = {{HierarchicalForecast: A Python Benchmarking Framework for Hierarchical Forecasting}},
  journal = {arXiv preprint arXiv:XXX.XXX},
  year    = {2022}
}
```
