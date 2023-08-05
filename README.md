# Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)

<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png">
<h1 align="center">NixtlaTS</h1>
<h3 align="center">Forecast using TimeGPT</h3>
    
[![CI](https://github.com/Nixtla/nixtlats/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/nixtlats/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/nixtlats)](https://pypi.org/project/nixtlats/)
[![PyPi](https://img.shields.io/pypi/v/nixtlats?color=blue)](https://pypi.org/project/nixtlats/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/nixtlats/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nixtla.github.io/nixtlats.svg?label=docs)](https://nixtla.github.io/nixtlats/)
[![Downloads](https://pepy.tech/badge/nixtlats)](https://pepy.tech/project/nixtlats)
    
**NixtlaTS** offers a collection of classes and methods to interact with the API of TimeGPT.
</div>

## Installation

You can install `NixtlaTS` with:

```python
pip install nixtlats
```

## Quick Start


```python
from nixtlats import TimeGPT
timegpt = TimeGPT(token=os.environ['TIMEGPT_TOKEN'], api_url=os.environ['TIMEGPT_API_URL'])
fcst_df = timegpt.forecast(df, h=24, freq='H', level=[80, 90])
```

![](./nbs/img/forecast_readme.png)