# Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)

<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png">
<h1 align="center">Nixtla</h1>
<h3 align="center">Forecast using TimeGPT</h3>
    
[![CI](https://github.com/Nixtla/nixtla/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/nixtla/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/nixtla)](https://pypi.org/project/nixtla/)
[![PyPi](https://img.shields.io/pypi/v/nixtla?color=blue)](https://pypi.org/project/nixtla/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/nixtla/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nixtla.github.io/nixtla.svg?label=docs)](https://nixtla.github.io/nixtla/)
[![Downloads](https://pepy.tech/badge/nixtla)](https://pepy.tech/project/nixtla)
    
**Nixtla** offers a collection of classes and methods to interact with the API of TimeGPT.
</div>

# 🕰️ TimeGPT: Revolutionizing Time-Series Analysis

Developed by Nixtla, TimeGPT is a cutting-edge generative pre-trained transformer model dedicated to prediction tasks. 🚀 By leveraging the most extensive dataset ever – financial, weather, energy, and sales data – TimeGPT brings unparalleled time-series analysis right to your terminal! 👩‍💻👨‍💻

In seconds, TimeGPT can discern complex patterns and predict future data points, transforming the landscape of data science and predictive analytics.

## ⚙️ Fine-Tuning: For Precision Prediction

In addition to its core capabilities, TimeGPT supports fine-tuning, enhancing its specialization for specific prediction tasks. 🎯 This feature is like training a machine learning model on a targeted data subset to improve its task-specific performance, making TimeGPT an even more versatile tool for your predictive needs.

## 🔄 `Nixtla`: Your Gateway to TimeGPT

With `Nixtla`, you can easily interact with TimeGPT through simple API calls, making the power of TimeGPT readily accessible in your projects.

## 💻 Installation

Get `Nixtla` up and running with a simple pip command:

```python
pip install nixtla>=0.4.0
```

## 🎈 Quick Start

Get started with TimeGPT now:

```python
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short.csv')

from nixtla import NixtlaClient
nixtla_client = NixtlaClient(
    # defaults to os.environ.get("NIXTLA_API_KEY")
    api_key = 'my_api_key_provided_by_nixtla'
)
fcst_df = nixtla_client.forecast(df, h=24, level=[80, 90])
```

![](./nbs/img/forecast_readme.png)
