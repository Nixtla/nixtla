# Using Prophet on Zillow data: exploring better options


## TLDR

Recently there has been controversy in the data science community about the zillow case. There has been [speculation that the Zillow team may have used](https://ryxcommar.com/2021/11/06/zillow-prophet-time-series-and-prices/) [Prophet](https://github.com/facebook/prophet) to generate forecasts of their time series. Although we do not (and will not) know if the above is true, we contribute to the discussion by showing that it is fundamental in forecasting tasks to create benchmarks. Furthermore, that `Prophet` does not turn out to be a good benchmark on [Zillow Home Value Index data](https://www.zillow.com/research/data/) data. Better alternatives to create benchmarks are [auto.arima](https://github.com/robjhyndman/forecast) or [statsforecast](https://github.com/Nixtla/statsforecast).

## Introduction

Recently, Zillow announced that it would exit the market because its models were not being able to correctly anticipate price changes. Since this news, [several opinions](https://twitter.com/vhranger/status/1456064415845990408) have been published about the alleged technology used by them for forecasting. In particular, opinions criticize the fact that they requested `Prophet` in their job offers.

Forecasting time series is a complicated task, and there is no single model that fits all business needs and data characteristics. [Best practices](https://towardsdatascience.com/time-series-forecasting-with-statistical-models-f08dcd1d24d1) always suggest starting with a simple model as a benchmark; such a model will allow, on the one hand, to build models with better performance and, on the other hand, to measure the value-added of such models (Forecast Value Added).

In this blog post, we have set ourselves the goal of empirically determining whether `Prophet` is a good choice (or at least a good benchmark) for modeling the data used in the context of Zillow. As we will see, `auto.arima` and even the `naive` model turn out to be better baseline strategies than `Prophet` for the particular dataset we use.
That `Prophet` does not perform well compared to other models is consistent with the evidence found by other practitioners (for example [here](https://www.microprediction.com/blog/prophet) and [here](https://kourentzes.com/forecasting/2017/07/29/benchmarking-facebooks-prophet/)).

## Dataset

The dataset we use to evaluate Prophet is the Zillow Home Value Index (ZHVI), which can be downloaded directly from the [Zillow research website](https://www.zillow.com/research/data/). According to the page, the ZHVI is "a smoothed, seasonally adjusted measure of typical home value and market changes for a given region and housing type. It reflects the typical value of homes in the 35th to 65th percentile range".

We chose this data set as it reflects changes in prices that can potentially be used to make decisions. The dataset consists of 909 Monthly series for different aggregations of regions and states. The data version was downloaded on November 4, 2021 and a copy can be found [here](https://github.com/FedericoGarza/zillow/tree/main/data).

## Experiments

To test the effectiveness of `Prophet` in forecasting the ZHVI, we use the last 4 observations as the test set and the remaining observations as the training set. In addition to `Prophet`, we ran `auto.arima` of R and some models of `statsforecast` (random walk with drift, naive, simple exponential smoothing, window average, seasonal naive and historic average).

### Reproducing results

Just follow the next steps. The whole process is automized using Docker, conda and Make.

1. `make init`. This instruction will create a docker container based on `environment.yml` which contains R and python needed libraries.
2. `make run_module module="python -m src.prepare_data"`. The module splits data in train and test sets. You cand find the generated data in `data/prepared-data-train.csv` and `data/prepared-data-test.csv` respectively.
3. `make run_module module="python -m src.forecast_prophet"`. Fits `Prophet` model (forecasts in `data/prophet-forecasts.csv`).
4. `make run_module module="python -m src.forecast_statsforecast"`. Fits `statsforecast` models (forecasts in  `data/statsforecast-forecasts.csv`).
5. `make run_module module="Rscript src/forecast_arima.R"`. Fits `auto.arima` model (forecasts in `data/arima-forecasts.csv`).

## Results

The following table summarizes the results.

|                                 |     mape |     rmse |    smape |      mae |
|:--------------------------------|---------:|---------:|---------:|---------:|
| `auto.arima`                      |  **1.01637** |  **6135.36** |  **1.0273**  | **2702.71** |
| `random_walk_with_drift`          |  2.70799 | 14246    |  2.77414 |  7848.35 |
| `naive`                           |  3.1611  | 15758.5  |  3.24514 |  8967.52 |
| `ses_alpha-0.9`                   |  3.27845 | 16238.7  |  3.36773 |  9296.87 |
| `prophet`                    |  4.42807 | 23444.3  |  4.59759 | 12941.3  |
| `window_average_window_size-4`    |  4.66278 | 21675    |  4.82723 | 13080    |
| `seasonal_naive_season_length-12` | 10.7527  | 42875.8  | 11.5389  | 28783.9  |
| `historic_average`                | 27.0533  | 96878.2  | 32.008   | 68741.7  |

As can we see, the best model is `auto.arima` for `mape`, `rmse`, `smape`, and `mae` metrics. Surprisingly, a very simple model such as `naive` (takes the last value as forecasts) turns out to be better in this experiment than `Prophet`.
