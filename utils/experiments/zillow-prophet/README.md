# Using prophet on Zillow data: exploring better options


## TLDR

Contributing to the discussion on [Zillow and Prophet](https://ryxcommar.com/2021/11/06/zillow-prophet-time-series-and-prices/), we empirically prove that it is not a good idea to use that model to forecast [Zillow Home Value Index](https://www.zillow.com/research/data/) data. Better alternatives are `auto.arima` or [statsforecast](https://github.com/Nixtla/statsforecast).

## Introduction

Recently, Zillow announced that it would exit the market because its models were not being able to correctly anticipate price changes. Since this news, several opinions have been published about the alleged technology used by them for forecasting. In particular, opinions criticize the fact that they requested Prophet in their job offers.

In this blog post, we have set ourselves the goal of empirically determining whether `Prophet` is a good choice (or at least a good benchmark) for modeling the data used in the context of Zillow. As we will see, `auto.arima` and even the `naive` model turn out to be better baseline strategies than Prophet for the particular dataset we use.

## Dataset

The dataset we use to evaluate Prophet is the Zillow Home Value Index (ZHVI), which can be downloaded directly from the [Zillow research website](https://www.zillow.com/research/data/). According to the page, the ZHVI is "a smoothed, seasonally adjusted measure of typical home value and market changes for a given region and housing type. It reflects the typical value of homes in the 35th to 65th percentile range."

We chose this data set as it reflects changes in prices that can potentially be used to make decisions. The dataset consists of 909 Monthly series for different aggregations of regions and states. The data version was downloaded on November 4, 2021 and a copy can be found [here](https://github.com/FedericoGarza/zillow/tree/main/data).

## Experiments

To test the effectiveness of `Prophet` in forecasting the ZHVI, we use the last 4 observations as the test set and the remaining observations as the training set. In addition to `Prophet`, we ran `auto.arima` of R and some models of `statsforecast` (random walk with drift, naive, simple exponential smoothing, window average, seasonal naive and historic average).

### Reproducing results

Just follow the next steps. The whole process is automized using Docker, conda and Make.

1. `make init`. This instruction will create a docker container based on `environment.yml` which contains R and python needed libraries.
2. `make run_module module="python -m src.prepare_data"`. The module splits data in train and test sets. You cand find the generated data in `data/prepared-data-train.csv` and `data/prepared-data-test.csv` respectively.
3. `make run_module module="python -m src.forecast_prophet"`. Fits `prophet` model (forecasts in `data/prophet-forecasts.csv`).
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

As can we see, the best models is `auto.arima` for `mape`, `rmse`, `smape`, and `mae` metrics.
