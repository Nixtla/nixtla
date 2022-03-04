# Prophet vs Linear Regression on Real Estate: The Zillow Case

## TL;DR

Recently there has been controversy in the data science community about the Zillow case. There has been [speculation that the Zillow team may have used](https://ryxcommar.com/2021/11/06/zillow-prophet-time-series-and-prices/) [Prophet](https://github.com/facebook/prophet) to generate forecasts of their time series. Although we do not know if the above is true, we contribute to the discussion by showing that creating good benchmarks is fundamental in forecasting tasks. Furthermore we show that `Prophet` does not turn out to be a good solution on [Zillow Home Value Index](https://www.zillow.com/research/data/) data. Better alternatives are simpler and faster models like [auto.arima](https://github.com/robjhyndman/forecast) or [statsforecast](https://github.com/Nixtla/statsforecast), and to improve them [mlforecast](https://github.com/Nixtla/mlforecast) is an excellent option because it makes forecasting with machine learning fast and easy and it allows practitioners to focus on the model and features instead of implementation details.

## Introduction

Recently, Zillow announced that it would close its [home-buying business](https://www.cnbc.com/2021/11/02/zillow-shares-plunge-after-announcing-it-will-close-home-buying-business.html) because its models were not being able to correctly anticipate price changes. The Zillow CEO Rich Barton said *"We’ve determined the unpredictability in forecasting home prices far exceeds what we anticipated"*. Since this news, [several opinions](https://twitter.com/vhranger/status/1456064415845990408) have been published about the alleged technology used by them for forecasting. In particular, opinions criticize the fact that they requested `Prophet` in their job offers.

Forecasting time series is a complicated task, and there is no single model that fits all business needs and data characteristics. [Best practices](https://towardsdatascience.com/time-series-forecasting-with-statistical-models-f08dcd1d24d1) always suggest starting with a simple model as a benchmark; such a model will allow, on the one hand, to build models with better performance and, on the other hand, to measure the value-added of such models (data scientists should obtain a lower loss of their more complex models compared to the benchmark's loss).

In this blog post, we have set ourselves the goal of empirically determining whether `Prophet` is a good choice (or at least a good benchmark) for modeling the data used in the context of Zillow. As we will see, `auto.arima` and even the [`naive`](https://otexts.com/fpp2/simple-methods.html#na%C3%AFve-method) model turn out to be better baseline strategies than `Prophet` for the particular dataset we use. We reveal that `Prophet` does not perform well compared to other models, which is consistent with the evidence found by other practitioners (for example [here](https://www.microprediction.com/blog/prophet) and [here](https://kourentzes.com/forecasting/2017/07/29/benchmarking-facebooks-prophet/)). Also, we show how using `mlforecast` (and [`LinearRegression` from `sklearn`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) as training model) can beat `auto.arima` and `Prophet` in no more than 3 seconds.

## Dataset

The dataset we use to evaluate `Prophet` is the Zillow Home Value Index (ZHVI), which can be downloaded directly from the [Zillow research website](https://www.zillow.com/research/data/). According to the page, the ZHVI is *"a smoothed, seasonally adjusted measure of typical home value and market changes for a given region and housing type. It reflects the typical value of homes in the 35th to 65th percentile range"* and [*"represents the “typical” home value for a region"*](https://www.zillow.com/research/zhvi-user-guide/).

The dataset reflects price changes, so we decided to experiment with it because a stakeholder can potentially use it to make decisions. The dataset consists of 909 Monthly series for different aggregations of regions and states. We downloaded it on November 4, 2021 and anybody interested can find a copy of it [here](https://github.com/Nixtla/nixtla/blob/main/utils/experiments/zillow-prophet/data/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv).

## Experiments

To test the effectiveness of `Prophet` in forecasting the ZHVI, we use the last 4 observations as the test set and the remaining observations as the training set. We performed a hyperparameter optimization over each time series using the last 4 observations of the training set as a validation for `Prophet`. In addition to `Prophet`, we ran `auto.arima` of R, some models of `statsforecast` (random walk with drift, naive, simple exponential smoothing, window average, seasonal naive and historic average) and `mlforecast`.

`mlforecast` is a framework that helps practitioners forecast time series using machine learning models. They need to give it a model (in this case, we use `LinearRegression` from `sklearn`), define which features to use and let `mlforecast` do the rest.

## Reproducing results

Just follow the next steps. The whole process is automized using Docker, conda and Make.

1. `make init`. This instruction will create a docker container based on `environment.yml` which contains R and python needed libraries.
2. `make run_module module="python -m src.prepare_data"`. The module splits data in train and test sets. You cand find the generated data in `data/prepared-data-train.csv` and `data/prepared-data-test.csv` respectively.
3. `make run_module module="python -m src.forecast_prophet"`. Fits `Prophet` model (forecasts in `data/prophet-forecasts.csv`).
4. `make run_module module="python -m src.forecast_statsforecast"`. Fits `statsforecast` models (forecasts in  `data/statsforecast-forecasts.csv`).
5. `make run_module module="Rscript src/forecast_arima.R"`. Fits `auto.arima` model (forecasts in `data/arima-forecasts.csv`).
6. `make run_module module="python -m src.forecast_mlforecast"`. Fits `mlforecast` model using `LinearRegression` (forecasts in `data/mlforecast-forecasts.csv`).

## Results

### Performance

The following table summarizes the results in terms of performance.

|                                                 |      mape |     rmse |     smape |      mae |
|:------------------------------------------------|----------:|---------:|----------:|---------:|
| `mlforecast.linear_regression`                  |  **0.942855** |  **3037.31** |  **0.951257** |  **2595.47** |
| `auto.arima`                                    |  1.01637  |  3221.03 |  1.0273   |  2702.71 |
| `statsforecast.random_walk_with_drift`          |  2.70799  |  8624.85 |  2.77414  |  7848.35 |
| `statsforecast.naive`                           |  3.1611   |  9842.39 |  3.24514  |  8967.52 |
| `statsforecast.ses_alpha-0.9`                   |  3.27845  | 10145.1  |  3.36773  |  9296.87 |
| `Prophet`                                       |  4.26159  | 13491.6  |  4.42465  | 12429.4  |
| `statsforecast.window_average_window_size-4`    |  4.66278  | 13707.7  |  4.82723  | 13080    |
| `statsforecast.seasonal_naive_season_length-12` | 10.7527   | 28986.9  | 11.5389   | 28783.9  |
| `statsforecast.historic_average`                | 27.0533   | 68887.4  | 32.008    | 68741.7  |


As can we see, the best model is `mlforecast.linear_regression` for `mape`, `rmse`, `smape`, and `mae` metrics. Surprisingly, a very simple model such as `naive` (takes the last value as forecasts) turns out to be better in this experiment than `Prophet`.

### Computational cost

The following table summarizes the results in terms of computational cost.

| model          | time (segs)  |  cost (USD) |
|:---------------|-------------:|------------:|
|`mlforecast`    | **2.68**     | **0.0034**  |
|`statsforecast` | 3.99         | 0.0051      |
|`auto.arima`    | 957.6        | 1.2257      |
|`Prophet`       | 46,980.5     | 60.1350     |

To run our experiments we used a [c5d.24xlarge AWS instance (96 vCPU, 192 RAM)](https://aws.amazon.com/ec2/instance-types/c5/). It costs 4.608 USD each hour. As can we see, `mlforecast` takes no more than 3 seconds and beats `Prophet` and `auto.arima` in performance.

## Conclusion

This post showed in the context of the Zillow controversy that doing benchmarks is fundamental to addressing any time series forecasting problem. Those benchmarks must be computationally efficient to iterate fast and build more complex models on top of them. The libraries [statsforecast](https://github.com/Nixtla/statsforecast) and [mlforecast](https://github.com/Nixtla/mlforecast) are excellent tools for the task. We also showed better options than `Prophet` to run benchmarks, which is consistent with previous findings by the data science community. 

**Build benchmarks. Always.**

