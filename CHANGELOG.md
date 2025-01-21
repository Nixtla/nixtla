# Changelog

## 0.6.6

### 🚀 Feature Enhancements

#### Online anomaly detection

We introduce the `online_anomaly_detection` method, which allows you to define a `detection_size` on which to look for anomalies.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.6.6).*

## 0.6.5

### 🚀 Feature Enhancements

#### Persisting fine-tuned models

You can now run an isolated fine-tuning process, save the model and use it afterwards in all of our methods, i.e. `forecast`, `cross_validation` and `detect_anomalies`.

#### zstd compression

All requests above 1MB are now automatically compressed using [zstd](https://github.com/facebook/zstd), which will help when sending large amounts of data, as well as users with slow connections.

#### refit argument in cross_validation

It's now possible to fine-tune the model only on the first window when performing `cross_validation` by setting `refit=False`. This will dramatically decrease the time it takes to run cross validation with fine-tuning.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.6.5).*

## 0.6.4

### 🚀 Feature Enhancements

#### Integer and custom pandas frequencies

It's now possible to provide integer timestamps and an integer frequency as well as non-standard timestamps and custom pandas frequencies like [CustomBusinessHour](https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.CustomBusinessHour.html)

#### Usage method

You can now programatically get the number of calls you've made to our API as well as your limits by calling the `usage` method.

#### Historic exogenous in cross validation

The `cross_validation` method now supports the `hist_exog_list` parameter, which allows you to define historical exogenous features.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.6.4).*

## 0.6.2

### 🚀 Feature Enhancements

#### Finetune depth

You can now specify the depth of the fine-tuning process by setting `finetune_depth` in the `forecast` and `cross_validation` methods.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.6.2).*

## 0.6.0

### 🚀 Feature Enhancements

#### V2 API endpoints

The client now makes requests to the the V2 API endpoints, which have a lower latency.

#### orjson serialization

We now use [orjson](https://github.com/ijl/orjson) to serialize the payloads, which provides a performance improvement, especially when using exogenous features.

#### Historical exogenous features

It's now possible to define historical exogenous features in the `forecast` method by providing the feature names through `hist_exog_list`.

#### Feature contributions

You can now get the contributions of your exogenous features to the forecast by setting `feature_contributions=True` in the `forecast` method.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.6.0).*

## 0.5.0

### 🚀 Feature Enhancements

#### Cross Validation endpoint

The `cross_validation` method used to make one API call per window, now the method makes a single API call.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.5.0).*

## 0.4.0

### 🔄 Changes & Deprecations

- **Deprecation of `nixtlats`package:**
  The `nixtlats` package has been deprecated in favor of the new `nixtla` package. Please make sure to install the `nixtla` package and update your imports like the following:

  **Before:**
  ```python
  from nixtlats import NixtlaClient
  ```

  **After:**
  ```python
  from nixtla import NixtlaClient
  ```

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.4.0).*

## 0.3.0

### 🔄 Changes & Deprecations

- **Deprecation of `TimeGPT` Class:**
  In an effort to streamline our API and align with industry best practices, we're deprecating the `TimeGPT` class in favor of the new `NixtlaClient` class. This change is designed to provide a more intuitive and powerful interface for interacting with our services.

  **Before:**
  ```python
  from nixtlats import TimeGPT

  # Initialize the TimeGPT model
  timegpt = TimeGPT()
  ```

  **After:**
  ```python
  from nixtlats import NixtlaClient

  # Initialize the NixtlaClient
  nixtla = NixtlaClient()
  ```

- **Renaming of Configuration Parameters:**
  To enhance clarity and consistency with other SDKs, we've renamed the `token` parameter to `api_key` and `environment` to `base_url`.

  **Before:**
  ```python
  timegpt = TimeGPT(token='YOUR_TOKEN', environment='YOUR_ENVIRONMENT_URL')
  ```

  **After:**
  ```python
  nixtla = NixtlaClient(api_key='YOUR_API_KEY', base_url='YOUR_BASE_URL')
  ```

- **Introduction of `NixtlaClient.validate_api_key`:**
  Replacing the previous `NixtlaClient.validate_token` method, this update aligns with the new authentication parameter naming and offers a straightforward way to validate API keys.

  **Before:**
  ```python
  timegpt.validate_token()
  ```

  **After:**
  ```python
  nixtla.validate_api_key()
  ```

- **Environment Variable Changes:**
  In line with the renaming of parameters, we've updated the environment variables to set up the API key and base URL. The `TIMEGPT_TOKEN` is now replaced with `NIXTLA_API_KEY`, and we've introduced `NIXTLA_BASE_URL` for custom API URLs.

#### **Backward Compatibility & Future Warnings:**

These changes are designed to be backward compatible. However, users can expect to see future warnings when utilizing deprecated features, such as the `TimeGPT` class.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.3.0).*

## 0.2.0 (Previously Released)

### 🔄 Changes & Deprecations

- **Renaming of Fine-Tuning Parameters:**
  The `finetune_steps` and `finetune_loss` parameters were renamed to `fewshot_steps` and `fewshot_loss`. Additionally, the model parameter values changed from `short-horizon` and `long-horizon` to `timegpt-1` and `timegpt-1-long-horizon`, with an emphasis on preserving backward compatibility. In version 0.3.0, these changes are deprecated in favor of reverting to the original parameter names and values, ensuring a seamless transition for existing users.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.2.0).*

## 0.1.21

### 🚀 Feature Enhancements

#### Introduction of Quantile Forecasts in `forecast` and `cross_validation` Methods 📈

We're thrilled to announce the integration of the `quantiles` argument into TimeGP's `forecast` and `cross_validation` methods. This feature allows users to specify a list of quantiles, offering a comprehensive view of potential future values under uncertainty.

- **Quantile Forecasting Capability:**
  By providing a list of quantiles, users can now obtain forecasts at various percentiles of the forecast distribution. This is crucial for understanding the range of possible outcomes and assessing risks more effectively.

``` python
# Generate quantile forecasts
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
timegpt_quantile_fcst_df = timegpt.forecast(df=df, h=12, quantiles=quantiles, ...)
```

- **Enhanced Cross-Validation with Quantiles:**
  The `cross_validation` method has been updated to support quantile forecasting, enabling a more nuanced validation of model performance across different percentiles.

``` python
# Apply quantile forecasting in cross-validation
timegpt_cv_quantile_fcst_df = timegpt.cross_validation(df=df, h=12, n_windows=5, quantiles=quantiles, ...)
```

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.1.21).*

## 0.1.20

### 🚀 Feature Enhancements

#### Enhanced Model Fine-tuning with `finetune_loss` and `finetune_steps` 🛠️

The latest update brings a significant enhancement to the fine-tuning capabilities of our forecasting models. With the introduction of the `finetune_loss`, users now have the ability to not only specify the number of steps for fine-tunning (with `finetune_steps`) but also to define the target loss for fine-tunning, offering more granular control over model optimization.

- **`finetune_loss` Options:**
  - `default`: Adopts the model's preset loss function, optimized during initial training.
  - `mae` (Mean Absolute Error): Focuses on the mean of the absolute differences between predicted and actual values.
  - `mse` (Mean Squared Error): Emphasizes the mean of the squares of the differences between predicted and actual values.
  - `rmse` (Root Mean Squared Error): Provides the square root of MSE, offering error terms in the same units as the predictions.
  - `mape` (Mean Absolute Percentage Error): Measures the mean absolute percent difference between predicted and actual values.
  - `smape` (Symmetric Mean Absolute Percentage Error): Offers a symmetric version of MAPE, ensuring equal treatment of over and underestimations.

- **`finetune_steps`:** Determines the number of steps to execute during the fine-tuning process. It is crucial to set `finetune_steps` to a value greater than 0 to activate the fine-tuning mechanism with the chosen `finetune_loss` function. This allows for a more tailored optimization, aligning the model closely with specific forecasting requirements and improving its predictive performance.

``` python
# Configure model fine-tuning with custom loss function and steps
fcst_df = timegpt.forecast(df, model='timegpt-1-long-horizon', finetune_loss='mape', finetune_steps=50)

# Apply fine-tuning to cross-validation for enhanced model validation
cv_df = timegpt.cross_validation(df, model='timegpt-1', finetune_loss='smape', finetune_steps=50)
```

This update opens up new possibilities for refining forecasting models, ensuring they are finely tuned to the specific characteristics and challenges of the forecasting task at hand.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.1.20).*

## 0.1.19

### 🚀 Feature Enhancements

#### Advanced Data Partitioning with `num_partitions` 🔄

We're excited to introduce the `num_partitions` argument for our `forecast`, `cross_validation`, and `detect_anomalies` methods, offering more control over data processing and parallelization:

- **Optimized Resource Utilization in Distributed Environments:** For Spark, Ray, or Dask dataframes, `num_partitions` enables the system to either leverage all available parallel resources or to specify the number of parallel processes. This ensures efficient resource allocation and utilization across distributed computing environments.

``` python
# Utilize num_partitions in distributed environments
fcst_df = timegpt.forecast(df, model='timegpt-1-long-horizon', num_partitions=10)
```

- **Efficient Handling of Large Pandas Dataframes:** When working with Pandas dataframes, `num_partitions` groups series into specified partitions, allowing for sequential API calls. This is particularly useful for large dataframes that are impractical to send over the internet in one go, enhancing performance and efficiency.

``` python
# Efficiently process large Pandas dataframes
cv_df = timegpt.cross_validation(df, model='timegpt-1', num_partitions=5)
```

This new feature provides a flexible approach to handling data across different environments, ensuring optimal performance and resource management.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.1.19).*

## 0.1.18

### 🚀 Feature Enhancements

#### Forecast Using Diverse Models 🌐

Release of new forecasting methods. Among the updates, we've unveiled the **`timegpt-1-long-horizon`** model, crafted specifically for long-term forecasts that span multiple seasonalities. To use it, simply specify the model in your methods like so:

``` python
from nixtlats import TimeGPT

# Initialize the TimeGPT model
timegpt = TimeGPT()

# Generate forecasts using the long-horizon model
fcst_df = timegpt.forecast(..., model='timegpt-1-long-horizon')

# Perform cross-validation with the long-horizon model
cv_df = timegpt.cross_validation(..., model='timegpt-1-long-horizon')

# Detect anomalies with the long-horizon model
anomalies_df = timegpt.detect_anomalies(..., model='timegpt-1-long-horizon')
```

Choose between `timegpt-1` for the first version of `TimeGPT` or `timegpt-1-long-horizon` for long horizon tasks..

#### Cross-Validation Methodology 📊

You can dive deeper into your forecasting pipelines with the new `cross_validation` feature. This method enables you to validate forecasts across different windows efficiently:

``` python
# Set up cross-validation with a custom horizon, number of windows, and step size
cv_df = timegpt.cross_validation(df, h=35, n_windows=5, step_size=5)
```

This will generate 5 distinct forecast sets, each with a horizon of 35, stepping through your data every 5 timestamps.

### 🔁 Retry Behavior for Robust API Calls

The new retry mechanism allows the making of more robust API calls (preventing them from crashing with large-scale tasks).

- **`max_retries`**: Number of max retries for an API call.
- **`retry_interval`**: Pause between retries.
- **`max_wait_time`**:  Total duration of retries.

``` python
timegpt = TimeGPT(max_retries=10, retry_interval=5, max_wait_time=360)
```

### 🔑 Token Inference Made Simple

The `TimeGPT` class now automatically infers your `TIMEGPT_TOKEN` using `os.environ.get('TIMEGPT_TOKEN')`, streamlining your setup:

``` python
# No more manual token handling - TimeGPT has got you covered
timegpt = TimeGPT()
```
For more information visit our [FAQS](https://docs.nixtla.io/docs/faqs#setting-up-your-authentication-token-for-nixtla-sdk) section.

### 📘 Introducing the FAQ Section

Questions? We've got answers! Our new [FAQ section](https://docs.nixtla.io/docs/faqs) tackles the most common inquiries, from integrating exogenous variables to configuring authorization tokens and understanding long-horizon forecasts.

*See full changelog [here](https://github.com/Nixtla/nixtla/releases/v0.1.18).*
