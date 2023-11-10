# Changelog

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



