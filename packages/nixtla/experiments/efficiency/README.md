# 🚀 TimeGPT API v2: Faster, Smarter, and More Powerful Time Series Forecasting! 🚀

We’re excited to introduce **v2 of the TimeGPT API**, featuring a significant boost in performance, enhanced flexibility, and new capabilities that make time series forecasting faster and more insightful than ever before.

In this release, you will find:
- **Dramatic speed improvements** across all major endpoints 🏎️
- **Scalable forecasting** that handles 1 billion time series in just 6 hours 📊
- **Advanced handling of exogenous variables**, both historical and future 🌐
- **Enhanced explainability** through SHAP values 🧠
- **New integration with Polars**, a high-performance DataFrame library ⚡

## Key Performance Highlights 🔥

We've optimized the core functionalities—forecasting, anomaly detection, and cross-validation—with v2 showing significant speedups compared to v1. Below are the benchmark results:

| Endpoint          | Features   | Level   | v1   | v2   | Speedup   |
|:------------------|:-----------|:--------|:-----|:-----|:----------|
| anomaly_detection | exog       | [80]    | 24s  | 3s   | 9x        |
| anomaly_detection | none       | [80]    | 13s  | 2s   | 8x        |
| cross_validation  | exog       | None    | 22s  | 4s   | 6x        |
| cross_validation  | exog       | [80]    | 31s  | 6s   | 5x        |
| cross_validation  | none       | None    | 5s   | 1s   | 9x        |
| cross_validation  | none       | [80]    | 9s   | 2s   | 4x        |
| forecast          | exog       | None    | 18s  | 1s   | 13x       |
| forecast          | exog       | [80]    | 20s  | 2s   | 10x       |
| forecast          | none       | None    | 1s   | 0s   | 6x        |
| forecast          | none       | [80]    | 3s   | 1s   | 6x        |

These results represent the huge leap in efficiency v2 provides, allowing you to analyze vast datasets and derive insights faster than ever before. 🚀

## How to Reproduce Results

### Installation 🛠️

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code 🏃‍♀️

This script benchmarks **forecasting**, **anomaly detection**, and **cross-validation** across both v1 and v2 of the TimeGPT API. You can run the script and compare performance results by executing:

```bash
python main.py
```

## Performance Breakdown 🏎️

With v2, you get **up to 13x speed improvements** on key operations like forecasting with exogenous variables. This makes the API ideal for production environments where performance and scalability are paramount.

### New Features in v2

- **Advanced Exogenous Variable Handling**: Leverage both historical and future exogenous data for more accurate forecasts.
- **SHAP Values**: Improve model interpretability with SHAP value integration.
- **Polars Integration**: Benefit from lightning-fast data processing with Polars, especially useful for big datasets. 

## Conclusion 🚀

With TimeGPT API v2, you’re not just getting a faster API—you’re gaining the tools to scale up your time series analysis effortlessly, with greater precision and deeper insights. Whether it’s detecting anomalies, validating models, or producing reliable forecasts, v2 ensures you get results **faster and smarter** than ever before.

Happy forecasting! 
