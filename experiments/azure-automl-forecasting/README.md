# Nixtla TimeGPT vs. Azure AutoML: A Comprehensive Performance Analysis

This experiment evaluates the performance of **Nixtla TimeGPT's zero-shot inference** against **Microsoft's Azure AutoML** in the domain of time series forecasting. Our analysis shows that TimeGPT **surpasses Azure AutoML by 12%, 12%, and 10% in MAE, RMSE, and MASE metrics** and has **300x improvement in computational efficiency**. This evaluation spanned over 3,000 distinct time series across various data frequencies, with considerations for Azure AutoML's cost constraints.

# Introduction

[Azure AutoML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automl-forecasting-methods?view=azureml-api-2), a product of Microsoft, offers a robust automated machine-learning solution that caters to a wide array of predictive tasks, including time series forecasting. TimeGPT is a foundational model for time series forecasting that can be accessed [through an API](https://docs.nixtla.io/). While Azure AutoML is known for its adaptability and ease of use, our findings reveal that TimeGPT offers superior accuracy and efficiency, especially in the context of time series data.

## Empirical Evaluation

Our study involved a detailed comparison of both models across various datasets, including Hourly, Daily, Weekly, and Monthly data frequencies. The datasets were chosen from the test set of the [TimeGPT-1 paper](https://arxiv.org/abs/2310.03589), ensuring a diverse set of time series for evaluation. The selection process was designed to manage computational complexity and adhere to Azure AutoML's dataset size requirements, with a cap of 3,000 observations to maintain cost-effectiveness.

## Results

The following table shows the main findings of our analysis, presenting a comparison of performance metrics (MASE, MAE, RMSE) and computational time (in seconds) across different datasets. The best results are highlighted in **bold** for clarity.

<img width="632" alt="image" src="https://github.com/Nixtla/nixtla/assets/10517170/0cc4285e-2572-4f08-9846-94c68ad72e8b">


## Reproducibility

All experiments were conducted in controlled environments to uphold the integrity and reproducibility of our results. TimeGPT evaluations were performed using a 2020 MacBook Air with an M1 chip, ensuring accessibility and practicality. In contrast, Azure AutoML experiments were carried out on a cluster of 11 STANDARD_DS5_V2 virtual machines equipped with substantial computational resources to showcase its scalability and power.

### Instructions

1. Configure Azure AutoML according to the official Microsoft documentation.
2. Set the environment variables in a `.env` file using `.env.example` as example.
3. Set up a conda environment using:

```bash
mamba create -n azure-automl-fcst python=3.10
conda activate azure-automl-fcst
pip install uv
uv pip install -r requirements.txt
```

4. Download the data using

```python
python -m src.utils.download_data
```

If you're interested in replicating the results, write us at `support@nixtla.io` to give you access to the data.

5. Filter the datasets to prevent AzureML from crashing

```
make filter_data
```

6. Run the forecasting tasks for TimeGPT, SeasonalNaive, and AzureAutoML using the following:

```
make run_methods
```

Notice that AzureAutoML will send the job to the predefined cluster. 

7. Retrieve AzureAutoML forecasts once they are ready:

```
make download_automl_forecasts
```

8. Run evaluation

```
make evaluate_experiments
```


### References
- [TimeGPT 1](https://arxiv.org/abs/2310.03589)
- [StatsForecast](https://github.com/Nixtla/statsforecast/)
- [Distributed AzureAutoML for forecasting](https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/pipelines/1k_demand_forecasting_with_pipeline_components/automl-forecasting-demand-many-models-in-pipeline/automl-forecasting-demand-many-models-in-pipeline.ipynb)
