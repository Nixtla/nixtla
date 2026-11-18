# Salesforce's Moirai performs great in hourly data and is much faster than Chronos but is still up to 33% less accurate and less efficient than statistical models when considering monthly, weekly, and yearly data

We present a comprehensive, reproducible evaluation demonstrating that a Statistical Ensemble—comprising AutoARIMA, AutoETS, AutoCES, and DynamicOptimizedTheta—substantially surpasses [Salesforce Moirai](https://github.com/SalesforceAIResearch/uni2ts), a foundational model for time series forecasting with over 311 million parameters. The **Statistical Ensemble achieves 33%, 33%, and 15% superior performance in CRPS, MASE, and SMAPE metrics, respectively**, across benchmark datasets including M1, M3, M4, and Tourism. A  **Simple Seasonal Naive achieves 17% and 0.5%, superior performance in MASE, and SMAPE metrics, respectively. However, Morai is 25% more accurate than a seasonal naive in terms of CRPS**.  Benchmark datasets include M1, M3, M4, and Tourism. 
These datasets cover more than **100,000 unique time series**, offering a robust comparison of the models. Efficiency-wise, **Moirai is 3.5x faster than the Statistical Ensemble but 160x slower than a seasonal naive forecast**, marking a trade-off between speed and accuracy in different forecasting frequencies.


# Introduction

Following our recent [benchmark demonstrating Amazon Chronos's lesser accuracy and slower speed compared to classical statistical models](https://github.com/Nixtla/nixtla/tree/main/experiments/amazon-chronos), the community sought a similar analysis for Moirai. We commend the Salesforce AI team for releasing the first fully open-source foundational time series model, complete with weights, data, and code. Morai's accuracy shines with hourly data, a noteworthy achievement we're eager to highlight. Our acknowledgment extends to Salesforce for recognizing our prior contributions to this research field.

Foundational models like Salesforce's Moirai signify a notable advance in time series forecasting, leveraging deep learning and extensive datasets for pre-training to enhance predictions. Despite Moirai's impressive parameter count (311 million) and scope, our findings suggest that traditional forecasting methods grouped into a Statistical Ensemble often outperform in accuracy. This benchmark continues our exploration of statistical versus deep learning models in forecasting.

In our assessment, Salesforece's Moirai shows a more promising path than Amazon Chronos in handling hourly data, hinting at the potential to surpass classical statistical methods eventually.


## Empirical Evaluation

Expanding upon our prior work, this study evaluates over 100,000 unique time series from the M1, M3, M4, and Tourism datasets across various frequencies. Our analysis also benchmarks against the Seasonal Naive model, a staple in traditional forecasting methods.

## Results

The **Statistical Ensemble achieves 33%, 33%, and 15% superior performance in CRPS, MASE, and SMAPE metrics, respectively**, across benchmark datasets including M1, M3, M4, and Tourism. A  **Simple Seasonal Naive achieves 17% and 0.5%, superior performance in MASE, and SMAPE metrics, respectively. However, Morai is 25% more accurate than a seasonal naive in terms of CRPS**.

Efficiency-wise, **Moirai is 3.5x faster than the Statistical Ensemble but 160x slower than a seasonal naive forecast**, marking a trade-off between speed and accuracy in different forecasting frequencies.

It is critical to highlight that Morai may possess an unfair advantage over the statistical ensemble due to its training methodology. Specifically, Morai was trained using all the datasets that are currently being used for evaluation. In contrast, the statistical ensemble was not exposed to the test dataset during its training phase.

![image (27)](https://github.com/Nixtla/nixtla-backup/assets/4086186/71cf04f5-a48d-455e-8508-a0c393beed6e)

The complete code to replicate all results is available at [GitHub](https://github.com/Nixtla/nixtla/tree/main/experiments/salesforce-moirai). This study underscores statistical models' continued relevance and superiority in specific scenarios, challenging the assumption that foundational deep-learning models are always the best solution for time series forecasting.

This revision integrates the comparative performance of the Statistical Ensemble and Salesforce's Moirai, highlighting key findings from your data. Please ensure to replace the placeholder for the new table image with an actual image link or embed the table directly if the platform supports LaTeX rendering.

## Reproducibility

To ensure the reproducibility of our findings, the Statistical Ensemble experiments were conducted on an AWS c5a.24xlarge instance, equipped with 96 vCPUs and 192 GiB of RAM. In contrast, the experiments for Salesforce Moirai were carried out on an AWS g5.4xlarge GPU instance, which includes 16 vCPUs, 64 GiB of RAM, and an NVIDIA A10G Tensor Core GPU with 24 GiB. All necessary code and detailed instructions for reproducing the experiments are available in this directory.

### Instructions

1. Set up a Python environment:
   
```bash
mamba env create -f environment.yml
conda activate moirai
pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
```

2. Run the experiments as reported in the table:
   
```bash
python -m src.main --mode fcst_statsforecast
python -m src.main --mode fcst_moirai
```

3. Evaluate the results using:

```bash
python -m src.main --mode evaluation
```

### References
- **Statistical Ensemble Paper**: [A Simple Combination of Univariate Models](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300585?via%3Dihub)
- **Salesforce Moirai Paper**: [nified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592)
