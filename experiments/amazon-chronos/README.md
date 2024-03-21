# Amazon Chronos is 10% less accurate and 500% slower than training classical statistical models.

We present a fully reproducible comprehensive evaluation showcasing that a Statistical Ensemble, consisting of AutoARIMA, AutoETS, AutoCES, and DynamicOptimizedTheta, outperforms Amazon Chronos—a foundational model for time series forecasting with over 710 million parameters. Specifically, the **Statistical Ensemble demonstrates 10%, 10%, and 11% superior performance in CRPS, MASE, and SMAPE metrics, respectively**, and it is **5x faster**. This analysis spans over **50,000 unique time series** across M1, M3, M4, and Tourism datasets, robustly comparing these models.

# Introduction

The rise of foundational models in time series forecasting, such as Amazon Chronos, represents a significant leap forward, leveraging deep learning and massive datasets for model pre-training to enhance predictive accuracy. Amazon Chronos, in particular, is noteworthy for its extensive parameterization and ambitious scope. However, our study shows that a comparatively simpler approach, employing a Statistical Ensemble of traditional forecasting methods, yields better accuracy and computational efficiency. One year ago, we used the same [benchmark](https://github.com/Nixtla/statsforecast/tree/main/experiments/m3) to showcase how statistical models outperformed deep learning models. 

## Empirical Evaluation

This study considers over 50,000 unique time series from the M1, M3, M4, and Tourism datasets, spanning various time series frequencies. Chronos did not use these datasets in the training phase. We have also included comparisons to the Seasonal Naive model to provide a benchmark for traditional forecasting methods.

## Results

Our findings are shown in the following table, showcasing the performance across different metrics: CRPS, MASE, SMAPE, and computational time (in seconds). The best results are highlighted in **bold** for ease of reference.

<img width="1099" alt="image" src="https://github.com/Nixtla/nixtla/assets/10517170/4d4fe9f3-4251-4b95-bd9b-248fc283e97b">


## Reproducibility

To ensure the reproducibility of our findings, the Statistical Ensemble experiments were conducted on an AWS c5a.24xlarge instance, equipped with 96 vCPUs and 192 GiB of RAM. In contrast, the experiments for Amazon Chronos were carried out on an AWS g5.4xlarge GPU instance, which includes 16 vCPUs, 64 GiB of RAM, and an NVIDIA A10G Tensor Core GPU with 24 GiB. All necessary code and detailed instructions for reproducing the experiments are available in this directory.

### Instructions

1. Set up a Python environment:
   
```bash
mamba env create -f environment.yml
conda activate amazon-chronos
```

2. Run the experiments as reported in the table:
   
```bash
python -m src.main --mode fcst_statsforecast
python -m src.main --mode fcst_chronos
```

3. Evaluate the results using:

```bash
python -m src.main --mode evaluation
```

### References
- **Statistical Ensemble Paper**: [A Simple Combination of Univariate Models](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300585?via%3Dihub)
- **Amazon Chronos Paper**: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
