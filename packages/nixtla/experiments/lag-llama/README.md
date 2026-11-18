# LagLLama is 40% less accurate than a simple SeasonalNaive and 1000x slower.

We present a fully reproducible experiment showing that SeasonalNaive significantly outperforms LagLlama, a recently introduced open-source foundational model for time series forecasting (a deep learning architecture pre-trained on time series datasets). Specifically, **SeasonalNaive achieves 42%, 24%, and 16% better performance** in terms of MASE, MAPE, and CRPS respectively, and boasts **a 1,000x speed advantage**. These findings are based on an extensive analysis covering 105,289 unique time series from the M1, M3, M4, and Tourism datasets, which were omitted in the original LagLlama paper.

# Introduction

In the field of time series forecasting, recent developments have introduced foundational models such as LagLlama, which utilizes deep learning and extensive data for pretraining, aiming to enhance predictive performance and model complexity. LagLLama is to be praised as one of the first open-source foundational models. However, contrary to expectations, our analysis indicates that the traditional SeasonalNaive model, known for its straightforward approach of extending past seasonal trends into future predictions, outperforms LagLlama in terms of both accuracy and computational efficiency. 

## Empirical Evaluation

The original paper uses 3,113 time series to assess the model performance. The original paper only reports CRPS and omits point forecast error metrics widely used in academia and industry, e.g. MASE and MAPE.

Our evaluation encompasses 105,289 unique time series from different datasets, including M1, M3, M4, and Tourism, covering yearly, quarterly, monthly, weekly, daily, and hourly frequencies. This diverse dataset selection allows for a robust assessment of the models across various time series characteristics and forecasting horizons. We also reproduce results for Pedestrian Counts and Weather originally included in the paper/code to show that we are running LagLlama correctly. 

## Results

The results are summarized in the following table, highlighting the performance metrics of MASE, MAPE, CRPS, and TIME (measured in seconds). The best results are indicated in **bold** for easy reference.

<img width="953" alt="image" src="https://github.com/Nixtla/nixtla/assets/10517170/8e65338d-930e-4837-8bf5-2e7aeddad5cc">


## Reproducibility

To ensure the reproducibility of our findings, the experiments were conducted on an AWS g5.4xlarge GPU instance equipped with 16 vCPUs, 64 GiB of RAM, and an NVIDIA A10G Tensor Core GPU (24 GiB). The complete code can be found in this repo.

### Instructions

1. Create a python environment using:
```
mamba env create -f environment.yml
conda activate lag-llama
```

2. Add lag-llama code to your environment

```
make download_lag_llama_code
```

5. Download lag-llama model

```
make download_lag_llama_model
```

4. Install lag-llama requirements

```
pip install -r lag-llama-requirements.txt
```

5. Run complete experiments reported in the table

```
python -m src.main
```

### References
- **Lag-Llama Paper**: [Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278)
- **SeasonalNaive Implementation**: [GitHub Repository](https://github.com/nixtla/statsforecast/)
- **CRPS Replication Note**: The CRPS performance for `LagLlama` is replicated from the model's publicly available [Colab notebook](https://colab.research.google.com/drive/13HHKYL_HflHBKxDWycXgIUAHSeHRR5eo?usp=sharing), ensuring a fair comparison.
