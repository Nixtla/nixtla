# Benchmarking foundation models for time series

> TL;DR: Foundation models for time series outperform alternatives and are ready to be tested in production. TimeGPT-1 is (so far) the most accurate and fastest model but TimesFM from Google comes very close. Some models are still outperformed by classical alternatives.

Notes: 
* The Amazon team responded to the original benchmark with this [PR](https://github.com/Nixtla/nixtla/pull/382) that shows, according to them, that by changing some parameters, Chronos is significantly faster and more accurate.
* The SalesForce team also responded with this [PR](https://github.com/Nixtla/nixtla/pull/389) showing improved accuracy and perfomance.
We are currently reviewing both PRs.

# Introduction

We present a reproducible benchmark comparing different foundation models across a wide variety of models in a large scale dataset.  

We conclude that [TimeGPT-1](https://arxiv.org/abs/2310.03589b) ranks first in terms of accuracy and speed inference compared to the latest foundation models, including [TimesFM](https://arxiv.org/pdf/2310.10688) (Google), [Chronos](https://arxiv.org/abs/2403.07815) (Amazon), [Moirai](https://arxiv.org/abs/2402.02592) (SalesForece), and [Lag-LLama](https://arxiv.org/pdf/2310.08278) (Service Now). `TimeGPT-1` and `TimesFM` also outperform established statistical, machine learning, and deep-learning models, with comparable inference times to a `SeasonalNaive`. `Chronos`, `Moirai` and `Lag-Llama` still need some further improvements and can be outperformed by other classical methods.

This analysis spans over **30,000 unique time series** across various domains and frequencies from M-Competitions, Monash Repository, and Wikipedia page views, among others, robustly comparing these models.

# Zero-shot foundation models
The rise of zero-shot foundational models in time series forecasting, such as `TimeGPT-` by [Nixtla](https://github.com/Nixtla/), `TimesFM` by Google or `Chronos` by Amazon, represents a significant leap forward in our field. The promise of this innovation is to allow practitioners to accurately forecast without having to train their own models. If foundation models succeed, this would make forecasting and anomaly detection much easier, faster, and, in many cases, more accurate than state-of-the-art alternatives. 

We have also seen some of these models being offered as out-of-the-box solutions. We at Nixtla [recently announced](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/announcing-timegen-1-in-azure-ai-leap-forward-in-time-series/ba-p/4140446) that `TimeGPT-1` and `TimeGEN-1` are now available on both Azure and our own platform. Google will also release a version of `TimesFM` on VertexAI, and it wouldn't be surprising if Amazon is trying to do the same for Bedrock.

We at Nixtla have provided some [early success stories](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/announcing-timegen-1-in-azure-ai-leap-forward-in-time-series/ba-p/4140446) of real companies leveraging the simplicity and accuracy of `TimeGPT-1` and we are sure that more positive examples will follow from other models.

However, the field [is still divided](https://news.ycombinator.com/item?id=39235983) on how all the different foundation models compare against each other. In the spirit of collaboration, we are starting a new project, `xiuhmolpilli`, in honor of how ancient civilizations celebrated the end of cycles, to build a benchmark to compare all the different foundation models for time series data in a large scale dataset and against classical, ML and Deep Learning Models.



# Empirical Evaluation

This study considers **over 30,000 unique time series** from the Monash Repository, M-Competitions, Wikipedia page views, among others, spanning various time series frequencies: Monthly, Weekly, Daily, and Hourly. Our evaluation compares five foundation models for time series data in terms of accuracy and inference times. We have also included comparisons to a large battery of statistical, machine learning, and deep-learning models, to provide a benchmark against traditional forecasting methods.

We include the following models in our comprehensive evaluation:

- [Statistical](https://github.com/Nixtla/statsforecast/): `SeasonalNaive`, `HistoricAverage`, `ZeroModel`, `AutoARIMA`, `Prophet`, `AutoCES`, `AutoETS`, `Theta`, `DynamicOptimizedTheta`, `ADIDA`, `IMAPA`, and `CrostonClassic`.
- [Machine Learning](https://github.com/Nixtla/mlforecast/): `AutoLGBM`. 
- [Deep Learning](https://github.com/Nixtla/neuralforecast/): `AutoTFT`, `AutoNHITS`.
- Foundation: `Chronos`, `Lag-Llama`, `Moirai`, `TimeGPT`, `TimeGPT` (long horizon), and `TimesFM`. 

## Results

`TimeGPT-1` ranks first in terms of accuracy and speed inference compared to the latest foundation models, including `TimesFM`, `Chronos`, `Moirai`, and `Lag-Llama`. `TimesFM` by Google ranks second in accuracy and outperfoms `TimeGPT-1` in inference speed. Amazon `Chronos` ranks third in accuracy but shows a significant drop in inference speed. Both Salesforces's and ServiceNow's models are far more efficient in terms of inference speed than `Chronos`, but they rank lower in terms of accuracy.

The following image shows the average performance ranking based on `MASE` for the four frequencies:

![image](https://github.com/Nixtla/nixtla/assets/10517170/ed9b66f9-afd5-49c6-b736-48cb9b239de8)


Our findings are shown in the following table, showcasing the performance measured by `MASE` and computational inference time (in minutes). The best model is highlighted in **bold** and the second best is underlined for ease of reference.

![image](https://github.com/Nixtla/nixtla/assets/10517170/1c042591-0585-4a5b-a548-2017a28f2d4f)


We also present a plot comparing accuracy and speed across foundation models.

![image](https://github.com/Nixtla/nixtla/assets/10517170/2a6a630e-c9db-4530-8ef2-86db3c85d8a9)


Some noteworthy observations from the results include:
- `TimeGPT-1` consistenly achieves the best overall ranking among all models, with comparable inference times to the simplest baselines and `TimesFM`.
- `TimesFM` ranks second among all foundation models, with performance slightly worse than `TimeGPT-1`.
- `Chronos` ranks third in performance, but with extremelly high inference times, reducing it's utility as a pre-trained foundation model. For reference, it is possible to fully train while performming automatic hyperparameter selection of state-of-the-art deep-learning models in less time than `Chronos` zero-shot inference time.
- `Moirai` and `Lag-Llama` rank last among foundation models and are often outperformed by almost all statistical, ML, and deep learning models.
- While `Prophet` is still widely used by practitioners, it consistently ranks lower than all the methods considered.


## Challenges to benchmark foundation models

There are two main challenges we faced to correctly compare foundation models:

- creating a brand-new framework capable of running various methods and algorithms, the framework presented here, xiuhmolpilli, was designed as an abstraction of the different ways foundation models were developed, including classic statistical, machine learning, and deep learning models. To that end, we based our architecture on the nixtlaverse approach to forecasting.
- finding appropriate and novel data unseen by all models considered in the analysis. For the current results, we guaranteed that all the timestamps for all the time series were completely unseen to TimeGPT-1 during training, including the train set part of the time series commonly used for in-distribution evaluation. 

Based on the datasets used for training reported in the papers of other foundation models:
- We can't discard that `TimesFM` potentially observed during training the time series made available here from Monthly, Weekly, and Daily frequencies.
- `Chronos` could have observed the train part of a small fraction of the time series (as reported in `Table 2 in-domain evaluation`) for all frequencies. 
- Based on the data contained in `LOTSA`, `Moirai` could have observed the train part of a small fraction of the time series for Monthly, Weekly, and Daily frequencies.
- `Lag-Llama` could have observed a small fraction of the time series for Monthly, Weekly, and Daily frequencies.

Note: we updated the initial benchmark table to reduce the chance of leakage.

Given these observations, we expect the current evaluation setting to at least favour other foundation models, as having observed the time series during training will likely lead to increased performance. 

This also underscores a common concern among practitioners: for models like `TimeGPT-1`, where the exact training data remains undisclosed, evaluating the model using public datasets is problematic due to the potential for overfitting. We recognize this as a limitation of closed-source models; however, we appeal for the reader's understanding. As a small startup competing with the largest companies in the world, we have had to keep some of our methods confidential.

That being said we celebrate the open-source nature of the other foundation models and we encourage the community to continue to push for more transparency in the field. We have been doing our part and will continue to do so in due time. 

We would also like to conclude that it is essential for practitioners to test these models on their datasets and form their conclusions.


## Preliminary Conclusions

In addition to the success stories we have heard from our users, this benchmark indicates that other models are also ready to be used in production for time series forecasting and anomaly detection tasks. By no means are we claiming that `TimeGPT-1` is the best model for all tasks, but it is the most accurate and fastest model we have benchmarked so far. It also production ready and can be used and tested by anyone out of the box.

That being said, it is well known among practitioners that before deploying any model into production, you should do some benchmarking. It is also known that there is no such thing as magic in time-series. And finally, it is also known that `Prophet` is not a good benchmarking model.

## Special Conclusion for Hacker News readers

* You can't forecast stocks with ANY of these models if you don't have additional data to include as covariates. Please stop [asking](https://news.ycombinator.com/item?id=37874891).

# Acknowledgments

We at Nixtla believe that all of our work is due to other great researchers that have paved the way. In that spirit, we would like to thank the more than 50 researchers from Google Cloud, AWS AI, AWS, UC Berkeley, NYU, CMU, Salesforce Research, Salesforce, McGill University, ServiceNow, Quebec AI Institute, Morgan Stanley, CERC AAI lab, MILA, Zalando who have been ardently working to improve the 
field of foundation models in time series.


# ToDo's [WIP]

> This is a first attempt to provide a fully reproducible benchmark foundation. A lot of work still needs to be done:
* Include probabilistic benchmarks
* Explore the distribution of errors across MASE and other metrics
* Experiment with different cross validation windows
* Include more classical and deep learning models
* Include more foundation models (Moment (CMU) and Tiny Time Mixers(IBM))
* Benchmark anomaly detection
* Include finetuning

# Reproducibility

To ensure the reproducibility of our findings, the experiments were conducted on an AWS c5a.24xlarge instance, equipped with 96 vCPUs and 192 GiB of RAM for non-cpu workloads. In contrast, the experiments for foundation and deep learning models were carried out on an AWS g5.4xlarge GPU instance, which includes 16 vCPUs, 64 GiB of RAM, and an NVIDIA A10G Tensor Core GPU with 24 GiB. All necessary code and detailed instructions for reproducing the experiments are available in this directory.

## Get started with Nixtla's models on the public API

It must be noted, that to use `TimeGPT-1` you will need an account on Nixtla platform to access our models and get a [30 days free trial](https://dashboard.nixtla.io/freetrial). You can also access `TimeGPT-1` through [Azure AI Studio](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-models-timegen-1?view=azureml-api-2) and Azure Machine Learning (named `TimeGEN-1`).

## Instructions to reproduce the benchmark

### Download data

#### Configure aws cli

The data lives in s3. You can download it using the `aws` cli. Follow the instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install it.

#### Download data from s3

```
make download_data
```

### Download extra code3

```
make download_lag_llama_code
```

### Download data

```
make download_data
```

### Create environment

```
mamba create -n foundation-ts python=3.10
conda activate foundation-ts
pip install uv
uv pip install -r requirements.txt
```

### Run

```
python -m xiuhmolpilli.arena
```

### New Features

Since the release, we received more than 15,000 requests from companies, organizations, and researchers, eager to try `TimeGPT-1`. We are thrilled for the overall feedback, as `TimeGPT-1` has proven valuable for hundreds of applications, from predicting river flows, to forecast demand sales of thousands of products.

After several months of beta testing, we have new updates and improvements based on user feedback:

- New Features: Introduction of `model="timegpt-1-long-horizon"` for improved long-term forecast accuracy. Enhanced uncertainty quantification for forecasting and anomaly detection. Advanced model fine-tuning with diverse loss functions (e.g., `"mae"`, `"mse"`, `"rmse"`, `"mape"`, `"smape"`). Support for distributed computing and big data (Spark, Ray, and Dask).
- [Documentation Improvements](https://docs.nixtla.io/): Revamped layout, new tutorials, and Google Colab compatibility. Expanded documentation covers What-if scenarios, electricity and financial forecasting, and anomaly detection.
- [New R SDK](https://github.com/Nixtla/nixtlar). Forecast using `TimeGPT-1` with R.

### Partnering with Microsoft to provide our models on Azure

At Nixtla, our mission is to make frontier AI ubiquitous. Last week we announce the availability of `TimeGEN-1` on Azure. Our models are now accessible through:

- Public API: hosted safely on Nixtla infrastructure, this access point enables developers to create applications and services across our range of models. Create your account [here](https://dashboard.nixtla.io/)

- [Azure](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-models-timegen-1?view=azureml-api-2): `TimeGEN-1` is available through Azure AI Studio and Azure Machine Learning, offering a seamless user experience comparable to our API. Beta customers have experienced significant success. 

- Self-deployment: our models can also be deployed in your environment for the most sensitive use cases. Contact our team for further details. Please contact 


# References
- **TimeGPT-1**: [TimeGPT-1](https://arxiv.org/abs/2310.03589b)
- **Chronos**: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- **TimesFM**: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/pdf/2310.10688)
- **Moirai**: [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592)
- **Lag-LLama**: [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/pdf/2310.08278)
