
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<div align="center">
<h1> 
Open source time series forecasting suite  
</h1>
<img src="utils/misc/BannerGit.png">

[Features](#Features) •
[Where](#How?) •
[Getting Started ](#Getting-Started-(SDK))

</div >

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pmp4rqiwiPL-ambxTrJGBiNMS-7vm3v6?ts=616700c4)

Open-source time-series pipeline capable of achieving 1% of the performance in the [M5 competition](https://en.wikipedia.org/wiki/Makridakis_Competitions). 

Our open-source solution has a 25% better accuracy than Amazon Forecast and is 20% more accurate than fbprophet. It also performs 4x faster than Amazon Forecast and is less expensive.

Read this [Medium Post](https://aws.plainenglish.io/automated-time-series-forecasting-pipeline-662e0feadd98) for a Step-by-Step guide .


## 🧰 Features 

>[tspreprocess](#tspreprocess) to preprocess time-series data such as missing values imputation

>[tsfeatures](#tsfeatures) to generate features to include in the models, 

>[tsforecast](#tsforecast) to perform forecast at scale

>[tsbenchmarks](#tsbenchmarks) to easily calculate accuracy baselines.

>[NeurosalForecast](github.com/nixtla/neuralforecast)


## ✨ Purpose?
Help data scientists and developers to have access to open source state-of-the-art forecasting pipelines. 

## How?
We built a complete pipeline that can be deployed in the cloud ☁️  using AWS and consumed via APIs or consumed as a service. 

### Build your own Infra [![Terraform](https://img.shields.io/badge/terraform-%235835CC.svg?style=for-the-badge&logo=terraform&logoColor=white)](#Build-your-own-Nixtla-in-AWS)
If you want to set up your own infrastructure, follow the instructions in the repository (Azure coming soon). With our Infrastructure as Code written in Terraform, you can deploy our solution in minutes without much effort.

### Use our APIs [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pmp4rqiwiPL-ambxTrJGBiNMS-7vm3v6?ts=616700c4)
You can use our fully hosted version as a service through our [python SDK](https://github.com/Nixtla/nixtla/tree/main/sdk/) ([autotimeseries](https://pypi.org/project/autotimeseries/)). To consume the APIs on our own infrastructure just request tokens by sending an email to federico@nixtla.io or opening a GitHub issue. **We currently have free resources available for anyone interested.**



# Getting Started (SDK)
[![CI python sdk](https://github.com/Nixtla/nixtla/actions/workflows/python-sdk.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/python-sdk.yml)

Check the following [example](https://github.com/Nixtla/autotimeseries/tree/main/examples/m5) for a full pipeline:

>Install with
`pip install autotimeseries`


<details>
 <summary markdown="span">Import libraries and config AWS </summary>

```python
import os

from autotimeseries.core import AutoTS

autotimeseries = AutoTS(bucket_name=os.environ['BUCKET_NAME'],
                        api_id=os.environ['API_ID'],
                        api_key=os.environ['API_KEY'],
                        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
```
</details>

<details>

<summary> Upload dataset to S3 </summary>

```python
train_dir = '../data/m5/parquet/train'
# File with target variables
filename_target = autotimeseries.upload_to_s3(f'{train_dir}/target.parquet')
# File with static variables
filename_static = autotimeseries.upload_to_s3(f'{train_dir}/static.parquet')
# File with temporal variables
filename_temporal = autotimeseries.upload_to_s3(f'{train_dir}/temporal.parquet')
```


Each time series of the uploaded datasets is defined by the column `item_id`. Meanwhile the time column is defined by `timestamp` and the target column by `demand`. We need to pass this arguments to each call.

```python
columns = dict(unique_id_column='item_id',
               ds_column='timestamp',
               y_column='demand')
```
</details>

<details>
<summary> Send the job to make forecasts and Download </summary>

```python
response_forecast = autotimeseries.tsforecast(filename_target=filename_target,
                                              freq='D',
                                              horizon=28,
                                              filename_static=filename_static,
                                              filename_temporal=filename_temporal,
                                              objective='tweedie',
                                              metric='rmse',
                                              n_estimators=170,
                                              **columns)
```

#### Download forecasts

```python
autotimeseries.download_from_s3(filename='forecasts_2021-10-12_19-04-32.csv', filename_output='../data/forecasts.csv')
```
</details>


# Forecasting Pipeline as a Service

Our forecasting pipeline is modular and built upon simple APIs:

## tspreprocess

[![CI/CD tspreprocess Lambda](https://github.com/Nixtla/nixtla/actions/workflows/tspreprocess-lambda.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tspreprocess-lambda.yml)
[![CI/CD tspreprocess docker image](https://github.com/Nixtla/nixtla/actions/workflows/tspreprocess-docker-image.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tspreprocess-docker-image.yml)

Time series usually contain missing values. This is the case for sales data where only the events that happened are recorded. In these cases it is convenient to balance the panel, i.e., to include the missing values to correctly determine the value of future sales.

The [tspreprocess](https://github.com/Nixtla/nixtla/tree/main/tspreprocess) API allows you to do this quickly and easily. In addition, it allows one-hot encoding of static variables (specific to each time series, such as the product family in case of sales) automatically.

## tsfeatures

[![CI/CD tsfeatures Lambda](https://github.com/Nixtla/nixtla/actions/workflows/tsfeatures-lambda.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsfeatures-lambda.yml)
[![CI/CD tsfeatures docker image](https://github.com/Nixtla/nixtla/actions/workflows/tsfeatures-docker-image.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsfeatures-docker-image.yml)

It is usually good practice to create features of the target variable so that they can be consumed by machine learning models. This API allows users to create features at the time series level (or static features) and also at the temporal level.

The [tsfeatures](https://github.com/Nixtla/nixtla/tree/main/tsfeatures) API is based on the [tsfeatures](https://github.com/Nixtla/tsfeatures) library also developed by the Nixtla team (inspired by the R package [tsfeatures](https://github.com/robjhyndman/tsfeatures)) and the [tsfresh](https://github.com/blue-yonder/tsfresh) library.

With this API the user can also generate holiday variables. Just enter the country of the special dates or a file with the specific dates and the API will return dummy variables of those dates for each observation in the dataset.

## tsforecast

[![CI/CD tsforecast Lambda](https://github.com/Nixtla/nixtla/actions/workflows/tsforecast-lambda.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsforecast-lambda.yml)
[![CI/CD tsforecast docker image](https://github.com/Nixtla/nixtla/actions/workflows/tsforecast-docker-image.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsforecast-docker-image.yml)

The [tsforecast](https://github.com/Nixtla/nixtla/tree/main/tsforecast) API is responsible for generating the time series forecasts. It receives as input the target data and can also receive static variables and time variables. At the moment, the API uses the [mlforecast](https://github.com/Nixtla/mlforecast) library developed by the Nixtla team using LightGBM as a model.

In future iterations, the user will be able to choose different Deep Learning models based on the [nixtlats](https://github.com/Nixtla/nixtlats) library developed by the Nixtla team.

## tsbenchmarks

[![CI/CD tsbenchmarks Lambda](https://github.com/Nixtla/nixtla/actions/workflows/tsbenchmarks-lambda.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsbenchmarks-lambda.yml)
[![CI/CD tsbenchmarks docker image](https://github.com/Nixtla/nixtla/actions/workflows/tsbenchmarks-docker-image.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsbenchmarks-docker-image.yml)

The [tsbenchmarks](https://github.com/Nixtla/nixtla/tree/main/tsbenchmarks) API is designed to easily compare the performance of models based on time series competition datasets. In particular, the API offers the possibility to evaluate forecasts of any frequency of the M4 competition and also of the M5 competition.


These APIs, written in Python and can be consumed through an [SDK](https://github.com/Nixtla/nixtla/tree/main/sdk/python-autotimeseries) also written in Python. The following diagram summarizes the structure of our pipeline:

<img src="https://raw.githubusercontent.com/Nixtla/nixtla/main/.github/images/sdk.png">

# Build your own Nixtla in AWS

## Why ?
We want to contribute to open source and help data scientists and developers to achieve great forecasting results without the need to implement complex pipelines.

## How?

If you want to use our hosted version send us an email or open a github issue and ask for API Keys.

If you want to deploy Nixtla on your own AWS Cloud you will need:

- API Gateway (to handle API calls).
- Lambda (or some computational unit).
- SageMaker (or some bigger computational unit).
- ECR (to store Docker images).
- S3 (for inputs and outputs).

You will end with an architecture that looks like the following diagram

<img src="https://raw.githubusercontent.com/Nixtla/nixtla/main/.github/images/Architecture.png">


Each call to the API executes a particular Lambda function depending on the endpoint. That particular lambda function instantiates a SageMaker job using a predefined type of instance. Finally, SageMaker reads the input data from S3 and writes the processed data to S3, using a predefined Docker image stored in ECR.

To create that infrastructue you can use our own Terraform code (infrastructure as code) or you can create the services from the console.

## 1. Terraform (infrastructure as Code)

Terraform is an open-source Infrastructure as Code tool that allows you to synthesize all the manual development into an automatic script. We have written all the needed steps to facilitate the deployment of Nixlta in your infrastructure. The Terraform code to create your infrastructure can be found [at this link](https://github.com/Nixtla/nixtla/tree/main/iac/terraform/aws). Just follow the next steps:

1. Define your AWS credentials. You can define them using:

```
export AWS_ACCESS_KEY_ID="anaccesskey"
export AWS_SECRET_ACCESS_KEY="asecretkey"
```

These credentials require permissions to use the S3, ECR, lambda and API Gateway services; in addition, you must be able to create IAM users.

2. To use Terraform, you must install it. [Here is an excellent guide](https://learn.hashicorp.com/tutorials/terraform/install-cli) to do so.

3. Position yourself in the `iac/terraform/aws` folder.

4. Run the command `terraform init`. This command will initialize the working directory with the necessary configuration.

5. Finally, you just need to use `terraform apply`. First, the list of services to be built will be displayed. You will have to accept to start the build. Once finished, you will get the API key needed to run the process, as well as the addresses of each of the APIs.

## 2. Create AWS resources using the console

### Create S3 buckets

For each service:
1. Create an S3 bucket. The code of each lambda function will be uploaded here.

### Create ECR repositorires

For each service:

1. Create a private repository for each service.

### Lambda Function

For each service:

1. Create a lambda function with `Python 3.7` runtime.
2. Modify the runtime setting and enter `main.handler` in the handler.
3. Go to the configuration:
	- Edit the general configuration and add a timeout of `9:59`.
	- Add an existing role capable of reading/writing from/to S3 and running Sagemaker services.
4. Add the following environment variables:
	- `PROCESSING_REPOSITORY_URI`: ECR URI of the docker image corresponding to the service.
	- `ROLE`: A  role capable of reading/writing from/to S3 and also running Sagemaker services.
 	- `INSTANCE_COUNT`
	- `INSTANCE_TYPE`

### API Gateway

1. Create a public REST API (Regional).
2. For each endpoint in `api/main.py`… add a resource.
3. For each created method add an ANY method:
	- Select lambda function.
	- Select Use Lambda Proxy Integration.
	- Introduce the name of the lambda function linked to that resource.
	- Once the method is created select Method Request and set API key required to true.
4. Deploy the API.

### Usage plan

1. Create a usage plan based on your needs.
2. Add your API stage.

### API Keys

1. Generate API keys as needed.

## Deployment

### GitHub secrets

1. Set the following secrets in your repo:
	- `AWS_ACCESS_KEY_ID`
	- `AWS_SECRET_ACCESS_KEY`
	- `AWS_DEFAULT_REGION`

## Run the API locally

  1. Create the environment using `make init`.
  2. Launch the app using `make app`.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/mergenthaler"><img src="https://avatars.githubusercontent.com/u/4086186?v=4?s=100" width="100px;" alt=""/><br /><sub><b>mergenthaler</b></sub></a><br /><a href="#ideas-mergenthaler" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!