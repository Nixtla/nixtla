# Nixtla
> Nixtla is an open-source time series forecasting library.

We are trying to help any data scientist or developer to have access to state-of-the-art forecasting pipelines. Either by setting up their own infrastructure in AWS by following the instructions in the repository. Or using our fully hosted version ([python SDK](https://github.com/Nixtla/nixtla/tree/main/sdk/)python-autotimeseries) which is in private beta right now. Just ask for free tokens to test the solution by sending an email to federico@nixtla.io or opening a GitHub issue.

We built a fully open-source time-series pipeline capable of achieving 1% of the performance in the [M5 competition](https://en.wikipedia.org/wiki/Makridakis_Competitions). Our open source solution has a 25% better accuracy than Amazon Forecast and is 20% more accurate than fbprophet. It also perfoms 4 times faster than Amazon Forecast and is less expensive.

To reproduce the results: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pmp4rqiwiPL-ambxTrJGBiNMS-7vm3v6?ts=616700c4) or you can read this [Medium Post](https://aws.plainenglish.io/automated-time-series-forecasting-pipeline-662e0feadd98)

At Nixtla we strongly believe in open-source, so we have released all the necessary code to set up your own time-series processing service in the cloud (using AWS, Azure is WIP). This repository uses continuous integration and deployment to deploy the APIs on our infrastructure.

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

# Build your own time-series processing service using AWS

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

## Run the API locally

1. Create the environment using `make init`.
2. Launch the app using `make app`.

## Create AWS resources

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


## Usage plan

1. Create a usage plan based on your needs.
2. Add your API stage.

## API Keys

1. Generate API keys as needed.

## Deployment

### GitHub secrets

1. Set the following secrets in your repo:
	- `AWS_ACCESS_KEY_ID`
	- `AWS_SECRET_ACCESS_KEY`
	- `AWS_DEFAULT_REGION`
