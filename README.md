# Nixtla
> Nixtla is an open-source time series forecasting library for the AWS cloud. 

[![CI/CD tspreprocess Lambda](https://github.com/Nixtla/nixtla/actions/workflows/tspreprocess-lambda.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tspreprocess-lambda.yml)
[![CI/CD tspreprocess docker image](https://github.com/Nixtla/nixtla/actions/workflows/tspreprocess-docker-image.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tspreprocess-docker-image.yml)

[![CI/CD tsfeatures Lambda](https://github.com/Nixtla/nixtla/actions/workflows/tsfeatures-lambda.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsfeatures-lambda.yml)
[![CI/CD tsfeatures docker image](https://github.com/Nixtla/nixtla/actions/workflows/tsfeatures-docker-image.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsfeatures-docker-image.yml)

[![CI/CD tsbenchmarks Lambda](https://github.com/Nixtla/nixtla/actions/workflows/tsbenchmarks-lambda.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsbenchmarks-lambda.yml)
[![CI/CD tsbenchmarks docker image](https://github.com/Nixtla/nixtla/actions/workflows/tsbenchmarks-docker-image.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsbenchmarks-docker-image.yml)

[![CI/CD tsforecast Lambda](https://github.com/Nixtla/nixtla/actions/workflows/tsforecast-lambda.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsforecast-lambda.yml)
[![CI/CD tsforecast docker image](https://github.com/Nixtla/nixtla/actions/workflows/tsforecast-docker-image.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/tsforecast-docker-image.yml)

# Current wraped functionalities

- tspreprocess: library for Time Series preprocessing
- tsfeatures: library for automatic feature extraction
- tsforecast: library for Time Series forecasting
- tsbenchmarks: library for Benchmarking


# Why?
We want to contribute to open source and help data scientists and developers to achieve great forecasting results without the need to implement complex pipelines.

# How?

We are testing the concept and are eager to gain Beta users for the hosted version of Nixtla. If you want to test the hosted Nixtla version of Nixtla, ask for API Keys.

If you want to deploy Nixtla on your own AWS Cloud you will need:

- API Gateway (to handle API calls)
- Lambda (or some computantional unit)
- Sagemaker (or some bigger computational unit)
- S3 (for inputs and outputs)

You will end with an architecture that looks like the following diagram

<img src="https://raw.githubusercontent.com/nixtla/fasttsfeatures/main/.github/images/Architecture.png">


The idea is that every module of the forecasting suite is an independent API endpoint that invokes a Lambda function that turns on a dedicated Sagemaker instance to run the specific processing job. 


# Run the API locally

1. Create the environment using `make init`.
2. Launch the app using `make app`.

# Create AWS resources

## Create S3 buckets

For each service:
1. Create an S3 bucket. The code of each lambda function will be uploaded here.

## Create ECR repositorires

For each service:

1. Create a private repository for each service.

## Lambda Function

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

## API Gateway

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

# Deployment

## GitHub secrets

1. Set the following secrets in your repo:
	- `AWS_ACCESS_KEY_ID`
	- `AWS_SECRET_ACCESS_KEY`
	- `AWS_DEFAULT_REGION`
