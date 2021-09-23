# Nixtla
Build a ...

# Test the api locally

1. Create the environment using `make init`.
2. Launch the app using `make app`.

# Create AWS resources

## Create ECR repositorires

For each service:

1. Create a private repository for each service.

## Lambda Function

For each service:

1. Create a lambda function from scrath and `Python 3.7` runtime.
2. Modify the runtime setting and enter `main.handler` in the handler.
3. Go to configuration:
	- Edit the general configuration and add a timeout of `9:59`.
	- Add an existing role capable of reading/writing from/to S3 and also running sagemaker services.
4. Add the following environment variables:
	- `PROCESSING_REPOSITORY_URI`: ECR URI of the docker image particular for the service.
	- `ROLE`: An existing role capable of reading/writing from/to S3 and also running sagemaker services.
 	- `INSTANCE_COUNT`
	- `INSTANCE_TYPE`

## API Gateway

1. Create a public REST API (Regional).
2. For each enpoint in `api/main.py` add a resource.
3. For each created method add an ANY method:
	- Select lambda function.
	- Select Use Lambda Proxy Integration.
	- Introduce the name of the lambda function linked to that resource.
	- Once the method is created select Method Request and set API key requiered to true.
4. Deploy the API.


## Usage plan

1. Create a usage plan based on your needs.
2. Add your API stage.

## Api Keys

1. Generate API keys as needed.
