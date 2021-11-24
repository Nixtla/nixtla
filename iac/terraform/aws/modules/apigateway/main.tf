variable "prefix" {}
variable "tsbenchmarks_lambda_invoke_arn" {}
variable "tsbenchmarks_lambda_name" {}
variable "tsfeatures_lambda_invoke_arn" {}
variable "tsfeatures_lambda_name" {}
variable "tsforecast_lambda_invoke_arn" {}
variable "tsforecast_lambda_name" {}
variable "tspreprocess_lambda_invoke_arn" {}
variable "tspreprocess_lambda_name" {}

resource "aws_api_gateway_rest_api" "api" {
  name = "${var.prefix}-nixtla-api"
}

# tsbenchmarks
resource "aws_api_gateway_resource" "tsbenchmarks_resource" {
  path_part   = "tsbenchmarks"
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  rest_api_id = aws_api_gateway_rest_api.api.id
}

resource "aws_api_gateway_method" "tsbenchmarks_method" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.tsbenchmarks_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "tsbenchmarks_integration" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.tsbenchmarks_resource.id
  http_method             = aws_api_gateway_method.tsbenchmarks_method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = var.tsbenchmarks_lambda_invoke_arn
}

resource "aws_lambda_permission" "tsbenchmarks_lambda_permission" {
  action        = "lambda:InvokeFunction"
  function_name = var.tsbenchmarks_lambda_name
  principal     = "apigateway.amazonaws.com"

  # The /*/*/* part allows invocation from any stage, method and resource path
  # within API Gateway REST API.
  source_arn = "${aws_api_gateway_rest_api.api.execution_arn}/*/*/*"
}

# tsfeatures
resource "aws_api_gateway_resource" "tsfeatures_resource" {
  path_part   = "tsfeatures"
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  rest_api_id = aws_api_gateway_rest_api.api.id
}

resource "aws_api_gateway_method" "tsfeatures_method" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.tsfeatures_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "tsfeatures_integration" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.tsfeatures_resource.id
  http_method             = aws_api_gateway_method.tsfeatures_method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = var.tsfeatures_lambda_invoke_arn
}

resource "aws_lambda_permission" "tsfeatures_lambda_permission" {
  action        = "lambda:InvokeFunction"
  function_name = var.tsfeatures_lambda_name
  principal     = "apigateway.amazonaws.com"

  # The /*/*/* part allows invocation from any stage, method and resource path
  # within API Gateway REST API.
  source_arn = "${aws_api_gateway_rest_api.api.execution_arn}/*/*/*"
}

# tsforecast
resource "aws_api_gateway_resource" "tsforecast_resource" {
  path_part   = "tsforecast"
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  rest_api_id = aws_api_gateway_rest_api.api.id
}

resource "aws_api_gateway_method" "tsforecast_method" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.tsforecast_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "tsforecast_integration" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.tsforecast_resource.id
  http_method             = aws_api_gateway_method.tsforecast_method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = var.tsforecast_lambda_invoke_arn
}

resource "aws_lambda_permission" "tsforecast_lambda_permission" {
  action        = "lambda:InvokeFunction"
  function_name = var.tsforecast_lambda_name
  principal     = "apigateway.amazonaws.com"

  # The /*/*/* part allows invocation from any stage, method and resource path
  # within API Gateway REST API.
  source_arn = "${aws_api_gateway_rest_api.api.execution_arn}/*/*/*"
}

# tspreprocess
resource "aws_api_gateway_resource" "tspreprocess_resource" {
  path_part   = "tspreprocess"
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  rest_api_id = aws_api_gateway_rest_api.api.id
}

resource "aws_api_gateway_method" "tspreprocess_method" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.tspreprocess_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "tspreprocess_integration" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.tspreprocess_resource.id
  http_method             = aws_api_gateway_method.tspreprocess_method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = var.tspreprocess_lambda_invoke_arn
}

resource "aws_lambda_permission" "tspreprocess_lambda_permission" {
  action        = "lambda:InvokeFunction"
  function_name = var.tspreprocess_lambda_name
  principal     = "apigateway.amazonaws.com"

  # The /*/*/* part allows invocation from any stage, method and resource path
  # within API Gateway REST API.
  source_arn = "${aws_api_gateway_rest_api.api.execution_arn}/*/*/*"
}

# apigateway deployment
resource "aws_api_gateway_deployment" "apigateway_deployment" {
  rest_api_id = aws_api_gateway_rest_api.api.id

  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.tsbenchmarks_resource.id,
      aws_api_gateway_method.tsbenchmarks_method.id,
      aws_api_gateway_integration.tsbenchmarks_integration.id,
      aws_api_gateway_resource.tsfeatures_resource.id,
      aws_api_gateway_method.tsfeatures_method.id,
      aws_api_gateway_integration.tsfeatures_integration.id,
      aws_api_gateway_method.tsforecast_method.id,
      aws_api_gateway_integration.tsforecast_integration.id,
      aws_api_gateway_integration.tsforecast_integration.id,
      aws_api_gateway_resource.tspreprocess_resource.id,
      aws_api_gateway_method.tspreprocess_method.id,
      aws_api_gateway_integration.tspreprocess_integration.id
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "apigateway_stage" {
  deployment_id = aws_api_gateway_deployment.apigateway_deployment.id
  rest_api_id   = aws_api_gateway_rest_api.api.id
  stage_name    = "prod"
}

# usage plan
resource "aws_api_gateway_usage_plan" "apigateway_usage_plan" {
  name         = "${var.prefix}-nixtla-usage-plan"
  description  = "Nixtla usage plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.api.id
    stage  = aws_api_gateway_stage.apigateway_stage.stage_name
  }

  quota_settings {
    limit  = 100
    period = "MONTH"
  }

  throttle_settings {
    burst_limit = 50
    rate_limit  = 100
  }
}

# api key
resource "aws_api_gateway_api_key" "apigateway_api_key" {
  name = "${var.prefix}-nixtla-api-key"
}

resource "aws_api_gateway_usage_plan_key" "apigateway_usage_plan_key" {
  key_id        = aws_api_gateway_api_key.apigateway_api_key.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.apigateway_usage_plan.id
}