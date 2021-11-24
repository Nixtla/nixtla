variable "prefix" {}
variable "bucket" {}
variable "lambda_role_arn" {}
variable "sagemaker_role_arn" {}
variable "repository_url" {}
variable "instance_count" {}
variable "instance_type" {}

resource "aws_lambda_function" "lambda" {
    function_name = "${var.prefix}_nixtla_tsfeatures_lambda"
    s3_bucket     = "${var.bucket}"
    s3_key        = "functions/tsfeatures/api.zip"
    role          = var.lambda_role_arn
    handler       = "main.handler"
    memory_size   = 10240

    runtime = "python3.7"

    environment {
      variables = {
        PROCESSING_REPOSITORY_URI = var.repository_url,
        ROLE = var.sagemaker_role_arn,
        INSTANCE_COUNT = var.instance_count,
        INSTANCE_TYPE = var.instance_type
      }
    }
}