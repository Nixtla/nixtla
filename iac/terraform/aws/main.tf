module "s3" {
  source = "./modules/s3"
  prefix = var.prefix
}

module "ecr" {
  source = "./modules/ecr"
  prefix = var.prefix
  region = var.region
}

module "iam" {
  source           = "./modules/iam"
  prefix           = var.prefix
  bucket           = module.s3.bucket
  arn_repositories = module.ecr.arn_repositories
}

module "tsbenchmarks" {
  source = "./modules/tsbenchmarks"

  bucket             = module.s3.bucket
  repository_url     = module.ecr.tsbenchmarks_repository_url
  lambda_role_arn    = module.iam.lambda_role_arn
  sagemaker_role_arn = module.iam.sagemaker_role_arn
  prefix             = var.prefix
  instance_count     = var.instance_count
  instance_type      = var.instace_type
}

module "tsfeatures" {
  source = "./modules/tsfeatures"

  bucket             = module.s3.bucket
  repository_url     = module.ecr.tsfeatures_repository_url
  lambda_role_arn    = module.iam.lambda_role_arn
  sagemaker_role_arn = module.iam.sagemaker_role_arn
  prefix             = var.prefix
  instance_count     = var.instance_count
  instance_type      = var.instace_type
}

module "tsforecast" {
  source = "./modules/tsforecast"

  bucket             = module.s3.bucket
  repository_url     = module.ecr.tsforecast_repository_url
  lambda_role_arn    = module.iam.lambda_role_arn
  sagemaker_role_arn = module.iam.sagemaker_role_arn
  prefix             = var.prefix
  instance_count     = var.instance_count
  instance_type      = var.instace_type
}

module "tspreprocess" {
  source = "./modules/tspreprocess"

  bucket             = module.s3.bucket
  repository_url     = module.ecr.tspreprocess_repository_url
  lambda_role_arn    = module.iam.lambda_role_arn
  sagemaker_role_arn = module.iam.sagemaker_role_arn
  prefix             = var.prefix
  instance_count     = var.instance_count
  instance_type      = var.instace_type
}

module "apigateway" {
  source                         = "./modules/apigateway"
  prefix                         = var.prefix
  tsbenchmarks_lambda_invoke_arn = module.tsbenchmarks.lambda_invoke_arn
  tsbenchmarks_lambda_name       = module.tsbenchmarks.lambda_name
  tsfeatures_lambda_invoke_arn   = module.tsfeatures.lambda_invoke_arn
  tsfeatures_lambda_name         = module.tsfeatures.lambda_name
  tsforecast_lambda_invoke_arn   = module.tsforecast.lambda_invoke_arn
  tsforecast_lambda_name         = module.tsforecast.lambda_name
  tspreprocess_lambda_invoke_arn = module.tspreprocess.lambda_invoke_arn
  tspreprocess_lambda_name       = module.tspreprocess.lambda_name
}