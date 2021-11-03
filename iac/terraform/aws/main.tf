module "s3" {
  source = "./modules/s3"
  prefix = var.prefix
}

module "ecr" {
  source = "./modules/ecr"
  prefix = var.prefix
}