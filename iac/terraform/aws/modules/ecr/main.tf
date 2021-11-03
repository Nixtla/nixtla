variable "prefix" {}

resource "aws_ecr_repository" "ecr" {
  name                 = "${var.prefix}-nixtla-registry"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}