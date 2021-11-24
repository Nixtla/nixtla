variable "prefix" {}

resource "aws_s3_bucket" "s3" {
  bucket        = "${var.prefix}-nixtla"
  acl           = "private"

  tags = {
    Name        = "${var.prefix}-nixtla"
  }

  provisioner "local-exec" {
    working_dir = "${path.module}"
    command     = "/bin/bash api_lambda.sh"

    environment = {
      S3_BUCKET = resource.aws_s3_bucket.s3.bucket
    }
  }
}