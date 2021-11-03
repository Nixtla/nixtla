variable "prefix" {}

resource "aws_s3_bucket" "s3" {
  bucket = "${var.prefix}-nixtla"
  acl    = "private"

  tags = {
    Name        = "${var.prefix}-nixtla"
  }
}