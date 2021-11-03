resource "aws_s3_bucket" "s3" {
  bucket = "my-tf-test-bucket-nov3"
  acl    = "private"

  tags = {
    Name        = "My bucket"
    Environment = "Dev"
  }
}