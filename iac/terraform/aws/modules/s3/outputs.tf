output "bucket" {
    value = aws_s3_bucket.s3.bucket
    description = "s3 bucket name"
}