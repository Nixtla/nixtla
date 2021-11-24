#! /bin/bash
set -e

cd ../../../../../

# create zip files
make create_api_zip

# upload zip files
make upload_api_zip s3_bucket=$S3_BUCKET

# clean zip files
make clean_api_zip