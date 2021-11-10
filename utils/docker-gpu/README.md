# Prerequisites

1. nvidia AMI.

To launch an NVIDIA AMI you can use:

`aws ec2 run-instances --image-id ami-05e329519be512f1b --count 1 --instance-type g4dn.2xlarge --key-name <YOUR_KEY_NAME> --security-groups <YOUR_SECURITY_GROUP>`

# Usage

1. Create docker image: `make init`.
2. Run your custom python module using: `make run_module module="python -m <YOUR_MODULE>"`.

You can test the correct behavior using: `make run_module module="python -m test".
