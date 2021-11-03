# Prerequisites

1. nvidia AMI.

# Usage

1. Create docker image: `make init`.
2. Run your custom python module using: `make run_module module="python -m <YOUR_MODULE>"`.

You can test the correct behavior using: `make run_module module="python -m test".
