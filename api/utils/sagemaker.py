import os
from typing import List
from uuid import uuid4

from dotenv import load_dotenv
from sagemaker.processing import (
    ScriptProcessor, ProcessingInput, ProcessingOutput
)


load_dotenv()

def run_sagemaker(url: str, dest_url: str,
                  output_name: str,
                  script: str, arguments: List[str]) -> None:
    """Run sagemaker ScriptProcessor.

    Parameters
    ----------
    url: str
        S3 uri of the files to process.
    dest_url: str
        S3 uri for output.
    output_name: str
        Name of the ouput file.
    script: str
        Script to use to process `url`.
    arguments: List[str]
        Arguments for the `script`.
    """
    dest_url_ = url if dest_url is None else dest_url
    # Setting ScriptProcessor job
    id_job = str(uuid4())
    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri=os.environ['PROCESSING_REPOSITORY_URI'],
        role=os.environ['ROLE'],
        instance_count=int(os.environ['INSTANCE_COUNT']),
        instance_type=os.environ['INSTANCE_TYPE'],
        base_job_name=id_job,
    )

    # Running job
    script_processor.run(
        code=script,
        inputs=[
            ProcessingInput(
                source=url,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                source='/opt/ml/processing/output',
                destination=dest_url_
            )
        ],
        arguments=arguments,
        wait=False,
    )

    output = {'id_job': id_job,
              'dest_url': f'{dest_url_}/{output_name}',
              'status': 200,
              'message': 'Check job status at GET /jobs/{job_id}'}

    return output
