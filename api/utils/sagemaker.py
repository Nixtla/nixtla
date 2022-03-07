import datetime
import os
from typing import Dict, List, Optional
from uuid import uuid4

import boto3
from dotenv import load_dotenv
from sagemaker.processing import (
    Processor, ScriptProcessor, ProcessingInput, ProcessingOutput
)


load_dotenv()

def run_sagemaker(url: str, dest_url: str,
                  output_name: str,
                  arguments: List[str],
                  script: Optional[str] = None) -> None:
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
    processor_args = dict(
        image_uri=os.environ['PROCESSING_REPOSITORY_URI'],
        role=os.environ['ROLE'],
        instance_count=int(os.environ['INSTANCE_COUNT']),
        instance_type=os.environ['INSTANCE_TYPE'],
        base_job_name=id_job,
    )
    dest_input = '/opt/ml/processing/input'
    source_output = '/opt/ml/processing/output'

    run_args = dict(
        inputs=[
            ProcessingInput(
                source=url,
                destination=dest_input
            )
        ],
        outputs=[
            ProcessingOutput(
                source=source_output,
                destination=dest_url_
            )
        ],
        arguments=arguments,
        wait=False,
        logs=False,
    )
    try:
        if script is not None:
            script_processor = ScriptProcessor(
                    command=['python3'],
                    **processor_args,
            )

            # Running job
            script_processor.run(
                    code=script,
                    **run_args,
            )
        else:
            processor = Processor(
                    entrypoint=['python', '/opt/ml/code/train.py'],
                    **processor_args,
            )

            # Running job
            processor.run(
                    **run_args,
            )


        output = {'id_job': id_job,
                  'dest_url': f'{dest_url_}/{output_name}',
                  'status': 200,
                  'message': 'Check job status at GET /jobs/{job_id}'}
    except:
        output = {'status': 403, 
                  'message': 'This demo service is full at the moment, please try later or contact federico AT nixtla.io'}


    return output


def run_sagemaker_hpo(url: str, dest_url: str,
                      dict_arguments: Dict[str, str]):
    """Run sagemaker HPO.

    Parameters
    ----------
    url: str
        S3 uri of the files to process.
    dest_url: str
        S3 uri for output.
    dict_arguments: Dict[str, str]
        Static Arguments.
    """
    now = datetime.datetime.now()
    id_job = now.strftime('%Y-%m-%d-%H-%M-%S')
    s3_output = url if dest_url is None else dest_url
    s3_output = f'{s3_output}{id_job}'

    sagemaker = boto3.client('sagemaker')
    
    sagemaker.create_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=id_job,
            HyperParameterTuningJobConfig={
                'Strategy': 'Bayesian',
                'HyperParameterTuningJobObjective': {
                    'Type': 'Minimize',
                    'MetricName': 'validation:error',
                },
                'ResourceLimits': {
                    'MaxNumberOfTrainingJobs': 50,
                    'MaxParallelTrainingJobs': 8,
                },
                'ParameterRanges': {
                    'IntegerParameterRanges': [
                        {
                            'Name': 'n-estimators',
                            'MinValue': '100',
                            'MaxValue': '1000',
                            'ScalingType': 'Linear',
                        },
                        {
                            'Name': 'num-leaves',
                            'MinValue': '30',
                            'MaxValue': '200',
                            'ScalingType': 'Linear',
                        },
                        {
                            'Name': 'min-data-in-leaf',
                            'MinValue': '20',
                            'MaxValue': '200',
                            'ScalingType': 'Linear',
                        },
                        {
                            'Name': 'bagging-freq',
                            'MinValue': '1',
                            'MaxValue': '5',
                            'ScalingType': 'Linear',
                        }
                    ],
                    'ContinuousParameterRanges': [
                        {
                            'Name': 'learning-rate',
                            'MinValue': '0.01',
                            'MaxValue': '0.2',
                            'ScalingType': 'Logarithmic',
                        },
                        {
                            'Name': 'bagging-fraction',
                            'MinValue': '0.5',
                            'MaxValue': '0.95',
                            'ScalingType': 'Linear',
                        },
                    ],
                    'CategoricalParameterRanges': [
                        {
                            'Name': 'objective',
                            'Values': ['poisson', 'tweedie'],
                        },
                    ]
                },
                'TrainingJobEarlyStoppingType': 'Off',
            },
            TrainingJobDefinition={
                'StaticHyperParameters': dict_arguments,
                'AlgorithmSpecification': {
                    'TrainingImage': os.environ['PROCESSING_REPOSITORY_URI'],
                    'TrainingInputMode': 'File',
                    'MetricDefinitions': [
                        {
                            'Name': 'validation:error',
                            'Regex': 'RMSE: (\d*\.\d*)',
                        }
                    ]
                },
                'RoleArn': os.environ['ROLE'],
                'InputDataConfig': [
                    {
                        'ChannelName': 'train',
                        'DataSource': {
                            'S3DataSource':
                                {
                                    'S3DataType': 'S3Prefix',
                                    'S3Uri': url,
                                }
                        }
                    },
                ],
                'OutputDataConfig': {
                    'S3OutputPath': s3_output,
                },
                'ResourceConfig': {
                    'InstanceType': os.environ['INSTANCE_TYPE'],
                    'InstanceCount': int(os.environ['INSTANCE_COUNT']),
                    'VolumeSizeInGB': 20,
                },
                'StoppingCondition': {
                    'MaxRuntimeInSeconds': 60 * 60,
                },
            },
        )

    output = {'id_job': id_job,
              'status': 200}

    return output
