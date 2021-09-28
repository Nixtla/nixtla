import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import boto3
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from mangum import Mangum

from utils.sagemaker import run_sagemaker, run_sagemaker_hpo
from utils.utils import parse_args


app = FastAPI()


class S3Args(BaseModel):
    """File arguments."""
    s3_url: str
    s3_dest_url: Optional[str] = None

class BaseArgs(BaseModel):
    """Base model arguments."""
    filename: str
    unique_id_column: Optional[str] = 'unique_id'
    ds_column: Optional[str] = 'ds'
    y_column: Optional[str] = 'y'


class TSPreprocessArgs(BaseArgs):
    """Arguments to compute feature at scale."""
    kind: str

@app.post('/tspreprocess')
def compute_tspreprocess(s3_args: S3Args, args: TSPreprocessArgs):
    """Calculates features using sagemaker."""
    sagemaker_response = run_sagemaker(url=s3_args.s3_url,
                                       dest_url=s3_args.s3_dest_url,
                                       output_name=f'{args.kind}-preprocessed.csv',
                                       script='preprocess/make_preprocess.py',
                                       arguments=parse_args(args))

    return sagemaker_response


class TSFeaturesArgs(BaseArgs):
    """Arguments to compute feature at scale."""
    freq: int
    kind: str

@app.post('/tsfeatures')
def compute_tsfeatures(s3_args: S3Args, args: TSFeaturesArgs):
    """Calculates features using sagemaker."""
    sagemaker_response = run_sagemaker(url=s3_args.s3_url,
                                       dest_url=s3_args.s3_dest_url,
                                       output_name=f'{args.kind}-features.csv',
                                       script='features/make_features.py',
                                       arguments=parse_args(args))

    return sagemaker_response


class CalendarTSFeaturesArgs(BaseArgs):
    """Arguments to compute feature at scale."""
    country: str
    events: Optional[str] = None
    scale: Optional[bool] = False

@app.post('/calendartsfeatures')
def compute_calendartsfeatures(s3_args: S3Args, args: CalendarTSFeaturesArgs):
    """Calculates features using sagemaker."""
    sagemaker_response = run_sagemaker(url=s3_args.s3_url,
                                       dest_url=s3_args.s3_dest_url,
                                       output_name=f'calendar-features.csv',
                                       script='calendar/make_holidays.py',
                                       arguments=parse_args(args))

    return sagemaker_response


class LGBArgs(BaseModel):
    """Arguments fo LGB Model."""
    objective: Optional[str] = 'l2'
    metric: Optional[str] = 'rmse'
    learning_rate: Optional[float] = 0.1
    n_estimators: Optional[int] = 100
    num_leaves: Optional[int] = 128
    min_data_in_leaf: Optional[int] = 20
    bagging_freq: Optional[int] = 0
    bagging_fraction: Optional[float] = 1.

class TSForecastDataArgs(BaseArgs):
    """Data Arguments for LGB model."""
    freq: str
    horizon: int = 28
    filename_static: Optional[str] = None
    filename_temporal: Optional[str] = None
   
@app.post('/tsforecast')
def compute_tsforecast(s3_args: S3Args, data_args: TSForecastDataArgs,
                       model_args: LGBArgs):
    """calculates forecast using sagemaker."""
    args = parse_args(data_args) + parse_args(model_args)
    args += ['--dir-train', '/opt/ml/processing/input']
    args += ['--dir-output', '/opt/ml/processing/output']

    sagemaker_response = run_sagemaker(url=s3_args.s3_url,
                                       dest_url=s3_args.s3_dest_url,
                                       output_name=f'forecasts.csv',
                                       arguments=args)
    
    return sagemaker_response


class TSForecastHPOArgs(TSForecastDataArgs):
    """HPO arguments."""
    backtest_windows: int = 1

@app.post('/tsforecasthpo')
def compute_tsforecast_hpo(s3_args: S3Args, data_args: TSForecastHPOArgs):
    """calculates forecast using sagemaker."""
    args = data_args.dict()
    args = {key.replace("_", "-"): str(value) for key, value in args.items() if value is not None}
    sagemaker_response = run_sagemaker_hpo(url=s3_args.s3_url,
                                           dest_url=s3_args.s3_dest_url,
                                           dict_arguments=args)
    
    return sagemaker_response


class TSBenchmarksArgs(BaseArgs):
    """Arguments to compute benchmarks."""
    dataset: str

@app.post('/tsbenchmarks')
def compute_tsforecast(s3_args: S3Args, args: TSBenchmarksArgs):
    """Calculates forecast using sagemaker."""
    sagemaker_response = run_sagemaker(url=s3_args.s3_url,
                                       dest_url=s3_args.s3_dest_url,
                                       output_name=f'benchmarks.csv',
                                       script='benchmarks/compute_benchmarks.py',
                                       arguments=parse_args(args))
    
    return sagemaker_response


@app.get('/jobs/')
async def get_status_job(job_id: str):
    """Gets job status."""

    # Sagemaker status
    sagemaker = boto3.client('sagemaker')
    status = sagemaker.list_processing_jobs(NameContains=job_id)

    status = status['ProcessingJobSummaries'][0]
    if not status.get('ProcessingEndTime'):
        processing_time = datetime.now(tz=status['CreationTime'].tzinfo) - status['CreationTime']
    else:
        processing_time = status['ProcessingEndTime'] - status['CreationTime']
    processing_time = processing_time.seconds

    status = status['ProcessingJobStatus']

    # Cloudwatch status
    logs = boto3.client('logs')

    query = f"""
        fields @message
            | parse @message "*:__main__:*" as loggingType, loggingMessage
            | filter loggingType = "INFO"
            | filter @logStream like /{job_id}*/
            | display @timestamp, loggingMessage
        """

    start_query_response = logs.start_query(
        logGroupName='/aws/sagemaker/ProcessingJobs',
        startTime=int((datetime.today() - timedelta(hours=24)).timestamp()),
        endTime=int(datetime.now().timestamp()),
        queryString=query,
    )

    query_response = None
    while query_response is None or query_response['status'] == 'Running':
        print('Waiting for query to complete ...')
        time.sleep(1)
        query_response = logs.get_query_results(
            queryId=start_query_response['queryId']
        )

    logs = []
    for event in query_response['results']:
        for field in event:
            if field['field'] == 'loggingMessage':
                logs.append(field['value'])

    response = {'status': status, 'processing_time_seconds': processing_time,
                'logs': json.dumps(logs)}

    return response

handler = Mangum(app=app)
