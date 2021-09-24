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

from utils.sagemaker import run_sagemaker
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

    response = {'status': 200,
                'message': 'Check job status at GET /tsfeatures/jobs/{job_id}'}

    response = {**response, **sagemaker_response}

    return response


class CalendarTSFeaturesArgs(BaseArgs):
    """Arguments to compute feature at scale."""
    country: str
    events: Optional[str] = None

@app.post('/calendartsfeatures')
def compute_tsfeatures(s3_args: S3Args, args: CalendarTSFeaturesArgs):
    """Calculates features using sagemaker."""
    sagemaker_response = run_sagemaker(url=s3_args.s3_url,
                                       dest_url=s3_args.s3_dest_url,
                                       output_name=f'calendar-features.csv',
                                       script='calendar/make_holidays.py',
                                       arguments=parse_args(args))

    response = {'status': 200,
                'message': 'Check job status at GET /tsfeatures/jobs/{job_id}'}

    response = {**response, **sagemaker_response}

    return response


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
