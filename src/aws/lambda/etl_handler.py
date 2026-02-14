"""
AWS Lambda ETL Handler
Serverless entry point for ETL pipeline execution.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for ETL pipeline execution.
    
    Supports multiple trigger types:
    - S3 event (new file arrival)
    - CloudWatch scheduled event
    - API Gateway request
    - Step Functions state machine
    
    Args:
        event: Lambda event payload
        context: Lambda context
        
    Returns:
        Response with execution status
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Determine trigger type
        trigger_type = _identify_trigger(event)
        logger.info(f"Trigger type: {trigger_type}")
        
        # Extract parameters
        params = _extract_parameters(event, trigger_type)
        
        # Execute appropriate pipeline
        if trigger_type == "s3":
            result = _handle_s3_trigger(params)
        elif trigger_type == "schedule":
            result = _handle_schedule_trigger(params)
        elif trigger_type == "api":
            result = _handle_api_trigger(params)
        elif trigger_type == "step_functions":
            result = _handle_step_functions_trigger(params)
        else:
            result = _handle_manual_trigger(params)
        
        # Log metrics
        _publish_metrics(result)
        
        # Send success notification
        if result.get("status") == "success":
            _send_notification(
                subject="ETL Pipeline Completed",
                message=f"Pipeline completed successfully.\n"
                        f"Rows processed: {result.get('rows_processed', 0):,}\n"
                        f"Duration: {result.get('duration_seconds', 0):.2f}s"
            )
        
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        
        # Send failure notification
        _send_notification(
            subject="ETL Pipeline Failed",
            message=f"Pipeline failed with error:\n{str(e)}",
            severity="error"
        )
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "failed",
                "error": str(e)
            })
        }


def _identify_trigger(event: Dict[str, Any]) -> str:
    """Identify the type of trigger that invoked the Lambda."""
    if "Records" in event and event["Records"]:
        record = event["Records"][0]
        if "s3" in record:
            return "s3"
        elif "eventSource" in record:
            return record["eventSource"]
    
    if "source" in event and event["source"] == "aws.events":
        return "schedule"
    
    if "httpMethod" in event:
        return "api"
    
    if "execution_id" in event:
        return "step_functions"
    
    return "manual"


def _extract_parameters(event: Dict[str, Any], trigger_type: str) -> Dict[str, Any]:
    """Extract parameters from event based on trigger type."""
    params = {
        "trigger_type": trigger_type,
        "timestamp": datetime.now().isoformat(),
        "execution_id": os.environ.get("AWS_REQUEST_ID", "local")
    }
    
    if trigger_type == "s3":
        record = event["Records"][0]["s3"]
        params["bucket"] = record["bucket"]["name"]
        params["key"] = record["object"]["key"]
        params["size"] = record["object"].get("size", 0)
        
    elif trigger_type == "api":
        if "body" in event and event["body"]:
            body = json.loads(event["body"])
            params.update(body)
        if "queryStringParameters" in event and event["queryStringParameters"]:
            params.update(event["queryStringParameters"])
            
    elif trigger_type == "step_functions":
        params.update(event)
    
    return params


def _handle_s3_trigger(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle S3 file arrival trigger."""
    logger.info(f"Processing S3 file: s3://{params['bucket']}/{params['key']}")
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Download file
    local_path = f"/tmp/{os.path.basename(params['key'])}"
    s3.download_file(params['bucket'], params['key'], local_path)
    
    # Run ETL pipeline
    from src.etl.pipeline import ETLPipeline
    from src.etl.extract import DataExtractor
    from src.etl.transform import DataTransformer
    from src.etl.load import S3Loader
    from src.data_quality.expectations import DataQualityChecker
    
    # Load configuration
    config = _load_config()
    
    # Build pipeline
    pipeline = ETLPipeline(name="s3_triggered_pipeline", config=config)
    pipeline.add_stage(DataExtractor(name="extract"))
    pipeline.add_stage(DataTransformer(name="transform"))
    pipeline.add_stage(DataQualityChecker(name="quality_check", fail_on_error=False))
    pipeline.add_stage(S3Loader(
        name="load_to_silver",
        bucket=params['bucket'],
        prefix="silver/claims",
        layer="silver"
    ))
    
    # Execute
    result = pipeline.run({"input_file": local_path})
    
    return {
        "status": result.status.value,
        "pipeline_id": result.pipeline_id,
        "rows_processed": result.total_rows_processed,
        "duration_seconds": result.total_duration_seconds,
        "source": f"s3://{params['bucket']}/{params['key']}"
    }


def _handle_schedule_trigger(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle CloudWatch scheduled trigger."""
    logger.info("Running scheduled ETL pipeline")
    
    from src.etl.pipeline import ETLPipeline
    from src.etl.extract import DataExtractor
    from src.etl.transform import DataTransformer
    from src.etl.load import DataLoader
    from src.data_quality.expectations import DataQualityChecker
    
    config = _load_config()
    
    # Build full pipeline
    pipeline = ETLPipeline(name="scheduled_pipeline", config=config)
    pipeline.add_stage(DataExtractor(name="extract"))
    pipeline.add_stage(DataTransformer(name="transform"))
    pipeline.add_stage(DataQualityChecker(name="quality_check", fail_on_error=False))
    pipeline.add_stage(DataLoader(name="load"))
    
    result = pipeline.run()
    
    return {
        "status": result.status.value,
        "pipeline_id": result.pipeline_id,
        "rows_processed": result.total_rows_processed,
        "duration_seconds": result.total_duration_seconds,
        "trigger": "schedule"
    }


def _handle_api_trigger(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle API Gateway trigger."""
    logger.info(f"Running API-triggered pipeline with params: {params}")
    
    action = params.get("action", "run")
    
    if action == "run":
        return _handle_schedule_trigger(params)
    
    elif action == "status":
        return _get_pipeline_status(params.get("pipeline_id"))
    
    elif action == "quality":
        return _run_quality_checks(params)
    
    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}"
        }


def _handle_step_functions_trigger(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle Step Functions state machine trigger."""
    stage = params.get("stage", "extract")
    logger.info(f"Running Step Functions stage: {stage}")
    
    config = _load_config()
    
    if stage == "extract":
        from src.etl.extract import DataExtractor
        extractor = DataExtractor()
        result = extractor.execute(params)
        
    elif stage == "transform":
        from src.etl.transform import DataTransformer
        transformer = DataTransformer()
        result = transformer.execute(params)
        
    elif stage == "quality":
        from src.data_quality.expectations import DataQualityChecker
        checker = DataQualityChecker(fail_on_error=False)
        result = checker.execute(params)
        
    elif stage == "load":
        from src.etl.load import DataLoader
        loader = DataLoader()
        result = loader.execute(params)
    
    else:
        return {"status": "error", "message": f"Unknown stage: {stage}"}
    
    return {
        "status": result.status.value,
        "stage": stage,
        "rows_processed": result.rows_processed,
        "output_path": result.output_path
    }


def _handle_manual_trigger(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle manual/test trigger."""
    logger.info("Running manual ETL pipeline")
    return _handle_schedule_trigger(params)


def _load_config() -> Dict[str, Any]:
    """Load pipeline configuration."""
    import yaml
    
    config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {
            "aws": {
                "region": os.environ.get("AWS_REGION", "us-east-1"),
                "s3": {
                    "data_lake_bucket": os.environ.get("S3_BUCKET", "claims-data-lake")
                }
            }
        }


def _publish_metrics(result: Dict[str, Any]) -> None:
    """Publish metrics to CloudWatch."""
    try:
        cloudwatch = boto3.client('cloudwatch')
        
        metrics = [
            {
                'MetricName': 'RowsProcessed',
                'Value': result.get('rows_processed', 0),
                'Unit': 'Count'
            },
            {
                'MetricName': 'PipelineDuration',
                'Value': result.get('duration_seconds', 0),
                'Unit': 'Seconds'
            },
            {
                'MetricName': 'PipelineSuccess' if result.get('status') == 'success' else 'PipelineFailure',
                'Value': 1,
                'Unit': 'Count'
            }
        ]
        
        cloudwatch.put_metric_data(
            Namespace='ClaimsPipeline',
            MetricData=metrics
        )
        
        logger.info("Published metrics to CloudWatch")
        
    except Exception as e:
        logger.warning(f"Failed to publish metrics: {e}")


def _send_notification(
    subject: str,
    message: str,
    severity: str = "info"
) -> None:
    """Send notification via SNS."""
    try:
        sns = boto3.client('sns')
        topic_arn = os.environ.get("SNS_TOPIC_ARN")
        
        if topic_arn:
            sns.publish(
                TopicArn=topic_arn,
                Subject=f"[{severity.upper()}] {subject}",
                Message=message
            )
            logger.info(f"Sent notification: {subject}")
            
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")


def _get_pipeline_status(pipeline_id: Optional[str]) -> Dict[str, Any]:
    """Get status of a pipeline execution."""
    # In production, this would query DynamoDB or similar
    return {
        "status": "not_implemented",
        "pipeline_id": pipeline_id,
        "message": "Pipeline status tracking requires DynamoDB setup"
    }


def _run_quality_checks(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run data quality checks only."""
    from src.data_quality.expectations import DataQualityChecker, ExpectationSuite
    
    # Build expectation suite
    suite = ExpectationSuite("claims_validation")
    suite.expect_table_row_count_to_be_between(1000, 50000000)
    suite.expect_column_values_to_be_positive("amt_paid", mostly=0.99)
    suite.expect_column_values_to_be_positive("amt_billed", mostly=0.99)
    
    checker = DataQualityChecker(suite=suite, fail_on_error=False)
    result = checker.execute(params)
    
    return {
        "status": "success" if result.status.value == "success" else "failed",
        "quality_checks": result.metadata
    }
