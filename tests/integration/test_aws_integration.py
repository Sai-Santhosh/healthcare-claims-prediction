"""
AWS Integration Tests
Tests for AWS service integration using moto for mocking.
"""

import os
import sys
import pytest
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import moto for AWS mocking
try:
    import boto3
    from moto import mock_aws
    MOTO_AVAILABLE = True
except ImportError:
    MOTO_AVAILABLE = False
    mock_aws = lambda: lambda x: x  # Dummy decorator

from src.aws.s3_handler import S3Handler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'claim_id': range(1, 101),
        'amount': np.random.normal(500, 100, 100),
        'date': pd.date_range('2016-01-01', periods=100)
    })


@pytest.fixture
def aws_credentials():
    """Mock AWS credentials."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'


# =============================================================================
# S3 Tests
# =============================================================================

@pytest.mark.skipif(not MOTO_AVAILABLE, reason="moto not installed")
class TestS3Integration:
    """Integration tests for S3 operations."""
    
    @mock_aws
    def test_create_bucket(self, aws_credentials):
        """Test bucket creation."""
        s3 = boto3.client('s3', region_name='us-east-1')
        
        handler = S3Handler(bucket_name='test-bucket', region='us-east-1')
        handler.create_bucket()
        
        # Verify bucket exists
        response = s3.list_buckets()
        bucket_names = [b['Name'] for b in response['Buckets']]
        
        assert 'test-bucket' in bucket_names
    
    @mock_aws
    def test_upload_dataframe(self, aws_credentials, sample_dataframe):
        """Test DataFrame upload to S3."""
        s3 = boto3.client('s3', region_name='us-east-1')
        
        # Create bucket first
        s3.create_bucket(Bucket='test-bucket')
        
        handler = S3Handler(bucket_name='test-bucket', region='us-east-1')
        
        # Upload DataFrame
        key = 'test/data.parquet'
        handler.upload_dataframe(sample_dataframe, key)
        
        # Verify file exists
        response = s3.list_objects_v2(Bucket='test-bucket', Prefix='test/')
        keys = [obj['Key'] for obj in response.get('Contents', [])]
        
        assert key in keys
    
    @mock_aws
    def test_download_dataframe(self, aws_credentials, sample_dataframe):
        """Test DataFrame download from S3."""
        s3 = boto3.client('s3', region_name='us-east-1')
        
        # Create bucket and upload data
        s3.create_bucket(Bucket='test-bucket')
        
        handler = S3Handler(bucket_name='test-bucket', region='us-east-1')
        
        key = 'test/data.parquet'
        handler.upload_dataframe(sample_dataframe, key)
        
        # Download and verify
        downloaded_df = handler.download_dataframe(key)
        
        assert len(downloaded_df) == len(sample_dataframe)
        assert list(downloaded_df.columns) == list(sample_dataframe.columns)
    
    @mock_aws
    def test_list_objects(self, aws_credentials):
        """Test listing S3 objects."""
        s3 = boto3.client('s3', region_name='us-east-1')
        
        # Create bucket and upload files
        s3.create_bucket(Bucket='test-bucket')
        s3.put_object(Bucket='test-bucket', Key='bronze/file1.parquet', Body=b'data1')
        s3.put_object(Bucket='test-bucket', Key='bronze/file2.parquet', Body=b'data2')
        s3.put_object(Bucket='test-bucket', Key='silver/file3.parquet', Body=b'data3')
        
        handler = S3Handler(bucket_name='test-bucket', region='us-east-1')
        
        # List objects in bronze prefix
        objects = handler.list_objects(prefix='bronze/')
        
        assert len(objects) == 2
    
    @mock_aws
    def test_medallion_architecture(self, aws_credentials, sample_dataframe):
        """Test medallion architecture data flow."""
        s3 = boto3.client('s3', region_name='us-east-1')
        
        # Create bucket
        s3.create_bucket(Bucket='data-lake')
        
        handler = S3Handler(bucket_name='data-lake', region='us-east-1')
        
        # Simulate medallion architecture flow
        # Bronze layer
        handler.upload_dataframe(sample_dataframe, 'bronze/claims/data.parquet')
        
        # Silver layer (transformed)
        silver_df = sample_dataframe.copy()
        silver_df['amount_rounded'] = silver_df['amount'].round(2)
        handler.upload_dataframe(silver_df, 'silver/claims/data.parquet')
        
        # Gold layer (aggregated)
        gold_df = silver_df.groupby(silver_df['date'].dt.month).agg({
            'claim_id': 'count',
            'amount': 'sum'
        }).reset_index()
        handler.upload_dataframe(gold_df, 'gold/claims/monthly_summary.parquet')
        
        # Verify all layers have data
        bronze_objects = handler.list_objects(prefix='bronze/')
        silver_objects = handler.list_objects(prefix='silver/')
        gold_objects = handler.list_objects(prefix='gold/')
        
        assert len(bronze_objects) == 1
        assert len(silver_objects) == 1
        assert len(gold_objects) == 1


# =============================================================================
# Lambda Handler Tests
# =============================================================================

@pytest.mark.skipif(not MOTO_AVAILABLE, reason="moto not installed")
class TestLambdaHandler:
    """Tests for Lambda ETL handler."""
    
    def test_identify_s3_trigger(self):
        """Test S3 trigger identification."""
        from src.aws.lambda.etl_handler import _identify_trigger
        
        s3_event = {
            "Records": [{
                "s3": {
                    "bucket": {"name": "test-bucket"},
                    "object": {"key": "test/file.parquet"}
                }
            }]
        }
        
        trigger_type = _identify_trigger(s3_event)
        assert trigger_type == "s3"
    
    def test_identify_schedule_trigger(self):
        """Test schedule trigger identification."""
        from src.aws.lambda.etl_handler import _identify_trigger
        
        schedule_event = {
            "source": "aws.events",
            "detail-type": "Scheduled Event"
        }
        
        trigger_type = _identify_trigger(schedule_event)
        assert trigger_type == "schedule"
    
    def test_identify_api_trigger(self):
        """Test API Gateway trigger identification."""
        from src.aws.lambda.etl_handler import _identify_trigger
        
        api_event = {
            "httpMethod": "POST",
            "body": "{\"action\": \"run\"}"
        }
        
        trigger_type = _identify_trigger(api_event)
        assert trigger_type == "api"
    
    def test_extract_parameters_s3(self):
        """Test parameter extraction from S3 event."""
        from src.aws.lambda.etl_handler import _extract_parameters
        
        s3_event = {
            "Records": [{
                "s3": {
                    "bucket": {"name": "test-bucket"},
                    "object": {"key": "raw/claims.parquet", "size": 1000000}
                }
            }]
        }
        
        params = _extract_parameters(s3_event, "s3")
        
        assert params["bucket"] == "test-bucket"
        assert params["key"] == "raw/claims.parquet"
        assert params["size"] == 1000000


# =============================================================================
# CloudWatch Tests
# =============================================================================

@pytest.mark.skipif(not MOTO_AVAILABLE, reason="moto not installed")
class TestCloudWatchIntegration:
    """Tests for CloudWatch metrics integration."""
    
    @mock_aws
    def test_put_metric(self, aws_credentials):
        """Test publishing metric to CloudWatch."""
        from src.monitoring.metrics import CloudWatchMetrics
        
        cw = CloudWatchMetrics(namespace='TestPipeline', region='us-east-1')
        
        result = cw.put_metric(
            metric_name='RowsProcessed',
            value=1000,
            unit='Count',
            dimensions={'Pipeline': 'test'}
        )
        
        # In mock environment, this should succeed
        assert result is True


# =============================================================================
# SNS Tests
# =============================================================================

@pytest.mark.skipif(not MOTO_AVAILABLE, reason="moto not installed")
class TestSNSIntegration:
    """Tests for SNS alerting integration."""
    
    @mock_aws
    def test_create_topic(self, aws_credentials):
        """Test SNS topic creation."""
        from src.monitoring.alerting import SNSAlerter
        
        alerter = SNSAlerter(topic_name='test-alerts', region='us-east-1')
        topic_arn = alerter._get_topic_arn()
        
        assert topic_arn is not None
        assert 'test-alerts' in topic_arn


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
