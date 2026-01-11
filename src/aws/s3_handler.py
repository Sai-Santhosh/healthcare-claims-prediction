"""
AWS S3 Handler Module
Handles interactions with Amazon S3 for data storage and retrieval.
"""

import io
import json
import pickle
from pathlib import Path
from typing import Optional, List, Union, BinaryIO
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError
import pandas as pd

from ..utils.logger import get_logger
from ..utils.helpers import timer

logger = get_logger(__name__)


@dataclass
class S3Object:
    """Represents an S3 object."""
    bucket: str
    key: str
    size: int
    last_modified: str
    
    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class S3Handler:
    """
    Handler for AWS S3 operations.
    Provides methods for uploading, downloading, and managing data in S3.
    """
    
    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        Initialize S3 handler.
        
        Args:
            bucket_name: Default S3 bucket name
            region: AWS region
            aws_access_key_id: AWS access key (optional, uses env/profile if not provided)
            aws_secret_access_key: AWS secret key (optional)
        """
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        session_kwargs = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        
        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client("s3")
        self.s3_resource = self.session.resource("s3")
        
        logger.info(f"Initialized S3Handler for bucket: {bucket_name}")
    
    def bucket_exists(self, bucket_name: Optional[str] = None) -> bool:
        """Check if bucket exists."""
        bucket = bucket_name or self.bucket_name
        try:
            self.s3_client.head_bucket(Bucket=bucket)
            return True
        except ClientError:
            return False
    
    def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Create S3 bucket if it doesn't exist.
        
        Args:
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            True if created or already exists
        """
        bucket = bucket_name or self.bucket_name
        
        if self.bucket_exists(bucket):
            logger.info(f"Bucket already exists: {bucket}")
            return True
        
        try:
            if self.region == "us-east-1":
                self.s3_client.create_bucket(Bucket=bucket)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": self.region}
                )
            logger.info(f"Created bucket: {bucket}")
            return True
        except ClientError as e:
            logger.error(f"Failed to create bucket: {e}")
            return False
    
    @timer
    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            True if successful
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            self.s3_client.upload_file(local_path, bucket, s3_key)
            logger.info(f"Uploaded: {local_path} -> s3://{bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    @timer
    def download_file(
        self,
        s3_key: str,
        local_path: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            True if successful
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            # Ensure local directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"Downloaded: s3://{bucket}/{s3_key} -> {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False
    
    @timer
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        s3_key: str,
        file_format: str = "parquet",
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        Upload a DataFrame to S3.
        
        Args:
            df: pandas DataFrame
            s3_key: S3 object key
            file_format: Output format ('parquet', 'csv', 'json')
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            True if successful
        """
        bucket = bucket_name or self.bucket_name
        buffer = io.BytesIO()
        
        try:
            if file_format == "parquet":
                df.to_parquet(buffer, index=False)
            elif file_format == "csv":
                df.to_csv(buffer, index=False)
                buffer = io.BytesIO(buffer.getvalue().encode())
            elif file_format == "json":
                df.to_json(buffer, orient="records")
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            buffer.seek(0)
            self.s3_client.upload_fileobj(buffer, bucket, s3_key)
            logger.info(f"Uploaded DataFrame ({len(df):,} rows) to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {e}")
            return False
    
    @timer
    def download_dataframe(
        self,
        s3_key: str,
        file_format: str = "parquet",
        bucket_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Download a DataFrame from S3.
        
        Args:
            s3_key: S3 object key
            file_format: Input format ('parquet', 'csv', 'json')
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            DataFrame or None if failed
        """
        bucket = bucket_name or self.bucket_name
        buffer = io.BytesIO()
        
        try:
            self.s3_client.download_fileobj(bucket, s3_key, buffer)
            buffer.seek(0)
            
            if file_format == "parquet":
                df = pd.read_parquet(buffer)
            elif file_format == "csv":
                df = pd.read_csv(buffer)
            elif file_format == "json":
                df = pd.read_json(buffer)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            logger.info(f"Downloaded DataFrame ({len(df):,} rows) from s3://{bucket}/{s3_key}")
            return df
        except Exception as e:
            logger.error(f"Failed to download DataFrame: {e}")
            return None
    
    def upload_model(
        self,
        model: object,
        s3_key: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        Upload a trained model to S3.
        
        Args:
            model: Trained model object
            s3_key: S3 object key
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            True if successful
        """
        bucket = bucket_name or self.bucket_name
        buffer = io.BytesIO()
        
        try:
            pickle.dump(model, buffer)
            buffer.seek(0)
            self.s3_client.upload_fileobj(buffer, bucket, s3_key)
            logger.info(f"Uploaded model to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return False
    
    def download_model(
        self,
        s3_key: str,
        bucket_name: Optional[str] = None
    ) -> Optional[object]:
        """
        Download a trained model from S3.
        
        Args:
            s3_key: S3 object key
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            Model object or None if failed
        """
        bucket = bucket_name or self.bucket_name
        buffer = io.BytesIO()
        
        try:
            self.s3_client.download_fileobj(bucket, s3_key, buffer)
            buffer.seek(0)
            model = pickle.load(buffer)
            logger.info(f"Downloaded model from s3://{bucket}/{s3_key}")
            return model
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None
    
    def list_objects(
        self,
        prefix: str = "",
        bucket_name: Optional[str] = None
    ) -> List[S3Object]:
        """
        List objects in S3 bucket.
        
        Args:
            prefix: Key prefix to filter
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            List of S3Object instances
        """
        bucket = bucket_name or self.bucket_name
        objects = []
        
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    objects.append(S3Object(
                        bucket=bucket,
                        key=obj["Key"],
                        size=obj["Size"],
                        last_modified=str(obj["LastModified"])
                    ))
            
            logger.info(f"Listed {len(objects)} objects in s3://{bucket}/{prefix}")
            return objects
        except ClientError as e:
            logger.error(f"Failed to list objects: {e}")
            return []
    
    def delete_object(
        self,
        s3_key: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """Delete an object from S3."""
        bucket = bucket_name or self.bucket_name
        
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
            logger.info(f"Deleted: s3://{bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            return False
    
    def copy_object(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None
    ) -> bool:
        """Copy an object within S3."""
        source_bucket = source_bucket or self.bucket_name
        dest_bucket = dest_bucket or self.bucket_name
        
        try:
            copy_source = {"Bucket": source_bucket, "Key": source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
            logger.info(f"Copied: s3://{source_bucket}/{source_key} -> s3://{dest_bucket}/{dest_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to copy object: {e}")
            return False
    
    def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        bucket_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds
            bucket_name: Bucket name (uses default if not provided)
        
        Returns:
            Presigned URL or None if failed
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
