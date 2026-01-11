"""
AWS Glue Handler Module
Handles interactions with AWS Glue for ETL operations.
"""

import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GlueJobRun:
    """Represents a Glue job run."""
    job_name: str
    run_id: str
    state: str
    started_on: Optional[str]
    completed_on: Optional[str]
    error_message: Optional[str] = None


class GlueHandler:
    """
    Handler for AWS Glue operations.
    Manages Glue jobs, crawlers, and data catalog operations.
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        database_name: str = "claims_database"
    ):
        """
        Initialize Glue handler.
        
        Args:
            region: AWS region
            database_name: Glue data catalog database name
        """
        self.region = region
        self.database_name = database_name
        
        self.glue_client = boto3.client("glue", region_name=region)
        logger.info(f"Initialized GlueHandler for database: {database_name}")
    
    # =========================================================================
    # Database Operations
    # =========================================================================
    
    def create_database(
        self,
        database_name: Optional[str] = None,
        description: str = "Medical Claims ML Database"
    ) -> bool:
        """
        Create a Glue data catalog database.
        
        Args:
            database_name: Database name (uses default if not provided)
            description: Database description
        
        Returns:
            True if successful
        """
        db_name = database_name or self.database_name
        
        try:
            self.glue_client.create_database(
                DatabaseInput={
                    "Name": db_name,
                    "Description": description
                }
            )
            logger.info(f"Created database: {db_name}")
            return True
        except self.glue_client.exceptions.AlreadyExistsException:
            logger.info(f"Database already exists: {db_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to create database: {e}")
            return False
    
    def delete_database(self, database_name: Optional[str] = None) -> bool:
        """Delete a Glue data catalog database."""
        db_name = database_name or self.database_name
        
        try:
            self.glue_client.delete_database(Name=db_name)
            logger.info(f"Deleted database: {db_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete database: {e}")
            return False
    
    # =========================================================================
    # Crawler Operations
    # =========================================================================
    
    def create_crawler(
        self,
        crawler_name: str,
        s3_path: str,
        iam_role: str,
        database_name: Optional[str] = None,
        table_prefix: str = "claims_"
    ) -> bool:
        """
        Create a Glue crawler to catalog S3 data.
        
        Args:
            crawler_name: Name of the crawler
            s3_path: S3 path to crawl (e.g., s3://bucket/prefix/)
            iam_role: IAM role ARN for the crawler
            database_name: Target database name
            table_prefix: Prefix for created tables
        
        Returns:
            True if successful
        """
        db_name = database_name or self.database_name
        
        try:
            self.glue_client.create_crawler(
                Name=crawler_name,
                Role=iam_role,
                DatabaseName=db_name,
                TablePrefix=table_prefix,
                Targets={
                    "S3Targets": [{"Path": s3_path}]
                },
                SchemaChangePolicy={
                    "UpdateBehavior": "UPDATE_IN_DATABASE",
                    "DeleteBehavior": "DEPRECATE_IN_DATABASE"
                }
            )
            logger.info(f"Created crawler: {crawler_name}")
            return True
        except self.glue_client.exceptions.AlreadyExistsException:
            logger.info(f"Crawler already exists: {crawler_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to create crawler: {e}")
            return False
    
    def start_crawler(self, crawler_name: str) -> bool:
        """Start a Glue crawler."""
        try:
            self.glue_client.start_crawler(Name=crawler_name)
            logger.info(f"Started crawler: {crawler_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to start crawler: {e}")
            return False
    
    def get_crawler_status(self, crawler_name: str) -> Optional[str]:
        """Get the current status of a crawler."""
        try:
            response = self.glue_client.get_crawler(Name=crawler_name)
            state = response["Crawler"]["State"]
            logger.info(f"Crawler {crawler_name} status: {state}")
            return state
        except ClientError as e:
            logger.error(f"Failed to get crawler status: {e}")
            return None
    
    def wait_for_crawler(
        self,
        crawler_name: str,
        timeout: int = 600,
        poll_interval: int = 30
    ) -> bool:
        """
        Wait for a crawler to complete.
        
        Args:
            crawler_name: Name of the crawler
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
        
        Returns:
            True if completed successfully
        """
        elapsed = 0
        
        while elapsed < timeout:
            status = self.get_crawler_status(crawler_name)
            
            if status == "READY":
                logger.info(f"Crawler {crawler_name} completed successfully")
                return True
            elif status in ["STOPPING", "STOPPED"]:
                logger.warning(f"Crawler {crawler_name} stopped")
                return False
            
            logger.info(f"Waiting for crawler... ({elapsed}s elapsed)")
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        logger.error(f"Crawler {crawler_name} timed out after {timeout}s")
        return False
    
    # =========================================================================
    # Job Operations
    # =========================================================================
    
    def create_etl_job(
        self,
        job_name: str,
        script_location: str,
        iam_role: str,
        default_arguments: Optional[Dict[str, str]] = None,
        max_capacity: float = 2.0,
        timeout: int = 60
    ) -> bool:
        """
        Create a Glue ETL job.
        
        Args:
            job_name: Name of the job
            script_location: S3 path to the job script
            iam_role: IAM role ARN for the job
            default_arguments: Default job arguments
            max_capacity: DPU capacity
            timeout: Job timeout in minutes
        
        Returns:
            True if successful
        """
        default_args = {
            "--job-language": "python",
            "--enable-metrics": "true",
            "--enable-continuous-cloudwatch-log": "true",
        }
        if default_arguments:
            default_args.update(default_arguments)
        
        try:
            self.glue_client.create_job(
                Name=job_name,
                Role=iam_role,
                Command={
                    "Name": "glueetl",
                    "ScriptLocation": script_location,
                    "PythonVersion": "3"
                },
                DefaultArguments=default_args,
                MaxCapacity=max_capacity,
                Timeout=timeout,
                GlueVersion="4.0"
            )
            logger.info(f"Created ETL job: {job_name}")
            return True
        except self.glue_client.exceptions.AlreadyExistsException:
            logger.info(f"Job already exists: {job_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to create job: {e}")
            return False
    
    def start_job(
        self,
        job_name: str,
        arguments: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Start a Glue job.
        
        Args:
            job_name: Name of the job
            arguments: Job arguments
        
        Returns:
            Job run ID or None if failed
        """
        try:
            response = self.glue_client.start_job_run(
                JobName=job_name,
                Arguments=arguments or {}
            )
            run_id = response["JobRunId"]
            logger.info(f"Started job {job_name} with run ID: {run_id}")
            return run_id
        except ClientError as e:
            logger.error(f"Failed to start job: {e}")
            return None
    
    def get_job_run_status(
        self,
        job_name: str,
        run_id: str
    ) -> Optional[GlueJobRun]:
        """Get the status of a job run."""
        try:
            response = self.glue_client.get_job_run(
                JobName=job_name,
                RunId=run_id
            )
            run = response["JobRun"]
            
            return GlueJobRun(
                job_name=job_name,
                run_id=run_id,
                state=run["JobRunState"],
                started_on=str(run.get("StartedOn")),
                completed_on=str(run.get("CompletedOn")),
                error_message=run.get("ErrorMessage")
            )
        except ClientError as e:
            logger.error(f"Failed to get job run status: {e}")
            return None
    
    def wait_for_job(
        self,
        job_name: str,
        run_id: str,
        timeout: int = 3600,
        poll_interval: int = 60
    ) -> bool:
        """
        Wait for a job to complete.
        
        Args:
            job_name: Name of the job
            run_id: Job run ID
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
        
        Returns:
            True if completed successfully
        """
        elapsed = 0
        
        while elapsed < timeout:
            job_run = self.get_job_run_status(job_name, run_id)
            
            if job_run is None:
                return False
            
            if job_run.state == "SUCCEEDED":
                logger.info(f"Job {job_name} completed successfully")
                return True
            elif job_run.state in ["FAILED", "STOPPED", "ERROR", "TIMEOUT"]:
                logger.error(f"Job {job_name} failed: {job_run.error_message}")
                return False
            
            logger.info(f"Job {job_name} state: {job_run.state} ({elapsed}s elapsed)")
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        logger.error(f"Job {job_name} timed out after {timeout}s")
        return False
    
    # =========================================================================
    # Table Operations
    # =========================================================================
    
    def get_tables(
        self,
        database_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all tables in a database."""
        db_name = database_name or self.database_name
        
        try:
            response = self.glue_client.get_tables(DatabaseName=db_name)
            tables = response.get("TableList", [])
            logger.info(f"Found {len(tables)} tables in {db_name}")
            return tables
        except ClientError as e:
            logger.error(f"Failed to get tables: {e}")
            return []
    
    def get_table_schema(
        self,
        table_name: str,
        database_name: Optional[str] = None
    ) -> Optional[List[Dict[str, str]]]:
        """Get the schema of a table."""
        db_name = database_name or self.database_name
        
        try:
            response = self.glue_client.get_table(
                DatabaseName=db_name,
                Name=table_name
            )
            columns = response["Table"]["StorageDescriptor"]["Columns"]
            logger.info(f"Got schema for {table_name}: {len(columns)} columns")
            return columns
        except ClientError as e:
            logger.error(f"Failed to get table schema: {e}")
            return None


# =============================================================================
# Glue ETL Script Generator
# =============================================================================

def generate_etl_script(
    source_s3_path: str,
    target_s3_path: str,
    transformations: List[str]
) -> str:
    """
    Generate a Glue ETL script.
    
    Args:
        source_s3_path: Source S3 path
        target_s3_path: Target S3 path
        transformations: List of transformation descriptions
    
    Returns:
        Python script content
    """
    script = '''
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read source data
source_path = "{source_path}"
datasource = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={{"paths": [source_path]}},
    format="parquet"
)

# Apply transformations
df = datasource.toDF()

# Data cleaning transformations
df = df.dropna(subset=['AMT_PAID', 'AMT_BILLED'])
df = df.filter(df['AMT_PAID'] >= 0)
df = df.filter(df['AMT_BILLED'] >= 0)

# Convert back to DynamicFrame
transformed = DynamicFrame.fromDF(df, glueContext, "transformed")

# Write to target
target_path = "{target_path}"
glueContext.write_dynamic_frame.from_options(
    frame=transformed,
    connection_type="s3",
    connection_options={{"path": target_path}},
    format="parquet"
)

job.commit()
'''.format(source_path=source_s3_path, target_path=target_s3_path)
    
    return script
