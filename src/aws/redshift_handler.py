"""
AWS Redshift Handler Module
Handles interactions with Amazon Redshift for data warehousing.
"""

import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RedshiftCluster:
    """Represents a Redshift cluster."""
    identifier: str
    status: str
    endpoint: Optional[str]
    port: int
    database: str
    node_type: str
    number_of_nodes: int


class RedshiftHandler:
    """
    Handler for AWS Redshift operations.
    Manages cluster operations and data loading.
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        cluster_identifier: str = "claims-analytics-cluster",
        database: str = "claims_db",
        port: int = 5439
    ):
        """
        Initialize Redshift handler.
        
        Args:
            region: AWS region
            cluster_identifier: Redshift cluster identifier
            database: Database name
            port: Database port
        """
        self.region = region
        self.cluster_identifier = cluster_identifier
        self.database = database
        self.port = port
        
        self.redshift_client = boto3.client("redshift", region_name=region)
        self.redshift_data = boto3.client("redshift-data", region_name=region)
        
        logger.info(f"Initialized RedshiftHandler for cluster: {cluster_identifier}")
    
    # =========================================================================
    # Cluster Operations
    # =========================================================================
    
    def create_cluster(
        self,
        master_username: str,
        master_password: str,
        node_type: str = "dc2.large",
        number_of_nodes: int = 2,
        iam_roles: Optional[List[str]] = None
    ) -> bool:
        """
        Create a Redshift cluster.
        
        Args:
            master_username: Master database username
            master_password: Master database password
            node_type: Node type
            number_of_nodes: Number of nodes
            iam_roles: List of IAM role ARNs
        
        Returns:
            True if successful
        """
        try:
            params = {
                "ClusterIdentifier": self.cluster_identifier,
                "ClusterType": "multi-node" if number_of_nodes > 1 else "single-node",
                "NodeType": node_type,
                "MasterUsername": master_username,
                "MasterUserPassword": master_password,
                "DBName": self.database,
                "Port": self.port,
                "PubliclyAccessible": True,
            }
            
            if number_of_nodes > 1:
                params["NumberOfNodes"] = number_of_nodes
            
            if iam_roles:
                params["IamRoles"] = iam_roles
            
            self.redshift_client.create_cluster(**params)
            logger.info(f"Creating cluster: {self.cluster_identifier}")
            return True
            
        except self.redshift_client.exceptions.ClusterAlreadyExistsFault:
            logger.info(f"Cluster already exists: {self.cluster_identifier}")
            return True
        except ClientError as e:
            logger.error(f"Failed to create cluster: {e}")
            return False
    
    def get_cluster_status(self) -> Optional[RedshiftCluster]:
        """Get cluster status and details."""
        try:
            response = self.redshift_client.describe_clusters(
                ClusterIdentifier=self.cluster_identifier
            )
            
            cluster = response["Clusters"][0]
            endpoint = cluster.get("Endpoint", {})
            
            return RedshiftCluster(
                identifier=cluster["ClusterIdentifier"],
                status=cluster["ClusterStatus"],
                endpoint=endpoint.get("Address"),
                port=endpoint.get("Port", self.port),
                database=cluster["DBName"],
                node_type=cluster["NodeType"],
                number_of_nodes=cluster.get("NumberOfNodes", 1)
            )
        except ClientError as e:
            logger.error(f"Failed to get cluster status: {e}")
            return None
    
    def wait_for_cluster(
        self,
        target_status: str = "available",
        timeout: int = 1800,
        poll_interval: int = 60
    ) -> bool:
        """
        Wait for cluster to reach target status.
        
        Args:
            target_status: Target status to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
        
        Returns:
            True if target status reached
        """
        elapsed = 0
        
        while elapsed < timeout:
            cluster = self.get_cluster_status()
            
            if cluster is None:
                return False
            
            if cluster.status == target_status:
                logger.info(f"Cluster {self.cluster_identifier} is {target_status}")
                return True
            
            logger.info(f"Cluster status: {cluster.status} ({elapsed}s elapsed)")
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        logger.error(f"Cluster did not reach {target_status} within {timeout}s")
        return False
    
    def delete_cluster(self, skip_snapshot: bool = True) -> bool:
        """Delete the Redshift cluster."""
        try:
            self.redshift_client.delete_cluster(
                ClusterIdentifier=self.cluster_identifier,
                SkipFinalClusterSnapshot=skip_snapshot
            )
            logger.info(f"Deleting cluster: {self.cluster_identifier}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete cluster: {e}")
            return False
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def execute_query(
        self,
        sql: str,
        db_user: str,
        wait_for_result: bool = True,
        timeout: int = 300
    ) -> Optional[str]:
        """
        Execute a SQL query using Redshift Data API.
        
        Args:
            sql: SQL query to execute
            db_user: Database user
            wait_for_result: Whether to wait for query completion
            timeout: Query timeout in seconds
        
        Returns:
            Query ID
        """
        try:
            cluster = self.get_cluster_status()
            if cluster is None or cluster.status != "available":
                logger.error("Cluster not available")
                return None
            
            response = self.redshift_data.execute_statement(
                ClusterIdentifier=self.cluster_identifier,
                Database=self.database,
                DbUser=db_user,
                Sql=sql
            )
            
            query_id = response["Id"]
            logger.info(f"Executed query: {query_id}")
            
            if wait_for_result:
                self._wait_for_query(query_id, timeout)
            
            return query_id
            
        except ClientError as e:
            logger.error(f"Failed to execute query: {e}")
            return None
    
    def _wait_for_query(self, query_id: str, timeout: int = 300) -> bool:
        """Wait for a query to complete."""
        elapsed = 0
        poll_interval = 5
        
        while elapsed < timeout:
            response = self.redshift_data.describe_statement(Id=query_id)
            status = response["Status"]
            
            if status == "FINISHED":
                logger.info(f"Query {query_id} completed successfully")
                return True
            elif status in ["FAILED", "ABORTED"]:
                error = response.get("Error", "Unknown error")
                logger.error(f"Query {query_id} failed: {error}")
                return False
            
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        logger.error(f"Query {query_id} timed out")
        return False
    
    def get_query_results(
        self,
        query_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Get results of a completed query.
        
        Args:
            query_id: Query ID to get results for
        
        Returns:
            DataFrame with results
        """
        try:
            # Get column metadata
            response = self.redshift_data.describe_statement(Id=query_id)
            if response["Status"] != "FINISHED":
                logger.error("Query not finished")
                return None
            
            # Get results
            paginator = self.redshift_data.get_paginator("get_statement_result")
            
            records = []
            columns = None
            
            for page in paginator.paginate(Id=query_id):
                if columns is None:
                    columns = [col["name"] for col in page["ColumnMetadata"]]
                
                for record in page["Records"]:
                    row = []
                    for field in record:
                        # Extract value from field
                        if "isNull" in field and field["isNull"]:
                            row.append(None)
                        else:
                            # Get first non-null value from field
                            for key in ["stringValue", "longValue", "doubleValue", "booleanValue"]:
                                if key in field:
                                    row.append(field[key])
                                    break
                    records.append(row)
            
            df = pd.DataFrame(records, columns=columns)
            logger.info(f"Retrieved {len(df)} rows from query {query_id}")
            return df
            
        except ClientError as e:
            logger.error(f"Failed to get query results: {e}")
            return None
    
    # =========================================================================
    # Data Loading Operations
    # =========================================================================
    
    def create_table(
        self,
        table_name: str,
        columns: List[Tuple[str, str]],
        db_user: str,
        primary_key: Optional[str] = None,
        sort_keys: Optional[List[str]] = None
    ) -> bool:
        """
        Create a Redshift table.
        
        Args:
            table_name: Table name
            columns: List of (column_name, data_type) tuples
            db_user: Database user
            primary_key: Primary key column
            sort_keys: Sort key columns
        
        Returns:
            True if successful
        """
        column_defs = ", ".join([f"{name} {dtype}" for name, dtype in columns])
        
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_defs}"
        
        if primary_key:
            sql += f", PRIMARY KEY ({primary_key})"
        
        sql += ")"
        
        if sort_keys:
            sql += f" SORTKEY ({', '.join(sort_keys)})"
        
        query_id = self.execute_query(sql, db_user)
        return query_id is not None
    
    def copy_from_s3(
        self,
        table_name: str,
        s3_path: str,
        iam_role: str,
        db_user: str,
        file_format: str = "PARQUET",
        region: Optional[str] = None
    ) -> bool:
        """
        Copy data from S3 to Redshift table.
        
        Args:
            table_name: Target table name
            s3_path: S3 path to data
            iam_role: IAM role ARN with S3 access
            db_user: Database user
            file_format: Data format (PARQUET, CSV, JSON)
            region: S3 region
        
        Returns:
            True if successful
        """
        region = region or self.region
        
        sql = f"""
        COPY {table_name}
        FROM '{s3_path}'
        IAM_ROLE '{iam_role}'
        FORMAT AS {file_format}
        REGION '{region}'
        """
        
        query_id = self.execute_query(sql, db_user, wait_for_result=True, timeout=1800)
        return query_id is not None
    
    def unload_to_s3(
        self,
        query: str,
        s3_path: str,
        iam_role: str,
        db_user: str,
        file_format: str = "PARQUET",
        parallel: bool = True
    ) -> bool:
        """
        Unload query results to S3.
        
        Args:
            query: SELECT query
            s3_path: S3 destination path
            iam_role: IAM role ARN with S3 access
            db_user: Database user
            file_format: Output format
            parallel: Whether to use parallel unload
        
        Returns:
            True if successful
        """
        parallel_opt = "PARALLEL ON" if parallel else "PARALLEL OFF"
        
        sql = f"""
        UNLOAD ('{query.replace("'", "''")}')
        TO '{s3_path}'
        IAM_ROLE '{iam_role}'
        FORMAT AS {file_format}
        {parallel_opt}
        """
        
        query_id = self.execute_query(sql, db_user, wait_for_result=True, timeout=1800)
        return query_id is not None


# =============================================================================
# Table Schema Definitions
# =============================================================================

CLAIMS_TABLE_SCHEMA = [
    ("claim_id", "BIGINT"),
    ("service_date", "DATE"),
    ("member_age", "INTEGER"),
    ("gender_code", "INTEGER"),
    ("form_type", "VARCHAR(50)"),
    ("sv_stat", "VARCHAR(50)"),
    ("product_type", "VARCHAR(100)"),
    ("icd_category", "VARCHAR(10)"),
    ("amt_billed", "DECIMAL(12,2)"),
    ("amt_paid", "DECIMAL(12,2)"),
    ("amt_deduct", "DECIMAL(12,2)"),
    ("amt_coins", "DECIMAL(12,2)"),
    ("client_los", "INTEGER"),
    ("num_diagnoses", "INTEGER"),
    ("created_at", "TIMESTAMP DEFAULT GETDATE()"),
]

PREDICTIONS_TABLE_SCHEMA = [
    ("prediction_id", "BIGINT IDENTITY(1,1)"),
    ("claim_id", "BIGINT"),
    ("predicted_amount", "DECIMAL(12,2)"),
    ("actual_amount", "DECIMAL(12,2)"),
    ("model_version", "VARCHAR(50)"),
    ("prediction_date", "TIMESTAMP DEFAULT GETDATE()"),
]
