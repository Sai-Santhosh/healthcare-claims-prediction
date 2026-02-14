"""
Data Loading Module
Handles loading data to various destinations: S3, RDS, local storage.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import logging

from .pipeline import PipelineStage
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader(PipelineStage):
    """
    Base data loader for writing to local storage.
    Supports parquet, CSV, and JSON formats.
    """
    
    def __init__(
        self,
        name: str = "load",
        output_path: str = None,
        output_format: str = "parquet",
        partition_by: Optional[List[str]] = None,
        mode: str = "overwrite",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.output_path = output_path
        self.output_format = output_format
        self.partition_by = partition_by
        self.mode = mode
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data loading.
        
        Args:
            context: Pipeline context with dataframe
            
        Returns:
            Load results
        """
        df = context.get("dataframe")
        
        if df is None:
            # Try to load from silver layer
            silver_path = "data/silver/claims_transformed.parquet"
            if os.path.exists(silver_path):
                df = pd.read_parquet(silver_path)
            else:
                raise ValueError("No dataframe found in context or silver layer")
        
        output_path = self.output_path or "data/gold/claims_final.parquet"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write data
        rows_written = self._write_data(df, output_path)
        
        logger.info(f"Loaded {rows_written:,} rows to {output_path}")
        
        return {
            "rows_processed": rows_written,
            "output_path": output_path,
            "metadata": {
                "format": self.output_format,
                "mode": self.mode,
                "file_size_mb": os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
            }
        }
    
    def _write_data(self, df: pd.DataFrame, output_path: str) -> int:
        """Write dataframe to specified format."""
        if self.output_format == "parquet":
            if self.partition_by:
                # Partitioned write
                for partition_values, partition_df in df.groupby(self.partition_by):
                    if not isinstance(partition_values, tuple):
                        partition_values = (partition_values,)
                    
                    partition_path = output_path
                    for col, val in zip(self.partition_by, partition_values):
                        partition_path = os.path.join(
                            os.path.dirname(partition_path),
                            f"{col}={val}",
                            os.path.basename(partition_path)
                        )
                    
                    os.makedirs(os.path.dirname(partition_path), exist_ok=True)
                    partition_df.to_parquet(partition_path, index=False)
            else:
                df.to_parquet(output_path, index=False)
                
        elif self.output_format == "csv":
            df.to_csv(output_path, index=False)
            
        elif self.output_format == "json":
            df.to_json(output_path, orient="records", lines=True)
        
        return len(df)


class S3Loader(PipelineStage):
    """
    Load data to AWS S3 Data Lake.
    Supports medallion architecture (bronze, silver, gold layers).
    """
    
    def __init__(
        self,
        name: str = "s3_load",
        bucket: str = None,
        prefix: str = None,
        layer: str = "gold",  # bronze, silver, gold
        output_format: str = "parquet",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.bucket = bucket
        self.prefix = prefix
        self.layer = layer
        self.output_format = output_format
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Upload data to S3."""
        df = context.get("dataframe")
        
        if df is None:
            raise ValueError("No dataframe found in context")
        
        try:
            import boto3
            from io import BytesIO
            
            s3 = boto3.client('s3')
            
            # Generate S3 key
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"{self.layer}/{self.prefix}/data_{timestamp}.{self.output_format}"
            
            # Serialize dataframe
            buffer = BytesIO()
            if self.output_format == "parquet":
                df.to_parquet(buffer, index=False)
            elif self.output_format == "csv":
                df.to_csv(buffer, index=False)
            
            buffer.seek(0)
            
            # Upload to S3
            s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            
            s3_path = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"Uploaded {len(df):,} rows to {s3_path}")
            
            return {
                "rows_processed": len(df),
                "output_path": s3_path,
                "metadata": {
                    "bucket": self.bucket,
                    "key": s3_key,
                    "layer": self.layer
                }
            }
            
        except ImportError:
            logger.warning("boto3 not available. Saving locally instead.")
            local_path = f"data/{self.layer}/claims_{datetime.now().strftime('%Y%m%d')}.parquet"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            df.to_parquet(local_path, index=False)
            
            return {
                "rows_processed": len(df),
                "output_path": local_path,
                "metadata": {"local_fallback": True}
            }
        
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise


class DatabaseLoader(PipelineStage):
    """
    Load data to RDS/PostgreSQL database.
    Supports upsert, append, and replace modes.
    """
    
    def __init__(
        self,
        name: str = "db_load",
        connection_string: str = None,
        table_name: str = None,
        schema: str = "public",
        mode: str = "append",  # append, replace, upsert
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.connection_string = connection_string
        self.table_name = table_name
        self.schema = schema
        self.mode = mode
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to database."""
        df = context.get("dataframe")
        
        if df is None:
            raise ValueError("No dataframe found in context")
        
        try:
            from sqlalchemy import create_engine
            
            engine = create_engine(self.connection_string)
            
            # Determine if_exists parameter
            if_exists = "append" if self.mode == "append" else "replace"
            
            # Write to database
            df.to_sql(
                name=self.table_name,
                con=engine,
                schema=self.schema,
                if_exists=if_exists,
                index=False,
                chunksize=10000
            )
            
            logger.info(f"Loaded {len(df):,} rows to {self.schema}.{self.table_name}")
            
            return {
                "rows_processed": len(df),
                "metadata": {
                    "table": f"{self.schema}.{self.table_name}",
                    "mode": self.mode
                }
            }
            
        except ImportError:
            logger.warning("sqlalchemy not available. Skipping database load.")
            return {"rows_processed": 0, "metadata": {"skipped": True}}
            
        except Exception as e:
            logger.error(f"Database load failed: {e}")
            raise


class RedshiftLoader(PipelineStage):
    """
    Load data to Amazon Redshift using COPY command.
    Optimized for large datasets using S3 staging.
    """
    
    def __init__(
        self,
        name: str = "redshift_load",
        connection_string: str = None,
        table_name: str = None,
        schema: str = "public",
        s3_staging_bucket: str = None,
        iam_role: str = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.connection_string = connection_string
        self.table_name = table_name
        self.schema = schema
        self.s3_staging_bucket = s3_staging_bucket
        self.iam_role = iam_role
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to Redshift via S3 COPY."""
        df = context.get("dataframe")
        
        if df is None:
            raise ValueError("No dataframe found in context")
        
        try:
            import boto3
            from sqlalchemy import create_engine
            from io import BytesIO
            
            # Stage data in S3
            s3 = boto3.client('s3')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"staging/{self.table_name}/{timestamp}.csv.gz"
            
            buffer = BytesIO()
            df.to_csv(buffer, index=False, compression='gzip')
            buffer.seek(0)
            
            s3.put_object(
                Bucket=self.s3_staging_bucket,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            
            s3_path = f"s3://{self.s3_staging_bucket}/{s3_key}"
            logger.info(f"Staged data to {s3_path}")
            
            # Execute COPY command
            engine = create_engine(self.connection_string)
            
            copy_sql = f"""
            COPY {self.schema}.{self.table_name}
            FROM '{s3_path}'
            IAM_ROLE '{self.iam_role}'
            CSV
            GZIP
            IGNOREHEADER 1
            DATEFORMAT 'auto'
            TIMEFORMAT 'auto';
            """
            
            with engine.connect() as conn:
                conn.execute(copy_sql)
            
            logger.info(f"COPY completed for {len(df):,} rows")
            
            return {
                "rows_processed": len(df),
                "metadata": {
                    "table": f"{self.schema}.{self.table_name}",
                    "s3_staging": s3_path
                }
            }
            
        except Exception as e:
            logger.error(f"Redshift load failed: {e}")
            raise


class FactTableLoader(PipelineStage):
    """
    Load fact table with dimension key lookups.
    Implements star schema loading pattern.
    """
    
    def __init__(
        self,
        name: str = "load_fact",
        fact_table_name: str = "fact_claims",
        dimension_lookups: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.fact_table_name = fact_table_name
        self.dimension_lookups = dimension_lookups or {}
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build and load fact table."""
        df = context.get("dataframe")
        
        if df is None:
            raise ValueError("No dataframe found in context")
        
        # Perform dimension key lookups
        for dim_name, lookup_col in self.dimension_lookups.items():
            dim_df = context.get(f"dim_{dim_name}")
            
            if dim_df is not None and lookup_col in df.columns:
                # Merge to get dimension key
                key_col = f"{dim_name}_key"
                df = df.merge(
                    dim_df[[key_col, lookup_col]],
                    on=lookup_col,
                    how='left'
                )
                logger.info(f"  Joined dimension: {dim_name}")
        
        # Save fact table
        output_path = f"data/gold/{self.fact_table_name}.parquet"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        return {
            "rows_processed": len(df),
            "output_path": output_path,
            "metadata": {
                "fact_table": self.fact_table_name,
                "dimensions_joined": list(self.dimension_lookups.keys())
            }
        }
