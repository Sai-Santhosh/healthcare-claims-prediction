"""
Data Extraction Module
Handles extraction from various sources: files, S3, databases.
"""

import os
from typing import Dict, Any, Optional, Iterator, List
from datetime import datetime
import pandas as pd
import logging

from .pipeline import PipelineStage
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataExtractor(PipelineStage):
    """
    Base data extractor with chunked reading capability.
    Handles large files efficiently with memory management.
    """
    
    def __init__(
        self,
        name: str = "extract",
        chunk_size: int = 100000,
        delimiter: str = "|",
        encoding: str = "utf-8",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.chunk_size = chunk_size
        self.delimiter = delimiter
        self.encoding = encoding
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute extraction from configured source.
        
        Args:
            context: Pipeline context with source configuration
            
        Returns:
            Dictionary with extracted data info
        """
        config = context.get("config", {})
        source_config = config.get("data_sources", {}).get("claims", {})
        
        file_path = source_config.get("file_path")
        
        if not file_path or not os.path.exists(file_path):
            # Generate sample data for demo
            logger.warning(f"Source file not found: {file_path}. Generating sample data.")
            return self._generate_sample_data(context)
        
        return self._extract_from_file(file_path, context)
    
    def _extract_from_file(
        self,
        file_path: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract data from file with chunked reading."""
        logger.info(f"Extracting data from: {file_path}")
        
        total_rows = 0
        chunks_processed = 0
        sample_size = context.get("config", {}).get("etl", {}).get("sample_size", 1000000)
        
        # First pass: get unique claim IDs
        logger.info("Phase 1: Scanning for unique claim IDs...")
        claim_ids = set()
        
        for chunk in pd.read_csv(
            file_path,
            sep=self.delimiter,
            encoding=self.encoding,
            chunksize=self.chunk_size,
            low_memory=False,
            on_bad_lines='skip'
        ):
            claim_ids.update(chunk['CLAIM_ID_KEY'].unique())
            chunks_processed += 1
            
            if len(claim_ids) >= sample_size:
                break
            
            if chunks_processed % 10 == 0:
                logger.info(f"  Processed {chunks_processed} chunks, found {len(claim_ids):,} unique claims")
        
        # Sample claim IDs
        import random
        sampled_ids = set(random.sample(list(claim_ids), min(sample_size, len(claim_ids))))
        logger.info(f"Sampled {len(sampled_ids):,} claim IDs")
        
        # Second pass: extract sampled claims
        logger.info("Phase 2: Extracting sampled claims...")
        dfs = []
        chunks_processed = 0
        
        for chunk in pd.read_csv(
            file_path,
            sep=self.delimiter,
            encoding=self.encoding,
            chunksize=self.chunk_size,
            low_memory=False,
            on_bad_lines='skip'
        ):
            filtered = chunk[chunk['CLAIM_ID_KEY'].isin(sampled_ids)]
            if len(filtered) > 0:
                dfs.append(filtered)
                total_rows += len(filtered)
            
            chunks_processed += 1
            if chunks_processed % 10 == 0:
                logger.info(f"  Processed {chunks_processed} chunks, extracted {total_rows:,} rows")
        
        # Combine all chunks
        df = pd.concat(dfs, ignore_index=True)
        
        # Save to bronze layer
        output_path = "data/bronze/claims_extracted.parquet"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        # Store in context for next stage
        context["dataframe"] = df
        context["schema"] = list(df.columns)
        
        return {
            "rows_processed": len(df),
            "output_path": output_path,
            "metadata": {
                "columns": len(df.columns),
                "unique_claims": df['CLAIM_ID_KEY'].nunique(),
                "file_size_mb": os.path.getsize(output_path) / (1024 * 1024)
            }
        }
    
    def _generate_sample_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sample data for demonstration."""
        import numpy as np
        
        logger.info("Generating sample claims data...")
        
        n_rows = 100000  # 100K sample rows
        n_claims = 50000  # 50K unique claims
        
        np.random.seed(42)
        
        df = pd.DataFrame({
            'CLAIM_ID_KEY': np.random.randint(1, n_claims + 1, n_rows),
            'MEMBER_STATE': np.random.choice(['NH', 'MA', 'VT', 'ME'], n_rows),
            'MEMBER_COUNTY': np.random.randint(1, 11, n_rows),
            'AGE': np.random.randint(0, 95, n_rows),
            'SEX': np.random.choice(['M', 'F'], n_rows),
            'FORM_TYPE': np.random.choice(['P', 'I', 'O'], n_rows),
            'SV_STAT': np.random.choice(['01', '02', '03', '04'], n_rows),
            'PRODUCT_TYPE': np.random.choice(['HMO', 'PPO', 'POS'], n_rows),
            'ICD_DIAG_01': ['A' + str(i).zfill(2) for i in np.random.randint(0, 100, n_rows)],
            'ICD_DIAG_02': ['B' + str(i).zfill(2) for i in np.random.randint(0, 100, n_rows)],
            'CPT': [str(i).zfill(5) for i in np.random.randint(10000, 99999, n_rows)],
            'POS': np.random.choice(['11', '21', '22', '23'], n_rows),
            'AMT_BILLED': np.abs(np.random.normal(500, 300, n_rows)).round(2),
            'AMT_PAID': np.abs(np.random.normal(350, 200, n_rows)).round(2),
            'AMT_DEDUCT': np.abs(np.random.normal(50, 30, n_rows)).round(2),
            'AMT_COINS': np.abs(np.random.normal(20, 15, n_rows)).round(2),
            'CLIENT_LOS': np.random.choice([0, 1, 2, 3, 5, 7], n_rows),
            'QTY': np.random.randint(1, 10, n_rows),
            'SERVICE_DATE': pd.date_range('2016-01-01', periods=n_rows, freq='h').strftime('%Y%m%d')
        })
        
        # Save to bronze layer
        output_path = "data/bronze/claims_extracted.parquet"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        context["dataframe"] = df
        context["schema"] = list(df.columns)
        
        logger.info(f"Generated {len(df):,} sample rows with {df['CLAIM_ID_KEY'].nunique():,} unique claims")
        
        return {
            "rows_processed": len(df),
            "output_path": output_path,
            "metadata": {
                "columns": len(df.columns),
                "unique_claims": df['CLAIM_ID_KEY'].nunique(),
                "sample_data": True
            }
        }


class S3Extractor(PipelineStage):
    """
    Extract data from AWS S3.
    Supports reading from data lake with prefix filtering.
    """
    
    def __init__(
        self,
        name: str = "s3_extract",
        bucket: str = None,
        prefix: str = None,
        file_format: str = "parquet",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.bucket = bucket
        self.prefix = prefix
        self.file_format = file_format
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from S3."""
        try:
            import boto3
            
            s3 = boto3.client('s3')
            
            # List objects in prefix
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
            
            dfs = []
            total_rows = 0
            
            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    
                    if key.endswith(f'.{self.file_format}'):
                        logger.info(f"Reading: s3://{self.bucket}/{key}")
                        
                        # Download and read
                        local_path = f"/tmp/{os.path.basename(key)}"
                        s3.download_file(self.bucket, key, local_path)
                        
                        if self.file_format == "parquet":
                            df = pd.read_parquet(local_path)
                        elif self.file_format == "csv":
                            df = pd.read_csv(local_path)
                        
                        dfs.append(df)
                        total_rows += len(df)
                        
                        os.remove(local_path)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                context["dataframe"] = combined_df
                
                return {
                    "rows_processed": total_rows,
                    "metadata": {"files_read": len(dfs)}
                }
            
            return {"rows_processed": 0}
            
        except Exception as e:
            logger.error(f"S3 extraction failed: {e}")
            raise


class DatabaseExtractor(PipelineStage):
    """
    Extract data from RDS/PostgreSQL database.
    """
    
    def __init__(
        self,
        name: str = "db_extract",
        connection_string: str = None,
        query: str = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.connection_string = connection_string
        self.query = query
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from database."""
        try:
            from sqlalchemy import create_engine
            
            engine = create_engine(self.connection_string)
            
            logger.info(f"Executing query: {self.query[:100]}...")
            
            df = pd.read_sql(self.query, engine)
            context["dataframe"] = df
            
            return {
                "rows_processed": len(df),
                "metadata": {"columns": len(df.columns)}
            }
            
        except Exception as e:
            logger.error(f"Database extraction failed: {e}")
            raise
