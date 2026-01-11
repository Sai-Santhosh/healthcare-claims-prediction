"""
Data Loading Module
Handles loading data from various sources including local files and AWS S3.
"""

import os
import math
import random
from pathlib import Path
from typing import Optional, Iterator, List, Set, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np

from ..utils.logger import get_logger, log_dataframe_info
from ..utils.helpers import timer, ProgressTracker, memory_usage

logger = get_logger(__name__)


@dataclass
class LoaderStats:
    """Statistics from data loading operation."""
    total_rows: int
    total_chunks: int
    unique_claims: int
    load_time_seconds: float
    memory_used: str


class DataLoader:
    """
    Standard data loader for manageable file sizes.
    """
    
    def __init__(
        self,
        delimiter: str = "|",
        encoding: str = "utf-8"
    ):
        self.delimiter = delimiter
        self.encoding = encoding
    
    @timer
    def load_csv(
        self,
        file_path: str,
        nrows: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        dtype: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.
        
        Args:
            file_path: Path to the CSV file
            nrows: Number of rows to load (optional)
            usecols: Columns to load (optional)
            dtype: Column data types (optional)
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading file: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep=self.delimiter,
            nrows=nrows,
            usecols=usecols,
            dtype=dtype,
            encoding=self.encoding,
            low_memory=False
        )
        
        log_dataframe_info(df, logger, "Loaded Data")
        return df
    
    @timer
    def load_parquet(
        self,
        file_path: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load a Parquet file into a DataFrame.
        
        Args:
            file_path: Path to the Parquet file
            columns: Columns to load (optional)
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading Parquet file: {file_path}")
        
        df = pd.read_parquet(file_path, columns=columns)
        
        log_dataframe_info(df, logger, "Loaded Data")
        return df
    
    def save_csv(
        self,
        df: pd.DataFrame,
        file_path: str,
        index: bool = False
    ) -> None:
        """Save DataFrame to CSV file."""
        logger.info(f"Saving to: {file_path}")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=index)
        logger.info(f"Saved {len(df):,} rows to {file_path}")
    
    def save_parquet(
        self,
        df: pd.DataFrame,
        file_path: str,
        compression: str = "snappy"
    ) -> None:
        """Save DataFrame to Parquet file."""
        logger.info(f"Saving to Parquet: {file_path}")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, compression=compression, index=False)
        logger.info(f"Saved {len(df):,} rows to {file_path}")


class ChunkedDataLoader:
    """
    Chunked data loader for large files that don't fit in memory.
    Implements memory-efficient loading strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = 100000,
        delimiter: str = "|",
        encoding: str = "utf-8"
    ):
        self.chunk_size = chunk_size
        self.delimiter = delimiter
        self.encoding = encoding
    
    def iterate_chunks(
        self,
        file_path: str,
        usecols: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over file in chunks.
        
        Args:
            file_path: Path to the file
            usecols: Columns to load (optional)
        
        Yields:
            DataFrame chunks
        """
        logger.info(f"Starting chunked iteration: {file_path}")
        logger.info(f"Chunk size: {self.chunk_size:,}")
        
        chunk_reader = pd.read_csv(
            file_path,
            sep=self.delimiter,
            chunksize=self.chunk_size,
            usecols=usecols,
            encoding=self.encoding,
            low_memory=False
        )
        
        for i, chunk in enumerate(chunk_reader):
            logger.debug(f"Processing chunk {i + 1}")
            yield chunk
    
    @timer
    def load_full_file(
        self,
        file_path: str,
        total_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load entire file using chunking and concatenation.
        
        Args:
            file_path: Path to the file
            total_rows: Expected total rows for progress tracking
        
        Returns:
            Complete DataFrame
        """
        logger.info(f"Loading full file with chunking: {file_path}")
        
        if total_rows:
            num_chunks = math.ceil(total_rows / self.chunk_size)
            logger.info(f"Expected chunks: {num_chunks}")
        
        chunks = []
        chunk_reader = pd.read_csv(
            file_path,
            sep=self.delimiter,
            chunksize=self.chunk_size,
            encoding=self.encoding,
            low_memory=False
        )
        
        for i, chunk in enumerate(chunk_reader):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1} chunks, Memory: {memory_usage()['rss']}")
        
        logger.info("Concatenating all chunks...")
        full_df = pd.concat(chunks, ignore_index=True)
        
        log_dataframe_info(full_df, logger, "Full Data")
        return full_df
    
    @timer
    def sample_by_claim_ids(
        self,
        file_path: str,
        claim_id_column: str,
        sample_size: int,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Sample data by unique claim IDs.
        
        This method:
        1. First pass: collect all unique claim IDs
        2. Sample the desired number of claim IDs
        3. Second pass: load only rows matching sampled IDs
        
        Args:
            file_path: Path to the file
            claim_id_column: Name of the claim ID column
            sample_size: Number of unique claims to sample
            random_seed: Random seed for reproducibility
        
        Returns:
            DataFrame with sampled claims
        """
        logger.info(f"Sampling {sample_size:,} unique claims from {file_path}")
        
        # First pass: collect unique claim IDs
        logger.info("Pass 1: Collecting unique claim IDs...")
        unique_ids: Set = set()
        
        for chunk in self.iterate_chunks(file_path, usecols=[claim_id_column]):
            unique_ids.update(chunk[claim_id_column].unique())
        
        logger.info(f"Found {len(unique_ids):,} unique claim IDs")
        
        # Sample claim IDs
        random.seed(random_seed)
        if sample_size > len(unique_ids):
            logger.warning(
                f"Sample size ({sample_size:,}) > unique IDs ({len(unique_ids):,}). "
                f"Using all available IDs."
            )
            sampled_ids = unique_ids
        else:
            sampled_ids = set(random.sample(list(unique_ids), sample_size))
        
        logger.info(f"Sampled {len(sampled_ids):,} claim IDs")
        
        # Second pass: load rows matching sampled IDs
        logger.info("Pass 2: Loading data for sampled claims...")
        sampled_chunks = []
        
        for chunk in self.iterate_chunks(file_path):
            mask = chunk[claim_id_column].isin(sampled_ids)
            if mask.any():
                sampled_chunks.append(chunk[mask])
        
        result_df = pd.concat(sampled_chunks, ignore_index=True)
        log_dataframe_info(result_df, logger, "Sampled Data")
        
        return result_df
    
    @timer
    def load_with_filter(
        self,
        file_path: str,
        filter_column: str,
        filter_values: Union[List, Set],
    ) -> pd.DataFrame:
        """
        Load data filtered by specific values in a column.
        
        Args:
            file_path: Path to the file
            filter_column: Column to filter on
            filter_values: Values to include
        
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Loading filtered data from {file_path}")
        logger.info(f"Filter: {filter_column} in {len(filter_values)} values")
        
        filter_set = set(filter_values)
        filtered_chunks = []
        
        for chunk in self.iterate_chunks(file_path):
            mask = chunk[filter_column].isin(filter_set)
            if mask.any():
                filtered_chunks.append(chunk[mask])
        
        result_df = pd.concat(filtered_chunks, ignore_index=True)
        log_dataframe_info(result_df, logger, "Filtered Data")
        
        return result_df


class ReferenceDataLoader:
    """
    Loader for reference/lookup tables.
    """
    
    def __init__(self, ref_tables_dir: str):
        self.ref_tables_dir = Path(ref_tables_dir)
    
    def load_reference_table(
        self,
        table_name: str,
        delimiter: str = "|"
    ) -> pd.DataFrame:
        """
        Load a reference table by name.
        
        Args:
            table_name: Name of the reference table (without path)
            delimiter: Column delimiter
        
        Returns:
            Reference table DataFrame
        """
        file_path = self.ref_tables_dir / table_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Reference table not found: {file_path}")
        
        logger.info(f"Loading reference table: {table_name}")
        df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8')
        log_dataframe_info(df, logger, f"Reference: {table_name}")
        
        return df
    
    def load_all_reference_tables(self) -> dict:
        """
        Load all reference tables from the directory.
        
        Returns:
            Dictionary mapping table names to DataFrames
        """
        tables = {}
        
        for file_path in self.ref_tables_dir.glob("*.txt"):
            table_name = file_path.stem
            try:
                tables[table_name] = self.load_reference_table(file_path.name)
            except Exception as e:
                logger.warning(f"Failed to load {table_name}: {e}")
        
        logger.info(f"Loaded {len(tables)} reference tables")
        return tables
