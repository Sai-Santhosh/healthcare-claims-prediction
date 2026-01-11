"""
Data Processing Module
Handles data cleaning, transformation, and preparation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from scipy.stats import zscore

from ..utils.logger import get_logger, log_dataframe_info
from ..utils.helpers import timer

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""
    initial_rows: int
    final_rows: int
    initial_columns: int
    final_columns: int
    rows_removed: int
    columns_removed: List[str]
    missing_values_handled: Dict[str, int]
    transformations_applied: List[str]


class DataCleaner:
    """
    Handles data cleaning operations.
    """
    
    def __init__(self, missing_threshold: int = 1000000):
        self.missing_threshold = missing_threshold
        self.cleaning_log = []
    
    def remove_high_missing_columns(
        self,
        df: pd.DataFrame,
        threshold: Optional[int] = None,
        keep_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove columns with missing values above threshold.
        
        Args:
            df: Input DataFrame
            threshold: Missing value threshold (uses default if not provided)
            keep_columns: Columns to keep regardless of missing values
        
        Returns:
            Tuple of (cleaned DataFrame, list of removed columns)
        """
        threshold = threshold or self.missing_threshold
        keep_columns = set(keep_columns or [])
        
        logger.info(f"Checking for columns with >{threshold:,} missing values")
        
        missing_counts = df.isnull().sum()
        columns_to_remove = []
        
        for col, count in missing_counts.items():
            if count > threshold and col not in keep_columns:
                columns_to_remove.append(col)
                logger.debug(f"  {col}: {count:,} missing values")
        
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
            logger.info(f"Removed {len(columns_to_remove)} columns with high missing values")
            self.cleaning_log.append(f"Removed columns: {columns_to_remove}")
        
        return df, columns_to_remove
    
    def remove_specified_columns(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Remove specified columns from DataFrame.
        
        Args:
            df: Input DataFrame
            columns: List of columns to remove
        
        Returns:
            DataFrame with columns removed
        """
        existing_columns = [col for col in columns if col in df.columns]
        
        if existing_columns:
            df = df.drop(columns=existing_columns)
            logger.info(f"Removed {len(existing_columns)} specified columns")
            self.cleaning_log.append(f"Removed columns: {existing_columns}")
        
        return df
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        fill_values: Optional[Dict[str, Any]] = None,
        drop_rows: bool = False
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            fill_values: Dictionary of column -> fill value mappings
            drop_rows: Whether to drop rows with any remaining missing values
        
        Returns:
            DataFrame with missing values handled
        """
        initial_rows = len(df)
        fill_values = fill_values or {}
        
        # Fill specified columns
        for col, value in fill_values.items():
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna(value)
                    logger.info(f"Filled {missing_count:,} missing values in {col} with {value}")
        
        # Drop rows with remaining missing values
        if drop_rows:
            df = df.dropna()
            rows_dropped = initial_rows - len(df)
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped:,} rows with missing values")
        
        return df
    
    def remove_negative_values(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Remove rows with negative values in specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for negative values
        
        Returns:
            DataFrame with negative value rows removed
        """
        initial_rows = len(df)
        
        for col in columns:
            if col in df.columns:
                df = df[df[col] >= 0]
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed:,} rows with negative values")
        
        return df
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicate to keep ('first', 'last', False)
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed:,} duplicate rows")
        
        return df
    
    def get_cleaning_report(self) -> List[str]:
        """Get the cleaning operations log."""
        return self.cleaning_log


class DataProcessor:
    """
    Handles data transformation and feature creation.
    """
    
    def __init__(self):
        self.transformations_log = []
    
    @timer
    def encode_gender(
        self,
        df: pd.DataFrame,
        column: str = 'SEX',
        new_column: str = 'Gender_Code'
    ) -> pd.DataFrame:
        """
        Encode gender column as binary.
        
        Args:
            df: Input DataFrame
            column: Original gender column name
            new_column: New encoded column name
        
        Returns:
            DataFrame with encoded gender
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return df
        
        df[new_column] = df[column].apply(lambda x: 1 if x == 'M' else 0)
        df = df.drop(columns=[column])
        
        logger.info(f"Encoded {column} -> {new_column}")
        self.transformations_log.append(f"Gender encoding: {column} -> {new_column}")
        
        return df
    
    @timer
    def encode_age(
        self,
        df: pd.DataFrame,
        column: str = 'AGE',
        new_column: str = 'Age'
    ) -> pd.DataFrame:
        """
        Convert age column to numeric, handling '90+' cases.
        
        Args:
            df: Input DataFrame
            column: Original age column name
            new_column: New numeric column name
        
        Returns:
            DataFrame with numeric age
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return df
        
        df[new_column] = df[column].apply(lambda x: 90 if x == '90+' else x)
        df[new_column] = df[new_column].astype(str).astype(int)
        df = df.drop(columns=[column])
        
        logger.info(f"Encoded {column} -> {new_column} (numeric)")
        self.transformations_log.append(f"Age encoding: {column} -> {new_column}")
        
        return df
    
    @timer
    def extract_code_category(
        self,
        df: pd.DataFrame,
        column: str,
        new_column: str,
        drop_original: bool = True
    ) -> pd.DataFrame:
        """
        Extract first character from code column as category.
        
        Args:
            df: Input DataFrame
            column: Original code column name
            new_column: New category column name
            drop_original: Whether to drop the original column
        
        Returns:
            DataFrame with extracted categories
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return df
        
        df[new_column] = df[column].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
        
        if drop_original:
            df = df.drop(columns=[column])
        
        logger.info(f"Extracted category from {column} -> {new_column}")
        self.transformations_log.append(f"Category extraction: {column} -> {new_column}")
        
        return df
    
    @timer
    def count_diagnoses(
        self,
        df: pd.DataFrame,
        diagnosis_columns: List[str],
        new_column: str = 'Num_diag'
    ) -> pd.DataFrame:
        """
        Count number of non-null diagnosis columns per row.
        
        Args:
            df: Input DataFrame
            diagnosis_columns: List of diagnosis column names
            new_column: Name for the count column
        
        Returns:
            DataFrame with diagnosis count column
        """
        existing_cols = [col for col in diagnosis_columns if col in df.columns]
        
        if not existing_cols:
            logger.warning("No diagnosis columns found")
            return df
        
        df[new_column] = df[existing_cols].notna().sum(axis=1)
        
        logger.info(f"Created {new_column} from {len(existing_cols)} diagnosis columns")
        self.transformations_log.append(f"Diagnosis count: {new_column}")
        
        return df
    
    @timer
    def merge_icd_categories(
        self,
        df: pd.DataFrame,
        column: str,
        merge_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Merge ICD code categories.
        
        Args:
            df: Input DataFrame
            column: Column to merge categories in
            merge_map: Dictionary mapping original -> merged values
        
        Returns:
            DataFrame with merged categories
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return df
        
        if merge_map is None:
            # Default merge map for ICD categories
            merge_map = {
                'W': 'V', 'X': 'V', 'Y': 'V',  # Merge W, X, Y into V
                'B': 'A',                        # Merge B into A
                'n': 'N',                        # Normalize case
            }
        
        for old_val, new_val in merge_map.items():
            df[column] = df[column].apply(lambda x: new_val if x == old_val else x)
        
        # Convert numeric codes to 'num' category
        df[column] = df[column].apply(lambda x: 'num' if str(x).isdigit() else x)
        
        logger.info(f"Merged categories in {column}")
        self.transformations_log.append(f"Category merge: {column}")
        
        return df
    
    @timer
    def create_dummy_variables(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_first: bool = False
    ) -> pd.DataFrame:
        """
        Create dummy variables for categorical columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode
            drop_first: Whether to drop first category (avoid multicollinearity)
        
        Returns:
            DataFrame with dummy variables
        """
        existing_cols = [col for col in columns if col in df.columns]
        
        if not existing_cols:
            logger.warning("No categorical columns found for encoding")
            return df
        
        initial_cols = len(df.columns)
        df = pd.get_dummies(df, columns=existing_cols, drop_first=drop_first)
        new_cols = len(df.columns) - initial_cols + len(existing_cols)
        
        logger.info(f"Created {new_cols} dummy variables from {len(existing_cols)} columns")
        self.transformations_log.append(f"Dummy encoding: {existing_cols}")
        
        return df
    
    @timer
    def standardize_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Standardize numerical columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to standardize
            method: Standardization method ('zscore', 'minmax')
        
        Returns:
            DataFrame with standardized columns
        """
        existing_cols = [col for col in columns if col in df.columns]
        
        if not existing_cols:
            logger.warning("No numerical columns found for standardization")
            return df
        
        if method == 'zscore':
            df[existing_cols] = df[existing_cols].apply(zscore, nan_policy='omit')
        elif method == 'minmax':
            for col in existing_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        logger.info(f"Standardized {len(existing_cols)} columns using {method}")
        self.transformations_log.append(f"Standardization ({method}): {existing_cols}")
        
        return df
    
    def get_transformation_report(self) -> List[str]:
        """Get the transformation operations log."""
        return self.transformations_log


class DataPipeline:
    """
    Complete data processing pipeline combining cleaning and transformation.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.cleaner = DataCleaner(
            missing_threshold=config.get('high_missing_threshold', 1000000)
        )
        self.processor = DataProcessor()
    
    @timer
    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Args:
            df: Raw input DataFrame
        
        Returns:
            Processed DataFrame ready for modeling
        """
        logger.info("Starting data processing pipeline")
        log_dataframe_info(df, logger, "Input Data")
        
        # Step 1: Count diagnoses before removing columns
        diagnosis_cols = [f'ICD_DIAG_{i:02d}' for i in range(1, 14)]
        diagnosis_cols.append('ICD_DIAG_01_PRIMARY')
        df = self.processor.count_diagnoses(df, diagnosis_cols)
        
        # Step 2: Remove columns with high missing values
        df, _ = self.cleaner.remove_high_missing_columns(
            df,
            keep_columns=['CLIENT_LOS']
        )
        
        # Step 3: Remove specified columns
        df = self.cleaner.remove_specified_columns(
            df,
            self.config.get('columns_to_remove', [])
        )
        
        # Step 4: Handle specific transformations
        df = self.processor.encode_gender(df)
        df = self.processor.encode_age(df)
        df = self.processor.extract_code_category(df, 'PROC_CODE', 'Proc_Code_letters')
        df = self.processor.extract_code_category(
            df, 'ICD_DIAG_01_PRIMARY', 'ICD_DIAG_01_PRIMARY_categories'
        )
        
        # Step 5: Handle missing values
        df = self.cleaner.handle_missing_values(
            df,
            fill_values={'CLIENT_LOS': 0}
        )
        
        # Step 6: Remove final columns
        df = self.cleaner.remove_specified_columns(
            df,
            self.config.get('final_columns_to_drop', [])
        )
        
        # Step 7: Merge ICD categories
        df = self.processor.merge_icd_categories(df, 'ICD_DIAG_01_PRIMARY_categories')
        
        # Step 8: Remove negative values
        df = self.cleaner.remove_negative_values(
            df,
            ['AMT_BILLED', 'AMT_PAID', 'AMT_COINS', 'AMT_DEDUCT']
        )
        
        # Step 9: Drop remaining missing values
        df = self.cleaner.handle_missing_values(df, drop_rows=True)
        
        log_dataframe_info(df, logger, "Processed Data")
        logger.info("Data processing pipeline complete")
        
        return df
