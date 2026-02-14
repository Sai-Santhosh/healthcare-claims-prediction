"""
Data Transformation Module
Handles data cleaning, standardization, and transformation for ELT pipelines.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

from .pipeline import PipelineStage
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformer(PipelineStage):
    """
    Main data transformer stage.
    Applies cleaning, standardization, and business transformations.
    """
    
    def __init__(
        self,
        name: str = "transform",
        transformations: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.transformations = transformations or [
            "clean_missing",
            "standardize_columns",
            "encode_categoricals",
            "create_derived_features"
        ]
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transformation pipeline.
        
        Args:
            context: Pipeline context with dataframe
            
        Returns:
            Transformation results
        """
        df = context.get("dataframe")
        
        if df is None:
            # Try to load from bronze layer
            bronze_path = "data/bronze/claims_extracted.parquet"
            if os.path.exists(bronze_path):
                df = pd.read_parquet(bronze_path)
            else:
                raise ValueError("No dataframe found in context or bronze layer")
        
        initial_rows = len(df)
        initial_cols = len(df.columns)
        
        logger.info(f"Starting transformation on {initial_rows:,} rows, {initial_cols} columns")
        
        # Apply transformations
        for transform_name in self.transformations:
            transform_method = getattr(self, f"_transform_{transform_name}", None)
            if transform_method:
                logger.info(f"  Applying transformation: {transform_name}")
                df = transform_method(df)
                logger.info(f"    Shape after: {df.shape}")
        
        # Save to silver layer
        output_path = "data/silver/claims_transformed.parquet"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        context["dataframe"] = df
        
        return {
            "rows_processed": len(df),
            "output_path": output_path,
            "metadata": {
                "initial_rows": initial_rows,
                "final_rows": len(df),
                "initial_cols": initial_cols,
                "final_cols": len(df.columns),
                "rows_removed": initial_rows - len(df)
            }
        }
    
    def _transform_clean_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        # Remove columns with >50% missing
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'UNKNOWN')
        
        return df
    
    def _transform_standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types."""
        # Lowercase column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Remove special characters
        df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]
        
        return df
    
    def _transform_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        # Gender encoding
        if 'sex' in df.columns:
            df['gender_code'] = df['sex'].map({'M': 1, 'F': 0}).fillna(-1).astype(int)
        
        # Age bands
        if 'age' in df.columns:
            df['age_band'] = pd.cut(
                df['age'],
                bins=[0, 18, 35, 50, 65, 120],
                labels=['0-17', '18-34', '35-49', '50-64', '65+'],
                include_lowest=True
            ).astype(str)
        
        return df
    
    def _transform_create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for analytics."""
        # ICD category from diagnosis code
        if 'icd_diag_01' in df.columns:
            df['icd_category'] = df['icd_diag_01'].str[0].fillna('X')
        
        # Payment ratio
        if 'amt_paid' in df.columns and 'amt_billed' in df.columns:
            df['payment_ratio'] = np.where(
                df['amt_billed'] > 0,
                df['amt_paid'] / df['amt_billed'],
                0
            ).round(4)
        
        # Number of diagnoses
        diag_cols = [col for col in df.columns if col.startswith('icd_diag_')]
        if diag_cols:
            df['num_diagnoses'] = df[diag_cols].notna().sum(axis=1)
        
        return df


class DataCleaner(PipelineStage):
    """
    Data cleaning stage with validation and quality checks.
    """
    
    def __init__(
        self,
        name: str = "clean",
        remove_duplicates: bool = True,
        remove_negative_amounts: bool = True,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.remove_duplicates = remove_duplicates
        self.remove_negative_amounts = remove_negative_amounts
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cleaning operations."""
        df = context.get("dataframe")
        
        if df is None:
            raise ValueError("No dataframe found in context")
        
        initial_rows = len(df)
        rows_removed = {}
        
        # Remove duplicates
        if self.remove_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            rows_removed["duplicates"] = before - len(df)
            logger.info(f"  Removed {rows_removed['duplicates']:,} duplicate rows")
        
        # Remove negative amounts
        if self.remove_negative_amounts:
            amount_cols = [col for col in df.columns if 'amt' in col.lower()]
            before = len(df)
            for col in amount_cols:
                if col in df.columns:
                    df = df[df[col] >= 0]
            rows_removed["negative_amounts"] = before - len(df)
            logger.info(f"  Removed {rows_removed['negative_amounts']:,} rows with negative amounts")
        
        context["dataframe"] = df
        
        return {
            "rows_processed": len(df),
            "metadata": {
                "initial_rows": initial_rows,
                "final_rows": len(df),
                "rows_removed": rows_removed
            }
        }


class DataAggregator(PipelineStage):
    """
    Aggregation stage for creating summary tables.
    """
    
    def __init__(
        self,
        name: str = "aggregate",
        group_by: Optional[List[str]] = None,
        aggregations: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.group_by = group_by or []
        self.aggregations = aggregations or {}
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation."""
        df = context.get("dataframe")
        
        if df is None:
            raise ValueError("No dataframe found in context")
        
        if not self.group_by:
            logger.warning("No group_by columns specified")
            return {"rows_processed": len(df)}
        
        # Build aggregation dictionary
        agg_dict = {}
        for col, funcs in self.aggregations.items():
            if col in df.columns:
                agg_dict[col] = funcs
        
        # Perform aggregation
        agg_df = df.groupby(self.group_by).agg(agg_dict).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in agg_df.columns]
        
        context["aggregated_dataframe"] = agg_df
        
        return {
            "rows_processed": len(agg_df),
            "metadata": {
                "input_rows": len(df),
                "output_rows": len(agg_df),
                "group_by": self.group_by
            }
        }


class DimensionBuilder(PipelineStage):
    """
    Build dimension tables for star schema.
    """
    
    def __init__(
        self,
        name: str = "build_dimension",
        dimension_name: str = None,
        source_columns: Optional[List[str]] = None,
        key_column: str = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dimension_name = dimension_name
        self.source_columns = source_columns or []
        self.key_column = key_column
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build dimension table."""
        df = context.get("dataframe")
        
        if df is None:
            raise ValueError("No dataframe found in context")
        
        # Select relevant columns
        dim_cols = [col for col in self.source_columns if col in df.columns]
        
        if not dim_cols:
            logger.warning(f"No matching columns found for dimension: {self.dimension_name}")
            return {"rows_processed": 0}
        
        # Create dimension table
        dim_df = df[dim_cols].drop_duplicates().reset_index(drop=True)
        
        # Add surrogate key
        dim_df.insert(0, f'{self.dimension_name}_key', range(1, len(dim_df) + 1))
        
        # Save dimension
        output_path = f"data/gold/dim_{self.dimension_name}.parquet"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dim_df.to_parquet(output_path, index=False)
        
        context[f"dim_{self.dimension_name}"] = dim_df
        
        return {
            "rows_processed": len(dim_df),
            "output_path": output_path,
            "metadata": {
                "dimension": self.dimension_name,
                "columns": list(dim_df.columns)
            }
        }
