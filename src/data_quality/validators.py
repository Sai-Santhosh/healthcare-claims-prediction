"""
Data Validation Module
Schema validation and data type checking.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnSchema:
    """Schema definition for a column."""
    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None


@dataclass
class TableSchema:
    """Schema definition for a table."""
    name: str
    columns: List[ColumnSchema]
    primary_key: Optional[List[str]] = None
    
    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Get column schema by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None


@dataclass
class ValidationResult:
    """Result of validation checks."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add warning message."""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics
        }


class DataValidator:
    """
    Validates data against schemas and business rules.
    """
    
    def __init__(
        self,
        schema: Optional[TableSchema] = None,
        strict_mode: bool = False
    ):
        self.schema = schema
        self.strict_mode = strict_mode
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame against schema and rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with all findings
        """
        result = ValidationResult(is_valid=True)
        
        # Basic checks
        result = self._validate_not_empty(df, result)
        
        # Schema validation if provided
        if self.schema:
            result = self._validate_schema(df, result)
        
        # Add statistics
        result.statistics = self._compute_statistics(df)
        
        return result
    
    def _validate_not_empty(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> ValidationResult:
        """Check that DataFrame is not empty."""
        if len(df) == 0:
            result.add_error("DataFrame is empty")
        
        if len(df.columns) == 0:
            result.add_error("DataFrame has no columns")
        
        return result
    
    def _validate_schema(
        self,
        df: pd.DataFrame,
        result: ValidationResult
    ) -> ValidationResult:
        """Validate DataFrame against schema."""
        # Check required columns
        for col_schema in self.schema.columns:
            if col_schema.name not in df.columns:
                if self.strict_mode:
                    result.add_error(f"Required column missing: {col_schema.name}")
                else:
                    result.add_warning(f"Expected column missing: {col_schema.name}")
                continue
            
            # Check nullable
            if not col_schema.nullable:
                null_count = df[col_schema.name].isnull().sum()
                if null_count > 0:
                    result.add_error(
                        f"Column '{col_schema.name}' has {null_count} null values "
                        f"but is marked as non-nullable"
                    )
            
            # Check unique
            if col_schema.unique:
                duplicate_count = df[col_schema.name].duplicated().sum()
                if duplicate_count > 0:
                    result.add_error(
                        f"Column '{col_schema.name}' has {duplicate_count} duplicate values "
                        f"but is marked as unique"
                    )
            
            # Check value range
            if col_schema.min_value is not None or col_schema.max_value is not None:
                if pd.api.types.is_numeric_dtype(df[col_schema.name]):
                    if col_schema.min_value is not None:
                        below_min = (df[col_schema.name] < col_schema.min_value).sum()
                        if below_min > 0:
                            result.add_error(
                                f"Column '{col_schema.name}' has {below_min} values "
                                f"below minimum {col_schema.min_value}"
                            )
                    
                    if col_schema.max_value is not None:
                        above_max = (df[col_schema.name] > col_schema.max_value).sum()
                        if above_max > 0:
                            result.add_error(
                                f"Column '{col_schema.name}' has {above_max} values "
                                f"above maximum {col_schema.max_value}"
                            )
            
            # Check allowed values
            if col_schema.allowed_values:
                invalid = ~df[col_schema.name].isin(col_schema.allowed_values)
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    result.add_error(
                        f"Column '{col_schema.name}' has {invalid_count} values "
                        f"not in allowed set"
                    )
        
        # Check primary key
        if self.schema.primary_key:
            pk_cols = [c for c in self.schema.primary_key if c in df.columns]
            if pk_cols:
                duplicate_pk = df.duplicated(subset=pk_cols).sum()
                if duplicate_pk > 0:
                    result.add_error(
                        f"Primary key violation: {duplicate_pk} duplicate rows "
                        f"for columns {pk_cols}"
                    )
        
        return result
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute validation statistics."""
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum()
        }
        
        # Numeric column stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        return stats
    
    def validate_incremental(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate incremental data against existing data.
        Checks for schema consistency and key overlaps.
        
        Args:
            new_df: New data to validate
            existing_df: Existing data to compare against
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        # Check column consistency
        new_cols = set(new_df.columns)
        existing_cols = set(existing_df.columns)
        
        missing_cols = existing_cols - new_cols
        extra_cols = new_cols - existing_cols
        
        if missing_cols:
            result.add_warning(f"Missing columns in new data: {missing_cols}")
        
        if extra_cols:
            result.add_warning(f"New columns not in existing data: {extra_cols}")
        
        # Check data type consistency for common columns
        common_cols = new_cols & existing_cols
        for col in common_cols:
            if new_df[col].dtype != existing_df[col].dtype:
                result.add_warning(
                    f"Data type mismatch for column '{col}': "
                    f"new={new_df[col].dtype}, existing={existing_df[col].dtype}"
                )
        
        return result


class SchemaInferrer:
    """
    Infers schema from DataFrame.
    """
    
    def __init__(self, sample_size: int = 10000):
        self.sample_size = sample_size
    
    def infer_schema(
        self,
        df: pd.DataFrame,
        table_name: str = "inferred_table"
    ) -> TableSchema:
        """
        Infer schema from DataFrame.
        
        Args:
            df: DataFrame to analyze
            table_name: Name for the schema
            
        Returns:
            Inferred TableSchema
        """
        columns = []
        
        for col in df.columns:
            col_schema = self._infer_column_schema(df[col], col)
            columns.append(col_schema)
        
        # Try to detect primary key
        primary_key = self._detect_primary_key(df)
        
        return TableSchema(
            name=table_name,
            columns=columns,
            primary_key=primary_key
        )
    
    def _infer_column_schema(
        self,
        series: pd.Series,
        name: str
    ) -> ColumnSchema:
        """Infer schema for a single column."""
        dtype = str(series.dtype)
        nullable = series.isnull().any()
        unique = series.nunique() == len(series)
        
        schema = ColumnSchema(
            name=name,
            dtype=dtype,
            nullable=nullable,
            unique=unique
        )
        
        # Infer value range for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            schema.min_value = float(series.min())
            schema.max_value = float(series.max())
        
        # Infer allowed values for low-cardinality columns
        if series.nunique() <= 20:
            schema.allowed_values = series.dropna().unique().tolist()
        
        return schema
    
    def _detect_primary_key(
        self,
        df: pd.DataFrame
    ) -> Optional[List[str]]:
        """Detect potential primary key columns."""
        # Check for columns with 'id' or 'key' in name
        potential_pk = []
        
        for col in df.columns:
            col_lower = col.lower()
            if 'id' in col_lower or 'key' in col_lower:
                if df[col].is_unique:
                    potential_pk.append(col)
        
        if potential_pk:
            return [potential_pk[0]]  # Return first match
        
        return None
    
    def save_schema(
        self,
        schema: TableSchema,
        filepath: str
    ) -> None:
        """Save schema to JSON file."""
        schema_dict = {
            "name": schema.name,
            "columns": [
                {
                    "name": col.name,
                    "dtype": col.dtype,
                    "nullable": col.nullable,
                    "unique": col.unique,
                    "min_value": col.min_value,
                    "max_value": col.max_value,
                    "allowed_values": col.allowed_values
                }
                for col in schema.columns
            ],
            "primary_key": schema.primary_key
        }
        
        with open(filepath, 'w') as f:
            json.dump(schema_dict, f, indent=2, default=str)
        
        logger.info(f"Saved schema to {filepath}")
