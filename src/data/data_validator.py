"""
Data Validation Module
Provides data quality checks and validation utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[ValidationResult]
    is_valid: bool
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Validation Report: {self.passed_checks}/{self.total_checks} checks passed. "
            f"Valid: {self.is_valid}"
        )


class DataValidator:
    """
    Validates data quality and integrity.
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def reset(self) -> None:
        """Reset validation results."""
        self.results = []
    
    def _add_result(
        self,
        check_name: str,
        passed: bool,
        severity: ValidationSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a validation result."""
        result = ValidationResult(
            check_name=check_name,
            passed=passed,
            severity=severity,
            message=message,
            details=details
        )
        self.results.append(result)
        
        log_func = logger.info if passed else (
            logger.warning if severity in [ValidationSeverity.INFO, ValidationSeverity.WARNING]
            else logger.error
        )
        status = "✓ PASS" if passed else "✗ FAIL"
        log_func(f"{status}: {check_name} - {message}")
    
    def check_not_empty(self, df: pd.DataFrame, name: str = "DataFrame") -> bool:
        """Check that DataFrame is not empty."""
        passed = len(df) > 0
        self._add_result(
            check_name="Not Empty Check",
            passed=passed,
            severity=ValidationSeverity.CRITICAL,
            message=f"{name} has {len(df):,} rows",
            details={"rows": len(df), "columns": len(df.columns)}
        )
        return passed
    
    def check_required_columns(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> bool:
        """Check that all required columns exist."""
        missing = set(required_columns) - set(df.columns)
        passed = len(missing) == 0
        
        self._add_result(
            check_name="Required Columns Check",
            passed=passed,
            severity=ValidationSeverity.ERROR,
            message=f"Missing columns: {missing}" if missing else "All required columns present",
            details={"missing_columns": list(missing), "total_required": len(required_columns)}
        )
        return passed
    
    def check_no_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        threshold: float = 0.01
    ) -> bool:
        """Check for duplicate rows."""
        n_duplicates = df.duplicated(subset=subset).sum()
        duplicate_ratio = n_duplicates / len(df) if len(df) > 0 else 0
        passed = duplicate_ratio <= threshold
        
        self._add_result(
            check_name="Duplicate Check",
            passed=passed,
            severity=ValidationSeverity.WARNING,
            message=f"{n_duplicates:,} duplicates ({duplicate_ratio:.2%})",
            details={"n_duplicates": n_duplicates, "ratio": duplicate_ratio}
        )
        return passed
    
    def check_missing_values(
        self,
        df: pd.DataFrame,
        max_missing_ratio: float = 0.5,
        columns: Optional[List[str]] = None
    ) -> bool:
        """Check missing value ratios in columns."""
        columns = columns or df.columns.tolist()
        columns = [c for c in columns if c in df.columns]
        
        issues = []
        for col in columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > max_missing_ratio:
                issues.append((col, missing_ratio))
        
        passed = len(issues) == 0
        
        self._add_result(
            check_name="Missing Values Check",
            passed=passed,
            severity=ValidationSeverity.WARNING,
            message=f"{len(issues)} columns exceed {max_missing_ratio:.0%} missing threshold",
            details={"columns_exceeding": issues}
        )
        return passed
    
    def check_data_types(
        self,
        df: pd.DataFrame,
        expected_types: Dict[str, type]
    ) -> bool:
        """Check that columns have expected data types."""
        type_issues = []
        
        for col, expected in expected_types.items():
            if col in df.columns:
                actual = df[col].dtype
                # Check if types are compatible
                if not np.issubdtype(actual, expected):
                    type_issues.append((col, str(actual), str(expected)))
        
        passed = len(type_issues) == 0
        
        self._add_result(
            check_name="Data Types Check",
            passed=passed,
            severity=ValidationSeverity.ERROR,
            message=f"{len(type_issues)} columns have unexpected types",
            details={"type_issues": type_issues}
        )
        return passed
    
    def check_value_ranges(
        self,
        df: pd.DataFrame,
        ranges: Dict[str, Tuple[Optional[float], Optional[float]]]
    ) -> bool:
        """Check that numeric columns are within expected ranges."""
        range_issues = []
        
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
            
            if min_val is not None and df[col].min() < min_val:
                range_issues.append((col, "min", df[col].min(), min_val))
            
            if max_val is not None and df[col].max() > max_val:
                range_issues.append((col, "max", df[col].max(), max_val))
        
        passed = len(range_issues) == 0
        
        self._add_result(
            check_name="Value Range Check",
            passed=passed,
            severity=ValidationSeverity.WARNING,
            message=f"{len(range_issues)} range violations found",
            details={"violations": range_issues}
        )
        return passed
    
    def check_no_negative_values(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> bool:
        """Check that specified columns have no negative values."""
        negative_issues = []
        
        for col in columns:
            if col in df.columns:
                n_negative = (df[col] < 0).sum()
                if n_negative > 0:
                    negative_issues.append((col, n_negative))
        
        passed = len(negative_issues) == 0
        
        self._add_result(
            check_name="Non-Negative Values Check",
            passed=passed,
            severity=ValidationSeverity.ERROR,
            message=f"{len(negative_issues)} columns have negative values",
            details={"columns_with_negatives": negative_issues}
        )
        return passed
    
    def check_categorical_values(
        self,
        df: pd.DataFrame,
        column: str,
        valid_values: List[Any]
    ) -> bool:
        """Check that categorical column contains only valid values."""
        if column not in df.columns:
            self._add_result(
                check_name=f"Categorical Check: {column}",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Column {column} not found"
            )
            return False
        
        unique_values = set(df[column].dropna().unique())
        invalid_values = unique_values - set(valid_values)
        passed = len(invalid_values) == 0
        
        self._add_result(
            check_name=f"Categorical Check: {column}",
            passed=passed,
            severity=ValidationSeverity.WARNING,
            message=f"{len(invalid_values)} invalid values found" if invalid_values else "All values valid",
            details={"invalid_values": list(invalid_values)[:10]}  # Limit to first 10
        )
        return passed
    
    def check_unique_values(
        self,
        df: pd.DataFrame,
        column: str,
        should_be_unique: bool = True
    ) -> bool:
        """Check uniqueness of a column."""
        n_unique = df[column].nunique()
        n_total = len(df)
        is_unique = n_unique == n_total
        
        if should_be_unique:
            passed = is_unique
            message = f"Column is {'unique' if is_unique else 'not unique'}: {n_unique:,}/{n_total:,}"
        else:
            passed = not is_unique
            message = f"Column has {n_unique:,} unique values out of {n_total:,}"
        
        self._add_result(
            check_name=f"Uniqueness Check: {column}",
            passed=passed,
            severity=ValidationSeverity.INFO,
            message=message,
            details={"n_unique": n_unique, "n_total": n_total}
        )
        return passed
    
    def generate_report(self) -> ValidationReport:
        """Generate validation report from all checks."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        # Data is valid if no ERROR or CRITICAL failures
        is_valid = all(
            r.passed or r.severity in [ValidationSeverity.INFO, ValidationSeverity.WARNING]
            for r in self.results
        )
        
        report = ValidationReport(
            total_checks=len(self.results),
            passed_checks=passed,
            failed_checks=failed,
            results=self.results.copy(),
            is_valid=is_valid
        )
        
        logger.info(report.summary())
        return report


def validate_claims_data(df: pd.DataFrame) -> ValidationReport:
    """
    Run comprehensive validation on claims data.
    
    Args:
        df: Claims DataFrame to validate
    
    Returns:
        ValidationReport with all check results
    """
    validator = DataValidator()
    
    logger.info("=" * 60)
    logger.info("Running Claims Data Validation")
    logger.info("=" * 60)
    
    # Basic checks
    validator.check_not_empty(df, "Claims Data")
    
    # Required columns
    required = ['AMT_BILLED', 'AMT_PAID']
    validator.check_required_columns(df, required)
    
    # Amount columns should be non-negative
    amount_cols = ['AMT_BILLED', 'AMT_PAID', 'AMT_DEDUCT', 'AMT_COINS']
    amount_cols = [c for c in amount_cols if c in df.columns]
    validator.check_no_negative_values(df, amount_cols)
    
    # Check for excessive missing values
    validator.check_missing_values(df, max_missing_ratio=0.5)
    
    # Check duplicates
    validator.check_no_duplicates(df)
    
    return validator.generate_report()
