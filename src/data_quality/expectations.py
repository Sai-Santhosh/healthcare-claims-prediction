"""
Data Quality Expectations Module
Implements Great Expectations-style data quality checks.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging

from ..utils.logger import get_logger
from ..etl.pipeline import PipelineStage

logger = get_logger(__name__)


@dataclass
class ExpectationResult:
    """Result of a single expectation check."""
    expectation_type: str
    column: Optional[str]
    success: bool
    observed_value: Any
    expected_value: Any = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expectation_type": self.expectation_type,
            "column": self.column,
            "success": self.success,
            "observed_value": str(self.observed_value),
            "expected_value": str(self.expected_value),
            "message": self.message
        }


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    dataset_name: str
    validation_time: datetime
    total_expectations: int
    successful_expectations: int
    failed_expectations: int
    success_rate: float
    results: List[ExpectationResult] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return self.failed_expectations == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "validation_time": self.validation_time.isoformat(),
            "total_expectations": self.total_expectations,
            "successful_expectations": self.successful_expectations,
            "failed_expectations": self.failed_expectations,
            "success_rate": self.success_rate,
            "success": self.success,
            "results": [r.to_dict() for r in self.results]
        }


class ExpectationSuite:
    """
    Collection of expectations for a dataset.
    Similar to Great Expectations suite concept.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.expectations: List[Dict[str, Any]] = []
    
    def expect_column_to_exist(self, column: str) -> "ExpectationSuite":
        """Expect a column to exist in the dataset."""
        self.expectations.append({
            "type": "column_exists",
            "column": column
        })
        return self
    
    def expect_column_values_to_not_be_null(
        self,
        column: str,
        mostly: float = 1.0
    ) -> "ExpectationSuite":
        """Expect column values to not be null."""
        self.expectations.append({
            "type": "not_null",
            "column": column,
            "params": {"mostly": mostly}
        })
        return self
    
    def expect_column_values_to_be_unique(
        self,
        column: str,
        mostly: float = 1.0
    ) -> "ExpectationSuite":
        """Expect column values to be unique."""
        self.expectations.append({
            "type": "unique",
            "column": column,
            "params": {"mostly": mostly}
        })
        return self
    
    def expect_column_values_to_be_in_set(
        self,
        column: str,
        value_set: List[Any],
        mostly: float = 1.0
    ) -> "ExpectationSuite":
        """Expect column values to be in a defined set."""
        self.expectations.append({
            "type": "in_set",
            "column": column,
            "params": {"value_set": value_set, "mostly": mostly}
        })
        return self
    
    def expect_column_values_to_be_between(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        mostly: float = 1.0
    ) -> "ExpectationSuite":
        """Expect column values to be within a range."""
        self.expectations.append({
            "type": "between",
            "column": column,
            "params": {
                "min_value": min_value,
                "max_value": max_value,
                "mostly": mostly
            }
        })
        return self
    
    def expect_column_values_to_be_positive(
        self,
        column: str,
        mostly: float = 1.0
    ) -> "ExpectationSuite":
        """Expect column values to be positive."""
        self.expectations.append({
            "type": "positive",
            "column": column,
            "params": {"mostly": mostly}
        })
        return self
    
    def expect_column_mean_to_be_between(
        self,
        column: str,
        min_value: float,
        max_value: float
    ) -> "ExpectationSuite":
        """Expect column mean to be within range."""
        self.expectations.append({
            "type": "mean_between",
            "column": column,
            "params": {"min_value": min_value, "max_value": max_value}
        })
        return self
    
    def expect_table_row_count_to_be_between(
        self,
        min_value: int,
        max_value: int
    ) -> "ExpectationSuite":
        """Expect table row count to be within range."""
        self.expectations.append({
            "type": "row_count_between",
            "column": None,
            "params": {"min_value": min_value, "max_value": max_value}
        })
        return self
    
    def expect_column_pair_values_to_be_greater(
        self,
        column_a: str,
        column_b: str,
        or_equal: bool = True,
        mostly: float = 1.0
    ) -> "ExpectationSuite":
        """Expect values in column_a to be >= column_b."""
        self.expectations.append({
            "type": "column_pair_greater",
            "column": column_a,
            "params": {
                "column_b": column_b,
                "or_equal": or_equal,
                "mostly": mostly
            }
        })
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Export suite as dictionary."""
        return {
            "name": self.name,
            "expectations": self.expectations
        }
    
    def save(self, filepath: str) -> None:
        """Save suite to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved expectation suite to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ExpectationSuite":
        """Load suite from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        suite = cls(data['name'])
        suite.expectations = data['expectations']
        return suite


class DataQualityChecker(PipelineStage):
    """
    Data quality checking stage.
    Runs expectation suites and generates validation reports.
    """
    
    def __init__(
        self,
        name: str = "quality_check",
        suite: Optional[ExpectationSuite] = None,
        fail_on_error: bool = True,
        reports_dir: str = "reports/quality",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.suite = suite
        self.fail_on_error = fail_on_error
        self.reports_dir = reports_dir
        
        os.makedirs(reports_dir, exist_ok=True)
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run quality checks on the dataframe.
        
        Args:
            context: Pipeline context with dataframe
            
        Returns:
            Quality check results
        """
        df = context.get("dataframe")
        
        if df is None:
            raise ValueError("No dataframe found in context")
        
        if self.suite is None:
            # Create default suite for claims data
            self.suite = self._create_default_suite()
        
        # Run validation
        report = self.validate(df, "claims_data")
        
        # Log results
        logger.info(f"Data Quality Results:")
        logger.info(f"  Total Checks: {report.total_expectations}")
        logger.info(f"  Passed: {report.successful_expectations}")
        logger.info(f"  Failed: {report.failed_expectations}")
        logger.info(f"  Success Rate: {report.success_rate:.2%}")
        
        # Log failed checks
        for result in report.results:
            if not result.success:
                logger.warning(f"  FAILED: {result.expectation_type} on {result.column}: {result.message}")
        
        # Save report
        report_path = os.path.join(
            self.reports_dir,
            f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        context["quality_report"] = report
        
        # Check if we should fail
        if self.fail_on_error and not report.success:
            raise ValueError(f"Data quality checks failed: {report.failed_expectations} failures")
        
        return {
            "rows_processed": len(df),
            "metadata": {
                "total_checks": report.total_expectations,
                "passed": report.successful_expectations,
                "failed": report.failed_expectations,
                "success_rate": report.success_rate
            }
        }
    
    def validate(self, df: pd.DataFrame, dataset_name: str) -> ValidationReport:
        """
        Validate dataframe against expectation suite.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for the dataset
            
        Returns:
            ValidationReport with all results
        """
        results = []
        
        for expectation in self.suite.expectations:
            result = self._run_expectation(df, expectation)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        
        return ValidationReport(
            dataset_name=dataset_name,
            validation_time=datetime.now(),
            total_expectations=len(results),
            successful_expectations=successful,
            failed_expectations=len(results) - successful,
            success_rate=successful / max(len(results), 1),
            results=results
        )
    
    def _run_expectation(
        self,
        df: pd.DataFrame,
        expectation: Dict[str, Any]
    ) -> ExpectationResult:
        """Run a single expectation."""
        exp_type = expectation["type"]
        column = expectation.get("column")
        params = expectation.get("params", {})
        
        try:
            if exp_type == "column_exists":
                return self._check_column_exists(df, column)
            
            elif exp_type == "not_null":
                return self._check_not_null(df, column, params.get("mostly", 1.0))
            
            elif exp_type == "unique":
                return self._check_unique(df, column, params.get("mostly", 1.0))
            
            elif exp_type == "in_set":
                return self._check_in_set(
                    df, column,
                    params["value_set"],
                    params.get("mostly", 1.0)
                )
            
            elif exp_type == "between":
                return self._check_between(
                    df, column,
                    params.get("min_value"),
                    params.get("max_value"),
                    params.get("mostly", 1.0)
                )
            
            elif exp_type == "positive":
                return self._check_positive(df, column, params.get("mostly", 1.0))
            
            elif exp_type == "mean_between":
                return self._check_mean_between(
                    df, column,
                    params["min_value"],
                    params["max_value"]
                )
            
            elif exp_type == "row_count_between":
                return self._check_row_count(
                    df,
                    params["min_value"],
                    params["max_value"]
                )
            
            elif exp_type == "column_pair_greater":
                return self._check_column_pair_greater(
                    df, column,
                    params["column_b"],
                    params.get("or_equal", True),
                    params.get("mostly", 1.0)
                )
            
            else:
                return ExpectationResult(
                    expectation_type=exp_type,
                    column=column,
                    success=False,
                    observed_value=None,
                    message=f"Unknown expectation type: {exp_type}"
                )
                
        except Exception as e:
            return ExpectationResult(
                expectation_type=exp_type,
                column=column,
                success=False,
                observed_value=None,
                message=f"Error running expectation: {str(e)}"
            )
    
    def _check_column_exists(
        self,
        df: pd.DataFrame,
        column: str
    ) -> ExpectationResult:
        """Check if column exists."""
        exists = column in df.columns
        return ExpectationResult(
            expectation_type="column_exists",
            column=column,
            success=exists,
            observed_value=exists,
            expected_value=True,
            message="" if exists else f"Column '{column}' not found"
        )
    
    def _check_not_null(
        self,
        df: pd.DataFrame,
        column: str,
        mostly: float
    ) -> ExpectationResult:
        """Check for null values."""
        if column not in df.columns:
            return self._column_not_found(column)
        
        non_null_rate = 1 - df[column].isnull().mean()
        success = non_null_rate >= mostly
        
        return ExpectationResult(
            expectation_type="not_null",
            column=column,
            success=success,
            observed_value=f"{non_null_rate:.4f}",
            expected_value=f">= {mostly}",
            message="" if success else f"Non-null rate {non_null_rate:.4f} below threshold {mostly}"
        )
    
    def _check_unique(
        self,
        df: pd.DataFrame,
        column: str,
        mostly: float
    ) -> ExpectationResult:
        """Check for unique values."""
        if column not in df.columns:
            return self._column_not_found(column)
        
        unique_rate = df[column].nunique() / len(df)
        success = unique_rate >= mostly
        
        return ExpectationResult(
            expectation_type="unique",
            column=column,
            success=success,
            observed_value=f"{unique_rate:.4f}",
            expected_value=f">= {mostly}",
            message="" if success else f"Unique rate {unique_rate:.4f} below threshold {mostly}"
        )
    
    def _check_in_set(
        self,
        df: pd.DataFrame,
        column: str,
        value_set: List[Any],
        mostly: float
    ) -> ExpectationResult:
        """Check if values are in set."""
        if column not in df.columns:
            return self._column_not_found(column)
        
        in_set_rate = df[column].isin(value_set).mean()
        success = in_set_rate >= mostly
        
        return ExpectationResult(
            expectation_type="in_set",
            column=column,
            success=success,
            observed_value=f"{in_set_rate:.4f}",
            expected_value=f">= {mostly}",
            message="" if success else f"In-set rate {in_set_rate:.4f} below threshold {mostly}"
        )
    
    def _check_between(
        self,
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float],
        max_value: Optional[float],
        mostly: float
    ) -> ExpectationResult:
        """Check if values are between range."""
        if column not in df.columns:
            return self._column_not_found(column)
        
        valid = pd.Series([True] * len(df))
        
        if min_value is not None:
            valid &= df[column] >= min_value
        if max_value is not None:
            valid &= df[column] <= max_value
        
        valid_rate = valid.mean()
        success = valid_rate >= mostly
        
        return ExpectationResult(
            expectation_type="between",
            column=column,
            success=success,
            observed_value=f"{valid_rate:.4f}",
            expected_value=f">= {mostly} in [{min_value}, {max_value}]",
            message="" if success else f"Valid rate {valid_rate:.4f} below threshold {mostly}"
        )
    
    def _check_positive(
        self,
        df: pd.DataFrame,
        column: str,
        mostly: float
    ) -> ExpectationResult:
        """Check if values are positive."""
        if column not in df.columns:
            return self._column_not_found(column)
        
        positive_rate = (df[column] >= 0).mean()
        success = positive_rate >= mostly
        
        return ExpectationResult(
            expectation_type="positive",
            column=column,
            success=success,
            observed_value=f"{positive_rate:.4f}",
            expected_value=f">= {mostly}",
            message="" if success else f"Positive rate {positive_rate:.4f} below threshold {mostly}"
        )
    
    def _check_mean_between(
        self,
        df: pd.DataFrame,
        column: str,
        min_value: float,
        max_value: float
    ) -> ExpectationResult:
        """Check if column mean is in range."""
        if column not in df.columns:
            return self._column_not_found(column)
        
        mean_val = df[column].mean()
        success = min_value <= mean_val <= max_value
        
        return ExpectationResult(
            expectation_type="mean_between",
            column=column,
            success=success,
            observed_value=f"{mean_val:.4f}",
            expected_value=f"[{min_value}, {max_value}]",
            message="" if success else f"Mean {mean_val:.4f} outside range [{min_value}, {max_value}]"
        )
    
    def _check_row_count(
        self,
        df: pd.DataFrame,
        min_value: int,
        max_value: int
    ) -> ExpectationResult:
        """Check row count."""
        count = len(df)
        success = min_value <= count <= max_value
        
        return ExpectationResult(
            expectation_type="row_count_between",
            column=None,
            success=success,
            observed_value=count,
            expected_value=f"[{min_value}, {max_value}]",
            message="" if success else f"Row count {count} outside range [{min_value}, {max_value}]"
        )
    
    def _check_column_pair_greater(
        self,
        df: pd.DataFrame,
        column_a: str,
        column_b: str,
        or_equal: bool,
        mostly: float
    ) -> ExpectationResult:
        """Check if column_a >= column_b."""
        if column_a not in df.columns or column_b not in df.columns:
            return ExpectationResult(
                expectation_type="column_pair_greater",
                column=column_a,
                success=False,
                observed_value=None,
                message=f"Column(s) not found: {column_a}, {column_b}"
            )
        
        if or_equal:
            valid_rate = (df[column_a] >= df[column_b]).mean()
        else:
            valid_rate = (df[column_a] > df[column_b]).mean()
        
        success = valid_rate >= mostly
        
        return ExpectationResult(
            expectation_type="column_pair_greater",
            column=column_a,
            success=success,
            observed_value=f"{valid_rate:.4f}",
            expected_value=f">= {mostly}",
            message="" if success else f"Comparison rate {valid_rate:.4f} below threshold {mostly}"
        )
    
    def _column_not_found(self, column: str) -> ExpectationResult:
        """Create result for missing column."""
        return ExpectationResult(
            expectation_type="column_exists",
            column=column,
            success=False,
            observed_value=False,
            message=f"Column '{column}' not found"
        )
    
    def _create_default_suite(self) -> ExpectationSuite:
        """Create default expectation suite for claims data."""
        suite = ExpectationSuite("claims_default")
        
        # Add common expectations
        suite.expect_table_row_count_to_be_between(1000, 50000000)
        suite.expect_column_to_exist("claim_id_key")
        suite.expect_column_values_to_not_be_null("claim_id_key")
        suite.expect_column_values_to_be_positive("amt_paid", mostly=0.99)
        suite.expect_column_values_to_be_positive("amt_billed", mostly=0.99)
        
        return suite
