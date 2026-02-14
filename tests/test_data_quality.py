"""
Unit Tests for Data Quality Module
Tests for expectations, validators, and profilers.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_quality.expectations import (
    DataQualityChecker,
    ExpectationSuite,
    ExpectationResult,
    ValidationReport
)
from src.data_quality.validators import (
    DataValidator,
    ValidationResult,
    SchemaInferrer,
    ColumnSchema,
    TableSchema
)
from src.data_quality.profiler import DataProfiler, DataProfile


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Sample claims dataframe for testing."""
    np.random.seed(42)
    n_rows = 100
    
    return pd.DataFrame({
        'claim_id_key': range(1, n_rows + 1),
        'member_state': np.random.choice(['NH', 'MA', 'VT'], n_rows),
        'age': np.random.randint(0, 95, n_rows),
        'sex': np.random.choice(['M', 'F'], n_rows),
        'amt_billed': np.abs(np.random.normal(500, 200, n_rows)).round(2),
        'amt_paid': np.abs(np.random.normal(350, 150, n_rows)).round(2),
        'icd_diag_01': ['A' + str(i).zfill(2) for i in np.random.randint(0, 100, n_rows)]
    })


@pytest.fixture
def dataframe_with_issues():
    """Dataframe with data quality issues."""
    return pd.DataFrame({
        'claim_id_key': [1, 2, 2, 4, 5],  # Duplicate ID
        'age': [25, -5, 150, 30, None],  # Negative, out of range, null
        'amt_paid': [100, 200, -50, 300, 400],  # Negative value
        'gender': ['M', 'F', 'X', 'M', 'F']  # Invalid value
    })


# =============================================================================
# ExpectationSuite Tests
# =============================================================================

class TestExpectationSuite:
    """Tests for ExpectationSuite."""
    
    def test_suite_creation(self):
        """Test suite creation."""
        suite = ExpectationSuite("test_suite")
        assert suite.name == "test_suite"
        assert len(suite.expectations) == 0
    
    def test_add_expectations(self):
        """Test adding expectations to suite."""
        suite = ExpectationSuite("test")
        
        suite.expect_column_to_exist("col1")
        suite.expect_column_values_to_not_be_null("col1")
        suite.expect_column_values_to_be_unique("col1")
        
        assert len(suite.expectations) == 3
    
    def test_expect_column_values_to_be_in_set(self):
        """Test in-set expectation."""
        suite = ExpectationSuite("test")
        suite.expect_column_values_to_be_in_set("gender", ['M', 'F'])
        
        assert suite.expectations[0]["type"] == "in_set"
        assert suite.expectations[0]["params"]["value_set"] == ['M', 'F']
    
    def test_expect_column_values_to_be_between(self):
        """Test between expectation."""
        suite = ExpectationSuite("test")
        suite.expect_column_values_to_be_between("age", min_value=0, max_value=120)
        
        assert suite.expectations[0]["type"] == "between"
        assert suite.expectations[0]["params"]["min_value"] == 0
        assert suite.expectations[0]["params"]["max_value"] == 120
    
    def test_suite_to_dict(self):
        """Test suite serialization."""
        suite = ExpectationSuite("test")
        suite.expect_column_to_exist("col1")
        
        result = suite.to_dict()
        
        assert result["name"] == "test"
        assert len(result["expectations"]) == 1
    
    def test_suite_save_and_load(self, tmp_path):
        """Test suite save and load."""
        suite = ExpectationSuite("test")
        suite.expect_column_to_exist("col1")
        suite.expect_column_values_to_not_be_null("col1")
        
        filepath = str(tmp_path / "suite.json")
        suite.save(filepath)
        
        loaded_suite = ExpectationSuite.load(filepath)
        
        assert loaded_suite.name == "test"
        assert len(loaded_suite.expectations) == 2


# =============================================================================
# DataQualityChecker Tests
# =============================================================================

class TestDataQualityChecker:
    """Tests for DataQualityChecker."""
    
    def test_checker_initialization(self):
        """Test checker initialization."""
        suite = ExpectationSuite("test")
        checker = DataQualityChecker(suite=suite)
        
        assert checker.suite == suite
    
    def test_check_column_exists_pass(self, sample_dataframe):
        """Test column exists check passes."""
        checker = DataQualityChecker()
        result = checker._check_column_exists(sample_dataframe, "claim_id_key")
        
        assert result.success is True
        assert result.expectation_type == "column_exists"
    
    def test_check_column_exists_fail(self, sample_dataframe):
        """Test column exists check fails."""
        checker = DataQualityChecker()
        result = checker._check_column_exists(sample_dataframe, "nonexistent")
        
        assert result.success is False
    
    def test_check_not_null_pass(self, sample_dataframe):
        """Test not null check passes."""
        checker = DataQualityChecker()
        result = checker._check_not_null(sample_dataframe, "claim_id_key", mostly=1.0)
        
        assert result.success is True
    
    def test_check_not_null_fail(self, dataframe_with_issues):
        """Test not null check fails."""
        checker = DataQualityChecker()
        result = checker._check_not_null(dataframe_with_issues, "age", mostly=1.0)
        
        assert result.success is False
    
    def test_check_unique_pass(self, sample_dataframe):
        """Test unique check passes."""
        checker = DataQualityChecker()
        result = checker._check_unique(sample_dataframe, "claim_id_key", mostly=1.0)
        
        assert result.success is True
    
    def test_check_unique_fail(self, dataframe_with_issues):
        """Test unique check fails."""
        checker = DataQualityChecker()
        result = checker._check_unique(dataframe_with_issues, "claim_id_key", mostly=1.0)
        
        assert result.success is False
    
    def test_check_positive_pass(self, sample_dataframe):
        """Test positive check passes."""
        checker = DataQualityChecker()
        result = checker._check_positive(sample_dataframe, "amt_paid", mostly=1.0)
        
        assert result.success is True
    
    def test_check_positive_fail(self, dataframe_with_issues):
        """Test positive check fails."""
        checker = DataQualityChecker()
        result = checker._check_positive(dataframe_with_issues, "amt_paid", mostly=1.0)
        
        assert result.success is False
    
    def test_check_between_pass(self, sample_dataframe):
        """Test between check passes."""
        checker = DataQualityChecker()
        result = checker._check_between(sample_dataframe, "age", 0, 120, mostly=1.0)
        
        assert result.success is True
    
    def test_check_between_fail(self, dataframe_with_issues):
        """Test between check fails."""
        checker = DataQualityChecker()
        result = checker._check_between(dataframe_with_issues, "age", 0, 120, mostly=1.0)
        
        assert result.success is False
    
    def test_validate_full_suite(self, sample_dataframe):
        """Test full validation with suite."""
        suite = ExpectationSuite("claims_test")
        suite.expect_column_to_exist("claim_id_key")
        suite.expect_column_values_to_not_be_null("claim_id_key")
        suite.expect_column_values_to_be_unique("claim_id_key")
        suite.expect_column_values_to_be_positive("amt_paid")
        
        checker = DataQualityChecker(suite=suite)
        report = checker.validate(sample_dataframe, "test_claims")
        
        assert report.success is True
        assert report.total_expectations == 4
        assert report.successful_expectations == 4
    
    def test_validate_with_failures(self, dataframe_with_issues):
        """Test validation reports failures."""
        suite = ExpectationSuite("test")
        suite.expect_column_values_to_be_unique("claim_id_key")
        suite.expect_column_values_to_be_positive("amt_paid")
        suite.expect_column_values_to_be_between("age", 0, 120)
        
        checker = DataQualityChecker(suite=suite)
        report = checker.validate(dataframe_with_issues, "test")
        
        assert report.success is False
        assert report.failed_expectations > 0


# =============================================================================
# DataValidator Tests
# =============================================================================

class TestDataValidator:
    """Tests for DataValidator."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DataValidator()
        assert validator.schema is None
    
    def test_validate_not_empty_pass(self, sample_dataframe):
        """Test not empty validation passes."""
        validator = DataValidator()
        result = ValidationResult(is_valid=True)
        result = validator._validate_not_empty(sample_dataframe, result)
        
        assert result.is_valid is True
    
    def test_validate_not_empty_fail(self):
        """Test not empty validation fails."""
        empty_df = pd.DataFrame()
        validator = DataValidator()
        result = ValidationResult(is_valid=True)
        result = validator._validate_not_empty(empty_df, result)
        
        assert result.is_valid is False
    
    def test_validate_with_schema(self, sample_dataframe):
        """Test validation with schema."""
        schema = TableSchema(
            name="claims",
            columns=[
                ColumnSchema(name="claim_id_key", dtype="int64", nullable=False, unique=True),
                ColumnSchema(name="amt_paid", dtype="float64", nullable=False, min_value=0)
            ]
        )
        
        validator = DataValidator(schema=schema)
        result = validator.validate(sample_dataframe)
        
        assert result.is_valid is True
    
    def test_compute_statistics(self, sample_dataframe):
        """Test statistics computation."""
        validator = DataValidator()
        stats = validator._compute_statistics(sample_dataframe)
        
        assert "row_count" in stats
        assert stats["row_count"] == len(sample_dataframe)
        assert "null_counts" in stats
        assert "duplicate_rows" in stats


# =============================================================================
# SchemaInferrer Tests
# =============================================================================

class TestSchemaInferrer:
    """Tests for SchemaInferrer."""
    
    def test_infer_schema(self, sample_dataframe):
        """Test schema inference."""
        inferrer = SchemaInferrer()
        schema = inferrer.infer_schema(sample_dataframe, "claims")
        
        assert schema.name == "claims"
        assert len(schema.columns) == len(sample_dataframe.columns)
    
    def test_infer_column_schema(self, sample_dataframe):
        """Test column schema inference."""
        inferrer = SchemaInferrer()
        col_schema = inferrer._infer_column_schema(
            sample_dataframe['amt_paid'],
            'amt_paid'
        )
        
        assert col_schema.name == 'amt_paid'
        assert col_schema.min_value is not None
        assert col_schema.max_value is not None
    
    def test_detect_primary_key(self, sample_dataframe):
        """Test primary key detection."""
        inferrer = SchemaInferrer()
        pk = inferrer._detect_primary_key(sample_dataframe)
        
        assert pk is not None
        assert 'claim_id_key' in pk


# =============================================================================
# DataProfiler Tests
# =============================================================================

class TestDataProfiler:
    """Tests for DataProfiler."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = DataProfiler(sample_size=1000)
        assert profiler.sample_size == 1000
    
    def test_profile_dataframe(self, sample_dataframe):
        """Test dataframe profiling."""
        profiler = DataProfiler()
        profile = profiler.profile(sample_dataframe, "test_claims")
        
        assert profile.name == "test_claims"
        assert profile.row_count == len(sample_dataframe)
        assert profile.column_count == len(sample_dataframe.columns)
        assert len(profile.columns) == len(sample_dataframe.columns)
    
    def test_profile_column(self, sample_dataframe):
        """Test column profiling."""
        profiler = DataProfiler()
        col_profile = profiler._profile_column(
            sample_dataframe['amt_paid'],
            sample_dataframe['amt_paid'],
            'amt_paid'
        )
        
        assert col_profile.name == 'amt_paid'
        assert col_profile.count == len(sample_dataframe)
        assert col_profile.mean is not None
        assert col_profile.std is not None
    
    def test_compute_correlations(self, sample_dataframe):
        """Test correlation computation."""
        profiler = DataProfiler()
        correlations = profiler._compute_correlations(sample_dataframe)
        
        assert correlations is not None
        assert 'amt_paid' in correlations
        assert 'amt_billed' in correlations['amt_paid']
    
    def test_save_profile_json(self, sample_dataframe, tmp_path):
        """Test profile saving to JSON."""
        profiler = DataProfiler(reports_dir=str(tmp_path))
        profile = profiler.profile(sample_dataframe, "test")
        
        filepath = profiler.save_profile(profile, format="json")
        
        assert os.path.exists(filepath)
    
    def test_save_profile_html(self, sample_dataframe, tmp_path):
        """Test profile saving to HTML."""
        profiler = DataProfiler(reports_dir=str(tmp_path))
        profile = profiler.profile(sample_dataframe, "test")
        
        filepath = profiler.save_profile(profile, format="html")
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.html')
    
    def test_compare_profiles(self, sample_dataframe):
        """Test profile comparison."""
        profiler = DataProfiler()
        
        # Create two profiles with slight differences
        profile1 = profiler.profile(sample_dataframe, "baseline")
        
        df2 = sample_dataframe.copy()
        df2 = pd.concat([df2, df2.head(10)])  # Add rows
        profile2 = profiler.profile(df2, "current")
        
        comparison = profiler.compare_profiles(profile1, profile2)
        
        assert comparison["row_count_change"] == 10
        assert comparison["row_count_pct_change"] == 10.0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
