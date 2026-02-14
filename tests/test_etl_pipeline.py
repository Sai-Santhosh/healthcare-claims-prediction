"""
Unit Tests for ETL Pipeline
Comprehensive tests for extract, transform, load operations.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.etl.pipeline import (
    ETLPipeline,
    PipelineStage,
    PipelineStatus,
    StageStatus,
    StageResult,
    PipelineResult,
    ETLPipelineBuilder
)
from src.etl.extract import DataExtractor
from src.etl.transform import DataTransformer, DataCleaner
from src.etl.load import DataLoader


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample pipeline configuration."""
    return {
        "etl": {
            "chunk_size": 1000,
            "sample_size": 100,
            "retry_attempts": 2
        },
        "aws": {
            "region": "us-east-1",
            "s3": {
                "data_lake_bucket": "test-bucket"
            }
        }
    }


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
        'icd_diag_01': ['A' + str(i).zfill(2) for i in np.random.randint(0, 100, n_rows)],
        'service_date': pd.date_range('2016-01-01', periods=n_rows).strftime('%Y%m%d')
    })


@pytest.fixture
def temp_data_dirs(tmp_path):
    """Create temporary data directories."""
    dirs = {
        'bronze': tmp_path / 'data' / 'bronze',
        'silver': tmp_path / 'data' / 'silver',
        'gold': tmp_path / 'data' / 'gold'
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipelineStage:
    """Tests for PipelineStage base class."""
    
    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = PipelineStage(name="test_stage", retry_attempts=3)
        assert stage.name == "test_stage"
        assert stage.retry_attempts == 3
    
    def test_stage_execute_not_implemented(self):
        """Test that base _run raises NotImplementedError."""
        stage = PipelineStage(name="test")
        
        with pytest.raises(NotImplementedError):
            stage._run({})


class TestETLPipeline:
    """Tests for ETLPipeline orchestrator."""
    
    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initialization."""
        pipeline = ETLPipeline(name="test_pipeline", config=sample_config)
        
        assert pipeline.name == "test_pipeline"
        assert pipeline.config == sample_config
        assert len(pipeline.stages) == 0
        assert pipeline.pipeline_id is not None
    
    def test_add_stage(self, sample_config):
        """Test adding stages to pipeline."""
        pipeline = ETLPipeline(name="test", config=sample_config)
        
        mock_stage = Mock(spec=PipelineStage)
        mock_stage.name = "mock_stage"
        
        pipeline.add_stage(mock_stage)
        
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0] == mock_stage
    
    def test_pipeline_builder(self, sample_config):
        """Test pipeline builder pattern."""
        mock_extractor = Mock(spec=PipelineStage)
        mock_transformer = Mock(spec=PipelineStage)
        mock_loader = Mock(spec=PipelineStage)
        
        pipeline = (
            ETLPipelineBuilder("test", sample_config)
            .add_extract(mock_extractor)
            .add_transform(mock_transformer)
            .add_load(mock_loader)
            .build()
        )
        
        assert len(pipeline.stages) == 3
    
    def test_pipeline_run_success(self, sample_config, sample_dataframe):
        """Test successful pipeline execution."""
        pipeline = ETLPipeline(name="test", config=sample_config)
        
        # Create mock stage that succeeds
        mock_stage = Mock(spec=PipelineStage)
        mock_stage.name = "mock_stage"
        mock_stage.execute.return_value = StageResult(
            stage_name="mock_stage",
            status=StageStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
            rows_processed=100
        )
        
        pipeline.add_stage(mock_stage)
        
        result = pipeline.run({"dataframe": sample_dataframe})
        
        assert result.status == PipelineStatus.SUCCESS
        assert result.total_rows_processed == 100
    
    def test_pipeline_run_failure(self, sample_config):
        """Test pipeline handles stage failure."""
        pipeline = ETLPipeline(name="test", config=sample_config)
        
        # Create mock stage that fails
        mock_stage = Mock(spec=PipelineStage)
        mock_stage.name = "failing_stage"
        mock_stage.execute.return_value = StageResult(
            stage_name="failing_stage",
            status=StageStatus.FAILED,
            start_time=datetime.now(),
            error_message="Test error"
        )
        
        pipeline.add_stage(mock_stage)
        
        result = pipeline.run()
        
        assert result.status == PipelineStatus.FAILED
        assert "Test error" in result.error_message


# =============================================================================
# Extractor Tests
# =============================================================================

class TestDataExtractor:
    """Tests for DataExtractor."""
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = DataExtractor(chunk_size=5000, delimiter=",")
        
        assert extractor.chunk_size == 5000
        assert extractor.delimiter == ","
    
    def test_generate_sample_data(self, sample_config):
        """Test sample data generation."""
        extractor = DataExtractor()
        
        context = {"config": sample_config}
        result = extractor._generate_sample_data(context)
        
        assert result["rows_processed"] > 0
        assert "dataframe" in context
        assert isinstance(context["dataframe"], pd.DataFrame)
    
    def test_extractor_creates_bronze_output(self, sample_config, temp_data_dirs, monkeypatch):
        """Test that extractor saves to bronze layer."""
        # Change to temp directory
        monkeypatch.chdir(temp_data_dirs['bronze'].parent.parent)
        
        extractor = DataExtractor()
        context = {"config": sample_config}
        
        result = extractor._run(context)
        
        assert result["output_path"] is not None
        assert "bronze" in result["output_path"]


# =============================================================================
# Transformer Tests
# =============================================================================

class TestDataTransformer:
    """Tests for DataTransformer."""
    
    def test_transformer_initialization(self):
        """Test transformer initialization."""
        transformer = DataTransformer()
        
        assert "clean_missing" in transformer.transformations
        assert "standardize_columns" in transformer.transformations
    
    def test_clean_missing_values(self, sample_dataframe):
        """Test missing value handling."""
        # Add some missing values
        df = sample_dataframe.copy()
        df.loc[0, 'amt_paid'] = None
        df.loc[1, 'sex'] = None
        
        transformer = DataTransformer()
        result_df = transformer._transform_clean_missing(df)
        
        assert result_df['amt_paid'].isnull().sum() == 0
        assert result_df['sex'].isnull().sum() == 0
    
    def test_standardize_columns(self, sample_dataframe):
        """Test column standardization."""
        df = sample_dataframe.copy()
        df.columns = ['CLAIM_ID_KEY', 'Member State', 'AGE', 'sex', 
                      'AMT_BILLED', 'AMT_PAID', 'ICD_DIAG_01', 'SERVICE_DATE']
        
        transformer = DataTransformer()
        result_df = transformer._transform_standardize_columns(df)
        
        # All columns should be lowercase
        for col in result_df.columns:
            assert col == col.lower()
    
    def test_encode_categoricals(self, sample_dataframe):
        """Test categorical encoding."""
        transformer = DataTransformer()
        result_df = transformer._transform_encode_categoricals(sample_dataframe)
        
        assert 'gender_code' in result_df.columns
        assert 'age_band' in result_df.columns
    
    def test_create_derived_features(self, sample_dataframe):
        """Test derived feature creation."""
        df = sample_dataframe.copy()
        df.columns = [col.lower() for col in df.columns]
        
        transformer = DataTransformer()
        result_df = transformer._transform_create_derived_features(df)
        
        assert 'icd_category' in result_df.columns
        assert 'payment_ratio' in result_df.columns


class TestDataCleaner:
    """Tests for DataCleaner."""
    
    def test_remove_duplicates(self, sample_dataframe):
        """Test duplicate removal."""
        df = pd.concat([sample_dataframe, sample_dataframe.head(5)])
        
        cleaner = DataCleaner(remove_duplicates=True)
        context = {"dataframe": df}
        
        result = cleaner._run(context)
        
        assert result["metadata"]["rows_removed"]["duplicates"] == 5
    
    def test_remove_negative_amounts(self, sample_dataframe):
        """Test negative amount removal."""
        df = sample_dataframe.copy()
        df.loc[0, 'amt_paid'] = -100
        df.loc[1, 'amt_billed'] = -50
        
        cleaner = DataCleaner(remove_negative_amounts=True)
        context = {"dataframe": df}
        
        result = cleaner._run(context)
        
        assert result["metadata"]["rows_removed"]["negative_amounts"] == 2


# =============================================================================
# Loader Tests
# =============================================================================

class TestDataLoader:
    """Tests for DataLoader."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = DataLoader(output_format="parquet", mode="overwrite")
        
        assert loader.output_format == "parquet"
        assert loader.mode == "overwrite"
    
    def test_write_parquet(self, sample_dataframe, tmp_path):
        """Test parquet file writing."""
        output_path = str(tmp_path / "output.parquet")
        
        loader = DataLoader(output_path=output_path, output_format="parquet")
        rows = loader._write_data(sample_dataframe, output_path)
        
        assert rows == len(sample_dataframe)
        assert os.path.exists(output_path)
        
        # Verify data can be read back
        loaded_df = pd.read_parquet(output_path)
        assert len(loaded_df) == len(sample_dataframe)
    
    def test_write_csv(self, sample_dataframe, tmp_path):
        """Test CSV file writing."""
        output_path = str(tmp_path / "output.csv")
        
        loader = DataLoader(output_format="csv")
        rows = loader._write_data(sample_dataframe, output_path)
        
        assert rows == len(sample_dataframe)
        assert os.path.exists(output_path)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for complete pipeline."""
    
    def test_full_pipeline_execution(self, sample_config, tmp_path, monkeypatch):
        """Test full ETL pipeline execution."""
        # Setup temp directories
        (tmp_path / "data" / "bronze").mkdir(parents=True)
        (tmp_path / "data" / "silver").mkdir(parents=True)
        (tmp_path / "data" / "gold").mkdir(parents=True)
        
        monkeypatch.chdir(tmp_path)
        
        # Build pipeline
        pipeline = ETLPipeline(name="integration_test", config=sample_config)
        pipeline.add_stage(DataExtractor(name="extract"))
        pipeline.add_stage(DataTransformer(name="transform"))
        pipeline.add_stage(DataLoader(
            name="load",
            output_path=str(tmp_path / "data" / "gold" / "output.parquet")
        ))
        
        # Execute
        result = pipeline.run()
        
        assert result.status == PipelineStatus.SUCCESS
        assert result.total_rows_processed > 0
        assert len(result.stages) == 3
        
        # Verify output file exists
        output_file = tmp_path / "data" / "gold" / "output.parquet"
        assert output_file.exists()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
