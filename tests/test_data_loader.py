"""
Tests for Data Loader Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader, ChunkedDataLoader


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            for i in range(100):
                f.write(f"{i},{i*2},{i*3}\n")
            return f.name
    
    def test_load_csv(self, sample_csv):
        """Test loading a CSV file."""
        loader = DataLoader(delimiter=',')
        df = loader.load_csv(sample_csv)
        
        assert len(df) == 100
        assert list(df.columns) == ['col1', 'col2', 'col3']
        
        os.unlink(sample_csv)
    
    def test_load_csv_with_nrows(self, sample_csv):
        """Test loading limited rows."""
        loader = DataLoader(delimiter=',')
        df = loader.load_csv(sample_csv, nrows=10)
        
        assert len(df) == 10
        
        os.unlink(sample_csv)


class TestChunkedDataLoader:
    """Tests for ChunkedDataLoader class."""
    
    @pytest.fixture
    def large_csv(self):
        """Create a larger CSV file for chunked testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,value,category\n")
            for i in range(1000):
                f.write(f"{i % 100},{i*1.5},{'A' if i % 2 == 0 else 'B'}\n")
            return f.name
    
    def test_iterate_chunks(self, large_csv):
        """Test chunked iteration."""
        loader = ChunkedDataLoader(chunk_size=100, delimiter=',')
        
        chunks = list(loader.iterate_chunks(large_csv))
        
        assert len(chunks) == 10
        assert all(len(chunk) == 100 for chunk in chunks)
        
        os.unlink(large_csv)
    
    def test_load_full_file(self, large_csv):
        """Test loading full file via chunking."""
        loader = ChunkedDataLoader(chunk_size=100, delimiter=',')
        
        df = loader.load_full_file(large_csv, total_rows=1000)
        
        assert len(df) == 1000
        
        os.unlink(large_csv)


class TestDataValidation:
    """Tests for data validation."""
    
    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        df = pd.DataFrame()
        
        validator.check_not_empty(df)
        
        report = validator.generate_report()
        assert not report.is_valid
    
    def test_valid_dataframe(self):
        """Test validation of valid DataFrame."""
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        df = pd.DataFrame({
            'AMT_BILLED': [100, 200, 300],
            'AMT_PAID': [50, 100, 150]
        })
        
        validator.check_not_empty(df)
        validator.check_required_columns(df, ['AMT_BILLED', 'AMT_PAID'])
        
        report = validator.generate_report()
        assert report.is_valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
