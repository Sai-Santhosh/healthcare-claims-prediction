"""
Tests for Model Training Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for model testing."""
        np.random.seed(42)
        n = 1000
        X = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.randn(n),
        })
        y = 100 + 20 * X['feature_1'] + 10 * X['feature_2'] + np.random.randn(n) * 5
        return X, pd.Series(y, name='target')
    
    def test_split_data(self, sample_data):
        """Test train/test split."""
        X, y = sample_data
        trainer = ModelTrainer(random_state=42, test_size=0.2)
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        assert len(X_train) == 800
        assert len(X_test) == 200
        assert len(y_train) == 800
        assert len(y_test) == 200
    
    def test_train_lasso(self, sample_data):
        """Test training Lasso model."""
        X, y = sample_data
        trainer = ModelTrainer(random_state=42)
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        result = trainer.train_model('lasso', X_train, y_train, X_test, y_test)
        
        assert result is not None
        assert result.model is not None
        assert result.metadata.model_type == 'lasso'
        assert 'r2' in result.metadata.validation_metrics
    
    def test_train_random_forest(self, sample_data):
        """Test training Random Forest model."""
        X, y = sample_data
        trainer = ModelTrainer(random_state=42)
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        result = trainer.train_model(
            'random_forest', 
            X_train, y_train, X_test, y_test,
            params={'n_estimators': 10, 'max_depth': 5}
        )
        
        assert result is not None
        assert result.feature_importances is not None
    
    def test_get_best_model(self, sample_data):
        """Test getting best model."""
        X, y = sample_data
        trainer = ModelTrainer(random_state=42)
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        trainer.train_model('lasso', X_train, y_train, X_test, y_test)
        trainer.train_model('ridge', X_train, y_train, X_test, y_test)
        
        best = trainer.get_best_model()
        assert best is not None


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert metrics.r2 > 0.9  # Should be high correlation
        assert metrics.mae < 20  # Average error should be small
        assert metrics.rmse < 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
