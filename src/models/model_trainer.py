"""
Model Training Module
Handles training, hyperparameter tuning, and model management.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..utils.logger import get_logger
from ..utils.helpers import timer, get_timestamp

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_name: str
    model_type: str
    version: str
    created_at: str
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    feature_names: List[str]
    training_samples: int
    random_state: int


@dataclass
class TrainingResult:
    """Result of model training."""
    model: Any
    metadata: ModelMetadata
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    feature_importances: Optional[Dict[str, float]] = None


class ModelTrainer:
    """
    Handles model training and hyperparameter tuning.
    """
    
    # Supported models
    MODELS = {
        'lasso': Lasso,
        'ridge': Ridge,
        'elasticnet': ElasticNet,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'adaboost': AdaBoostRegressor,
    }
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        'lasso': {'alpha': 0.1, 'max_iter': 1000},
        'ridge': {'alpha': 0.5, 'max_iter': 1000},
        'elasticnet': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000},
        'random_forest': {
            'n_estimators': 300,
            'max_depth': 30,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42
        },
        'gradient_boosting': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'adaboost': {
            'n_estimators': 200,
            'learning_rate': 1.0,
            'random_state': 42
        },
    }
    
    # Hyperparameter grids for tuning
    PARAM_GRIDS = {
        'lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'ridge': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        },
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
    }
    
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2
    ):
        self.random_state = random_state
        self.test_size = test_size
        self.trained_models: Dict[str, TrainingResult] = {}
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        stratify: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and validation sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Test/validation size
            stratify: Column for stratified splitting
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        test_size = test_size or self.test_size
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        logger.info(f"Data split: Train={len(X_train):,}, Val={len(X_val):,}")
        return X_train, X_val, y_train, y_val
    
    @timer
    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> TrainingResult:
        """
        Train a single model.
        
        Args:
            model_type: Type of model ('lasso', 'ridge', 'random_forest', etc.)
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model hyperparameters
            version: Model version string
        
        Returns:
            TrainingResult containing model and metrics
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Training {model_type} model...")
        
        # Get parameters
        model_params = self.DEFAULT_PARAMS.get(model_type, {}).copy()
        if params:
            model_params.update(params)
        
        # Initialize and train model
        model_class = self.MODELS[model_type]
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_metrics = {
            'r2': r2_score(y_train, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'mae': mean_absolute_error(y_train, train_pred)
        }
        
        val_metrics = {
            'r2': r2_score(y_val, val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred)
        }
        
        # Get feature importances if available
        feature_importances = None
        if hasattr(model, 'feature_importances_'):
            feature_importances = dict(zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importances = dict(zip(X_train.columns, np.abs(model.coef_)))
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=f"{model_type}_{version or get_timestamp()}",
            model_type=model_type,
            version=version or get_timestamp(),
            created_at=datetime.now().isoformat(),
            hyperparameters=model_params,
            training_metrics=train_metrics,
            validation_metrics=val_metrics,
            feature_names=list(X_train.columns),
            training_samples=len(X_train),
            random_state=self.random_state
        )
        
        result = TrainingResult(
            model=model,
            metadata=metadata,
            train_predictions=train_pred,
            val_predictions=val_pred,
            feature_importances=feature_importances
        )
        
        # Store result
        self.trained_models[model_type] = result
        
        logger.info(f"Training R²: {train_metrics['r2']:.4f}")
        logger.info(f"Validation R²: {val_metrics['r2']:.4f}")
        logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")
        
        return result
    
    @timer
    def tune_hyperparameters(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        cv: int = 5,
        n_iter: Optional[int] = None,
        scoring: str = 'r2'
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune model hyperparameters using grid or random search.
        
        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training target
            param_grid: Hyperparameter grid
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search (None for grid search)
            scoring: Scoring metric
        
        Returns:
            Tuple of (best parameters, best score)
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Tuning {model_type} hyperparameters...")
        
        # Get parameter grid
        grid = param_grid or self.PARAM_GRIDS.get(model_type, {})
        if not grid:
            logger.warning(f"No parameter grid for {model_type}")
            return {}, 0.0
        
        # Initialize base model
        model_class = self.MODELS[model_type]
        base_params = self.DEFAULT_PARAMS.get(model_type, {}).copy()
        model = model_class(**base_params)
        
        # Choose search method
        if n_iter:
            search = RandomizedSearchCV(
                model, grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state
            )
        else:
            search = GridSearchCV(
                model, grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_params_, search.best_score_
    
    @timer
    def cross_validate(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model_type: Type of model
            X: Features DataFrame
            y: Target Series
            cv: Number of folds
            params: Model parameters
        
        Returns:
            Dictionary with CV scores
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Cross-validating {model_type}...")
        
        model_params = self.DEFAULT_PARAMS.get(model_type, {}).copy()
        if params:
            model_params.update(params)
        
        model = self.MODELS[model_type](**model_params)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
        
        results = {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'scores': scores.tolist()
        }
        
        logger.info(f"CV R²: {results['mean_r2']:.4f} (+/- {results['std_r2']:.4f})")
        
        return results
    
    @timer
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, TrainingResult]:
        """
        Train multiple models and compare results.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            model_types: List of model types to train
        
        Returns:
            Dictionary of training results
        """
        model_types = model_types or list(self.MODELS.keys())
        
        results = {}
        for model_type in model_types:
            try:
                result = self.train_model(
                    model_type, X_train, y_train, X_val, y_val
                )
                results[model_type] = result
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
        
        # Log comparison
        logger.info("\nModel Comparison:")
        logger.info("-" * 50)
        for name, result in sorted(
            results.items(),
            key=lambda x: x[1].metadata.validation_metrics['r2'],
            reverse=True
        ):
            r2 = result.metadata.validation_metrics['r2']
            rmse = result.metadata.validation_metrics['rmse']
            logger.info(f"{name:20s}: R²={r2:.4f}, RMSE={rmse:.4f}")
        
        return results
    
    def get_best_model(self) -> Optional[TrainingResult]:
        """Get the best performing model based on validation R²."""
        if not self.trained_models:
            return None
        
        return max(
            self.trained_models.values(),
            key=lambda x: x.metadata.validation_metrics['r2']
        )


class ModelRegistry:
    """
    Manages model versioning, storage, and retrieval.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "production": None}
    
    def _save_registry(self) -> None:
        """Save model registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def save_model(
        self,
        result: TrainingResult,
        model_name: Optional[str] = None
    ) -> str:
        """
        Save a trained model to disk.
        
        Args:
            result: TrainingResult to save
            model_name: Optional custom name
        
        Returns:
            Model path
        """
        name = model_name or result.metadata.model_name
        version = result.metadata.version
        
        # Create model directory
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"model_v{version}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(result.model, f)
        
        # Save metadata
        metadata_path = model_dir / f"metadata_v{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(result.metadata), f, indent=2)
        
        # Update registry
        if name not in self.registry["models"]:
            self.registry["models"][name] = {"versions": [], "latest": None}
        
        self.registry["models"][name]["versions"].append(version)
        self.registry["models"][name]["latest"] = version
        self._save_registry()
        
        logger.info(f"Saved model to {model_path}")
        return str(model_path)
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Tuple[Any, ModelMetadata]:
        """
        Load a model from disk.
        
        Args:
            model_name: Name of the model
            version: Specific version (latest if not specified)
        
        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_name}")
        
        version = version or self.registry["models"][model_name]["latest"]
        
        model_dir = self.models_dir / model_name
        model_path = model_dir / f"model_v{version}.pkl"
        metadata_path = model_dir / f"metadata_v{version}.json"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
            metadata = ModelMetadata(**metadata_dict)
        
        logger.info(f"Loaded model from {model_path}")
        return model, metadata
    
    def set_production_model(self, model_name: str, version: Optional[str] = None) -> None:
        """Set a model as the production model."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_name}")
        
        version = version or self.registry["models"][model_name]["latest"]
        self.registry["production"] = {"model": model_name, "version": version}
        self._save_registry()
        
        logger.info(f"Set production model: {model_name} v{version}")
    
    def get_production_model(self) -> Optional[Tuple[Any, ModelMetadata]]:
        """Get the current production model."""
        if not self.registry["production"]:
            return None
        
        return self.load_model(
            self.registry["production"]["model"],
            self.registry["production"]["version"]
        )
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models."""
        return self.registry["models"]
