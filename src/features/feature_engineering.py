"""
Feature Engineering Module
Handles feature creation, transformation, and selection.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import zscore

from ..utils.logger import get_logger, log_dataframe_info
from ..utils.helpers import timer

logger = get_logger(__name__)


@dataclass
class FeatureSet:
    """Container for feature set with metadata."""
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    target_name: str
    
    def shape_summary(self) -> str:
        return f"Features: {self.X.shape}, Target: {self.y.shape}"


@dataclass 
class TransformerState:
    """Stores fitted transformer state for persistence."""
    scalers: Dict[str, Any] = field(default_factory=dict)
    encoders: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    categorical_mapping: Dict[str, Dict] = field(default_factory=dict)


class FeatureEngineer:
    """
    Handles feature engineering operations.
    Provides methods for creating, transforming, and preparing features.
    """
    
    def __init__(self):
        self.state = TransformerState()
        self.is_fitted = False
    
    def save_state(self, filepath: str) -> None:
        """Save transformer state to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.state, f)
        logger.info(f"Saved transformer state to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load transformer state from file."""
        with open(filepath, 'rb') as f:
            self.state = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded transformer state from {filepath}")
    
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
            drop_first: Whether to drop first category
        
        Returns:
            DataFrame with dummy variables
        """
        existing = [c for c in columns if c in df.columns]
        if not existing:
            logger.warning("No categorical columns found")
            return df
        
        initial_cols = len(df.columns)
        df = pd.get_dummies(df, columns=existing, drop_first=drop_first)
        
        # Store mapping for inference
        for col in existing:
            if col not in self.state.categorical_mapping:
                self.state.categorical_mapping[col] = None  # Will be populated during fit
        
        new_cols = len(df.columns) - initial_cols + len(existing)
        logger.info(f"Created {new_cols} dummy variables from {len(existing)} columns")
        
        return df
    
    @timer
    def fit_scalers(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Fit and apply scalers to numerical columns.
        
        Args:
            df: Input DataFrame
            numerical_columns: Columns to scale
            method: Scaling method ('standard', 'minmax', 'zscore')
        
        Returns:
            DataFrame with scaled columns
        """
        existing = [c for c in numerical_columns if c in df.columns]
        if not existing:
            logger.warning("No numerical columns found")
            return df
        
        if method == 'zscore':
            # Use scipy zscore (applied column-wise)
            for col in existing:
                df[col] = zscore(df[col].values, nan_policy='omit')
                # Store mean and std for inference
                self.state.scalers[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
        else:
            scaler_class = StandardScaler if method == 'standard' else MinMaxScaler
            
            for col in existing:
                scaler = scaler_class()
                df[col] = scaler.fit_transform(df[[col]])
                self.state.scalers[col] = scaler
        
        self.is_fitted = True
        logger.info(f"Fitted {method} scalers on {len(existing)} columns")
        
        return df
    
    @timer
    def transform_scalers(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply fitted scalers to new data.
        
        Args:
            df: Input DataFrame
            numerical_columns: Columns to scale
        
        Returns:
            DataFrame with scaled columns
        """
        if not self.is_fitted:
            raise RuntimeError("Transformers not fitted. Call fit_scalers first.")
        
        existing = [c for c in numerical_columns if c in df.columns and c in self.state.scalers]
        
        for col in existing:
            scaler = self.state.scalers[col]
            if isinstance(scaler, dict):
                # zscore method
                df[col] = (df[col] - scaler['mean']) / scaler['std']
            else:
                df[col] = scaler.transform(df[[col]])
        
        logger.info(f"Transformed {len(existing)} columns")
        return df
    
    @timer
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[Tuple[str, str]],
        operations: List[str] = ['multiply', 'ratio']
    ) -> pd.DataFrame:
        """
        Create interaction features between column pairs.
        
        Args:
            df: Input DataFrame
            column_pairs: List of column pairs for interaction
            operations: Operations to apply ('multiply', 'ratio', 'add', 'subtract')
        
        Returns:
            DataFrame with interaction features
        """
        for col1, col2 in column_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            if 'multiply' in operations:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            if 'ratio' in operations:
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
            
            if 'add' in operations:
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            
            if 'subtract' in operations:
                df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
        
        logger.info(f"Created interaction features for {len(column_pairs)} pairs")
        return df
    
    @timer
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features for numerical columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to transform
            degree: Polynomial degree
        
        Returns:
            DataFrame with polynomial features
        """
        existing = [c for c in columns if c in df.columns]
        
        for col in existing:
            for d in range(2, degree + 1):
                df[f'{col}_pow{d}'] = df[col] ** d
        
        logger.info(f"Created polynomial features (degree={degree}) for {len(existing)} columns")
        return df
    
    @timer
    def create_binned_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_bins: int = 5,
        strategy: str = 'quantile'
    ) -> pd.DataFrame:
        """
        Create binned versions of numerical columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to bin
            n_bins: Number of bins
            strategy: Binning strategy ('quantile', 'uniform')
        
        Returns:
            DataFrame with binned features
        """
        existing = [c for c in columns if c in df.columns]
        
        for col in existing:
            if strategy == 'quantile':
                df[f'{col}_binned'] = pd.qcut(
                    df[col], q=n_bins, labels=False, duplicates='drop'
                )
            else:
                df[f'{col}_binned'] = pd.cut(
                    df[col], bins=n_bins, labels=False
                )
        
        logger.info(f"Created binned features for {len(existing)} columns")
        return df
    
    @timer
    def create_log_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Create log-transformed features.
        
        Args:
            df: Input DataFrame
            columns: Columns to transform
        
        Returns:
            DataFrame with log features
        """
        existing = [c for c in columns if c in df.columns]
        
        for col in existing:
            # Add small constant to handle zeros
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
        logger.info(f"Created log features for {len(existing)} columns")
        return df
    
    @timer
    def prepare_feature_set(
        self,
        df: pd.DataFrame,
        target_column: str,
        categorical_columns: List[str],
        numerical_columns: List[str],
        scale_method: str = 'zscore',
        create_dummies: bool = True,
        fit: bool = True
    ) -> FeatureSet:
        """
        Prepare complete feature set for modeling.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            categorical_columns: Categorical columns
            numerical_columns: Numerical columns
            scale_method: Scaling method
            create_dummies: Whether to create dummy variables
            fit: Whether to fit transformers (True for training, False for inference)
        
        Returns:
            FeatureSet containing prepared features and target
        """
        logger.info("Preparing feature set")
        log_dataframe_info(df, logger, "Input")
        
        # Separate target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
        
        # Process categorical features
        cat_cols = [c for c in categorical_columns if c in X.columns]
        if create_dummies and cat_cols:
            X = self.create_dummy_variables(X, cat_cols)
        
        # Process numerical features
        num_cols = [c for c in numerical_columns if c in X.columns and c != target_column]
        if num_cols:
            if fit:
                X = self.fit_scalers(X, num_cols, method=scale_method)
            else:
                X = self.transform_scalers(X, num_cols)
        
        # Store feature names
        self.state.feature_names = list(X.columns)
        
        feature_set = FeatureSet(
            X=X,
            y=y,
            feature_names=list(X.columns),
            categorical_features=cat_cols,
            numerical_features=num_cols,
            target_name=target_column
        )
        
        logger.info(f"Prepared {feature_set.shape_summary()}")
        return feature_set


class FeatureSelector:
    """
    Handles feature selection operations.
    """
    
    def __init__(self):
        self.selected_features: List[str] = []
        self.feature_importances: Dict[str, float] = {}
    
    @timer
    def select_by_variance(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove low-variance features.
        
        Args:
            X: Feature DataFrame
            threshold: Variance threshold
        
        Returns:
            Tuple of (filtered DataFrame, removed column names)
        """
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        mask = selector.get_support()
        removed = [col for col, keep in zip(X.columns, mask) if not keep]
        
        X_filtered = X.loc[:, mask]
        logger.info(f"Removed {len(removed)} low-variance features")
        
        return X_filtered, removed
    
    @timer
    def select_k_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 20,
        score_func: str = 'f_regression'
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Select top-k features using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            score_func: Scoring function ('f_regression', 'mutual_info')
        
        Returns:
            Tuple of (selected features DataFrame, feature scores)
        """
        if score_func == 'f_regression':
            func = f_regression
        else:
            func = mutual_info_regression
        
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=func, k=k)
        selector.fit(X, y)
        
        # Get feature scores
        scores = dict(zip(X.columns, selector.scores_))
        self.feature_importances.update(scores)
        
        # Get selected features
        mask = selector.get_support()
        X_selected = X.loc[:, mask]
        self.selected_features = list(X_selected.columns)
        
        logger.info(f"Selected {k} best features")
        return X_selected, scores
    
    @timer
    def select_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        n_estimators: int = 100
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Select features based on Random Forest importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            n_estimators: Number of trees for importance estimation
        
        Returns:
            Tuple of (selected features DataFrame, feature importances)
        """
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = dict(zip(X.columns, rf.feature_importances_))
        self.feature_importances.update(importances)
        
        # Select top features
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:n_features]]
        self.selected_features = top_features
        
        X_selected = X[top_features]
        
        logger.info(f"Selected {n_features} features by importance")
        return X_selected, importances
    
    @timer
    def select_by_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        step: int = 5
    ) -> pd.DataFrame:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            step: Number of features to remove at each iteration
        
        Returns:
            DataFrame with selected features
        """
        estimator = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        )
        
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=step
        )
        selector.fit(X, y)
        
        mask = selector.get_support()
        self.selected_features = list(X.columns[mask])
        
        X_selected = X.loc[:, mask]
        
        logger.info(f"Selected {n_features} features using RFE")
        return X_selected
    
    @timer
    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
        
        Returns:
            Tuple of (filtered DataFrame, removed column names)
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find columns with correlation above threshold
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        
        X_filtered = X.drop(columns=to_drop)
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        return X_filtered, to_drop
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
