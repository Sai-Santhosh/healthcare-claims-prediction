"""
Model Evaluation Module
Handles model evaluation, metrics calculation, and visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, explained_variance_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logger import get_logger
from ..utils.helpers import timer

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    r2: float
    rmse: float
    mae: float
    mape: float
    explained_variance: float
    residual_mean: float
    residual_std: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'R²': self.r2,
            'RMSE': self.rmse,
            'MAE': self.mae,
            'MAPE': self.mape,
            'Explained Variance': self.explained_variance,
            'Residual Mean': self.residual_mean,
            'Residual Std': self.residual_std
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"R²: {self.r2:.4f} | RMSE: {self.rmse:.4f} | "
            f"MAE: {self.mae:.4f} | MAPE: {self.mape:.2%}"
        )


class ModelEvaluator:
    """
    Handles comprehensive model evaluation.
    """
    
    def __init__(self, figures_dir: str = "reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> EvaluationMetrics:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        
        Returns:
            EvaluationMetrics instance
        """
        residuals = y_true - y_pred
        
        # Handle MAPE for zero values
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            mape = np.nan
        
        metrics = EvaluationMetrics(
            r2=r2_score(y_true, y_pred),
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            mape=mape,
            explained_variance=explained_variance_score(y_true, y_pred),
            residual_mean=residuals.mean(),
            residual_std=residuals.std()
        )
        
        logger.info(f"Evaluation: {metrics.summary()}")
        return metrics
    
    @timer
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model"
    ) -> Tuple[EvaluationMetrics, np.ndarray]:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name for logging
        
        Returns:
            Tuple of (metrics, predictions)
        """
        logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        metrics = self.calculate_metrics(y_test.values, y_pred)
        
        return metrics, y_pred
    
    @timer
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test data.
        
        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test target
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models.items():
            metrics, _ = self.evaluate_model(model, X_test, y_test, name)
            results.append({
                'Model': name,
                **metrics.to_dict()
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        logger.info("\nModel Comparison:")
        logger.info(comparison_df.to_string())
        
        return comparison_df
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Model name for title
            save: Whether to save the figure
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title(f'{model_name}: Actual vs Predicted', fontsize=14)
        ax1.legend()
        
        # Residual plot
        ax2 = axes[1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title(f'{model_name}: Residual Plot', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / f"{model_name.lower().replace(' ', '_')}_predictions.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_residual_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot residual distribution.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Model name for title
            save: Whether to save the figure
        
        Returns:
            matplotlib Figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--', lw=2)
        ax1.axvline(x=residuals.mean(), color='g', linestyle='-', lw=2, label=f'Mean: {residuals.mean():.2f}')
        ax1.set_xlabel('Residual', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'{model_name}: Residual Distribution', fontsize=14)
        ax1.legend()
        
        # Q-Q plot
        ax2 = axes[1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'{model_name}: Q-Q Plot', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / f"{model_name.lower().replace(' ', '_')}_residuals.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_importances: Dict[str, float],
        top_n: int = 20,
        model_name: str = "Model",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importances: Dictionary of feature -> importance
            top_n: Number of top features to show
            model_name: Model name for title
            save: Whether to save the figure
        
        Returns:
            matplotlib Figure
        """
        # Sort and get top features
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        
        colors = ['#2ecc71' if imp >= 0 else '#e74c3c' for imp in importances]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{model_name}: Top {top_n} Feature Importances', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / f"{model_name.lower().replace(' ', '_')}_importance.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'R²',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to plot
            save: Whether to save the figure
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = comparison_df['Model'].values
        values = comparison_df[metric].values
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10
            )
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison: {metric}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / f"model_comparison_{metric.lower().replace(' ', '_')}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot learning curve.
        
        Args:
            train_sizes: Array of training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            model_name: Model name for title
            save: Whether to save the figure
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='#3498db', 
                label='Training Score')
        ax.fill_between(
            train_sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            alpha=0.2, color='#3498db'
        )
        
        ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='#e74c3c',
                label='Validation Score')
        ax.fill_between(
            train_sizes,
            val_scores.mean(axis=1) - val_scores.std(axis=1),
            val_scores.mean(axis=1) + val_scores.std(axis=1),
            alpha=0.2, color='#e74c3c'
        )
        
        ax.set_xlabel('Training Examples', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{model_name}: Learning Curve', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / f"{model_name.lower().replace(' ', '_')}_learning_curve.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def generate_evaluation_report(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
        feature_importances: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Model name
            feature_importances: Optional feature importances
        
        Returns:
            Dictionary with all evaluation results
        """
        logger.info(f"Generating evaluation report for {model_name}...")
        
        # Calculate metrics
        metrics, y_pred = self.evaluate_model(model, X_test, y_test, model_name)
        
        # Generate plots
        self.plot_predictions(y_test.values, y_pred, model_name)
        self.plot_residual_distribution(y_test.values, y_pred, model_name)
        
        if feature_importances:
            self.plot_feature_importance(feature_importances, model_name=model_name)
        
        report = {
            'model_name': model_name,
            'metrics': metrics.to_dict(),
            'predictions': y_pred,
            'actual': y_test.values,
            'residuals': y_test.values - y_pred,
            'feature_importances': feature_importances
        }
        
        logger.info("Evaluation report generated successfully")
        return report
