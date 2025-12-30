"""
Model Training Pipeline for QML Trading System
================================================
Implements walk-forward validation and hyperparameter optimization
for robust model training.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from config.settings import settings
from src.ml.model import ModelConfig, ModelMetrics, QMLModel


@dataclass
class TrainerConfig:
    """Configuration for model training."""
    
    # Walk-forward settings
    n_splits: int = 5
    train_ratio: float = 0.7
    purge_gap: int = 10  # Bars to skip between train/test
    
    # Minimum samples
    min_train_samples: int = 100
    min_test_samples: int = 30
    
    # Hyperparameter optimization
    enable_hyperparam_opt: bool = True
    n_trials: int = 50
    
    # Model configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold."""
    
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    train_samples: int
    test_samples: int
    
    metrics: ModelMetrics
    
    predictions: np.ndarray
    probabilities: np.ndarray
    actuals: np.ndarray


@dataclass
class TrainingResult:
    """Complete training result."""
    
    fold_results: List[WalkForwardResult]
    
    # Aggregate metrics
    mean_accuracy: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_auc: float = 0.0
    mean_win_rate: float = 0.0
    
    std_accuracy: float = 0.0
    std_auc: float = 0.0
    
    # Best model
    best_fold: int = 0
    best_model: Optional[QMLModel] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    
    # Feature importance (averaged)
    feature_importance: Dict[str, float] = field(default_factory=dict)


class ModelTrainer:
    """
    Trains QML models using walk-forward validation.
    
    Walk-forward validation is essential for time-series data
    to avoid look-ahead bias and test on truly out-of-sample data.
    
    The process:
    1. Split data into sequential folds
    2. For each fold: train on past, test on future
    3. Purge gap between train/test to prevent leakage
    4. Aggregate metrics across folds
    
    Optional hyperparameter optimization using Optuna.
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainerConfig()
    
    def train(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[pd.Series] = None
    ) -> TrainingResult:
        """
        Train model using walk-forward validation.
        
        Args:
            features: Feature DataFrame
            labels: Target labels (0/1)
            returns: Optional trade returns for trading metrics
            timestamps: Optional timestamps for time-based splitting
            
        Returns:
            TrainingResult with all fold results and best model
        """
        n_samples = len(features)
        
        if n_samples < self.config.min_train_samples + self.config.min_test_samples:
            raise ValueError(f"Insufficient samples: {n_samples}")
        
        logger.info(f"Starting walk-forward training with {n_samples} samples, {self.config.n_splits} splits")
        
        # Generate fold indices
        folds = self._generate_folds(n_samples, timestamps)
        
        fold_results: List[WalkForwardResult] = []
        best_auc = 0
        best_model = None
        best_fold = 0
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"Training fold {fold_idx + 1}/{len(folds)}")
            
            # Extract data
            X_train = features.iloc[train_idx]
            y_train = labels[train_idx]
            X_test = features.iloc[test_idx]
            y_test = labels[test_idx]
            
            returns_test = returns[test_idx] if returns is not None else None
            
            # Optimize hyperparameters if enabled
            if self.config.enable_hyperparam_opt and OPTUNA_AVAILABLE:
                best_params = self._optimize_hyperparams(X_train, y_train)
                model_config = self._params_to_config(best_params)
            else:
                model_config = self.config.model_config
                best_params = {}
            
            # Train model
            model = QMLModel(config=model_config)
            model.fit(
                X_train, y_train,
                feature_names=list(features.columns)
            )
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test, returns_test)
            
            # Store predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Create fold result
            fold_result = WalkForwardResult(
                fold=fold_idx,
                train_start=timestamps.iloc[train_idx[0]] if timestamps is not None else datetime.now(),
                train_end=timestamps.iloc[train_idx[-1]] if timestamps is not None else datetime.now(),
                test_start=timestamps.iloc[test_idx[0]] if timestamps is not None else datetime.now(),
                test_end=timestamps.iloc[test_idx[-1]] if timestamps is not None else datetime.now(),
                train_samples=len(train_idx),
                test_samples=len(test_idx),
                metrics=metrics,
                predictions=predictions,
                probabilities=probabilities,
                actuals=y_test
            )
            
            fold_results.append(fold_result)
            
            # Track best model
            if metrics.auc_roc > best_auc:
                best_auc = metrics.auc_roc
                best_model = model
                best_fold = fold_idx
            
            logger.info(
                f"Fold {fold_idx + 1}: AUC={metrics.auc_roc:.3f}, "
                f"Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}"
            )
        
        # Aggregate results
        result = self._aggregate_results(fold_results)
        result.best_fold = best_fold
        result.best_model = best_model
        result.best_params = best_params
        
        logger.info(
            f"Training complete: Mean AUC={result.mean_auc:.3f} (±{result.std_auc:.3f}), "
            f"Mean Win Rate={result.mean_win_rate:.3f}"
        )
        
        return result
    
    def _generate_folds(
        self,
        n_samples: int,
        timestamps: Optional[pd.Series] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward fold indices.
        
        Uses expanding window: each fold has more training data than the previous.
        """
        folds = []
        
        # Calculate fold sizes
        test_size = n_samples // self.config.n_splits
        
        for fold in range(self.config.n_splits):
            # Training: all data up to fold
            train_end = (fold + 1) * test_size
            train_start = 0
            
            # Ensure minimum training samples
            if train_end - train_start < self.config.min_train_samples:
                continue
            
            # Test: next fold
            test_start = train_end + self.config.purge_gap
            test_end = min(test_start + test_size, n_samples)
            
            # Ensure minimum test samples
            if test_end - test_start < self.config.min_test_samples:
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            folds.append((train_idx, test_idx))
        
        return folds
    
    def _optimize_hyperparams(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Uses cross-validation on training data to find best params.
        """
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
            
            config = self._params_to_config(params)
            
            # Simple holdout validation for speed
            split = int(len(X) * 0.8)
            X_tr, X_val = X.iloc[:split], X.iloc[split:]
            y_tr, y_val = y[:split], y[split:]
            
            model = QMLModel(config=config)
            model.fit(X_tr, y_tr, feature_names=list(X.columns))
            
            metrics = model.evaluate(X_val, y_val)
            
            return metrics.auc_roc
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        return study.best_params
    
    def _params_to_config(self, params: Dict[str, Any]) -> ModelConfig:
        """Convert parameter dict to ModelConfig."""
        config = ModelConfig()
        
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _aggregate_results(
        self,
        fold_results: List[WalkForwardResult]
    ) -> TrainingResult:
        """Aggregate metrics across folds."""
        
        accuracies = [f.metrics.accuracy for f in fold_results]
        precisions = [f.metrics.precision for f in fold_results]
        recalls = [f.metrics.recall for f in fold_results]
        aucs = [f.metrics.auc_roc for f in fold_results]
        win_rates = [f.metrics.win_rate for f in fold_results]
        
        # Aggregate feature importance
        all_importance: Dict[str, List[float]] = {}
        for f in fold_results:
            for feat, imp in f.metrics.feature_importance.items():
                if feat not in all_importance:
                    all_importance[feat] = []
                all_importance[feat].append(imp)
        
        avg_importance = {
            feat: np.mean(imps) for feat, imps in all_importance.items()
        }
        
        return TrainingResult(
            fold_results=fold_results,
            mean_accuracy=np.mean(accuracies),
            mean_precision=np.mean(precisions),
            mean_recall=np.mean(recalls),
            mean_auc=np.mean(aucs),
            mean_win_rate=np.mean([w for w in win_rates if w > 0]),
            std_accuracy=np.std(accuracies),
            std_auc=np.std(aucs),
            feature_importance=avg_importance
        )
    
    def save_results(
        self,
        result: TrainingResult,
        path: Optional[Path] = None
    ) -> Path:
        """
        Save training results and best model.
        
        Args:
            result: Training result
            path: Save directory
            
        Returns:
            Path where results were saved
        """
        if path is None:
            path = Path(settings.ml.model_path)
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        if result.best_model:
            model_path = result.best_model.save(path / "best_model.joblib")
        
        # Save metrics summary
        summary = {
            "mean_auc": result.mean_auc,
            "std_auc": result.std_auc,
            "mean_accuracy": result.mean_accuracy,
            "mean_precision": result.mean_precision,
            "mean_recall": result.mean_recall,
            "mean_win_rate": result.mean_win_rate,
            "best_fold": result.best_fold,
            "n_folds": len(result.fold_results),
            "feature_importance": result.feature_importance,
            "best_params": result.best_params,
        }
        
        import json
        with open(path / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {path}")
        
        return path


def create_trainer(config: Optional[TrainerConfig] = None) -> ModelTrainer:
    """Factory function for ModelTrainer."""
    return ModelTrainer(config=config)

