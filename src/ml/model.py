"""
XGBoost Model for QML Pattern Scoring
======================================
Implements the core ML model for predicting pattern quality
and trade outcomes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available, using sklearn fallback")

from config.settings import settings


@dataclass
class ModelConfig:
    """Configuration for XGBoost model."""
    
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    
    # Training settings
    early_stopping_rounds: int = 20
    eval_metric: str = "logloss"
    
    # Calibration
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"
    
    # Feature scaling
    scale_features: bool = True
    
    # Prediction threshold
    prediction_threshold: float = 0.5


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)


class QMLModel:
    """
    XGBoost-based model for QML pattern quality prediction.
    
    Predicts the probability that a detected pattern will result
    in a winning trade, enabling filtering of low-quality patterns.
    
    Features:
    - Probability calibration for reliable confidence scores
    - Feature importance analysis
    - Model versioning and serialization
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize model.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler() if self.config.scale_features else None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics: Optional[ModelMetrics] = None
    
    def _create_model(self) -> Any:
        """Create the XGBoost classifier."""
        if XGB_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                eval_metric=self.config.eval_metric,
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_child_weight,
                subsample=self.config.subsample,
                random_state=42
            )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "QMLModel":
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training labels (0/1)
            X_val: Optional validation features
            y_val: Optional validation labels
            feature_names: List of feature names
            
        Returns:
            Self for chaining
        """
        self.feature_names = feature_names or list(X.columns)
        
        # Convert to numpy if DataFrame
        X_train = X.values if isinstance(X, pd.DataFrame) else X
        
        # Handle missing values
        X_train = np.nan_to_num(X_train, nan=0.0)
        
        # Scale features
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
        
        # Create and fit model
        self.model = self._create_model()
        
        if X_val is not None and y_val is not None and XGB_AVAILABLE:
            X_val_processed = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            X_val_processed = np.nan_to_num(X_val_processed, nan=0.0)
            
            if self.scaler:
                X_val_processed = self.scaler.transform(X_val_processed)
            
            self.model.fit(
                X_train, y,
                eval_set=[(X_val_processed, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y)
        
        # Calibrate probabilities
        if self.config.calibrate_probabilities:
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method=self.config.calibration_method,
                cv="prefit"
            )
            self.calibrated_model.fit(X_train, y)
        
        self.is_fitted = True
        logger.info(f"Model trained on {len(y)} samples")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            
        Returns:
            Array of predictions (0/1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_processed = self._preprocess(X)
        probs = self.predict_proba(X)
        
        return (probs >= self.config.prediction_threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of probabilities for positive class
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_processed = self._preprocess(X)
        
        if self.calibrated_model:
            probs = self.calibrated_model.predict_proba(X_processed)[:, 1]
        else:
            probs = self.model.predict_proba(X_processed)[:, 1]
        
        return probs
    
    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features for prediction."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        if self.scaler:
            X_arr = self.scaler.transform(X_arr)
        
        return X_arr
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            returns: Optional array of trade returns for trading metrics
            
        Returns:
            ModelMetrics object
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Classification metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, predictions),
            precision=precision_score(y, predictions, zero_division=0),
            recall=recall_score(y, predictions, zero_division=0),
            auc_roc=roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else 0.0
        )
        
        # F1 score
        if metrics.precision + metrics.recall > 0:
            metrics.f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        
        # Trading metrics if returns provided
        if returns is not None:
            predicted_positive = predictions == 1
            actual_positive = y == 1
            
            # Win rate of predicted wins
            if np.sum(predicted_positive) > 0:
                metrics.win_rate = np.sum(predicted_positive & actual_positive) / np.sum(predicted_positive)
            
            # Average win/loss on predictions
            wins = returns[(predictions == 1) & (returns > 0)]
            losses = returns[(predictions == 1) & (returns <= 0)]
            
            if len(wins) > 0:
                metrics.avg_win = np.mean(wins)
            if len(losses) > 0:
                metrics.avg_loss = np.mean(losses)
            
            # Profit factor
            if np.sum(losses) != 0:
                metrics.profit_factor = abs(np.sum(wins) / np.sum(losses))
        
        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            metrics.feature_importance = dict(zip(self.feature_names, importance))
        
        self.metrics = metrics
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            return dict(sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            ))
        
        return {}
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save model to disk.
        
        Args:
            path: Save path (defaults to models directory)
            
        Returns:
            Path where model was saved
        """
        if path is None:
            path = Path(settings.ml.model_path) / f"qml_model_{self.version}.joblib"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config,
            "version": self.version,
            "metrics": self.metrics,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> "QMLModel":
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded QMLModel instance
        """
        model_data = joblib.load(path)
        
        instance = cls(config=model_data["config"])
        instance.model = model_data["model"]
        instance.calibrated_model = model_data["calibrated_model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.version = model_data["version"]
        instance.metrics = model_data["metrics"]
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {path} (version: {instance.version})")
        
        return instance


def create_model(config: Optional[ModelConfig] = None) -> QMLModel:
    """Factory function for QMLModel."""
    return QMLModel(config=config)

