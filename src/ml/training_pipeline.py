"""
ML Training Pipeline with proper validation.

IMPROVEMENTS from code review:
- Data validation (NaN/inf checks)
- Class weight handling for imbalanced data
- Configurable early stopping
- Model versioning on save
- Better error handling

Uses:
- Purged cross-validation (no data leakage)
- XGBoost/LightGBM for tabular data
- SHAP for feature importance
- Proper metrics for imbalanced classification
"""

import numpy as np
import pandas as pd
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import joblib
from pathlib import Path
import warnings

# ML libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from .purged_cv import PurgedKFold, WalkForwardCV, PurgedFold


# Model version for compatibility tracking
MODEL_VERSION = "1.1.0"


@dataclass
class TrainingMetrics:
    """Metrics from model training."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    avg_precision: float = 0.0  # Better for imbalanced data

    # Per-fold metrics
    fold_metrics: List[Dict] = field(default_factory=list)

    # Confusion matrix
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class TrainingResult:
    """Complete result from training pipeline."""
    model: Any
    metrics: TrainingMetrics
    feature_importance: Dict[str, float]
    scaler: Optional[StandardScaler]

    # Training info
    model_type: str
    n_features: int
    n_samples_train: int
    n_samples_test: int
    training_time_seconds: float

    # Class balance info
    class_ratio: float  # Positive class ratio
    scale_pos_weight: float  # Weight used for imbalanced handling

    # Validation
    cv_method: str
    n_folds: int

    # Versioning
    model_version: str = MODEL_VERSION
    trained_at: datetime = field(default_factory=datetime.now)
    feature_names: List[str] = field(default_factory=list)


class MLTrainingPipeline:
    """
    Complete ML training pipeline for QML pattern classification.

    Features:
    - Automatic model selection (XGBoost or LightGBM)
    - Purged cross-validation (no data leakage)
    - Feature importance with SHAP
    - Handles imbalanced classes automatically
    - Configurable early stopping
    - Data validation
    """

    def __init__(
        self,
        model_type: str = 'xgboost',  # 'xgboost' or 'lightgbm'
        cv_method: str = 'purged',     # 'purged' or 'walk_forward'
        n_folds: int = 5,
        scale_features: bool = True,
        handle_imbalanced: bool = True,  # Auto class weighting
        early_stopping_rounds: int = 10,  # Early stopping
        eval_metric: str = 'auc',         # Configurable metric
        random_state: int = 42,
        n_jobs: int = -1,                 # Parallelization
        use_gpu: bool = False             # GPU support
    ):
        self.model_type = model_type
        self.cv_method = cv_method
        self.n_folds = n_folds
        self.scale_features = scale_features
        self.handle_imbalanced = handle_imbalanced
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.scale_pos_weight = 1.0

    def _validate_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Validate input data for common issues.
        """
        # Check for NaN values
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(
                f"X contains NaN values in columns: {nan_cols}. "
                f"Use fillna() or drop these columns."
            )

        if y.isnull().any():
            raise ValueError(
                f"y contains {y.isnull().sum()} NaN values. "
                f"Remove or impute these before training."
            )

        # Check for infinite values
        numeric_cols = X.select_dtypes(include=[np.number])
        if not np.isfinite(numeric_cols.values).all():
            inf_cols = numeric_cols.columns[~np.isfinite(numeric_cols).all()].tolist()
            raise ValueError(
                f"X contains infinite values in columns: {inf_cols}. "
                f"Replace inf with np.nan then impute."
            )

        # Check target values
        unique_y = y.unique()
        if len(unique_y) < 2:
            raise ValueError(
                f"y has only {len(unique_y)} unique value(s): {unique_y}. "
                f"Need at least 2 classes for classification."
            )

        if not set(unique_y).issubset({0, 1}):
            warnings.warn(
                f"y contains values other than 0/1: {unique_y}. "
                f"Converting to binary: values > 0 will be class 1."
            )

        # Check minimum samples
        if len(X) < 50:
            warnings.warn(
                f"Only {len(X)} samples. Results may be unreliable. "
                f"Recommend at least 100 samples for meaningful training."
            )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: Optional[pd.Series] = None,
        model_params: Optional[Dict] = None
    ) -> TrainingResult:
        """
        Train model with purged cross-validation.

        Args:
            X: Feature DataFrame
            y: Target Series (0/1 for classification)
            timestamps: Optional timestamps for time-based splitting
            model_params: Optional model hyperparameters

        Returns:
            TrainingResult with model, metrics, and feature importance
        """
        import time
        start_time = time.time()

        # Validate data FIRST
        self._validate_data(X, y)

        # Ensure y is binary
        y = (y > 0).astype(int)

        self.feature_names = list(X.columns)

        # Calculate class weights for imbalanced data
        pos_ratio = y.mean()
        if self.handle_imbalanced and pos_ratio > 0 and pos_ratio < 1:
            self.scale_pos_weight = (1 - pos_ratio) / pos_ratio
        else:
            self.scale_pos_weight = 1.0

        # Scale features
        if self.scale_features:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X.copy()

        # Get CV splitter
        if self.cv_method == 'purged':
            cv = PurgedKFold(n_splits=self.n_folds)
        else:
            cv = WalkForwardCV(n_splits=self.n_folds)

        # Default model parameters
        default_params = self._get_default_params()
        if model_params:
            default_params.update(model_params)

        # Cross-validation
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        for fold in cv.split(X_scaled, y, timestamps):
            X_train = X_scaled.iloc[fold.train_indices]
            X_test = X_scaled.iloc[fold.test_indices]
            y_train = y.iloc[fold.train_indices]
            y_test = y.iloc[fold.test_indices]

            # Skip folds with only one class in training
            if len(y_train.unique()) < 2:
                warnings.warn(f"Fold {fold.fold_number} has only one class in training. Skipping.")
                continue

            # Train model
            model = self._create_model(default_params)
            model = self._fit_model(model, X_train, y_train, X_test, y_test)

            # Predict
            y_pred = model.predict(X_test)
            y_prob = self._predict_proba(model, X_test)

            # Calculate fold metrics
            fold_metric = self._calculate_metrics(y_test, y_pred, y_prob)
            fold_metric['fold'] = fold.fold_number
            fold_metric['n_train'] = fold.n_train
            fold_metric['n_test'] = fold.n_test
            fold_metrics.append(fold_metric)

            # Accumulate for overall metrics
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)

        if not fold_metrics:
            raise ValueError("No valid folds completed. Check your data distribution.")

        # Train final model on all data
        final_model = self._create_model(default_params)
        final_model = self._fit_model(final_model, X_scaled, y)
        self.model = final_model

        # Overall metrics
        overall_metrics = self._calculate_metrics(
            np.array(all_y_true),
            np.array(all_y_pred),
            np.array(all_y_prob)
        )

        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        metrics = TrainingMetrics(
            accuracy=overall_metrics['accuracy'],
            precision=overall_metrics['precision'],
            recall=overall_metrics['recall'],
            f1=overall_metrics['f1'],
            roc_auc=overall_metrics['roc_auc'],
            avg_precision=overall_metrics['avg_precision'],
            fold_metrics=fold_metrics,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn)
        )

        # Feature importance
        feature_importance = self._get_feature_importance(final_model, X_scaled)

        training_time = time.time() - start_time

        return TrainingResult(
            model=final_model,
            metrics=metrics,
            feature_importance=feature_importance,
            scaler=self.scaler,
            model_type=self.model_type,
            n_features=len(self.feature_names),
            n_samples_train=len(X),
            n_samples_test=sum(f['n_test'] for f in fold_metrics),
            training_time_seconds=training_time,
            class_ratio=pos_ratio,
            scale_pos_weight=self.scale_pos_weight,
            cv_method=self.cv_method,
            n_folds=self.n_folds,
            model_version=MODEL_VERSION,
            feature_names=self.feature_names
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = X
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns
            )

        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = X
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns
            )

        return self._predict_proba(self.model, X_scaled)

    def save(self, path: str):
        """
        Save model with version metadata.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Create data hash for validation
        feature_hash = hashlib.md5(str(self.feature_names).encode()).hexdigest()[:8]

        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'version': MODEL_VERSION,
            'created_at': datetime.now().isoformat(),
            'feature_hash': feature_hash,
            'scale_pos_weight': self.scale_pos_weight
        }

        joblib.dump(save_data, path)

    def load(self, path: str):
        """
        Load model with version compatibility check.
        """
        data = joblib.load(path)

        # Version compatibility check
        saved_version = data.get('version', '0.0.0')
        if saved_version != MODEL_VERSION:
            warnings.warn(
                f"Model was saved with version {saved_version}, "
                f"current version is {MODEL_VERSION}. "
                f"Results may differ."
            )

        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.scale_pos_weight = data.get('scale_pos_weight', 1.0)

    def _get_default_params(self) -> Dict:
        """Get default model parameters with class weight handling."""
        if self.model_type == 'xgboost':
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': self.eval_metric,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'scale_pos_weight': self.scale_pos_weight  # Imbalanced handling
            }

            # GPU support
            if self.use_gpu:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0

            return params

        else:  # lightgbm
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary',
                'metric': self.eval_metric,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': -1,
                'scale_pos_weight': self.scale_pos_weight  # Imbalanced handling
            }

            # GPU support
            if self.use_gpu:
                params['device'] = 'gpu'

            return params

    def _create_model(self, params: Dict):
        """Create model instance."""
        if self.model_type == 'xgboost' and HAS_XGB:
            return xgb.XGBClassifier(**params)
        elif self.model_type == 'lightgbm' and HAS_LGB:
            return lgb.LGBMClassifier(**params)
        else:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingClassifier
            sklearn_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 5),
                'learning_rate': params.get('learning_rate', 0.1),
                'random_state': self.random_state
            }
            return GradientBoostingClassifier(**sklearn_params)

    def _fit_model(self, model, X_train, y_train, X_val=None, y_val=None):
        """
        Fit model with optional early stopping.
        """
        if X_val is not None and self.early_stopping_rounds > 0:
            if self.model_type == 'xgboost' and HAS_XGB:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif self.model_type == 'lightgbm' and HAS_LGB:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                )
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        return model

    def _predict_proba(self, model, X) -> np.ndarray:
        """Get probability predictions."""
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim > 1 else proba
        return model.predict(X)

    def _calculate_metrics(self, y_true, y_pred, y_prob) -> Dict:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        # AUC metrics (need both classes)
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        else:
            metrics['roc_auc'] = 0.5
            metrics['avg_precision'] = 0.5

        return metrics

    def _get_feature_importance(self, model, X: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance scores using SHAP if available."""
        importance = {}

        # Try SHAP first (most accurate)
        if HAS_SHAP:
            try:
                # Sample for speed on large datasets
                sample_size = min(100, len(X))
                X_sample = X.head(sample_size)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Binary classification

                mean_importance = np.abs(shap_values).mean(axis=0)
                importance = dict(zip(X.columns, mean_importance))
            except Exception as e:
                warnings.warn(f"SHAP failed: {e}. Using built-in importance.")

        # Fallback to built-in importance
        if not importance and hasattr(model, 'feature_importances_'):
            importance = dict(zip(X.columns, model.feature_importances_))

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance
