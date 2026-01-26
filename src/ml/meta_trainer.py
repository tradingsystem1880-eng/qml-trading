"""
ML Meta-Labeling Trainer for Phase 8.0
======================================
Orchestrates walk-forward ML training for meta-labeling.

Uses:
- Existing PatternFeatureExtractor for 170+ features
- Existing WalkForwardCV for time-series splits
- Existing XGBoostPredictor as base model
- Triple-barrier labels with quality distinction (hit_tp_before_sl)

The meta-labeler predicts WIN probability for position sizing,
NOT to replace the working detector.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import warnings

import numpy as np
import pandas as pd

# sklearn imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. pip install xgboost")

# Project imports
from src.ml.labeler import TripleBarrierLabeler, LabelConfig, LabelResult, Label
from src.ml.purged_cv import WalkForwardCV


@dataclass
class MetaTrainerConfig:
    """Configuration for meta-labeling training."""
    # Cross-validation
    n_folds: int = 5
    min_train_samples: int = 100
    min_test_samples: int = 20

    # Feature pruning
    max_features: int = 20  # Max features to keep (prevent overfitting)
    min_feature_importance: float = 0.001  # Minimum importance to keep

    # XGBoost parameters (shallow to prevent overfitting)
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 100
    xgb_min_child_weight: int = 5
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # Label strategy
    use_quality_labels: bool = True  # Use hit_tp_before_sl for label=1

    # Validation thresholds
    min_auc_threshold: float = 0.55  # Must beat random
    max_auc_std: float = 0.10  # Stability across folds


@dataclass
class MetaTrainingResult:
    """Result of meta-labeling training."""
    # Status
    success: bool
    failure_reason: Optional[str] = None

    # Model
    model: Any = None  # Trained XGBoost model
    feature_names: List[str] = field(default_factory=list)
    selected_features: List[str] = field(default_factory=list)

    # Cross-validation metrics
    cv_aucs: List[float] = field(default_factory=list)
    cv_accuracies: List[float] = field(default_factory=list)
    mean_auc: float = 0.0
    std_auc: float = 0.0
    mean_accuracy: float = 0.0

    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Data stats
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    positive_rate: float = 0.0

    # Timing
    training_time_seconds: float = 0.0
    timestamp: str = ""

    def passes_validation(self, config: MetaTrainerConfig) -> bool:
        """Check if model passes validation criteria."""
        if not self.success:
            return False
        if self.mean_auc < config.min_auc_threshold:
            return False
        if self.std_auc > config.max_auc_std:
            return False
        if not all(auc > 0.50 for auc in self.cv_aucs):
            return False
        return True


class MetaTrainer:
    """
    Walk-forward training of ML meta-labeler.

    Trains a model to predict WIN probability for each trade signal.
    Uses time-series cross-validation to prevent data leakage.
    Performs aggressive feature pruning to prevent overfitting.
    """

    def __init__(self, config: Optional[MetaTrainerConfig] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or MetaTrainerConfig()

        if not HAS_XGBOOST:
            raise ImportError("XGBoost required: pip install xgboost")

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        timestamps: Optional[pd.Series] = None,
    ) -> MetaTrainingResult:
        """
        Train meta-labeling model with walk-forward CV.

        Args:
            features: Feature DataFrame (n_samples x n_features)
            labels: Binary labels (0/1)
            timestamps: Optional timestamps for time-series ordering

        Returns:
            MetaTrainingResult with trained model and metrics
        """
        start_time = datetime.now()
        cfg = self.config

        # Validate input
        if len(features) < cfg.min_train_samples:
            return MetaTrainingResult(
                success=False,
                failure_reason=f"Insufficient samples: {len(features)} < {cfg.min_train_samples}",
                timestamp=start_time.isoformat(),
            )

        # Clean features
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        feature_names = list(features.columns)

        # Convert to numpy
        X = features.values
        y = labels.values.astype(int)

        n_samples = len(X)
        n_positive = int(y.sum())
        n_negative = n_samples - n_positive
        positive_rate = n_positive / n_samples if n_samples > 0 else 0

        print(f"\n{'='*50}")
        print(f"META-LABELING TRAINING")
        print(f"{'='*50}")
        print(f"Samples: {n_samples} ({n_positive} positive, {n_negative} negative)")
        print(f"Positive rate: {positive_rate:.1%}")
        print(f"Features: {len(feature_names)}")

        # Check class balance
        if positive_rate < 0.1 or positive_rate > 0.9:
            print(f"WARNING: Highly imbalanced classes ({positive_rate:.1%} positive)")

        # Step 1: Feature pruning using first pass
        print(f"\n--- Step 1: Feature Pruning ---")
        selected_features, feature_importance = self._prune_features(
            X, y, feature_names
        )
        print(f"Selected {len(selected_features)} / {len(feature_names)} features")

        # Get indices of selected features
        selected_idx = [feature_names.index(f) for f in selected_features]
        X_selected = X[:, selected_idx]

        # Step 2: Walk-forward cross-validation
        print(f"\n--- Step 2: Walk-Forward CV ({cfg.n_folds} folds) ---")
        cv_aucs, cv_accuracies = self._walk_forward_cv(X_selected, y)

        mean_auc = np.mean(cv_aucs)
        std_auc = np.std(cv_aucs)
        mean_accuracy = np.mean(cv_accuracies)

        print(f"\nCV Results:")
        print(f"  AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  Accuracy: {mean_accuracy:.4f}")
        print(f"  Per-fold AUC: {[f'{a:.3f}' for a in cv_aucs]}")

        # Check validation criteria
        if mean_auc < cfg.min_auc_threshold:
            return MetaTrainingResult(
                success=False,
                failure_reason=f"AUC {mean_auc:.4f} < {cfg.min_auc_threshold} (not better than random)",
                cv_aucs=cv_aucs,
                cv_accuracies=cv_accuracies,
                mean_auc=mean_auc,
                std_auc=std_auc,
                n_samples=n_samples,
                n_positive=n_positive,
                n_negative=n_negative,
                positive_rate=positive_rate,
                timestamp=start_time.isoformat(),
            )

        # Step 3: Train final model on all data
        print(f"\n--- Step 3: Training Final Model ---")
        final_model = self._train_xgboost(X_selected, y)

        training_time = (datetime.now() - start_time).total_seconds()

        result = MetaTrainingResult(
            success=True,
            model=final_model,
            feature_names=feature_names,
            selected_features=selected_features,
            cv_aucs=cv_aucs,
            cv_accuracies=cv_accuracies,
            mean_auc=mean_auc,
            std_auc=std_auc,
            mean_accuracy=mean_accuracy,
            feature_importance=feature_importance,
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_rate=positive_rate,
            training_time_seconds=training_time,
            timestamp=start_time.isoformat(),
        )

        print(f"\n{'='*50}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Final AUC: {mean_auc:.4f}")
        print(f"Training time: {training_time:.1f}s")

        return result

    def _prune_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Prune features using XGBoost importance.

        Returns:
            (selected_feature_names, importance_dict)
        """
        cfg = self.config

        # Train a quick XGBoost model to get feature importance
        model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )
        model.fit(X, y)

        # Get importance
        importance = model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))

        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top features
        selected = []
        for name, imp in sorted_features:
            if imp >= cfg.min_feature_importance and len(selected) < cfg.max_features:
                selected.append(name)

        # Ensure at least 5 features
        if len(selected) < 5:
            selected = [name for name, _ in sorted_features[:5]]

        # Print top 10
        print(f"Top 10 features:")
        for name, imp in sorted_features[:10]:
            marker = "*" if name in selected else " "
            print(f"  {marker} {name}: {imp:.4f}")

        return selected, importance_dict

    def _walk_forward_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[List[float], List[float]]:
        """
        Run walk-forward cross-validation.

        Returns:
            (list of AUC scores, list of accuracy scores)
        """
        cfg = self.config

        # Use sklearn's TimeSeriesSplit for walk-forward
        tscv = TimeSeriesSplit(n_splits=cfg.n_folds)

        aucs = []
        accuracies = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Skip if insufficient samples
            if len(X_train) < cfg.min_train_samples or len(X_test) < cfg.min_test_samples:
                print(f"  Fold {fold+1}: Skipped (insufficient samples)")
                continue

            # Train model
            model = self._train_xgboost(X_train, y_train)

            # Predict
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                # Only one class in test set
                auc = 0.5

            accuracy = accuracy_score(y_test, y_pred)

            aucs.append(auc)
            accuracies.append(accuracy)

            print(f"  Fold {fold+1}: AUC={auc:.4f}, Acc={accuracy:.4f}, "
                  f"Train={len(train_idx)}, Test={len(test_idx)}")

        return aucs, accuracies

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier with configured parameters.
        """
        cfg = self.config

        # Calculate scale_pos_weight for class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = xgb.XGBClassifier(
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            n_estimators=cfg.xgb_n_estimators,
            min_child_weight=cfg.xgb_min_child_weight,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            random_state=42,
        )

        model.fit(X_train, y_train)
        return model

    def predict(
        self,
        model: Any,
        features: pd.DataFrame,
        selected_features: List[str],
    ) -> np.ndarray:
        """
        Predict win probability for new samples.

        Args:
            model: Trained XGBoost model
            features: Feature DataFrame
            selected_features: List of feature names used in training

        Returns:
            Array of win probabilities (0 to 1)
        """
        # Select only the features used in training
        X = features[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        return model.predict_proba(X)[:, 1]

    def save(
        self,
        result: MetaTrainingResult,
        output_dir: Path,
        model_name: str = "meta_labeler",
    ) -> Path:
        """
        Save trained model and metadata.

        Args:
            result: Training result
            output_dir: Output directory
            model_name: Base name for files

        Returns:
            Path to saved model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / f"{model_name}.json"
        if result.model is not None:
            result.model.save_model(str(model_path))

        # Convert numpy types to native Python for JSON serialization
        def to_native(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj

        # Save metadata
        meta = {
            'success': result.success,
            'failure_reason': result.failure_reason,
            'selected_features': result.selected_features,
            'feature_importance': to_native(result.feature_importance),
            'cv_aucs': to_native(result.cv_aucs),
            'cv_accuracies': to_native(result.cv_accuracies),
            'mean_auc': to_native(result.mean_auc),
            'std_auc': to_native(result.std_auc),
            'mean_accuracy': to_native(result.mean_accuracy),
            'n_samples': to_native(result.n_samples),
            'n_positive': to_native(result.n_positive),
            'n_negative': to_native(result.n_negative),
            'positive_rate': to_native(result.positive_rate),
            'training_time_seconds': to_native(result.training_time_seconds),
            'timestamp': result.timestamp,
        }

        meta_path = output_dir / f"{model_name}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {meta_path}")

        return model_path

    def load(self, model_path: Path) -> Tuple[Any, Dict[str, Any]]:
        """
        Load trained model and metadata.

        Args:
            model_path: Path to model file

        Returns:
            (model, metadata_dict)
        """
        model_path = Path(model_path)

        # Load model
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))

        # Load metadata
        meta_path = model_path.with_suffix('.meta.json')
        with open(meta_path) as f:
            meta = json.load(f)

        return model, meta


def create_labels_from_trades(
    trades: List[Dict[str, Any]],
    use_quality_labels: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Create binary labels from trade results.

    Args:
        trades: List of trade dictionaries with 'result', 'exit_reason' keys
        use_quality_labels: If True, label=1 only for quality wins (TP exit)

    Returns:
        (labels Series, weights Series)
    """
    labels = []
    weights = []

    for trade in trades:
        result = trade.get('result', 'LOSS')
        exit_reason = trade.get('exit_reason', '')
        r_multiple = abs(trade.get('pnl_r', trade.get('r_multiple', 1.0)))

        if use_quality_labels:
            # Label = 1 only for quality wins (TP hit before SL)
            # Check exit_reason for take_profit
            hit_tp_before_sl = exit_reason == 'take_profit'
            label = 1 if hit_tp_before_sl else 0
        else:
            # Simple binary: WIN=1, LOSS/BREAKEVEN=0
            label = 1 if result == 'WIN' else 0

        labels.append(label)
        weights.append(max(r_multiple, 0.1))  # Weight by magnitude of R-multiple

    return pd.Series(labels), pd.Series(weights)
