"""
ML Module for QML Trading System
================================
Machine learning infrastructure for signal prediction.

Components:
- PurgedKFold, WalkForwardCV: Cross-validation without data leakage
- MLTrainingPipeline: Complete training with proper validation
- XGBoostPredictor: Inference for trained models
"""

from src.ml.predictor import XGBoostPredictor
from src.ml.purged_cv import PurgedKFold, WalkForwardCV, PurgedFold
from src.ml.training_pipeline import (
    MLTrainingPipeline,
    TrainingResult,
    TrainingMetrics,
    MODEL_VERSION,
)

__all__ = [
    # Existing
    "XGBoostPredictor",

    # Phase 7: Cross-validation
    "PurgedKFold",
    "WalkForwardCV",
    "PurgedFold",

    # Phase 7: Training pipeline
    "MLTrainingPipeline",
    "TrainingResult",
    "TrainingMetrics",
    "MODEL_VERSION",
]
