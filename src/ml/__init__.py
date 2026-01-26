"""
ML Module for QML Trading System
================================
Machine learning infrastructure for signal prediction.

Components:
- PurgedKFold, WalkForwardCV: Cross-validation without data leakage
- MLTrainingPipeline: Complete training with proper validation
- XGBoostPredictor: Inference for trained models
- MetaTrainer: Phase 8.0 meta-labeling trainer
- KellySizer: Kelly criterion position sizing
- ProductionGate: ML vs baseline comparison
"""

from src.ml.predictor import XGBoostPredictor
from src.ml.purged_cv import PurgedKFold, WalkForwardCV, PurgedFold
from src.ml.training_pipeline import (
    MLTrainingPipeline,
    TrainingResult,
    TrainingMetrics,
    MODEL_VERSION,
)

# Phase 8.0: Meta-labeling components
from src.ml.meta_trainer import MetaTrainer, MetaTrainerConfig, MetaTrainingResult
from src.ml.kelly_sizer import KellySizer, KellyConfig, PositionSizeResult
from src.ml.production_gate import ProductionGate, ProductionGateConfig, GateResult
from src.ml.labeler import TripleBarrierLabeler, LabelConfig, LabelResult, Label

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

    # Phase 8.0: Meta-labeling
    "MetaTrainer",
    "MetaTrainerConfig",
    "MetaTrainingResult",
    "KellySizer",
    "KellyConfig",
    "PositionSizeResult",
    "ProductionGate",
    "ProductionGateConfig",
    "GateResult",
    "TripleBarrierLabeler",
    "LabelConfig",
    "LabelResult",
    "Label",
]
