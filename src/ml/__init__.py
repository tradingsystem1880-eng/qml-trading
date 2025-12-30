"""
Machine Learning Module for QML Trading System
================================================
XGBoost-based pattern quality scoring with walk-forward validation.
"""

from src.ml.model import QMLModel, ModelConfig
from src.ml.trainer import ModelTrainer, TrainerConfig
from src.ml.labeler import TripleBarrierLabeler, LabelConfig

__all__ = [
    "QMLModel",
    "ModelConfig",
    "ModelTrainer",
    "TrainerConfig",
    "TripleBarrierLabeler",
    "LabelConfig",
]

