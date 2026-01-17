"""
Pipeline Stages Package
=======================
Modular pipeline stages for validation orchestration.

Each stage is a self-contained module with single responsibility:
- Data synchronization
- Pattern detection
- Backtesting
- Validation (walk-forward, statistical tests)
- Reporting

Usage:
    from src.pipeline.stages import DataSyncStage, PatternDetectionStage
    
    stage = DataSyncStage()
    result = stage.run(context)
"""

from .base import BaseStage, StageContext, StageResult
from .feature_engineering import FeatureEngineeringStage
from .regime_detection import RegimeDetectionStage
from .walk_forward import WalkForwardStage
from .statistical_testing import StatisticalTestingStage
from .diagnostics import DiagnosticsStage

__all__ = [
    'BaseStage',
    'StageContext',
    'StageResult',
    'FeatureEngineeringStage',
    'RegimeDetectionStage',
    'WalkForwardStage',
    'StatisticalTestingStage',
    'DiagnosticsStage',
]
