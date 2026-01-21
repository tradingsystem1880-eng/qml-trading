"""
Feature Engineering module for QML Trading System - Phase 3
============================================================

Uses libraries for 90% of work:
- pandas-ta: All technical indicators (ATR, RSI, ADX, EMA, etc.)
- scipy: Statistical functions (percentileofscore, linregress)
- sklearn: Normalization (RobustScaler) and feature selection (SelectKBest)
- joblib: Model/scaler persistence

Only Tier 1 Pattern Geometry is custom (QML-specific calculations).
"""

from src.features.calculator import FeatureCalculator
from src.features.pipeline import FeaturePipeline
from src.features.normalizer import (
    FeatureNormalizer,
    FeatureSelector,
    FeaturePreprocessor,
)

__all__ = [
    'FeatureCalculator',
    'FeaturePipeline',
    'FeatureNormalizer',
    'FeatureSelector',
    'FeaturePreprocessor',
]
