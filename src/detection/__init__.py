"""
Detection Module
================
Pattern detection algorithms for the QML Trading System.

Available Detectors:
- RollingWindowDetector (v1.1.0): Fixed-size sliding window approach
- ATRDetector (v2.0.0): ATR-driven detection at confirmed swing points

Usage:
    # Using the factory (recommended)
    from src.detection import get_detector
    detector = get_detector("atr")  # or "rolling_window"
    signals = detector.detect(df, symbol='BTCUSDT', timeframe='4h')
    
    # Direct import
    from src.detection.v2_atr import ATRDetector
    detector = ATRDetector()
    signals = detector.detect(df)
"""

from src.detection.base import BaseDetector, DetectorConfig
from src.detection.factory import get_detector, get_default_detector, list_available_detectors
from src.detection.v1_rolling import RollingWindowDetector, RollingWindowConfig
from src.detection.v2_atr import ATRDetector, ATRDetectorConfig

__all__ = [
    # Base classes
    'BaseDetector',
    'DetectorConfig',
    
    # Factory functions
    'get_detector',
    'get_default_detector',
    'list_available_detectors',
    
    # V1 Rolling Window
    'RollingWindowDetector',
    'RollingWindowConfig',
    
    # V2 ATR (primary)
    'ATRDetector',
    'ATRDetectorConfig',
]
