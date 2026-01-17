"""
Pattern Detector
================
Unified interface for QML pattern detection.

Wraps existing detection infrastructure.
"""

import pandas as pd
from typing import List, Dict, Optional
from loguru import logger


class PatternDetector:
    """
    Unified QML pattern detector.
    
    Wraps existing detection infrastructure for clean access.
    
    Example:
        detector = PatternDetector()
        patterns = detector.detect(df, "BTC/USDT", "4h")
    """
    
    def __init__(self, config=None):
        """Initialize detector."""
        self.config = config
        self._detector = None
    
    @property
    def detector(self):
        """Lazy-load the underlying detector."""
        if self._detector is None:
            try:
                from src.detection.factory import create_detector
                self._detector = create_detector()
            except Exception:
                try:
                    from src.detection.detector import QMLDetector, DetectorConfig
                    config = DetectorConfig()
                    self._detector = QMLDetector(config)
                except Exception as e:
                    logger.error(f"Failed to create detector: {e}")
        return self._detector
    
    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "4h"
    ) -> List[Dict]:
        """
        Detect QML patterns in data.
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading pair
            timeframe: Timeframe
            
        Returns:
            List of detected patterns
        """
        if self.detector is None:
            logger.error("Detector not available")
            return []
        
        try:
            # Try different detector interfaces
            # Note: QMLDetector.detect signature is (symbol, timeframe, df=None)
            if hasattr(self.detector, 'detect'):
                patterns = self.detector.detect(symbol, timeframe, df=df)
            elif hasattr(self.detector, 'scan'):
                patterns = self.detector.scan(df)
            else:
                logger.warning("Unknown detector interface")
                patterns = []
            
            # Convert QMLPattern objects to dicts if needed
            if patterns and hasattr(patterns[0], '__dict__'):
                patterns = [self._pattern_to_dict(p) for p in patterns]
            
            logger.info(f"Detected {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []
    
    def _pattern_to_dict(self, pattern) -> Dict:
        """Convert a QMLPattern object to a dictionary."""
        try:
            # Get pattern type as string
            pattern_type = "bullish"
            if hasattr(pattern, 'pattern_type'):
                if hasattr(pattern.pattern_type, 'value'):
                    pattern_type = pattern.pattern_type.value.lower()
                else:
                    pattern_type = str(pattern.pattern_type).lower()
            
            # Get trading levels
            entry = 0
            stop_loss = 0
            take_profit = 0
            if hasattr(pattern, 'trading_levels') and pattern.trading_levels:
                tl = pattern.trading_levels
                entry = getattr(tl, 'entry', 0)
                stop_loss = getattr(tl, 'stop_loss', 0)
                take_profit = getattr(tl, 'take_profit_1', 0)
            
            # Calculate risk/reward
            risk_reward = 0
            if entry and stop_loss and take_profit:
                risk = abs(entry - stop_loss)
                reward = abs(take_profit - entry)
                if risk > 0:
                    risk_reward = reward / risk
            
            return {
                "type": pattern_type,
                "validity": getattr(pattern, 'validity_score', 0.7),
                "entry_price": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward": risk_reward,
                "detection_time": getattr(pattern, 'detection_time', None),
                "head_price": getattr(pattern, 'head_price', 0),
                "left_shoulder_price": getattr(pattern, 'left_shoulder_price', 0),
                "left_shoulder_time": getattr(pattern, 'left_shoulder_time', None),
                "head_time": getattr(pattern, 'head_time', None),
            }
        except Exception as e:
            logger.error(f"Pattern conversion failed: {e}")
            return {
                "type": "unknown",
                "validity": 0,
                "entry_price": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "risk_reward": 0
            }
    
    def scan_symbols(
        self,
        symbols: List[str],
        timeframe: str = "4h",
        min_validity: float = 0.5
    ) -> Dict[str, List[Dict]]:
        """
        Scan multiple symbols for patterns.
        
        Args:
            symbols: List of trading pairs
            timeframe: Timeframe to scan
            min_validity: Minimum pattern validity
            
        Returns:
            Dict mapping symbol to patterns
        """
        from qml.core.data import DataLoader
        
        loader = DataLoader()
        results = {}
        
        for symbol in symbols:
            try:
                df = loader.load_ohlcv(symbol, timeframe)
                patterns = self.detect(df, symbol, timeframe)
                # Filter by validity
                patterns = [p for p in patterns if p.get("validity", 0) >= min_validity]
                results[symbol] = patterns
            except Exception as e:
                logger.warning(f"Scan failed for {symbol}: {e}")
                results[symbol] = []
        
        return results
