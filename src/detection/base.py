"""
Base Detector Interface
=======================
Abstract base class that all pattern detectors must implement.
This enforces a consistent interface across detection algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core.models import Candle, CandleList, Signal


# =============================================================================
# DETECTOR CONFIGURATION
# =============================================================================

@dataclass
class DetectorConfig:
    """
    Base configuration for all detectors.
    
    Subclasses can extend this with algorithm-specific parameters.
    All parameters should be loaded from YAML config files.
    """
    # Identification
    name: str = "base_detector"
    version: str = "1.0.0"
    
    # Pattern detection parameters
    min_pattern_bars: int = 20
    max_pattern_bars: int = 200
    min_validity_score: float = 0.7
    
    # ATR parameters (common to most detectors)
    atr_period: int = 14
    
    # Risk management
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0
    
    # Filtering
    min_volume_ratio: float = 0.5  # Minimum volume vs average
    
    # Additional parameters (strategy-specific)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DetectorConfig':
        """Create config from dictionary."""
        known_fields = {
            'name', 'version', 'min_pattern_bars', 'max_pattern_bars',
            'min_validity_score', 'atr_period', 'stop_loss_atr_mult',
            'take_profit_atr_mult', 'min_volume_ratio'
        }
        
        base_config = {k: v for k, v in config_dict.items() if k in known_fields}
        extra = {k: v for k, v in config_dict.items() if k not in known_fields}
        
        return cls(**base_config, extra=extra)


# =============================================================================
# BASE DETECTOR (ABSTRACT)
# =============================================================================

class BaseDetector(ABC):
    """
    Abstract Base Class for all pattern detectors.
    
    All detection algorithms must inherit from this class and implement
    the `detect()` method. This ensures a consistent interface across
    different detection strategies (rolling window, ATR-driven, etc.).
    
    Usage:
        class MyDetector(BaseDetector):
            def detect(self, candles):
                # Implementation here
                return [Signal(...), Signal(...)]
        
        detector = MyDetector(config)
        signals = detector.detect(candle_data)
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detector configuration. If None, uses defaults.
        """
        self.config = config or DetectorConfig()
        self._is_initialized = False
    
    @property
    def name(self) -> str:
        """Get detector name."""
        return self.config.name
    
    @property
    def version(self) -> str:
        """Get detector version."""
        return self.config.version
    
    def initialize(self) -> None:
        """
        Initialize detector state.
        
        Override this in subclasses if initialization is needed
        (e.g., loading models, pre-computing values).
        """
        self._is_initialized = True
    
    @abstractmethod
    def detect(
        self, 
        candles: Union[List[Candle], CandleList, pd.DataFrame],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Signal]:
        """
        Detect patterns and generate trading signals.
        
        This is the main method that all detectors must implement.
        It receives candle data and returns a list of trading signals.
        
        Args:
            candles: Price data in one of the supported formats:
                     - List[Candle]: List of Candle dataclass instances
                     - CandleList: Wrapped list with helper methods
                     - pd.DataFrame: DataFrame with columns [time, open, high, low, close, volume]
            symbol: Trading pair (e.g., 'BTCUSDT'). Optional if included in candles.
            timeframe: Candle timeframe (e.g., '4h'). Optional if included in candles.
        
        Returns:
            List[Signal]: List of trading signals detected in the data.
                         Empty list if no patterns found.
        
        Raises:
            ValueError: If candle data is invalid or insufficient.
        """
        pass
    
    def _normalize_candles(
        self, 
        candles: Union[List[Candle], CandleList, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Convert any candle input format to DataFrame.
        
        Args:
            candles: Candle data in any supported format.
        
        Returns:
            pd.DataFrame with standardized columns.
        """
        if isinstance(candles, pd.DataFrame):
            df = candles.copy()
            # Standardize column names
            column_map = {
                'timestamp': 'time',
                'date': 'time',
                'datetime': 'time',
            }
            df.rename(columns=column_map, inplace=True)
            return df
        
        elif isinstance(candles, CandleList):
            return pd.DataFrame({
                'time': [c.timestamp for c in candles],
                'open': [c.open for c in candles],
                'high': [c.high for c in candles],
                'low': [c.low for c in candles],
                'close': [c.close for c in candles],
                'volume': [c.volume for c in candles],
            })
        
        elif isinstance(candles, list) and len(candles) > 0:
            if isinstance(candles[0], Candle):
                return pd.DataFrame({
                    'time': [c.timestamp for c in candles],
                    'open': [c.open for c in candles],
                    'high': [c.high for c in candles],
                    'low': [c.low for c in candles],
                    'close': [c.close for c in candles],
                    'volume': [c.volume for c in candles],
                })
            else:
                raise ValueError(f"Unsupported candle type: {type(candles[0])}")
        
        else:
            raise ValueError(f"Unsupported candles format: {type(candles)}")
    
    def _calculate_atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray,
        period: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: from config)
        
        Returns:
            ATR array (same length as input, NaN for initial period)
        """
        period = period or self.config.atr_period
        n = len(high)
        
        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # ATR (simple moving average of TR)
        atr = np.full(n, np.nan)
        for i in range(period - 1, n):
            atr[i] = np.mean(tr[i - period + 1:i + 1])
        
        return atr
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns and sufficient data.
        
        Args:
            df: DataFrame to validate.
        
        Returns:
            True if valid.
        
        Raises:
            ValueError: If data is invalid.
        """
        required = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) < self.config.min_pattern_bars:
            raise ValueError(
                f"Insufficient data: {len(df)} bars, "
                f"minimum {self.config.min_pattern_bars} required"
            )
        
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
