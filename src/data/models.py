"""
Data Models for QML Trading System
===================================
Pydantic models for type-safe data handling throughout the system.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Timeframe(str, Enum):
    """Supported timeframes for analysis."""
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    
    @property
    def minutes(self) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return mapping[self.value]


class SwingType(str, Enum):
    """Type of swing point."""
    HIGH = "high"
    LOW = "low"


class StructureType(str, Enum):
    """Market structure classification."""
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low


class TrendType(str, Enum):
    """Market trend classification."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    CONSOLIDATION = "consolidation"


class PatternType(str, Enum):
    """QML Pattern direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"


class PatternStatus(str, Enum):
    """QML Pattern lifecycle status."""
    FORMING = "forming"
    ACTIVE = "active"
    TRIGGERED = "triggered"
    INVALIDATED = "invalidated"
    COMPLETED = "completed"


class TradeOutcome(str, Enum):
    """Trade outcome classification."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"
    CANCELLED = "cancelled"


class OHLCV(BaseModel):
    """OHLCV candlestick data model."""
    model_config = ConfigDict(from_attributes=True)
    
    time: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None
    trades: Optional[int] = None
    
    @field_validator("high")
    @classmethod
    def high_ge_low(cls, v: float, info) -> float:
        """Validate high >= low."""
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v
    
    @field_validator("high", "low")
    @classmethod
    def price_in_range(cls, v: float, info) -> float:
        """Validate prices are within open-close range."""
        return v
    
    @property
    def body(self) -> float:
        """Candle body size (absolute)."""
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        """Upper wick size."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Lower wick size."""
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open
    
    @property
    def range(self) -> float:
        """Total candle range (high - low)."""
        return self.high - self.low


class SwingPoint(BaseModel):
    """Swing high/low point model."""
    model_config = ConfigDict(from_attributes=True)

    time: datetime
    symbol: str
    timeframe: str
    swing_type: SwingType
    price: float
    significance: float = Field(description="ATR-normalized significance")
    atr_at_point: float
    confirmed: bool = False
    bar_index: Optional[int] = None

    # Phase 7.5 additions for improved detection
    atr_at_formation: Optional[float] = Field(
        default=None,
        description="ATR at the time the swing was DETECTED (not current ATR)"
    )
    significance_zscore: float = Field(
        default=0.0,
        description="Statistical significance as z-score for cross-timeframe comparison"
    )
    
    @property
    def is_high(self) -> bool:
        """Check if this is a swing high."""
        return self.swing_type == SwingType.HIGH
    
    @property
    def is_low(self) -> bool:
        """Check if this is a swing low."""
        return self.swing_type == SwingType.LOW


class MarketStructure(BaseModel):
    """Market structure point (HH/HL/LH/LL)."""
    model_config = ConfigDict(from_attributes=True)
    
    time: datetime
    symbol: str
    timeframe: str
    structure_type: StructureType
    price: float
    previous_price: Optional[float] = None
    trend: TrendType = TrendType.CONSOLIDATION
    trend_strength: float = 0.0
    swing_point: Optional[SwingPoint] = None
    
    @property
    def is_bullish_structure(self) -> bool:
        """Check if structure is bullish (HH or HL)."""
        return self.structure_type in [StructureType.HH, StructureType.HL]
    
    @property
    def is_bearish_structure(self) -> bool:
        """Check if structure is bearish (LH or LL)."""
        return self.structure_type in [StructureType.LH, StructureType.LL]


class CHoCHEvent(BaseModel):
    """Change of Character (CHoCH) event."""
    model_config = ConfigDict(from_attributes=True)
    
    time: datetime
    symbol: str
    timeframe: str
    choch_type: PatternType
    break_level: float
    break_strength: float = Field(description="ATR-normalized break strength")
    volume_confirmation: bool = False
    confirmed: bool = False
    bar_index: Optional[int] = None


class BoSEvent(BaseModel):
    """Break of Structure (BoS) event."""
    model_config = ConfigDict(from_attributes=True)
    
    time: datetime
    symbol: str
    timeframe: str
    bos_type: PatternType
    break_level: float
    volume_spike: bool = False
    choch_event: Optional[CHoCHEvent] = None
    bar_index: Optional[int] = None


class TradingLevels(BaseModel):
    """Trading levels for a QML pattern."""
    
    entry: float
    stop_loss: float
    take_profit_1: float  # 1:1 R:R
    take_profit_2: float  # 2:1 R:R
    take_profit_3: float  # 3:1 R:R
    risk_amount: float = Field(description="Distance from entry to stop loss")
    
    @property
    def risk_reward_1(self) -> float:
        """Risk/Reward ratio for TP1."""
        return abs(self.take_profit_1 - self.entry) / self.risk_amount if self.risk_amount > 0 else 0
    
    @property
    def risk_reward_2(self) -> float:
        """Risk/Reward ratio for TP2."""
        return abs(self.take_profit_2 - self.entry) / self.risk_amount if self.risk_amount > 0 else 0
    
    @property
    def risk_reward_3(self) -> float:
        """Risk/Reward ratio for TP3."""
        return abs(self.take_profit_3 - self.entry) / self.risk_amount if self.risk_amount > 0 else 0


class QMLPattern(BaseModel):
    """
    Complete QML (Quasimodo) Pattern model.
    
    Represents a fully detected QML pattern with all components,
    trading levels, and quality metrics.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    detection_time: datetime
    symbol: str
    timeframe: str
    pattern_type: PatternType
    
    # Pattern Components
    left_shoulder_price: float
    left_shoulder_time: datetime
    head_price: float
    head_time: datetime
    right_shoulder_price: Optional[float] = None
    right_shoulder_time: Optional[datetime] = None
    neckline_start: Optional[float] = None
    neckline_end: Optional[float] = None
    
    # Trading Levels
    trading_levels: Optional[TradingLevels] = None
    
    # Quality Metrics
    validity_score: float = Field(ge=0, le=1)
    geometric_score: Optional[float] = None
    volume_score: Optional[float] = None
    context_score: Optional[float] = None
    
    # ML Prediction
    ml_confidence: Optional[float] = None
    ml_model_version: Optional[str] = None
    
    # Status
    status: PatternStatus = PatternStatus.FORMING
    invalidation_reason: Optional[str] = None
    
    # Outcome (filled after trade completes)
    outcome: Optional[TradeOutcome] = None
    actual_return_pct: Optional[float] = None
    bars_to_outcome: Optional[int] = None
    
    # Related Events
    choch_event: Optional[CHoCHEvent] = None
    bos_event: Optional[BoSEvent] = None
    
    @property
    def head_depth(self) -> float:
        """Calculate head depth from left shoulder."""
        if self.pattern_type == PatternType.BULLISH:
            return self.left_shoulder_price - self.head_price
        else:
            return self.head_price - self.left_shoulder_price
    
    @property
    def is_valid(self) -> bool:
        """Check if pattern meets minimum validity threshold."""
        return self.validity_score >= 0.7
    
    @property
    def combined_score(self) -> float:
        """Calculate combined quality score including ML confidence."""
        base_score = self.validity_score
        if self.ml_confidence is not None:
            # Weighted average: 40% pattern quality, 60% ML confidence
            return 0.4 * base_score + 0.6 * self.ml_confidence
        return base_score
    
    def invalidate(self, reason: str) -> None:
        """Mark pattern as invalidated."""
        self.status = PatternStatus.INVALIDATED
        self.invalidation_reason = reason


class PatternFeatures(BaseModel):
    """
    Engineered features for a QML pattern.
    Used as input for ML model prediction.
    """
    model_config = ConfigDict(from_attributes=True)
    
    pattern_id: int
    pattern_time: datetime
    
    # Geometric Features
    head_depth_ratio: Optional[float] = None
    shoulder_symmetry: Optional[float] = None
    neckline_slope: Optional[float] = None
    pattern_duration_bars: Optional[int] = None
    
    # Volume Features
    volume_at_head: Optional[float] = None
    volume_ratio_head_shoulders: Optional[float] = None
    obv_divergence: Optional[float] = None
    
    # Context Features
    atr_percentile: Optional[float] = None
    distance_from_daily_high: Optional[float] = None
    distance_from_daily_low: Optional[float] = None
    btc_correlation: Optional[float] = None
    
    # Regime Features
    market_regime: Optional[str] = None
    trend_strength: Optional[float] = None
    volatility_regime: Optional[str] = None
    
    # Crypto-specific Features
    funding_rate: Optional[float] = None
    open_interest_change: Optional[float] = None
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML model."""
        feature_values = [
            self.head_depth_ratio,
            self.shoulder_symmetry,
            self.neckline_slope,
            self.pattern_duration_bars,
            self.volume_at_head,
            self.volume_ratio_head_shoulders,
            self.obv_divergence,
            self.atr_percentile,
            self.distance_from_daily_high,
            self.distance_from_daily_low,
            self.btc_correlation,
            self.trend_strength,
        ]
        # Replace None with NaN for ML handling
        return np.array([v if v is not None else np.nan for v in feature_values])


class OHLCVDataFrame:
    """
    Wrapper for OHLCV DataFrame with helper methods.
    Provides a clean interface for accessing price data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        
        Expected columns: time, open, high, low, close, volume
        """
        self._df = df.copy()
        self._validate()
    
    def _validate(self) -> None:
        """Validate DataFrame has required columns."""
        required = ["time", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in self._df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(self._df["time"]):
            self._df["time"] = pd.to_datetime(self._df["time"])
        
        # Sort by time
        self._df = self._df.sort_values("time").reset_index(drop=True)
    
    @property
    def df(self) -> pd.DataFrame:
        """Get underlying DataFrame."""
        return self._df
    
    @property
    def open(self) -> np.ndarray:
        """Get open prices."""
        return self._df["open"].values
    
    @property
    def high(self) -> np.ndarray:
        """Get high prices."""
        return self._df["high"].values
    
    @property
    def low(self) -> np.ndarray:
        """Get low prices."""
        return self._df["low"].values
    
    @property
    def close(self) -> np.ndarray:
        """Get close prices."""
        return self._df["close"].values
    
    @property
    def volume(self) -> np.ndarray:
        """Get volume."""
        return self._df["volume"].values
    
    @property
    def time(self) -> np.ndarray:
        """Get timestamps."""
        return self._df["time"].values
    
    def __len__(self) -> int:
        """Get number of candles."""
        return len(self._df)
    
    def __getitem__(self, idx):
        """Get candle(s) by index."""
        return self._df.iloc[idx]
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """Get last n candles."""
        return self._df.tail(n)
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n candles."""
        return self._df.head(n)

