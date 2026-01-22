"""
QML (Quasimodo) Pattern Detector.

Detects the 5-point QML pattern:
- P1: First swing (left shoulder)
- P2: First retracement
- P3: Head (extends beyond P1)
- P4: Second retracement (Break of Structure)
- P5: Right shoulder (entry zone)

Integrates market regime filtering and multiple swing algorithms.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .swing_algorithms import (
    MultiAlgorithmSwingDetector,
    SwingPoint,
    SwingConfig,
    SwingAlgorithm
)
from .regime import MarketRegimeDetector, MarketRegime, RegimeResult


class PatternDirection(Enum):
    """Direction of the QML pattern."""
    BULLISH = "BULLISH"  # Head is HIGH - bearish reversal (short setup)
    BEARISH = "BEARISH"  # Head is LOW - bullish reversal (long setup)


@dataclass
class QMLPattern:
    """A detected QML pattern with all 5 swing points."""
    id: str
    direction: PatternDirection

    # The 5 swing points
    p1: SwingPoint  # Left shoulder
    p2: SwingPoint  # First retracement
    p3: SwingPoint  # Head
    p4: SwingPoint  # BOS (Break of Structure)
    p5: SwingPoint  # Right shoulder (entry zone)

    # Calculated trading levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None

    # Pattern quality metrics
    head_extension_atr: float = 0.0
    shoulder_symmetry: float = 0.0
    bos_count: int = 1
    pattern_strength: float = 0.0

    # Market context (DeepSeek addition)
    market_regime: str = "RANGING"
    regime_confidence: float = 0.5

    # Metadata
    detection_time: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class QMLConfig:
    """Configuration for QML pattern detection."""
    # Swing detection settings
    swing_algorithm: SwingAlgorithm = SwingAlgorithm.ROLLING
    swing_lookback: int = 5
    smoothing_window: int = 5

    # Pattern validation thresholds
    min_head_extension_atr: float = 0.5
    max_shoulder_tolerance_atr: float = 1.0
    bos_requirement: int = 1

    # Entry/Exit placement
    entry_buffer_atr: float = 0.1
    sl_buffer_atr: float = 0.5
    tp1_r_multiple: float = 1.5
    tp2_r_multiple: float = 2.5

    # Filters (DeepSeek recommendations)
    require_trend_alignment: bool = True
    require_volume_confirmation: bool = False
    min_bos_volume_ratio: float = 1.2

    # Duration limits
    min_pattern_bars: int = 10
    max_pattern_bars: int = 100


class QMLPatternDetector:
    """
    Detects QML patterns from OHLCV data.

    Uses configurable swing detection algorithms and
    market regime filtering for adaptive detection.
    """

    def __init__(self, config: Optional[QMLConfig] = None):
        self.config = config or QMLConfig()

        # Initialize swing detector
        swing_config = SwingConfig(
            algorithm=self.config.swing_algorithm,
            lookback=self.config.swing_lookback,
            smoothing_window=self.config.smoothing_window
        )
        self.swing_detector = MultiAlgorithmSwingDetector(swing_config)

        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector()

    def detect(self, df: pd.DataFrame) -> List[QMLPattern]:
        """
        Detect QML patterns in the data.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of detected QML patterns
        """
        if len(df) < self.config.min_pattern_bars:
            return []

        df = self._add_atr(df)

        # Check market regime (DeepSeek: filter strong trends)
        regime_result = self.regime_detector.get_regime(df)

        if self.config.require_trend_alignment:
            if regime_result.regime == MarketRegime.TRENDING:
                return []  # Skip detection in strong trends

        # Detect swing points
        swings = self.swing_detector.detect(df)

        if len(swings) < 5:
            return []

        # Find patterns in both directions
        patterns = []
        patterns.extend(self._find_patterns(df, swings, PatternDirection.BULLISH, regime_result))
        patterns.extend(self._find_patterns(df, swings, PatternDirection.BEARISH, regime_result))

        return patterns

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add ATR column using Wilder's smoothing (DeepSeek fix)."""
        df = df.copy()

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0/period, adjust=False).mean()

        if df['atr'].isna().all():
            df['atr'] = tr.rolling(period).mean()

        return df

    def _find_patterns(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        direction: PatternDirection,
        regime_result: RegimeResult
    ) -> List[QMLPattern]:
        """Find QML patterns for a given direction."""
        patterns = []

        highs = [s for s in swings if s.swing_type == 'HIGH']
        lows = [s for s in swings if s.swing_type == 'LOW']

        if direction == PatternDirection.BULLISH:
            # Bullish QML: Head is a HIGH (bearish setup - shorting opportunity)
            for p3_candidate in highs:
                pattern = self._validate_bullish_pattern(
                    df, swings, highs, lows, p3_candidate, regime_result
                )
                if pattern:
                    patterns.append(pattern)
        else:
            # Bearish QML: Head is a LOW (bullish setup - buying opportunity)
            for p3_candidate in lows:
                pattern = self._validate_bearish_pattern(
                    df, swings, highs, lows, p3_candidate, regime_result
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _validate_bullish_pattern(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        highs: List[SwingPoint],
        lows: List[SwingPoint],
        p3: SwingPoint,
        regime_result: RegimeResult
    ) -> Optional[QMLPattern]:
        """
        Validate bullish QML pattern (head is HIGH).

        Structure: P1(H) -> P2(L) -> P3(H/head) -> P4(L/BOS) -> P5(H)
        This is a bearish reversal setup (shorting opportunity).
        """
        if p3.index >= len(df):
            return None

        atr = df['atr'].iloc[p3.index]
        if atr <= 0 or np.isnan(atr):
            return None

        # P1: Previous high before P3 (left shoulder)
        p1_candidates = [h for h in highs if h.index < p3.index]
        if not p1_candidates:
            return None
        p1 = p1_candidates[-1]

        # P3 must be higher than P1 (head extends beyond left shoulder)
        head_extension = (p3.price - p1.price) / atr
        if head_extension < self.config.min_head_extension_atr:
            return None

        # P2: Low between P1 and P3 (first retracement)
        p2_candidates = [l for l in lows if p1.index < l.index < p3.index]
        if not p2_candidates:
            return None
        p2 = min(p2_candidates, key=lambda x: x.price)

        # P4: Low after P3 that breaks P2 level (BOS)
        p4_candidates = [l for l in lows if l.index > p3.index]
        bos_count = 0
        p4 = None

        for candidate in p4_candidates:
            if candidate.price < p2.price:  # Breaks P2 level
                bos_count += 1
                if p4 is None or candidate.price < p4.price:
                    p4 = candidate
                if bos_count >= self.config.bos_requirement:
                    break

        if bos_count < self.config.bos_requirement or p4 is None:
            return None

        # P5: High after P4 (right shoulder / entry zone)
        p5_candidates = [h for h in highs if h.index > p4.index]
        if not p5_candidates:
            return None
        p5 = p5_candidates[0]

        # Shoulder symmetry check
        shoulder_diff = abs(p5.price - p1.price) / atr
        if shoulder_diff > self.config.max_shoulder_tolerance_atr:
            return None

        # Pattern duration check
        pattern_bars = p5.index - p1.index
        if pattern_bars < self.config.min_pattern_bars:
            return None
        if pattern_bars > self.config.max_pattern_bars:
            return None

        # Calculate trading levels (SHORT setup)
        entry_price = p5.price + (self.config.entry_buffer_atr * atr)
        stop_loss = p3.price + (self.config.sl_buffer_atr * atr)
        risk = stop_loss - entry_price

        tp1 = entry_price - (risk * self.config.tp1_r_multiple)
        tp2 = entry_price - (risk * self.config.tp2_r_multiple)

        # Calculate pattern strength
        strength = self._calculate_strength(
            head_extension, shoulder_diff, bos_count,
            p1, p2, p3, p4, p5
        )

        return QMLPattern(
            id=f"QML_B_{p3.index}",
            direction=PatternDirection.BULLISH,
            p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            head_extension_atr=head_extension,
            shoulder_symmetry=shoulder_diff,
            bos_count=bos_count,
            pattern_strength=strength,
            market_regime=regime_result.regime.value,
            regime_confidence=regime_result.confidence,
            detection_time=pd.Timestamp.now()
        )

    def _validate_bearish_pattern(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        highs: List[SwingPoint],
        lows: List[SwingPoint],
        p3: SwingPoint,
        regime_result: RegimeResult
    ) -> Optional[QMLPattern]:
        """
        Validate bearish QML pattern (head is LOW).

        Structure: P1(L) -> P2(H) -> P3(L/head) -> P4(H/BOS) -> P5(L)
        This is a bullish reversal setup (buying opportunity).
        """
        if p3.index >= len(df):
            return None

        atr = df['atr'].iloc[p3.index]
        if atr <= 0 or np.isnan(atr):
            return None

        # P1: Previous low before P3 (left shoulder)
        p1_candidates = [l for l in lows if l.index < p3.index]
        if not p1_candidates:
            return None
        p1 = p1_candidates[-1]

        # P3 must be lower than P1 (head extends beyond left shoulder)
        head_extension = (p1.price - p3.price) / atr
        if head_extension < self.config.min_head_extension_atr:
            return None

        # P2: High between P1 and P3 (first retracement)
        p2_candidates = [h for h in highs if p1.index < h.index < p3.index]
        if not p2_candidates:
            return None
        p2 = max(p2_candidates, key=lambda x: x.price)

        # P4: High after P3 that breaks P2 level (BOS)
        p4_candidates = [h for h in highs if h.index > p3.index]
        bos_count = 0
        p4 = None

        for candidate in p4_candidates:
            if candidate.price > p2.price:  # Breaks P2 level
                bos_count += 1
                if p4 is None or candidate.price > p4.price:
                    p4 = candidate
                if bos_count >= self.config.bos_requirement:
                    break

        if bos_count < self.config.bos_requirement or p4 is None:
            return None

        # P5: Low after P4 (right shoulder / entry zone)
        p5_candidates = [l for l in lows if l.index > p4.index]
        if not p5_candidates:
            return None
        p5 = p5_candidates[0]

        # Shoulder symmetry check
        shoulder_diff = abs(p5.price - p1.price) / atr
        if shoulder_diff > self.config.max_shoulder_tolerance_atr:
            return None

        # Pattern duration check
        pattern_bars = p5.index - p1.index
        if pattern_bars < self.config.min_pattern_bars:
            return None
        if pattern_bars > self.config.max_pattern_bars:
            return None

        # Calculate trading levels (LONG setup)
        entry_price = p5.price - (self.config.entry_buffer_atr * atr)
        stop_loss = p3.price - (self.config.sl_buffer_atr * atr)
        risk = entry_price - stop_loss

        tp1 = entry_price + (risk * self.config.tp1_r_multiple)
        tp2 = entry_price + (risk * self.config.tp2_r_multiple)

        # Calculate pattern strength
        strength = self._calculate_strength(
            head_extension, shoulder_diff, bos_count,
            p1, p2, p3, p4, p5
        )

        return QMLPattern(
            id=f"QML_S_{p3.index}",
            direction=PatternDirection.BEARISH,
            p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            head_extension_atr=head_extension,
            shoulder_symmetry=shoulder_diff,
            bos_count=bos_count,
            pattern_strength=strength,
            market_regime=regime_result.regime.value,
            regime_confidence=regime_result.confidence,
            detection_time=pd.Timestamp.now()
        )

    def _calculate_strength(
        self,
        head_extension: float,
        shoulder_symmetry: float,
        bos_count: int,
        p1: SwingPoint,
        p2: SwingPoint,
        p3: SwingPoint,
        p4: SwingPoint,
        p5: SwingPoint
    ) -> float:
        """
        Calculate pattern strength using market-validated weights.

        DeepSeek research-based weights:
        - Head extension: 40% (most important)
        - BOS quality: 30%
        - Shoulder symmetry: 15%
        - Swing confidence: 15%
        """
        # Head extension score (optimal: 1-2 ATR)
        if 1.0 <= head_extension <= 2.0:
            head_score = 1.0
        else:
            head_score = min(1.0, head_extension / 1.5)

        # BOS quality score
        bos_cleanliness = 1.0 if bos_count >= 2 else 0.7
        bos_score = min(1.0, bos_cleanliness * (bos_count / 2.0))

        # Shoulder symmetry score (lower is better)
        symmetry_score = max(0.0, 1.0 - shoulder_symmetry / 1.5)

        # Swing confidence score (average of all swing strengths)
        swing_score = np.mean([
            p1.strength, p2.strength, p3.strength,
            p4.strength, p5.strength
        ])

        # Weighted combination (DeepSeek weights)
        strength = (
            head_score * 0.40 +
            bos_score * 0.30 +
            symmetry_score * 0.15 +
            swing_score * 0.15
        )

        return round(strength, 3)
