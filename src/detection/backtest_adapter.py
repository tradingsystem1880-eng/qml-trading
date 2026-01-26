"""
Backtest Adapter for Phase 7.5 Detection
=========================================
Converts Phase 7.5 detection outputs to backtest-compatible formats.

This module bridges:
- ValidationResult + ScoringResult → QMLPattern (for BacktestEngine)
- ValidationResult + ScoringResult → Signal (for detector interface)
"""

from datetime import datetime
from typing import List, Optional, Tuple
import pandas as pd

from src.detection.pattern_validator import ValidationResult, PatternDirection
from src.detection.pattern_scorer import ScoringResult, PatternTier
from src.detection.config import PatternValidationConfig
from src.data.models import (
    QMLPattern,
    TradingLevels,
    PatternType,
    PatternStatus,
)
from src.core.models import Signal, SignalType


class BacktestAdapter:
    """
    Adapter to convert Phase 7.5 detection outputs to backtest formats.

    Handles conversion of:
    - ValidationResult → QMLPattern (Pydantic model for backtest engine)
    - ValidationResult → Signal (dataclass for detector interface)
    """

    def __init__(
        self,
        entry_buffer_atr: float = 0.1,
        sl_buffer_atr: float = 0.5,
        tp1_r_multiple: float = 1.5,
        tp2_r_multiple: float = 2.5,
        tp3_r_multiple: float = 3.5,
    ):
        """
        Initialize adapter with trading level parameters.

        Args:
            entry_buffer_atr: Buffer from P5 for entry (in ATR units)
            sl_buffer_atr: Buffer beyond P3 for stop loss (in ATR units)
            tp1_r_multiple: Risk multiple for TP1 (e.g., 1.5 = 1.5R)
            tp2_r_multiple: Risk multiple for TP2
            tp3_r_multiple: Risk multiple for TP3
        """
        self.entry_buffer_atr = entry_buffer_atr
        self.sl_buffer_atr = sl_buffer_atr
        self.tp1_r_multiple = tp1_r_multiple
        self.tp2_r_multiple = tp2_r_multiple
        self.tp3_r_multiple = tp3_r_multiple

    def validation_to_qml_pattern(
        self,
        validation_result: ValidationResult,
        scoring_result: ScoringResult,
        symbol: str = "BTCUSDT",
        timeframe: str = "4h",
        pattern_id: Optional[int] = None,
    ) -> Optional[QMLPattern]:
        """
        Convert ValidationResult to QMLPattern for backtest engine.

        Args:
            validation_result: Valid pattern from PatternValidator
            scoring_result: Scores from PatternScorer
            symbol: Trading pair
            timeframe: Candle timeframe
            pattern_id: Optional unique ID

        Returns:
            QMLPattern ready for BacktestEngine, or None if invalid
        """
        if not validation_result.is_valid:
            return None

        vr = validation_result

        # Calculate trading levels
        trading_levels = self._calculate_trading_levels(vr)
        if trading_levels is None:
            return None

        # Map direction to PatternType
        # IMPORTANT: QML naming is opposite to trade direction!
        # - BULLISH pattern (head is HIGH) = bearish reversal = SHORT setup
        # - BEARISH pattern (head is LOW) = bullish reversal = LONG setup
        # The BacktestEngine uses PatternType.BULLISH for LONG trades,
        # so we need to INVERT the mapping.
        if vr.direction == PatternDirection.BULLISH:
            # BULLISH QML pattern = SHORT trade = use BEARISH PatternType
            pattern_type = PatternType.BEARISH
        else:
            # BEARISH QML pattern = LONG trade = use BULLISH PatternType
            pattern_type = PatternType.BULLISH

        # Build QMLPattern
        return QMLPattern(
            id=pattern_id,
            detection_time=vr.p5.timestamp.to_pydatetime(),
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,

            # Pattern components
            left_shoulder_price=vr.p1.price,
            left_shoulder_time=vr.p1.timestamp.to_pydatetime(),
            head_price=vr.p3.price,
            head_time=vr.p3.timestamp.to_pydatetime(),
            right_shoulder_price=vr.p5.price,
            right_shoulder_time=vr.p5.timestamp.to_pydatetime(),
            neckline_start=vr.p2.price,
            neckline_end=vr.p4.price,

            # Trading levels
            trading_levels=trading_levels,

            # Quality metrics (from scorer)
            validity_score=scoring_result.total_score,
            geometric_score=scoring_result.head_extension_score,
            volume_score=scoring_result.bos_efficiency_score,
            context_score=scoring_result.shoulder_symmetry_score,

            # Status
            status=PatternStatus.ACTIVE,
        )

    def validation_to_signal(
        self,
        validation_result: ValidationResult,
        scoring_result: ScoringResult,
        symbol: str = "BTCUSDT",
        timeframe: str = "4h",
    ) -> Optional[Signal]:
        """
        Convert ValidationResult to Signal for detector interface.

        Args:
            validation_result: Valid pattern from PatternValidator
            scoring_result: Scores from PatternScorer
            symbol: Trading pair
            timeframe: Candle timeframe

        Returns:
            Signal ready for trade execution, or None if invalid
        """
        if not validation_result.is_valid:
            return None

        vr = validation_result
        atr = vr.atr_p5

        # Calculate entry and stop loss
        if vr.direction == PatternDirection.BULLISH:
            # BULLISH pattern = bearish reversal = SHORT setup
            entry = vr.p5.price + (self.entry_buffer_atr * atr)
            sl = vr.p3.price + (self.sl_buffer_atr * atr)
            signal_type = SignalType.SELL
        else:
            # BEARISH pattern = bullish reversal = LONG setup
            entry = vr.p5.price - (self.entry_buffer_atr * atr)
            sl = vr.p3.price - (self.sl_buffer_atr * atr)
            signal_type = SignalType.BUY

        risk = abs(entry - sl)
        if risk <= 0:
            return None

        # Calculate take profits
        if signal_type == SignalType.BUY:
            tp1 = entry + (risk * self.tp1_r_multiple)
            tp2 = entry + (risk * self.tp2_r_multiple)
            tp3 = entry + (risk * self.tp3_r_multiple)
        else:
            tp1 = entry - (risk * self.tp1_r_multiple)
            tp2 = entry - (risk * self.tp2_r_multiple)
            tp3 = entry - (risk * self.tp3_r_multiple)

        # Pattern type string
        pattern_type_str = f"QML_{vr.direction.value}"

        return Signal(
            timestamp=vr.p5.timestamp.to_pydatetime(),
            signal_type=signal_type,
            price=entry,
            strategy_name="QML_Phase75",
            confidence=scoring_result.total_score,
            validity_score=scoring_result.total_score,
            stop_loss=sl,
            take_profit=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            pattern_type=pattern_type_str,
            pattern_id=f"P75_{vr.p3.bar_index}",
            symbol=symbol,
            timeframe=timeframe,
            atr_at_signal=atr,
            metadata={
                "tier": scoring_result.tier.value,
                "head_extension_atr": vr.head_extension_atr,
                "bos_efficiency": vr.bos_efficiency,
                "shoulder_diff_atr": vr.shoulder_diff_atr,
                "pattern_bars": vr.pattern_bars,
                "p1_bar": vr.p1.bar_index,
                "p2_bar": vr.p2.bar_index,
                "p3_bar": vr.p3.bar_index,
                "p4_bar": vr.p4.bar_index,
                "p5_bar": vr.p5.bar_index,
                "component_scores": {
                    "head_extension": scoring_result.head_extension_score,
                    "bos_efficiency": scoring_result.bos_efficiency_score,
                    "shoulder_symmetry": scoring_result.shoulder_symmetry_score,
                    "swing_significance": scoring_result.swing_significance_score,
                },
            },
        )

    def _calculate_trading_levels(
        self,
        vr: ValidationResult
    ) -> Optional[TradingLevels]:
        """
        Calculate trading levels from ValidationResult.

        Args:
            vr: Valid ValidationResult

        Returns:
            TradingLevels for backtest, or None if invalid
        """
        atr = vr.atr_p5
        if atr <= 0:
            return None

        if vr.direction == PatternDirection.BULLISH:
            # SHORT setup
            entry = vr.p5.price + (self.entry_buffer_atr * atr)
            sl = vr.p3.price + (self.sl_buffer_atr * atr)
        else:
            # LONG setup
            entry = vr.p5.price - (self.entry_buffer_atr * atr)
            sl = vr.p3.price - (self.sl_buffer_atr * atr)

        risk = abs(entry - sl)
        if risk <= 0:
            return None

        # Calculate TPs based on direction
        if vr.direction == PatternDirection.BEARISH:
            # LONG: TPs are above entry
            tp1 = entry + (risk * self.tp1_r_multiple)
            tp2 = entry + (risk * self.tp2_r_multiple)
            tp3 = entry + (risk * self.tp3_r_multiple)
        else:
            # SHORT: TPs are below entry
            tp1 = entry - (risk * self.tp1_r_multiple)
            tp2 = entry - (risk * self.tp2_r_multiple)
            tp3 = entry - (risk * self.tp3_r_multiple)

        return TradingLevels(
            entry=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_amount=risk,
        )

    def batch_convert_to_patterns(
        self,
        validation_results: List[ValidationResult],
        scoring_results: List[ScoringResult],
        symbol: str = "BTCUSDT",
        timeframe: str = "4h",
        min_tier: PatternTier = PatternTier.C,
    ) -> List[QMLPattern]:
        """
        Convert multiple patterns to QMLPattern list.

        Args:
            validation_results: List of ValidationResults
            scoring_results: List of corresponding ScoringResults
            symbol: Trading pair
            timeframe: Candle timeframe
            min_tier: Minimum tier to include (filters lower quality)

        Returns:
            List of QMLPatterns for backtest
        """
        patterns = []

        for i, (vr, sr) in enumerate(zip(validation_results, scoring_results)):
            # Filter by tier
            if min_tier == PatternTier.A and sr.tier != PatternTier.A:
                continue
            if min_tier == PatternTier.B and sr.tier not in [PatternTier.A, PatternTier.B]:
                continue
            if min_tier == PatternTier.C and sr.tier == PatternTier.REJECT:
                continue

            pattern = self.validation_to_qml_pattern(
                validation_result=vr,
                scoring_result=sr,
                symbol=symbol,
                timeframe=timeframe,
                pattern_id=i,
            )

            if pattern is not None:
                patterns.append(pattern)

        return patterns

    def batch_convert_to_signals(
        self,
        validation_results: List[ValidationResult],
        scoring_results: List[ScoringResult],
        symbol: str = "BTCUSDT",
        timeframe: str = "4h",
        min_tier: PatternTier = PatternTier.C,
    ) -> List[Signal]:
        """
        Convert multiple patterns to Signal list.

        Args:
            validation_results: List of ValidationResults
            scoring_results: List of corresponding ScoringResults
            symbol: Trading pair
            timeframe: Candle timeframe
            min_tier: Minimum tier to include

        Returns:
            List of Signals for trade execution
        """
        signals = []

        for vr, sr in zip(validation_results, scoring_results):
            # Filter by tier
            if min_tier == PatternTier.A and sr.tier != PatternTier.A:
                continue
            if min_tier == PatternTier.B and sr.tier not in [PatternTier.A, PatternTier.B]:
                continue
            if min_tier == PatternTier.C and sr.tier == PatternTier.REJECT:
                continue

            signal = self.validation_to_signal(
                validation_result=vr,
                scoring_result=sr,
                symbol=symbol,
                timeframe=timeframe,
            )

            if signal is not None:
                signals.append(signal)

        return signals


def create_adapter(config: Optional[PatternValidationConfig] = None) -> BacktestAdapter:
    """
    Factory function to create BacktestAdapter with config.

    Args:
        config: Optional PatternValidationConfig to use parameters from

    Returns:
        Configured BacktestAdapter
    """
    if config is None:
        return BacktestAdapter()

    return BacktestAdapter(
        entry_buffer_atr=config.entry_buffer_atr,
        sl_buffer_atr=config.sl_buffer_atr,
        tp1_r_multiple=config.tp1_r_multiple,
        tp2_r_multiple=config.tp2_r_multiple,
        tp3_r_multiple=config.tp3_r_multiple,
    )
