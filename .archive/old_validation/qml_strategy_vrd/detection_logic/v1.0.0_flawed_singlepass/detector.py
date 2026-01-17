"""
QML Pattern Detector - Main Detection Engine
=============================================
Combines swing detection, market structure analysis, CHoCH, and BoS
detection into a complete QML pattern detection pipeline.

This is the core module that orchestrates the entire detection process
and produces validated QML patterns with trading levels.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.database import DatabaseManager, get_database
from src.data.fetcher import DataFetcher, create_data_fetcher
from src.data.models import (
    BoSEvent,
    CHoCHEvent,
    MarketStructure,
    PatternStatus,
    PatternType,
    QMLPattern,
    SwingPoint,
    SwingType,
    TradingLevels,
    TrendType,
)
from src.detection.bos import BoSDetector
from src.detection.choch import CHoCHDetector
from src.detection.structure import StructureAnalyzer, TrendState
from src.detection.swing import SwingDetector
from src.utils.indicators import calculate_atr


@dataclass
class DetectorConfig:
    """Configuration for QML pattern detection."""
    
    # Minimum pattern validity score
    min_validity_score: float = 0.7
    
    # Head depth constraints (in ATR)
    min_head_depth_atr: float = 0.5
    max_head_depth_atr: float = 3.0
    
    # Risk/Reward settings
    default_risk_reward: float = 3.0
    stop_loss_atr_multiplier: float = 1.5
    
    # Pattern geometry constraints
    max_pattern_bars: int = 100
    min_pattern_bars: int = 5
    
    # Right shoulder tolerance (% from left shoulder level)
    right_shoulder_tolerance: float = 0.02  # 2%


class QMLDetector:
    """
    Complete QML Pattern Detection Engine.
    
    Orchestrates the detection pipeline:
    1. Fetch/load price data
    2. Detect swing points
    3. Analyze market structure
    4. Detect CHoCH events
    5. Detect BoS events
    6. Validate and score QML patterns
    7. Calculate trading levels
    
    The detector produces fully validated QML patterns with:
    - Pattern components (left shoulder, head, right shoulder)
    - Validity scores
    - Entry, stop loss, and take profit levels
    """
    
    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        db: Optional[DatabaseManager] = None,
        fetcher: Optional[DataFetcher] = None
    ):
        """
        Initialize QML detector.
        
        Args:
            config: Detection configuration
            db: Database manager
            fetcher: Data fetcher
        """
        self.config = config or DetectorConfig()
        self.db = db or get_database()
        self.fetcher = fetcher
        
        # Override config from settings
        self.config.min_validity_score = settings.detection.min_pattern_validity_score
        self.config.min_head_depth_atr = settings.detection.min_head_depth_atr
        self.config.max_head_depth_atr = settings.detection.max_head_depth_atr
        self.config.stop_loss_atr_multiplier = settings.trading.stop_loss_atr_multiplier
        self.config.default_risk_reward = settings.trading.default_risk_reward
        
        # Initialize sub-detectors
        self.swing_detector: Dict[str, SwingDetector] = {}
        self.structure_analyzer = StructureAnalyzer()
        self.choch_detector = CHoCHDetector()
        self.bos_detector = BoSDetector()
    
    def _get_swing_detector(self, timeframe: str) -> SwingDetector:
        """Get or create swing detector for timeframe."""
        if timeframe not in self.swing_detector:
            self.swing_detector[timeframe] = SwingDetector(timeframe=timeframe)
        return self.swing_detector[timeframe]
    
    def detect(
        self,
        symbol: str,
        timeframe: str,
        df: Optional[pd.DataFrame] = None,
        lookback_bars: int = 500
    ) -> List[QMLPattern]:
        """
        Detect QML patterns for a symbol/timeframe.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '4h')
            df: Optional pre-loaded DataFrame
            lookback_bars: Number of bars to analyze
            
        Returns:
            List of detected QML patterns
        """
        # Get data
        if df is None:
            if self.fetcher is None:
                self.fetcher = create_data_fetcher()
            
            df = self.fetcher.get_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback_bars,
                auto_sync=True
            )
        
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for {symbol} {timeframe}")
            return []
        
        logger.info(f"Running QML detection on {symbol} {timeframe} ({len(df)} bars)")
        
        # Step 1: Detect swing points
        swing_detector = self._get_swing_detector(timeframe)
        swing_points = swing_detector.detect(df, symbol)
        
        if len(swing_points) < 6:
            logger.info(f"Insufficient swing points for {symbol} {timeframe}")
            return []
        
        # Step 2: Analyze market structure
        structures, trend_state = self.structure_analyzer.analyze(
            swing_points, symbol, timeframe
        )
        
        if not structures:
            logger.info(f"No market structure detected for {symbol} {timeframe}")
            return []
        
        # Step 3: Detect CHoCH events
        choch_events = self.choch_detector.detect(
            df, swing_points, structures, trend_state, symbol, timeframe
        )
        
        if not choch_events:
            logger.debug(f"No CHoCH events for {symbol} {timeframe}")
            return []
        
        # Step 4: Detect BoS events
        bos_events = self.bos_detector.detect(df, choch_events, symbol, timeframe)
        
        if not bos_events:
            logger.debug(f"No BoS events for {symbol} {timeframe}")
            return []
        
        # Step 5: Build and validate QML patterns
        patterns = self._build_patterns(
            df, choch_events, bos_events, swing_points, symbol, timeframe
        )
        
        # Filter by validity score
        valid_patterns = [
            p for p in patterns
            if p.validity_score >= self.config.min_validity_score
        ]
        
        logger.info(
            f"Detected {len(valid_patterns)} valid QML patterns for {symbol} {timeframe}"
        )
        
        return valid_patterns
    
    def _build_patterns(
        self,
        df: pd.DataFrame,
        choch_events: List[CHoCHEvent],
        bos_events: List[BoSEvent],
        swing_points: List[SwingPoint],
        symbol: str,
        timeframe: str
    ) -> List[QMLPattern]:
        """
        Build QML patterns from detected components.
        
        Args:
            df: OHLCV DataFrame
            choch_events: CHoCH events
            bos_events: BoS events
            swing_points: All swing points
            symbol: Trading pair
            timeframe: Candle timeframe
            
        Returns:
            List of QML patterns
        """
        patterns = []
        
        # Calculate ATR for the dataset
        atr = calculate_atr(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            settings.detection.atr_period
        )
        
        # Match CHoCH with BoS events
        for bos in bos_events:
            choch = bos.choch_event
            if not choch:
                continue
            
            # Find head point
            head_price, head_time, head_idx = self.bos_detector.find_head_point(
                df, choch, bos
            )
            
            if head_price is None:
                continue
            
            # Get ATR at head point
            atr_at_head = atr[head_idx] if head_idx and not np.isnan(atr[head_idx]) else np.nanmean(atr[-50:])
            
            # Validate head depth
            if bos.bos_type == PatternType.BULLISH:
                head_depth = choch.break_level - head_price
            else:
                head_depth = head_price - choch.break_level
            
            head_depth_atr = head_depth / atr_at_head if atr_at_head > 0 else 0
            
            if head_depth_atr < self.config.min_head_depth_atr:
                logger.debug(f"Head depth too shallow: {head_depth_atr:.2f} ATR")
                continue
            
            if head_depth_atr > self.config.max_head_depth_atr:
                logger.debug(f"Head depth too deep: {head_depth_atr:.2f} ATR")
                continue
            
            # Calculate validity score
            validity_score = self._calculate_validity_score(
                choch, bos, head_price, head_depth_atr, df, atr
            )
            
            # Calculate trading levels
            trading_levels = self._calculate_trading_levels(
                choch, bos, head_price, atr_at_head
            )
            
            # Create pattern
            pattern = QMLPattern(
                detection_time=bos.time,
                symbol=symbol,
                timeframe=timeframe,
                pattern_type=bos.bos_type,
                
                # Pattern components
                left_shoulder_price=choch.break_level,
                left_shoulder_time=choch.time,
                head_price=head_price,
                head_time=head_time,
                neckline_start=choch.break_level,
                neckline_end=bos.break_level,
                
                # Trading levels
                trading_levels=trading_levels,
                
                # Quality metrics
                validity_score=validity_score,
                geometric_score=self._calculate_geometric_score(choch, bos, head_price, head_depth_atr),
                volume_score=1.0 if bos.volume_spike else 0.5,
                context_score=self._calculate_context_score(df, bos.bar_index, atr),
                
                # Status
                status=PatternStatus.ACTIVE,
                
                # Related events
                choch_event=choch,
                bos_event=bos,
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_validity_score(
        self,
        choch: CHoCHEvent,
        bos: BoSEvent,
        head_price: float,
        head_depth_atr: float,
        df: pd.DataFrame,
        atr: np.ndarray
    ) -> float:
        """
        Calculate overall pattern validity score (0-1).
        
        Components:
        - CHoCH strength (20%)
        - BoS confirmation (20%)
        - Head depth quality (30%)
        - Volume confirmation (15%)
        - Pattern geometry (15%)
        """
        scores = []
        weights = []
        
        # CHoCH strength (0-1 based on break strength)
        choch_score = min(1.0, choch.break_strength / 2.0)  # Normalize to 0-1
        scores.append(choch_score)
        weights.append(0.20)
        
        # BoS confirmation
        bos_score = 0.8 if bos.volume_spike else 0.5
        if choch.confirmed:
            bos_score += 0.2
        scores.append(min(1.0, bos_score))
        weights.append(0.20)
        
        # Head depth quality (optimal is 1-2 ATR)
        if 1.0 <= head_depth_atr <= 2.0:
            head_score = 1.0
        elif 0.5 <= head_depth_atr < 1.0:
            head_score = 0.7
        elif 2.0 < head_depth_atr <= 3.0:
            head_score = 0.7
        else:
            head_score = 0.4
        scores.append(head_score)
        weights.append(0.30)
        
        # Volume confirmation
        volume_score = 1.0 if bos.volume_spike else 0.6
        if choch.volume_confirmation:
            volume_score = min(1.0, volume_score + 0.2)
        scores.append(volume_score)
        weights.append(0.15)
        
        # Pattern geometry (based on structure clarity)
        geometry_score = self._calculate_geometric_score(choch, bos, head_price, head_depth_atr)
        scores.append(geometry_score)
        weights.append(0.15)
        
        # Weighted average
        validity = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        return round(validity, 3)
    
    def _calculate_geometric_score(
        self,
        choch: CHoCHEvent,
        bos: BoSEvent,
        head_price: float,
        head_depth_atr: float
    ) -> float:
        """
        Calculate geometric quality of the pattern.
        
        Evaluates:
        - Head position relative to shoulders
        - Pattern symmetry
        - Neckline characteristics
        """
        score = 0.5  # Base score
        
        # Check head is properly positioned
        if bos.bos_type == PatternType.BULLISH:
            # For bullish, head should be below both shoulders
            if head_price < choch.break_level and head_price < bos.break_level:
                score += 0.3
        else:
            # For bearish, head should be above both shoulders
            if head_price > choch.break_level and head_price > bos.break_level:
                score += 0.3
        
        # Optimal head depth bonus
        if 1.0 <= head_depth_atr <= 2.0:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_context_score(
        self,
        df: pd.DataFrame,
        bos_idx: Optional[int],
        atr: np.ndarray
    ) -> float:
        """
        Calculate context quality score.
        
        Evaluates:
        - Position relative to recent range
        - Volatility regime
        """
        if bos_idx is None or bos_idx < 20:
            return 0.5
        
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # Recent range
        lookback = min(50, bos_idx)
        recent_high = np.max(high[bos_idx - lookback:bos_idx])
        recent_low = np.min(low[bos_idx - lookback:bos_idx])
        range_size = recent_high - recent_low
        
        if range_size <= 0:
            return 0.5
        
        # Position in range
        current_price = close[bos_idx]
        range_position = (current_price - recent_low) / range_size
        
        # Better score if pattern forms near range extremes
        if range_position < 0.2 or range_position > 0.8:
            score = 0.8
        elif range_position < 0.3 or range_position > 0.7:
            score = 0.7
        else:
            score = 0.5
        
        return score
    
    def _calculate_trading_levels(
        self,
        choch: CHoCHEvent,
        bos: BoSEvent,
        head_price: float,
        atr_at_head: float
    ) -> TradingLevels:
        """
        Calculate trading levels for the pattern.
        
        Entry: Near the retest zone (between CHoCH and BoS levels)
        Stop Loss: Beyond the head with ATR buffer
        Take Profits: Based on risk/reward ratios
        """
        if bos.bos_type == PatternType.BULLISH:
            # Bullish pattern - long trade
            # Entry near the demand zone (between head and CHoCH level)
            entry = choch.break_level * 0.995  # Slightly below CHoCH level
            
            # Stop loss below head with buffer
            stop_loss = head_price - (atr_at_head * self.config.stop_loss_atr_multiplier)
            
            # Risk amount
            risk = entry - stop_loss
            
            # Take profits
            take_profit_1 = entry + risk  # 1:1
            take_profit_2 = entry + (2 * risk)  # 2:1
            take_profit_3 = entry + (self.config.default_risk_reward * risk)  # 3:1
            
        else:
            # Bearish pattern - short trade
            # Entry near the supply zone (between head and CHoCH level)
            entry = choch.break_level * 1.005  # Slightly above CHoCH level
            
            # Stop loss above head with buffer
            stop_loss = head_price + (atr_at_head * self.config.stop_loss_atr_multiplier)
            
            # Risk amount
            risk = stop_loss - entry
            
            # Take profits
            take_profit_1 = entry - risk  # 1:1
            take_profit_2 = entry - (2 * risk)  # 2:1
            take_profit_3 = entry - (self.config.default_risk_reward * risk)  # 3:1
        
        return TradingLevels(
            entry=round(entry, 8),
            stop_loss=round(stop_loss, 8),
            take_profit_1=round(take_profit_1, 8),
            take_profit_2=round(take_profit_2, 8),
            take_profit_3=round(take_profit_3, 8),
            risk_amount=round(abs(risk), 8)
        )
    
    def detect_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        save_to_db: bool = True
    ) -> Dict[str, List[QMLPattern]]:
        """
        Run detection across multiple symbols and timeframes.
        
        Args:
            symbols: List of symbols (defaults to configured)
            timeframes: List of timeframes (defaults to configured)
            save_to_db: Whether to save patterns to database
            
        Returns:
            Dictionary mapping 'symbol_timeframe' to patterns
        """
        symbols = symbols or settings.detection.symbols
        timeframes = timeframes or settings.detection.timeframes
        
        all_patterns: Dict[str, List[QMLPattern]] = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                try:
                    patterns = self.detect(symbol, timeframe)
                    all_patterns[key] = patterns
                    
                    if save_to_db and patterns:
                        for pattern in patterns:
                            self._save_pattern_to_db(pattern)
                    
                except Exception as e:
                    logger.error(f"Detection failed for {key}: {e}")
                    all_patterns[key] = []
        
        # Summary
        total = sum(len(p) for p in all_patterns.values())
        logger.info(f"Total patterns detected: {total}")
        
        return all_patterns
    
    def _save_pattern_to_db(self, pattern: QMLPattern) -> None:
        """Save pattern to database."""
        try:
            pattern_dict = {
                "detection_time": pattern.detection_time,
                "symbol": pattern.symbol,
                "timeframe": pattern.timeframe,
                "pattern_type": pattern.pattern_type.value,
                "left_shoulder_price": pattern.left_shoulder_price,
                "left_shoulder_time": pattern.left_shoulder_time,
                "head_price": pattern.head_price,
                "head_time": pattern.head_time,
                "right_shoulder_price": pattern.right_shoulder_price,
                "right_shoulder_time": pattern.right_shoulder_time,
                "neckline_start": pattern.neckline_start,
                "neckline_end": pattern.neckline_end,
                "entry_price": pattern.trading_levels.entry if pattern.trading_levels else None,
                "stop_loss": pattern.trading_levels.stop_loss if pattern.trading_levels else None,
                "take_profit_1": pattern.trading_levels.take_profit_1 if pattern.trading_levels else None,
                "take_profit_2": pattern.trading_levels.take_profit_2 if pattern.trading_levels else None,
                "take_profit_3": pattern.trading_levels.take_profit_3 if pattern.trading_levels else None,
                "validity_score": pattern.validity_score,
                "geometric_score": pattern.geometric_score,
                "volume_score": pattern.volume_score,
                "context_score": pattern.context_score,
                "ml_confidence": pattern.ml_confidence,
                "ml_model_version": pattern.ml_model_version,
                "status": pattern.status.value,
            }
            
            self.db.insert_pattern(pattern_dict)
            
        except Exception as e:
            logger.error(f"Failed to save pattern to database: {e}")


def create_detector(
    config: Optional[DetectorConfig] = None,
    db: Optional[DatabaseManager] = None,
    fetcher: Optional[DataFetcher] = None
) -> QMLDetector:
    """
    Factory function to create a QML detector.
    
    Args:
        config: Detection configuration
        db: Database manager
        fetcher: Data fetcher
        
    Returns:
        QMLDetector instance
    """
    return QMLDetector(config=config, db=db, fetcher=fetcher)

