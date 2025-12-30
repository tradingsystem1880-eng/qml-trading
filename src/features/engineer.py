"""
Feature Engineering for QML Trading System
===========================================
Calculates pattern-specific and market context features
used for ML model training and pattern quality scoring.

Feature Categories:
1. Geometric Features - Pattern shape and proportions
2. Temporal Features - Time-based characteristics
3. Volume Features - Volume profile and confirmation
4. Context Features - Market position and levels
5. Regime Features - Market state classification
6. Crypto-Specific Features - Funding, OI, etc.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.models import (
    PatternFeatures,
    PatternType,
    QMLPattern,
)
from src.utils.indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_obv,
    calculate_adx,
    calculate_volatility_percentile,
    detect_divergence,
)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # ATR period for calculations
    atr_period: int = 14
    
    # RSI period
    rsi_period: int = 14
    
    # ADX period for trend strength
    adx_period: int = 14
    
    # Lookback for context features
    context_lookback: int = 50
    
    # Lookback for volatility percentile
    volatility_lookback: int = 100
    
    # Volume rolling period
    volume_period: int = 20


class FeatureEngineer:
    """
    Engineers features for QML patterns.
    
    Produces a comprehensive feature set for each detected pattern,
    enabling ML models to assess pattern quality and predict outcomes.
    
    Features are designed to be:
    - Point-in-time (no look-ahead bias)
    - Normalized where appropriate
    - Robust to different market conditions
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature calculation configuration
        """
        self.config = config or FeatureConfig()
    
    def calculate_features(
        self,
        pattern: QMLPattern,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None
    ) -> PatternFeatures:
        """
        Calculate all features for a pattern.
        
        Args:
            pattern: QML pattern to analyze
            df: OHLCV DataFrame for the pattern's symbol
            btc_df: Optional BTC DataFrame for correlation features
            
        Returns:
            PatternFeatures object with all calculated features
        """
        # Initialize features
        features = PatternFeatures(
            pattern_id=pattern.id or 0,
            pattern_time=pattern.detection_time
        )
        
        # Calculate base indicators
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values
        
        atr = calculate_atr(high, low, close, self.config.atr_period)
        rsi = calculate_rsi(close, self.config.rsi_period)
        obv = calculate_obv(close, volume)
        adx = calculate_adx(high, low, close, self.config.adx_period)
        
        # Get bar index for pattern head
        head_idx = self._find_bar_index(df, pattern.head_time)
        
        if head_idx is None or head_idx < self.config.context_lookback:
            logger.warning(f"Insufficient data for feature calculation")
            return features
        
        # Calculate each feature category
        features = self._calculate_geometric_features(features, pattern, atr, head_idx)
        features = self._calculate_temporal_features(features, pattern, df)
        features = self._calculate_volume_features(features, pattern, df, volume, obv, head_idx)
        features = self._calculate_context_features(features, df, atr, rsi, adx, head_idx)
        
        if btc_df is not None:
            features = self._calculate_correlation_features(features, df, btc_df, head_idx)
        
        return features
    
    def _find_bar_index(
        self,
        df: pd.DataFrame,
        target_time: datetime
    ) -> Optional[int]:
        """Find bar index for a given timestamp."""
        times = pd.to_datetime(df["time"])
        target = pd.Timestamp(target_time)
        
        # Find closest bar
        time_diffs = abs(times - target)
        min_idx = time_diffs.argmin()
        
        # Check if reasonably close
        if time_diffs.iloc[min_idx] > timedelta(days=1):
            return None
        
        return int(min_idx)
    
    def _calculate_geometric_features(
        self,
        features: PatternFeatures,
        pattern: QMLPattern,
        atr: np.ndarray,
        head_idx: int
    ) -> PatternFeatures:
        """
        Calculate geometric features describing pattern shape.
        
        Features:
        - head_depth_ratio: Head depth normalized by ATR
        - shoulder_symmetry: Ratio of shoulder heights
        - neckline_slope: Slope of neckline (normalized)
        """
        atr_at_head = atr[head_idx] if not np.isnan(atr[head_idx]) else np.nanmean(atr[-50:])
        
        # Head depth ratio (ATR normalized)
        if pattern.pattern_type == PatternType.BULLISH:
            head_depth = pattern.left_shoulder_price - pattern.head_price
        else:
            head_depth = pattern.head_price - pattern.left_shoulder_price
        
        features.head_depth_ratio = head_depth / atr_at_head if atr_at_head > 0 else 0
        
        # Shoulder symmetry (right shoulder vs left shoulder)
        if pattern.right_shoulder_price and pattern.left_shoulder_price > 0:
            if pattern.pattern_type == PatternType.BULLISH:
                # For bullish, shoulders are swing lows
                left_height = pattern.left_shoulder_price - pattern.head_price
                right_height = pattern.right_shoulder_price - pattern.head_price
            else:
                # For bearish, shoulders are swing highs
                left_height = pattern.head_price - pattern.left_shoulder_price
                right_height = pattern.head_price - pattern.right_shoulder_price
            
            if left_height > 0:
                features.shoulder_symmetry = right_height / left_height
            else:
                features.shoulder_symmetry = 0
        else:
            features.shoulder_symmetry = None
        
        # Neckline slope (normalized by ATR)
        if pattern.neckline_start and pattern.neckline_end:
            slope = (pattern.neckline_end - pattern.neckline_start)
            features.neckline_slope = slope / atr_at_head if atr_at_head > 0 else 0
        
        return features
    
    def _calculate_temporal_features(
        self,
        features: PatternFeatures,
        pattern: QMLPattern,
        df: pd.DataFrame
    ) -> PatternFeatures:
        """
        Calculate temporal features describing pattern timing.
        
        Features:
        - pattern_duration_bars: Number of bars in pattern
        """
        # Calculate pattern duration
        left_idx = self._find_bar_index(df, pattern.left_shoulder_time)
        detection_idx = self._find_bar_index(df, pattern.detection_time)
        
        if left_idx is not None and detection_idx is not None:
            features.pattern_duration_bars = detection_idx - left_idx
        else:
            features.pattern_duration_bars = None
        
        return features
    
    def _calculate_volume_features(
        self,
        features: PatternFeatures,
        pattern: QMLPattern,
        df: pd.DataFrame,
        volume: np.ndarray,
        obv: np.ndarray,
        head_idx: int
    ) -> PatternFeatures:
        """
        Calculate volume-based features.
        
        Features:
        - volume_at_head: Relative volume at head formation
        - volume_ratio_head_shoulders: Volume comparison
        - obv_divergence: OBV divergence signal
        """
        # Average volume for normalization
        avg_volume = np.mean(volume[max(0, head_idx - self.config.volume_period):head_idx])
        
        # Volume at head (relative)
        if avg_volume > 0:
            features.volume_at_head = volume[head_idx] / avg_volume
        else:
            features.volume_at_head = 1.0
        
        # Volume ratio between head and shoulders
        left_idx = self._find_bar_index(df, pattern.left_shoulder_time)
        if left_idx is not None and left_idx < head_idx:
            shoulder_volume = np.mean(volume[left_idx:left_idx + 3])
            head_volume = np.mean(volume[head_idx - 1:head_idx + 2])
            
            if shoulder_volume > 0:
                features.volume_ratio_head_shoulders = head_volume / shoulder_volume
            else:
                features.volume_ratio_head_shoulders = 1.0
        
        # OBV divergence
        close = df["close"].values
        lookback = min(20, head_idx)
        obv_divergence = detect_divergence(close, obv, lookback)
        features.obv_divergence = float(obv_divergence[head_idx])
        
        return features
    
    def _calculate_context_features(
        self,
        features: PatternFeatures,
        df: pd.DataFrame,
        atr: np.ndarray,
        rsi: np.ndarray,
        adx: np.ndarray,
        head_idx: int
    ) -> PatternFeatures:
        """
        Calculate market context features.
        
        Features:
        - atr_percentile: Current volatility percentile
        - distance_from_daily_high: Position relative to recent high
        - distance_from_daily_low: Position relative to recent low
        - trend_strength: ADX-based trend strength
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        lookback = self.config.context_lookback
        start_idx = max(0, head_idx - lookback)
        
        # ATR percentile
        vol_percentile = calculate_volatility_percentile(atr, close, lookback)
        features.atr_percentile = float(vol_percentile[head_idx]) if not np.isnan(vol_percentile[head_idx]) else 50.0
        
        # Distance from recent high/low (normalized)
        recent_high = np.max(high[start_idx:head_idx + 1])
        recent_low = np.min(low[start_idx:head_idx + 1])
        recent_range = recent_high - recent_low
        
        if recent_range > 0:
            current_price = close[head_idx]
            features.distance_from_daily_high = (recent_high - current_price) / recent_range
            features.distance_from_daily_low = (current_price - recent_low) / recent_range
        else:
            features.distance_from_daily_high = 0.5
            features.distance_from_daily_low = 0.5
        
        # Trend strength (ADX)
        features.trend_strength = float(adx[head_idx]) / 100 if not np.isnan(adx[head_idx]) else 0.0
        
        return features
    
    def _calculate_correlation_features(
        self,
        features: PatternFeatures,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        head_idx: int
    ) -> PatternFeatures:
        """
        Calculate BTC correlation features.
        
        Features:
        - btc_correlation: Rolling correlation with BTC
        """
        lookback = min(20, head_idx)
        
        # Get aligned price series
        symbol_returns = pd.Series(df["close"].values).pct_change()
        btc_returns = pd.Series(btc_df["close"].values).pct_change()
        
        # Calculate rolling correlation
        if len(btc_returns) >= lookback:
            start_idx = max(0, head_idx - lookback)
            end_idx = head_idx + 1
            
            symbol_window = symbol_returns.iloc[start_idx:end_idx]
            btc_window = btc_returns.iloc[start_idx:end_idx]
            
            if len(symbol_window) > 5 and len(btc_window) > 5:
                correlation = symbol_window.corr(btc_window)
                features.btc_correlation = float(correlation) if not np.isnan(correlation) else 0.0
        
        return features
    
    def calculate_batch_features(
        self,
        patterns: List[QMLPattern],
        price_data: Dict[str, pd.DataFrame],
        btc_df: Optional[pd.DataFrame] = None
    ) -> List[PatternFeatures]:
        """
        Calculate features for multiple patterns.
        
        Args:
            patterns: List of QML patterns
            price_data: Dictionary mapping symbol to OHLCV DataFrame
            btc_df: Optional BTC DataFrame for correlation
            
        Returns:
            List of PatternFeatures objects
        """
        all_features = []
        
        for pattern in patterns:
            df = price_data.get(pattern.symbol)
            if df is None or df.empty:
                logger.warning(f"No price data for {pattern.symbol}")
                continue
            
            features = self.calculate_features(pattern, df, btc_df)
            all_features.append(features)
        
        logger.info(f"Calculated features for {len(all_features)} patterns")
        
        return all_features
    
    def features_to_dataframe(
        self,
        features_list: List[PatternFeatures]
    ) -> pd.DataFrame:
        """
        Convert list of PatternFeatures to DataFrame for ML.
        
        Args:
            features_list: List of PatternFeatures objects
            
        Returns:
            DataFrame with feature columns
        """
        records = []
        
        for f in features_list:
            record = {
                "pattern_id": f.pattern_id,
                "pattern_time": f.pattern_time,
                "head_depth_ratio": f.head_depth_ratio,
                "shoulder_symmetry": f.shoulder_symmetry,
                "neckline_slope": f.neckline_slope,
                "pattern_duration_bars": f.pattern_duration_bars,
                "volume_at_head": f.volume_at_head,
                "volume_ratio_head_shoulders": f.volume_ratio_head_shoulders,
                "obv_divergence": f.obv_divergence,
                "atr_percentile": f.atr_percentile,
                "distance_from_daily_high": f.distance_from_daily_high,
                "distance_from_daily_low": f.distance_from_daily_low,
                "btc_correlation": f.btc_correlation,
                "trend_strength": f.trend_strength,
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for ML model."""
        return [
            "head_depth_ratio",
            "shoulder_symmetry",
            "neckline_slope",
            "pattern_duration_bars",
            "volume_at_head",
            "volume_ratio_head_shoulders",
            "obv_divergence",
            "atr_percentile",
            "distance_from_daily_high",
            "distance_from_daily_low",
            "btc_correlation",
            "trend_strength",
        ]


def create_feature_engineer(
    config: Optional[FeatureConfig] = None
) -> FeatureEngineer:
    """Factory function for FeatureEngineer."""
    return FeatureEngineer(config=config)

