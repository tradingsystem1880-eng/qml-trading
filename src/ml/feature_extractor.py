"""
Pattern Feature Extractor
=========================
Extracts 170+ VRD features for pattern registration and ML training.

Combines market context features from FeatureLibrary with pattern-specific
geometric and temporal features.

Usage:
    from src.ml.feature_extractor import PatternFeatureExtractor
    
    extractor = PatternFeatureExtractor()
    features = extractor.extract_pattern_features(pattern_data, market_df)
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.features.library import FeatureLibrary, FeatureLibraryConfig
from src.features.regime import RegimeClassifier


class PatternFeatureExtractor:
    """
    Extract comprehensive features for QML pattern analysis.
    
    Produces 170+ features organized into categories:
    1. Market Context (50+): Technical indicators at detection time
    2. Pattern Geometry (15+): Shape and structure of the pattern
    3. Temporal (10+): Timing and seasonality features
    4. Volatility (15+): Volatility regime and structure
    5. Momentum (15+): Trend and momentum indicators
    6. Volume (10+): Volume profile features
    
    All features are point-in-time (no look-ahead bias).
    """
    
    def __init__(self, config: Optional[FeatureLibraryConfig] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration for feature library
        """
        self.feature_library = FeatureLibrary(config)
        self.regime_classifier = RegimeClassifier()
        self._feature_names: List[str] = []
    
    def extract_pattern_features(
        self,
        pattern_data: Dict[str, Any],
        market_context: pd.DataFrame,
        bar_idx: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Extract all features for a detected pattern.
        
        Args:
            pattern_data: Pattern information dict with keys:
                - pattern_type: 'bullish' or 'bearish'
                - left_shoulder_price, left_shoulder_idx
                - head_price, head_idx
                - right_shoulder_price, right_shoulder_idx
                - detection_time, detection_idx
                - entry_price, stop_loss, take_profit
                - validity_score
            market_context: OHLCV DataFrame with market data
            bar_idx: Optional bar index for feature extraction
                     (defaults to detection bar)
        
        Returns:
            Dictionary of feature_name -> value (170+ features)
        """
        if bar_idx is None:
            bar_idx = pattern_data.get('detection_idx', len(market_context) - 1)
        
        # Ensure we have enough history
        min_lookback = 200
        if bar_idx < min_lookback:
            logger.warning(f"Limited history ({bar_idx} bars), some features may be unavailable")
        
        features = {}
        
        # 1. Market Context Features (from FeatureLibrary)
        market_features = self._extract_market_features(market_context, bar_idx)
        features.update(market_features)
        
        # 2. Pattern Geometry Features
        geometry_features = self._extract_geometry_features(pattern_data, market_context, bar_idx)
        features.update(geometry_features)
        
        # 3. Temporal Features
        temporal_features = self._extract_temporal_features(pattern_data, market_context, bar_idx)
        features.update(temporal_features)
        
        # 4. Volatility Features
        volatility_features = self._extract_volatility_features(market_context, bar_idx)
        features.update(volatility_features)
        
        # 5. Momentum Features
        momentum_features = self._extract_momentum_features(market_context, bar_idx)
        features.update(momentum_features)
        
        # 6. Volume Features
        volume_features = self._extract_volume_features(pattern_data, market_context, bar_idx)
        features.update(volume_features)
        
        # 7. Regime Features
        regime_features = self._extract_regime_features(market_context, bar_idx)
        features.update(regime_features)
        
        # 8. Store RAW GEOMETRY for visualization (P1-P5 coordinates)
        # This is critical for pattern_viz.py to extract validated coordinates
        raw_geometry = self._extract_raw_geometry(pattern_data, market_context, bar_idx)
        features.update(raw_geometry)
        
        # Cache feature names
        if not self._feature_names:
            self._feature_names = list(features.keys())
        
        # Clean features (replace NaN/Inf with 0)
        features = self._clean_features(features)
        
        logger.debug(f"Extracted {len(features)} features for pattern")
        
        return features
    
    def _extract_market_features(
        self,
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Extract market context features using FeatureLibrary."""
        
        # Use a subset of data for efficiency
        start_idx = max(0, bar_idx - 500)
        subset = df.iloc[start_idx:bar_idx + 1].copy()
        
        # Compute features for the range
        feature_df = self.feature_library.compute_features_for_range(
            subset,
            start_idx=min(50, len(subset) - 1),
            end_idx=len(subset)
        )
        
        if len(feature_df) == 0:
            return {}
        
        # Get last row (current bar)
        last_row = feature_df.iloc[-1]
        
        # Convert to dict, excluding metadata columns
        features = {}
        for col in feature_df.columns:
            if col not in ['time', 'bar_idx']:
                val = last_row[col]
                if isinstance(val, (int, float, np.number)):
                    features[f"ctx_{col}"] = float(val)
        
        return features
    
    def _extract_geometry_features(
        self,
        pattern_data: Dict[str, Any],
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Extract pattern geometry features."""
        features = {}
        
        # Get prices
        left_price = pattern_data.get('left_shoulder_price', 0)
        head_price = pattern_data.get('head_price', 0)
        right_price = pattern_data.get('right_shoulder_price', 0)
        entry_price = pattern_data.get('entry_price', df.iloc[bar_idx]['close'] if bar_idx < len(df) else 0)
        
        # Get ATR for normalization
        atr = self._calculate_atr(df, bar_idx)
        
        if atr > 0 and left_price > 0 and head_price > 0:
            pattern_type = pattern_data.get('pattern_type', 'bullish')
            
            # Head depth (normalized by ATR)
            if pattern_type == 'bullish':
                head_depth = left_price - head_price
            else:
                head_depth = head_price - left_price
            
            features['geo_head_depth_atr'] = head_depth / atr
            features['geo_head_depth_pct'] = (head_depth / left_price) * 100
            
            # Shoulder symmetry
            if right_price > 0 and left_price > 0:
                if pattern_type == 'bullish':
                    features['geo_shoulder_symmetry'] = right_price / left_price
                else:
                    features['geo_shoulder_symmetry'] = left_price / right_price
                
                shoulder_diff = abs(left_price - right_price)
                features['geo_shoulder_diff_atr'] = shoulder_diff / atr
            else:
                features['geo_shoulder_symmetry'] = 1.0
                features['geo_shoulder_diff_atr'] = 0.0
            
            # Neckline slope
            left_idx = pattern_data.get('left_shoulder_idx', 0)
            right_idx = pattern_data.get('right_shoulder_idx', bar_idx)
            
            if right_idx > left_idx:
                bar_span = right_idx - left_idx
                features['geo_neckline_slope'] = (right_price - left_price) / (bar_span * atr) if bar_span > 0 else 0
            else:
                features['geo_neckline_slope'] = 0.0
            
            # Risk/Reward metrics
            stop_loss = pattern_data.get('stop_loss', 0)
            take_profit = pattern_data.get('take_profit', 0)
            
            if stop_loss > 0 and entry_price > 0:
                risk = abs(entry_price - stop_loss)
                features['geo_risk_atr'] = risk / atr
                
                if take_profit > 0:
                    reward = abs(take_profit - entry_price)
                    features['geo_reward_atr'] = reward / atr
                    features['geo_rr_ratio'] = reward / risk if risk > 0 else 0
                else:
                    features['geo_reward_atr'] = 0.0
                    features['geo_rr_ratio'] = 0.0
            else:
                features['geo_risk_atr'] = 0.0
                features['geo_reward_atr'] = 0.0
                features['geo_rr_ratio'] = 0.0
            
            # Pattern direction indicator
            features['geo_is_bullish'] = 1.0 if pattern_type == 'bullish' else 0.0
            features['geo_is_bearish'] = 1.0 if pattern_type == 'bearish' else 0.0
            
            # Validity score from detection
            features['geo_validity_score'] = float(pattern_data.get('validity_score', 0.5))
        
        return features
    
    def _extract_temporal_features(
        self,
        pattern_data: Dict[str, Any],
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Extract temporal and seasonality features."""
        features = {}
        
        # Get detection time
        detection_time = pattern_data.get('detection_time')
        if detection_time is None and bar_idx < len(df):
            if 'time' in df.columns:
                detection_time = df.iloc[bar_idx]['time']
        
        if detection_time is not None:
            if isinstance(detection_time, str):
                detection_time = pd.to_datetime(detection_time)
            
            # Ensure we have a pandas Timestamp (not datetime.datetime)
            if not isinstance(detection_time, pd.Timestamp):
                detection_time = pd.Timestamp(detection_time)
            
            # Hour of day (cyclical encoding)
            hour = detection_time.hour
            features['temp_hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['temp_hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Day of week (cyclical encoding)
            dow = detection_time.dayofweek
            features['temp_dow_sin'] = np.sin(2 * np.pi * dow / 7)
            features['temp_dow_cos'] = np.cos(2 * np.pi * dow / 7)
            
            # Weekend indicator
            features['temp_is_weekend'] = 1.0 if dow >= 5 else 0.0
            
            # Session indicators (crypto runs 24/7, but activity varies)
            features['temp_is_asia'] = 1.0 if 0 <= hour < 8 else 0.0
            features['temp_is_europe'] = 1.0 if 8 <= hour < 16 else 0.0
            features['temp_is_us'] = 1.0 if 16 <= hour < 24 else 0.0
        
        # Pattern duration in bars
        left_idx = pattern_data.get('left_shoulder_idx', 0)
        right_idx = pattern_data.get('right_shoulder_idx', bar_idx)
        detection_idx = pattern_data.get('detection_idx', bar_idx)
        
        features['temp_pattern_duration'] = float(right_idx - left_idx)
        features['temp_bars_since_head'] = float(detection_idx - pattern_data.get('head_idx', detection_idx))
        
        return features
    
    def _extract_volatility_features(
        self,
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Extract volatility-related features."""
        features = {}
        
        close = df['close'].values[:bar_idx + 1]
        high = df['high'].values[:bar_idx + 1]
        low = df['low'].values[:bar_idx + 1]
        
        if len(close) < 20:
            return features
        
        # Current ATR and percentiles
        atr_14 = self._calculate_atr(df, bar_idx, period=14)
        atr_21 = self._calculate_atr(df, bar_idx, period=21)
        
        features['vol_atr_14'] = atr_14
        features['vol_atr_21'] = atr_21
        
        # ATR ratio (expansion/contraction)
        if atr_21 > 0:
            features['vol_atr_ratio'] = atr_14 / atr_21
        else:
            features['vol_atr_ratio'] = 1.0
        
        # Realized volatility (rolling std of returns)
        returns = np.diff(close[-51:]) / close[-51:-1] if len(close) > 51 else np.array([0])
        
        if len(returns) >= 20:
            features['vol_realized_20'] = float(np.std(returns[-20:]) * np.sqrt(252))
            features['vol_realized_50'] = float(np.std(returns) * np.sqrt(252))
        else:
            features['vol_realized_20'] = 0.0
            features['vol_realized_50'] = 0.0
        
        # Current range vs average
        if bar_idx > 0 and bar_idx < len(df):
            current_range = high[bar_idx] - low[bar_idx]
            avg_range = np.mean(high[-20:] - low[-20:]) if len(high) >= 20 else current_range
            features['vol_range_ratio'] = current_range / avg_range if avg_range > 0 else 1.0
        
        return features
    
    def _extract_momentum_features(
        self,
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Extract momentum features."""
        features = {}
        
        close = df['close'].values[:bar_idx + 1]
        
        if len(close) < 50:
            return features
        
        # Returns over various lookbacks
        for lookback in [5, 10, 20, 50]:
            if len(close) > lookback:
                ret = (close[-1] - close[-lookback - 1]) / close[-lookback - 1] * 100
                features[f'mom_return_{lookback}'] = float(ret)
            else:
                features[f'mom_return_{lookback}'] = 0.0
        
        # Price position relative to range
        if len(close) >= 20:
            high_20 = np.max(df['high'].values[bar_idx - 19:bar_idx + 1])
            low_20 = np.min(df['low'].values[bar_idx - 19:bar_idx + 1])
            range_20 = high_20 - low_20
            
            if range_20 > 0:
                features['mom_position_20'] = (close[-1] - low_20) / range_20
            else:
                features['mom_position_20'] = 0.5
        
        # Trend strength (using linear regression slope)
        if len(close) >= 20:
            x = np.arange(20)
            y = close[-20:]
            if len(y) == 20:
                slope = np.polyfit(x, y, 1)[0]
                features['mom_trend_slope'] = slope / close[-1] * 100
            else:
                features['mom_trend_slope'] = 0.0
        
        return features
    
    def _extract_volume_features(
        self,
        pattern_data: Dict[str, Any],
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Extract volume-related features."""
        features = {}
        
        if 'volume' not in df.columns:
            return features
        
        volume = df['volume'].values[:bar_idx + 1]
        
        if len(volume) < 20:
            return features
        
        # Current volume vs averages
        current_vol = volume[-1]
        avg_vol_10 = np.mean(volume[-10:])
        avg_vol_20 = np.mean(volume[-20:])
        
        features['vol_relative_10'] = current_vol / avg_vol_10 if avg_vol_10 > 0 else 1.0
        features['vol_relative_20'] = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
        
        # Volume at pattern points
        head_idx = pattern_data.get('head_idx', bar_idx)
        if head_idx < len(volume):
            head_vol = volume[head_idx]
            features['vol_at_head_relative'] = head_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
        else:
            features['vol_at_head_relative'] = 1.0
        
        # Volume trend
        if len(volume) >= 10:
            recent_avg = np.mean(volume[-5:])
            prior_avg = np.mean(volume[-10:-5])
            features['vol_trend'] = recent_avg / prior_avg if prior_avg > 0 else 1.0
        else:
            features['vol_trend'] = 1.0
        
        return features
    
    def _extract_regime_features(
        self,
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Extract market regime features."""
        features = {}
        
        try:
            # Detect current regime
            subset = df.iloc[:bar_idx + 1].copy()
            
            if len(subset) >= 50:
                regime_state = self.regime_classifier.classify(subset)
                
                # Encode regime as features
                features['regime_is_trending'] = 1.0 if regime_state.confidence > 0.6 else 0.0
                features['regime_is_ranging'] = 1.0 if regime_state.confidence < 0.4 else 0.0
                features['regime_trend_strength'] = float(regime_state.trend_direction)
                features['regime_volatility'] = float(regime_state.atr_percentile / 100.0)
        except Exception as e:
            logger.debug(f"Regime detection failed: {e}")
            features['regime_is_trending'] = 0.5
            features['regime_is_ranging'] = 0.5
            features['regime_trend_strength'] = 0.5
            features['regime_volatility'] = 0.5
        
        return features
    
    def _extract_raw_geometry(
        self,
        pattern_data: Dict[str, Any],
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, Any]:
        """
        Extract RAW pattern geometry coordinates for visualization.
        
        This stores the actual P1-P5 timestamps, prices, and indices so that
        pattern_viz.py can extract validated data instead of inferring.
        
        Point mapping (QML Pattern):
            P1: Left Shoulder (trend extreme before reversal)
            P2: CHoCH Point (change of character)
            P3: Head (pattern extreme - lowest for bullish, highest for bearish)
            P4: BOS Point (break of structure)
            P5: Right Shoulder / Entry (retrace to neckline)
        """
        geometry = {}
        
        # Get pattern type
        pattern_type = pattern_data.get('pattern_type', 'bullish')
        geometry['pattern_type'] = pattern_type
        
        # Extract timestamps from DataFrame using indices
        time_col = 'time' if 'time' in df.columns else df.index.name
        
        def get_timestamp(idx):
            """Get timestamp for given bar index."""
            if idx is None or idx >= len(df):
                return None
            try:
                if 'time' in df.columns:
                    ts = df.iloc[idx]['time']
                else:
                    ts = df.index[idx]
                # Convert to ISO string for JSON serialization
                if hasattr(ts, 'isoformat'):
                    return ts.isoformat()
                return str(ts)
            except:
                return None
        
        # P1: Left Shoulder
        left_idx = pattern_data.get('left_shoulder_idx')
        left_price = pattern_data.get('left_shoulder_price')
        geometry['p1_idx'] = left_idx
        geometry['p1_price'] = left_price
        geometry['p1_timestamp'] = get_timestamp(left_idx)
        
        # P3: Head (the pattern extreme)
        head_idx = pattern_data.get('head_idx')
        head_price = pattern_data.get('head_price')
        geometry['p3_idx'] = head_idx
        geometry['p3_price'] = head_price
        geometry['p3_timestamp'] = get_timestamp(head_idx)
        
        # P5: Right Shoulder / Entry Point
        right_idx = pattern_data.get('right_shoulder_idx', pattern_data.get('detection_idx'))
        right_price = pattern_data.get('right_shoulder_price')
        geometry['p5_idx'] = right_idx
        geometry['p5_price'] = right_price
        geometry['p5_timestamp'] = get_timestamp(right_idx)
        
        # P2: CHoCH Point (interpolated between P1 and P3)
        if left_idx is not None and head_idx is not None:
            p2_idx = (left_idx + head_idx) // 2
            geometry['p2_idx'] = p2_idx
            geometry['p2_timestamp'] = get_timestamp(p2_idx)
            # P2 price: intermediate level
            if left_price is not None and head_price is not None:
                geometry['p2_price'] = (left_price + head_price) / 2
            else:
                geometry['p2_price'] = None
        else:
            geometry['p2_idx'] = None
            geometry['p2_timestamp'] = None
            geometry['p2_price'] = None
        
        # P4: BOS Point (interpolated between P3 and P5)
        if head_idx is not None and right_idx is not None:
            p4_idx = (head_idx + right_idx) // 2
            geometry['p4_idx'] = p4_idx
            geometry['p4_timestamp'] = get_timestamp(p4_idx)
            # P4 price: intermediate level
            if head_price is not None and right_price is not None:
                geometry['p4_price'] = (head_price + right_price) / 2
            else:
                geometry['p4_price'] = None
        else:
            geometry['p4_idx'] = None
            geometry['p4_timestamp'] = None
            geometry['p4_price'] = None
        
        # Trade levels
        geometry['entry_price'] = pattern_data.get('entry_price')
        geometry['stop_loss_price'] = pattern_data.get('stop_loss')
        geometry['take_profit_price'] = pattern_data.get('take_profit')
        
        # Store detection info
        geometry['detection_idx'] = bar_idx
        geometry['detection_timestamp'] = get_timestamp(bar_idx)
        
        return geometry
    
    
    def _calculate_atr(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        period: int = 14
    ) -> float:
        """Calculate ATR at a specific bar."""
        if bar_idx < period:
            return 0.0
        
        high = df['high'].values[:bar_idx + 1]
        low = df['low'].values[:bar_idx + 1]
        close = df['close'].values[:bar_idx + 1]
        
        # True Range
        tr = np.maximum(
            high[-period:] - low[-period:],
            np.maximum(
                np.abs(high[-period:] - np.roll(close, 1)[-period:]),
                np.abs(low[-period:] - np.roll(close, 1)[-period:])
            )
        )
        
        return float(np.mean(tr))
    
    def _clean_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Clean features by replacing NaN/Inf with 0."""
        cleaned = {}
        for key, value in features.items():
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value) or np.isinf(value):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = float(value)
            else:
                cleaned[key] = 0.0
        return cleaned
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self._feature_names
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self._feature_names)
    
    def features_to_json(self, features: Dict[str, float]) -> str:
        """Convert features dict to JSON string."""
        return json.dumps(features)
    
    def features_from_json(self, json_str: str) -> Dict[str, float]:
        """Parse features from JSON string."""
        return json.loads(json_str)


def create_feature_extractor(
    config: Optional[FeatureLibraryConfig] = None
) -> PatternFeatureExtractor:
    """Factory function for PatternFeatureExtractor."""
    return PatternFeatureExtractor(config)
