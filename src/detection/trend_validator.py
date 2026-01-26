"""
Prior Trend Validator
=====================
Validates that a meaningful prior trend exists before QML patterns.

QML patterns are reversal patterns, so they require:
- BULLISH QML: Prior uptrend to reverse (will short)
- BEARISH QML: Prior downtrend to reverse (will long)

This validator ensures patterns don't appear in choppy, trendless markets.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.detection.historical_detector import HistoricalSwingPoint
from src.detection.pattern_validator import PatternDirection


@dataclass
class TrendValidationConfig:
    """
    Configuration for prior trend validation.
    All parameters are ML-optimizable.
    """
    # Minimum ADX for a valid trend
    min_adx: float = 20.0

    # Minimum total move in ATR units
    min_trend_move_atr: float = 3.0

    # Minimum number of swings in the trend
    min_trend_swings: int = 3

    # Minimum trend duration in bars
    min_trend_bars: int = 15

    # Maximum trend duration (avoid ancient trends)
    max_trend_bars: int = 200

    # ADX calculation period
    adx_period: int = 14

    # ATR calculation period
    atr_period: int = 14

    # Trend direction consistency threshold
    # For uptrend: HH/HL ratio >= this
    # For downtrend: LH/LL ratio >= this
    direction_consistency: float = 0.6

    def __post_init__(self):
        """Validate configuration."""
        if self.min_adx < 0:
            raise ValueError("min_adx must be >= 0")
        if self.min_trend_move_atr <= 0:
            raise ValueError("min_trend_move_atr must be > 0")
        if self.min_trend_swings < 2:
            raise ValueError("min_trend_swings must be >= 2")


@dataclass
class TrendValidationResult:
    """Result of prior trend validation."""
    is_valid: bool
    trend_direction: Optional[str] = None  # 'UP' or 'DOWN'

    # Trend metrics
    trend_move_atr: float = 0.0
    trend_swings: int = 0
    trend_bars: int = 0
    trend_adx: float = 0.0

    # Swing sequence analysis
    higher_highs: int = 0
    lower_lows: int = 0
    lower_highs: int = 0
    higher_lows: int = 0

    # Rejection info
    rejection_reason: str = ""
    rejection_details: str = ""

    # Trend start/end
    trend_start_idx: int = 0
    trend_end_idx: int = 0


class TrendValidator:
    """
    Validates that a meaningful prior trend exists before a pattern.

    For QML patterns to be valid reversals, they need a trend to reverse.
    This validator examines the swing structure before P1 to verify:
    1. ADX indicates trending conditions
    2. Sufficient move size (in ATR terms)
    3. Proper swing sequence (HH/HL for uptrend, LH/LL for downtrend)
    """

    def __init__(self, config: Optional[TrendValidationConfig] = None):
        """
        Initialize the trend validator.

        Args:
            config: Validation configuration
        """
        self.config = config or TrendValidationConfig()

    def validate(
        self,
        swings: List[HistoricalSwingPoint],
        p1_bar_index: int,
        df: pd.DataFrame,
        pattern_direction: PatternDirection
    ) -> TrendValidationResult:
        """
        Validate that a proper prior trend exists.

        Args:
            swings: All detected swing points
            p1_bar_index: Bar index of P1 (pattern start)
            df: OHLCV DataFrame
            pattern_direction: Direction of the QML pattern

        Returns:
            TrendValidationResult with validity and metrics
        """
        # Get swings before P1
        prior_swings = [s for s in swings if s.bar_index < p1_bar_index]

        if len(prior_swings) < self.config.min_trend_swings:
            return TrendValidationResult(
                is_valid=False,
                rejection_reason="insufficient_swings",
                rejection_details=f"Only {len(prior_swings)} swings before P1, need {self.config.min_trend_swings}",
            )

        # Determine expected trend direction
        # BULLISH QML (head HIGH) reverses uptrend -> expect prior UPTREND
        # BEARISH QML (head LOW) reverses downtrend -> expect prior DOWNTREND
        if pattern_direction == PatternDirection.BULLISH:
            expected_trend = 'UP'
        else:
            expected_trend = 'DOWN'

        # Analyze the swing sequence
        trend_swings = self._get_trend_swings(prior_swings, p1_bar_index)

        if len(trend_swings) < self.config.min_trend_swings:
            return TrendValidationResult(
                is_valid=False,
                rejection_reason="insufficient_trend_swings",
                rejection_details=f"Only {len(trend_swings)} swings in lookback window",
            )

        # Analyze swing structure
        structure = self._analyze_swing_structure(trend_swings)

        # Check trend duration
        trend_bars = p1_bar_index - trend_swings[0].bar_index
        if trend_bars < self.config.min_trend_bars:
            return TrendValidationResult(
                is_valid=False,
                rejection_reason="trend_too_short",
                rejection_details=f"Trend spans {trend_bars} bars, need {self.config.min_trend_bars}",
                trend_bars=trend_bars,
                **structure,
            )

        # Calculate trend move size
        atr = self._get_atr_at_index(df, p1_bar_index)
        trend_move_atr = self._calculate_trend_move(trend_swings, atr)

        if trend_move_atr < self.config.min_trend_move_atr:
            return TrendValidationResult(
                is_valid=False,
                rejection_reason="trend_move_too_small",
                rejection_details=f"Trend move {trend_move_atr:.2f} ATR < {self.config.min_trend_move_atr}",
                trend_move_atr=trend_move_atr,
                trend_bars=trend_bars,
                **structure,
            )

        # Calculate average ADX during trend
        avg_adx = self._calculate_avg_adx(df, trend_swings[0].bar_index, p1_bar_index)

        # Validate direction consistency
        is_uptrend = self._is_uptrend(structure)
        is_downtrend = self._is_downtrend(structure)

        detected_trend = None
        if is_uptrend:
            detected_trend = 'UP'
        elif is_downtrend:
            detected_trend = 'DOWN'

        # Check if detected trend matches expected
        if detected_trend != expected_trend:
            return TrendValidationResult(
                is_valid=False,
                trend_direction=detected_trend,
                rejection_reason="trend_direction_mismatch",
                rejection_details=f"Expected {expected_trend} trend for {pattern_direction.value} pattern, detected {detected_trend}",
                trend_move_atr=trend_move_atr,
                trend_bars=trend_bars,
                trend_adx=avg_adx,
                trend_swings=len(trend_swings),
                trend_start_idx=trend_swings[0].bar_index,
                trend_end_idx=p1_bar_index,
                **structure,
            )

        # All checks passed
        return TrendValidationResult(
            is_valid=True,
            trend_direction=detected_trend,
            trend_move_atr=trend_move_atr,
            trend_bars=trend_bars,
            trend_adx=avg_adx,
            trend_swings=len(trend_swings),
            trend_start_idx=trend_swings[0].bar_index,
            trend_end_idx=p1_bar_index,
            **structure,
        )

    def _get_trend_swings(
        self,
        prior_swings: List[HistoricalSwingPoint],
        p1_bar_index: int
    ) -> List[HistoricalSwingPoint]:
        """
        Get the swings that form the prior trend.

        Looks back from P1 up to max_trend_bars.
        """
        min_bar = p1_bar_index - self.config.max_trend_bars

        return [
            s for s in prior_swings
            if s.bar_index >= min_bar
        ]

    def _analyze_swing_structure(
        self,
        swings: List[HistoricalSwingPoint]
    ) -> dict:
        """
        Analyze the HH/HL/LH/LL structure of swings.

        Returns:
            Dictionary with counts of HH, HL, LH, LL
        """
        highs = [s for s in swings if s.swing_type == 'HIGH']
        lows = [s for s in swings if s.swing_type == 'LOW']

        higher_highs = 0
        lower_highs = 0
        higher_lows = 0
        lower_lows = 0

        # Analyze highs
        for i in range(1, len(highs)):
            if highs[i].price > highs[i-1].price:
                higher_highs += 1
            else:
                lower_highs += 1

        # Analyze lows
        for i in range(1, len(lows)):
            if lows[i].price > lows[i-1].price:
                higher_lows += 1
            else:
                lower_lows += 1

        return {
            'higher_highs': higher_highs,
            'lower_highs': lower_highs,
            'higher_lows': higher_lows,
            'lower_lows': lower_lows,
        }

    def _is_uptrend(self, structure: dict) -> bool:
        """
        Check if the structure represents an uptrend.

        Uptrend: Higher Highs and Higher Lows
        """
        total_highs = structure['higher_highs'] + structure['lower_highs']
        total_lows = structure['higher_lows'] + structure['lower_lows']

        if total_highs == 0 or total_lows == 0:
            return False

        hh_ratio = structure['higher_highs'] / total_highs
        hl_ratio = structure['higher_lows'] / total_lows

        return (
            hh_ratio >= self.config.direction_consistency and
            hl_ratio >= self.config.direction_consistency
        )

    def _is_downtrend(self, structure: dict) -> bool:
        """
        Check if the structure represents a downtrend.

        Downtrend: Lower Highs and Lower Lows
        """
        total_highs = structure['higher_highs'] + structure['lower_highs']
        total_lows = structure['higher_lows'] + structure['lower_lows']

        if total_highs == 0 or total_lows == 0:
            return False

        lh_ratio = structure['lower_highs'] / total_highs
        ll_ratio = structure['lower_lows'] / total_lows

        return (
            lh_ratio >= self.config.direction_consistency and
            ll_ratio >= self.config.direction_consistency
        )

    def _calculate_trend_move(
        self,
        swings: List[HistoricalSwingPoint],
        atr: float
    ) -> float:
        """
        Calculate the total trend move in ATR units.

        Uses the price range from trend start to end.
        """
        if not swings or atr <= 0:
            return 0.0

        prices = [s.price for s in swings]
        move = max(prices) - min(prices)

        return move / atr

    def _get_atr_at_index(self, df: pd.DataFrame, idx: int) -> float:
        """Get ATR value at a specific bar index."""
        if 'ATR' in df.columns:
            return float(df.iloc[idx]['ATR'])

        # Calculate ATR if not present
        period = self.config.atr_period

        high = df['high'].values if 'high' in df.columns else df['High'].values
        low = df['low'].values if 'low' in df.columns else df['Low'].values
        close = df['close'].values if 'close' in df.columns else df['Close'].values

        # Simple ATR at index
        start = max(0, idx - period)
        tr_values = []

        for i in range(start, idx):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]) if i > 0 else high[i] - low[i],
                abs(low[i] - close[i-1]) if i > 0 else high[i] - low[i],
            )
            tr_values.append(tr)

        return np.mean(tr_values) if tr_values else 1.0

    def _calculate_avg_adx(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> float:
        """Calculate average ADX over a range."""
        period = self.config.adx_period

        high = df['high'].values if 'high' in df.columns else df['High'].values
        low = df['low'].values if 'low' in df.columns else df['Low'].values
        close = df['close'].values if 'close' in df.columns else df['Close'].values

        # Simplified ADX calculation for the range
        up_move = np.diff(high[start_idx:end_idx], prepend=high[start_idx])
        down_move = np.diff(-low[start_idx:end_idx], prepend=-low[start_idx])

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Simple average for ADX estimate
        avg_plus = np.mean(plus_dm)
        avg_minus = np.mean(minus_dm)
        avg_tr = np.mean(high[start_idx:end_idx] - low[start_idx:end_idx])

        if avg_tr < 0.001:
            return 0.0

        plus_di = 100 * avg_plus / avg_tr
        minus_di = 100 * avg_minus / avg_tr

        if (plus_di + minus_di) < 0.001:
            return 0.0

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        return float(dx)

    def find_trend_sequence(
        self,
        swings: List[HistoricalSwingPoint],
        end_idx: int,
        trend_direction: str
    ) -> List[HistoricalSwingPoint]:
        """
        Find the sequence of swings that form a trend.

        Useful for visualization of the prior trend.

        Args:
            swings: All swing points
            end_idx: Bar index where trend ends (P1)
            trend_direction: 'UP' or 'DOWN'

        Returns:
            List of swings forming the trend sequence
        """
        prior_swings = [s for s in swings if s.bar_index < end_idx]

        if len(prior_swings) < 2:
            return []

        # Get swings within max lookback
        min_bar = end_idx - self.config.max_trend_bars
        trend_swings = [s for s in prior_swings if s.bar_index >= min_bar]

        return trend_swings


# =============================================================================
# TREND REGIME VALIDATOR (Phase 7.7)
# =============================================================================

@dataclass
class TrendRegimeConfig:
    """
    Configuration for trend regime validation with R² linearity.

    Phase 7.7 addition: Validates trend quality using linear regression.
    """
    # R² linearity threshold (0-1, higher = more linear trend)
    min_r_squared: float = 0.6

    # Minimum trend slope (price change per bar, normalized)
    min_slope_magnitude: float = 0.001

    # Regression lookback period
    regression_lookback: int = 50

    # ADX threshold for trending market
    min_adx: float = 20.0

    # Minimum bars for regression
    min_regression_bars: int = 20

    # Trend strength multiplier (slope × R²)
    min_trend_strength: float = 0.3

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.min_r_squared <= 1:
            raise ValueError("min_r_squared must be between 0 and 1")
        if self.min_regression_bars < 10:
            raise ValueError("min_regression_bars must be >= 10")


@dataclass
class TrendRegimeResult:
    """Result of trend regime validation."""
    is_valid: bool
    r_squared: float = 0.0
    slope: float = 0.0
    slope_normalized: float = 0.0  # Normalized by price
    trend_strength: float = 0.0  # slope × R²
    adx: float = 0.0
    regime: str = ""  # 'TRENDING_UP', 'TRENDING_DOWN', 'RANGING'

    # Rejection details
    rejection_reason: str = ""
    rejection_details: str = ""


class TrendRegimeValidator:
    """
    Validates trend regime using R² linear regression.

    Phase 7.7 enhancement: Uses R² (coefficient of determination) to
    measure how linear the prior trend is. More linear trends produce
    more reliable reversal signals.

    R² = 1 - (SS_res / SS_tot)
    - R² = 1.0: Perfect linear trend
    - R² = 0.0: No linear relationship
    - R² > 0.6: Reasonably linear trend (good for reversals)
    """

    def __init__(self, config: Optional[TrendRegimeConfig] = None):
        """
        Initialize the trend regime validator.

        Args:
            config: Validation configuration
        """
        self.config = config or TrendRegimeConfig()

    def validate(
        self,
        df: pd.DataFrame,
        end_idx: int,
        expected_direction: str,  # 'UP' or 'DOWN'
    ) -> TrendRegimeResult:
        """
        Validate trend regime quality.

        Args:
            df: OHLCV DataFrame
            end_idx: Bar index where trend ends (pattern start)
            expected_direction: Expected trend direction

        Returns:
            TrendRegimeResult with R², slope, and validity
        """
        cfg = self.config

        # Calculate lookback range
        start_idx = max(0, end_idx - cfg.regression_lookback)
        actual_bars = end_idx - start_idx

        if actual_bars < cfg.min_regression_bars:
            return TrendRegimeResult(
                is_valid=False,
                rejection_reason="insufficient_bars",
                rejection_details=f"Only {actual_bars} bars, need {cfg.min_regression_bars}",
            )

        # Get close prices for regression
        close = df['close'].values if 'close' in df.columns else df['Close'].values
        prices = close[start_idx:end_idx]

        # Calculate linear regression
        r_squared, slope = self._calculate_regression(prices)

        # Normalize slope by average price
        avg_price = np.mean(prices)
        slope_normalized = slope / avg_price if avg_price > 0 else 0.0

        # Calculate trend strength (slope × R²)
        trend_strength = abs(slope_normalized) * r_squared

        # Calculate ADX
        adx = self._calculate_adx_at_index(df, end_idx)

        # Determine regime
        if r_squared >= cfg.min_r_squared:
            if slope_normalized > cfg.min_slope_magnitude:
                regime = 'TRENDING_UP'
            elif slope_normalized < -cfg.min_slope_magnitude:
                regime = 'TRENDING_DOWN'
            else:
                regime = 'RANGING'
        else:
            regime = 'RANGING'

        # Validate direction match
        direction_match = (
            (expected_direction == 'UP' and regime == 'TRENDING_UP') or
            (expected_direction == 'DOWN' and regime == 'TRENDING_DOWN')
        )

        if not direction_match:
            return TrendRegimeResult(
                is_valid=False,
                r_squared=r_squared,
                slope=slope,
                slope_normalized=slope_normalized,
                trend_strength=trend_strength,
                adx=adx,
                regime=regime,
                rejection_reason="direction_mismatch",
                rejection_details=f"Expected {expected_direction}, detected {regime}",
            )

        # Validate R² threshold
        if r_squared < cfg.min_r_squared:
            return TrendRegimeResult(
                is_valid=False,
                r_squared=r_squared,
                slope=slope,
                slope_normalized=slope_normalized,
                trend_strength=trend_strength,
                adx=adx,
                regime=regime,
                rejection_reason="low_r_squared",
                rejection_details=f"R²={r_squared:.3f} < {cfg.min_r_squared}",
            )

        # Validate trend strength
        if trend_strength < cfg.min_trend_strength:
            return TrendRegimeResult(
                is_valid=False,
                r_squared=r_squared,
                slope=slope,
                slope_normalized=slope_normalized,
                trend_strength=trend_strength,
                adx=adx,
                regime=regime,
                rejection_reason="weak_trend",
                rejection_details=f"Trend strength {trend_strength:.3f} < {cfg.min_trend_strength}",
            )

        # All validations passed
        return TrendRegimeResult(
            is_valid=True,
            r_squared=r_squared,
            slope=slope,
            slope_normalized=slope_normalized,
            trend_strength=trend_strength,
            adx=adx,
            regime=regime,
        )

    def _calculate_regression(self, prices: np.ndarray) -> Tuple[float, float]:
        """
        Calculate linear regression R² and slope.

        Args:
            prices: Array of prices

        Returns:
            (r_squared, slope)
        """
        n = len(prices)
        if n < 2:
            return 0.0, 0.0

        # X = bar indices (0, 1, 2, ...)
        x = np.arange(n)
        y = prices

        # Linear regression using least squares
        # slope = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0, 0.0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Predicted values
        y_pred = slope * x + intercept

        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        if ss_tot == 0:
            return 0.0, slope

        r_squared = 1 - (ss_res / ss_tot)

        return float(r_squared), float(slope)

    def _calculate_adx_at_index(
        self,
        df: pd.DataFrame,
        idx: int,
        period: int = 14,
    ) -> float:
        """Calculate ADX at a specific index."""
        start = max(0, idx - period * 3)

        # Try to get high/low columns, fall back to close if not available
        if 'high' in df.columns:
            high = df['high'].values
        elif 'High' in df.columns:
            high = df['High'].values
        else:
            # No high column - use close as approximation
            close = df['close'].values if 'close' in df.columns else df['Close'].values
            return 0.0  # Cannot calculate ADX without high/low

        if 'low' in df.columns:
            low = df['low'].values
        elif 'Low' in df.columns:
            low = df['Low'].values
        else:
            return 0.0  # Cannot calculate ADX without high/low

        if idx - start < period:
            return 0.0

        # Simplified ADX calculation
        up_move = np.diff(high[start:idx], prepend=high[start])
        down_move = np.diff(-low[start:idx], prepend=-low[start])

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        avg_plus = np.mean(plus_dm[-period:])
        avg_minus = np.mean(minus_dm[-period:])
        avg_tr = np.mean(high[start:idx][-period:] - low[start:idx][-period:])

        if avg_tr < 0.001:
            return 0.0

        plus_di = 100 * avg_plus / avg_tr
        minus_di = 100 * avg_minus / avg_tr

        if (plus_di + minus_di) < 0.001:
            return 0.0

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        return float(dx)
