"""
Technical Indicators - Powered by 'ta' Library
==============================================
Clean wrapper around the 'ta' library maintaining backward compatibility.

All indicators now use battle-tested implementations from the 'ta' library
while maintaining the same API for backward compatibility.

References:
- ta library: https://github.com/bukosabino/ta
- 3.7k+ stars, production-tested
- TradingView parity for all major indicators
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Import from ta library (battle-tested implementations)
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator


def calculate_atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """
    Calculate Average True Range (ATR).
    
    Now powered by 'ta' library for accuracy and reliability.
    Uses Wilder's smoothing method (exponential moving average).
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period (default: 14)
        
    Returns:
        Array of ATR values
    """
    # Convert to pandas Series (ta library requirement)
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })
    
    # Use ta library
    atr_indicator = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=period
    )
    
    return atr_indicator.average_true_range().to_numpy()


def calculate_atr_df(
    df: pd.DataFrame,
    period: int = 14,
    column_name: str = "atr"
) -> pd.DataFrame:
    """
    Calculate ATR and add to DataFrame.
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period
        column_name: Name for ATR column
        
    Returns:
        DataFrame with ATR column added
    """
    df = df.copy()
    
    atr_indicator = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=period
    )
    
    df[column_name] = atr_indicator.average_true_range()
    return df


def calculate_rsi(
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """
    Calculate Relative Strength Index (RSI).
    
    Powered by 'ta' library.
    
    Args:
        close: Array of close prices
        period: RSI period (default: 14)
        
    Returns:
        Array of RSI values (0-100)
    """
    close_series = pd.Series(close)
    rsi_indicator = RSIIndicator(close=close_series, window=period)
    return rsi_indicator.rsi().to_numpy()


def calculate_obv(
    close: NDArray[np.float64],
    volume: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate On-Balance Volume (OBV).
    
    Powered by 'ta' library.
    
    Args:
        close: Array of close prices
        volume: Array of volume
        
    Returns:
        Array of OBV values
    """
    close_series = pd.Series(close)
    volume_series = pd.Series(volume)
    
    obv_indicator = OnBalanceVolumeIndicator(
        close=close_series,
        volume=volume_series
    )
    
    return obv_indicator.on_balance_volume().to_numpy()


def calculate_ema(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """
    Calculate Exponential Moving Average (EMA).
    
    Powered by 'ta' library.
    
    Args:
        data: Array of values
        period: EMA period
        
    Returns:
        Array of EMA values
    """
    data_series = pd.Series(data)
    ema_indicator = EMAIndicator(close=data_series, window=period)
    return ema_indicator.ema_indicator().to_numpy()


def calculate_sma(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """
    Calculate Simple Moving Average (SMA).
    
    Powered by 'ta' library.
    
    Args:
        data: Array of values
        period: SMA period
        
    Returns:
        Array of SMA values
    """
    data_series = pd.Series(data)
    sma_indicator = SMAIndicator(close=data_series, window=period)
    return sma_indicator.sma_indicator().to_numpy()


def calculate_bollinger_bands(
    close: NDArray[np.float64],
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate Bollinger Bands.
    
    Powered by 'ta' library.
    
    Args:
        close: Array of close prices
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    close_series = pd.Series(close)
    
    bb_indicator = BollingerBands(
        close=close_series,
        window=period,
        window_dev=std_dev
    )
    
    upper = bb_indicator.bollinger_hband().to_numpy()
    middle = bb_indicator.bollinger_mavg().to_numpy()
    lower = bb_indicator.bollinger_lband().to_numpy()
    
    return upper, middle, lower


def calculate_macd(
    close: NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Powered by 'ta' library.
    
    Args:
        close: Array of close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    close_series = pd.Series(close)
    
    macd_indicator = MACD(
        close=close_series,
        window_fast=fast_period,
        window_slow=slow_period,
        window_sign=signal_period
    )
    
    macd_line = macd_indicator.macd().to_numpy()
    signal_line = macd_indicator.macd_signal().to_numpy()
    histogram = macd_indicator.macd_diff().to_numpy()
    
    return macd_line, signal_line, histogram


def calculate_adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """
    Calculate Average Directional Index (ADX).
    
    Powered by 'ta' library.
    Used for measuring trend strength.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ADX period
        
    Returns:
        Array of ADX values (0-100)
    """
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })
    
    adx_indicator = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=period
    )
    
    return adx_indicator.adx().to_numpy()


def calculate_volume_profile(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    period: int = 20
) -> NDArray[np.float64]:
    """
    Calculate rolling average volume (volume profile).
    
    Args:
        close: Array of close prices
        volume: Array of volume
        period: Rolling period
        
    Returns:
        Array of average volume values
    """
    # Simple rolling average (pandas is efficient for this)
    volume_series = pd.Series(volume)
    return volume_series.rolling(window=period).mean().to_numpy()


def calculate_volatility_percentile(
    atr: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int = 100
) -> NDArray[np.float64]:
    """
    Calculate ATR as percentile of recent history.
    
    Args:
        atr: Array of ATR values
        close: Array of close prices (for normalization)
        lookback: Lookback period for percentile calculation
        
    Returns:
        Array of percentile values (0-100)
    """
    # Normalize ATR by price
    atr_pct = (atr / close) * 100
    
    # Calculate rolling percentile
    atr_series = pd.Series(atr_pct)
    
    # Rolling percentile rank
    percentiles = np.zeros_like(atr)
    for i in range(lookback, len(atr)):
        window = atr_pct[i - lookback:i]
        percentiles[i] = (np.sum(window < atr_pct[i]) / lookback) * 100
    
    return percentiles


def detect_divergence(
    price: NDArray[np.float64],
    indicator: NDArray[np.float64],
    lookback: int = 20
) -> NDArray[np.float64]:
    """
    Detect price/indicator divergence.
    
    Returns positive values for bullish divergence,
    negative values for bearish divergence.
    
    Args:
        price: Array of prices
        indicator: Array of indicator values
        lookback: Lookback period
        
    Returns:
        Array of divergence scores
    """
    divergence = np.zeros_like(price)
    
    for i in range(lookback, len(price)):
        # Price direction
        price_window = price[i - lookback:i]
        price_trend = (price[i] - price[i - lookback]) / price[i - lookback]
        
        # Indicator direction
        ind_window = indicator[i - lookback:i]
        ind_trend = (indicator[i] - indicator[i - lookback]) / (indicator[i - lookback] + 1e-10)
        
        # Bullish divergence: price down, indicator up
        if price_trend < 0 and ind_trend > 0:
            divergence[i] = abs(price_trend) + abs(ind_trend)
        
        # Bearish divergence: price up, indicator down
        elif price_trend > 0 and ind_trend < 0:
            divergence[i] = -(abs(price_trend) + abs(ind_trend))
    
    return divergence
