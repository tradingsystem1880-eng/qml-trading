"""
Technical Indicators for QML Trading System
============================================
Optimized implementations of technical indicators used
throughout the detection and feature engineering pipeline.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def calculate_atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """
    Calculate Average True Range (ATR).
    
    Uses Wilder's smoothing method (exponential moving average).
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period (default: 14)
        
    Returns:
        Array of ATR values
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    # First value has no previous close
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Wilder's smoothing (EMA with alpha = 1/period)
    alpha = 1.0 / period
    atr = np.zeros_like(true_range)
    
    # Initialize with SMA for first 'period' values
    atr[:period] = np.nan
    atr[period - 1] = np.mean(true_range[:period])
    
    # EMA for rest
    for i in range(period, len(true_range)):
        atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i - 1]
    
    return atr


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
    df[column_name] = calculate_atr(
        df["high"].values,
        df["low"].values,
        df["close"].values,
        period
    )
    return df


def calculate_rsi(
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        close: Array of close prices
        period: RSI period (default: 14)
        
    Returns:
        Array of RSI values (0-100)
    """
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    alpha = 1.0 / period
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initialize with SMA
    avg_gain[:period] = np.nan
    avg_loss[:period] = np.nan
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])
    
    # EMA for rest
    for i in range(period, len(close)):
        avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]
    
    # Calculate RSI
    rs = np.divide(
        avg_gain,
        avg_loss,
        out=np.full_like(avg_gain, 100.0),
        where=avg_loss != 0
    )
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_obv(
    close: NDArray[np.float64],
    volume: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        close: Array of close prices
        volume: Array of volume
        
    Returns:
        Array of OBV values
    """
    # Calculate price direction
    direction = np.sign(np.diff(close, prepend=close[0]))
    direction[0] = 0  # First bar has no direction
    
    # Calculate OBV
    obv = np.cumsum(direction * volume)
    
    return obv


def calculate_ema(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        data: Array of values
        period: EMA period
        
    Returns:
        Array of EMA values
    """
    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(data)
    
    # Initialize with first value
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    return ema


def calculate_sma(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data: Array of values
        period: SMA period
        
    Returns:
        Array of SMA values
    """
    sma = np.full_like(data, np.nan)
    
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    
    return sma


def calculate_bollinger_bands(
    close: NDArray[np.float64],
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        close: Array of close prices
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(close, period)
    
    # Calculate rolling standard deviation
    rolling_std = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        rolling_std[i] = np.std(close[i - period + 1:i + 1])
    
    upper = middle + (std_dev * rolling_std)
    lower = middle - (std_dev * rolling_std)
    
    return upper, middle, lower


def calculate_macd(
    close: NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        close: Array of close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = calculate_ema(close, fast_period)
    slow_ema = calculate_ema(close, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """
    Calculate Average Directional Index (ADX).
    
    Used for measuring trend strength.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ADX period
        
    Returns:
        Array of ADX values (0-100)
    """
    # Calculate +DM and -DM
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Smooth +DM and -DM
    alpha = 1.0 / period
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period - 1] = np.sum(plus_dm[:period])
    minus_dm_smooth[period - 1] = np.sum(minus_dm[:period])
    
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = np.divide(
        100 * plus_dm_smooth,
        atr * period,
        out=np.zeros_like(plus_dm_smooth),
        where=atr * period != 0
    )
    minus_di = np.divide(
        100 * minus_dm_smooth,
        atr * period,
        out=np.zeros_like(minus_dm_smooth),
        where=atr * period != 0
    )
    
    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.divide(
        100 * di_diff,
        di_sum,
        out=np.zeros_like(di_diff),
        where=di_sum != 0
    )
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[:period * 2 - 1] = np.nan
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, len(dx)):
        adx[i] = ((adx[i - 1] * (period - 1)) + dx[i]) / period
    
    return adx


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
    return calculate_sma(volume, period)


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
    atr_pct = atr / close * 100
    
    percentile = np.full_like(atr_pct, np.nan)
    
    for i in range(lookback, len(atr_pct)):
        window = atr_pct[i - lookback:i]
        current = atr_pct[i]
        percentile[i] = (np.sum(window < current) / lookback) * 100
    
    return percentile


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
        price_window = price[i - lookback:i + 1]
        ind_window = indicator[i - lookback:i + 1]
        
        # Price making new low but indicator not
        price_new_low = price[i] <= np.min(price_window[:-1])
        ind_not_low = indicator[i] > np.min(ind_window[:-1])
        
        # Price making new high but indicator not
        price_new_high = price[i] >= np.max(price_window[:-1])
        ind_not_high = indicator[i] < np.max(ind_window[:-1])
        
        if price_new_low and ind_not_low:
            # Bullish divergence
            divergence[i] = (indicator[i] - np.min(ind_window[:-1])) / np.std(ind_window)
        elif price_new_high and ind_not_high:
            # Bearish divergence
            divergence[i] = -(np.max(ind_window[:-1]) - indicator[i]) / np.std(ind_window)
    
    return divergence

