"""
Enhanced Chart Renderer with Advanced Indicators
=================================================
Adds moving averages, Fibonacci, and volume profile to TradingView charts.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger


def calculate_fibonacci_levels(df: pd.DataFrame, swing_high_idx: int, swing_low_idx: int) -> List[Dict]:
    """
    Calculate Fibonacci retracement levels.
    
    Args:
        df: OHLCV DataFrame
        swing_high_idx: Index of swing high
        swing_low_idx: Index of swing low
        
    Returns:
        List of Fibonacci levels with prices
    """
    high_price = df.iloc[swing_high_idx]['high']
    low_price = df.iloc[swing_low_idx]['low']
    
    diff = high_price - low_price
    
    levels = {
        "0.0": low_price,
        "23.6": low_price + diff * 0.236,
        "38.2": low_price + diff * 0.382,
        "50.0": low_price + diff * 0.500,
        "61.8": low_price + diff * 0.618,
        "78.6": low_price + diff * 0.786,
        "100.0": high_price
    }
    
    fib_levels = []
    for label, price in levels.items():
        fib_levels.append({
            "level": label,
            "price": float(price),
            "color": "#fbbf24" if label in ["38.2", "50.0", "61.8"] else "#94a3b8"
        })
    
    logger.debug(f"Calculated {len(fib_levels)} Fibonacci levels")
    return fib_levels


def add_moving_averages(chart_config: Dict, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> Dict:
    """
    Add moving average lines to chart config.
    
    Args:
        chart_config: Existing chart configuration
        df: OHLCV DataFrame
        periods: MA periods to add
        
    Returns:
        Updated chart config with MA lines
    """
    colors = {
        20: "#fbbf24",   # Yellow/gold
        50: "#f59e0b",   # Orange
        200: "#ef4444"   # Red
    }
    
    for period in periods:
        if len(df) >= period:
            # Calculate EMA
            ma_values = df['close'].ewm(span=period, adjust=False).mean()
            
            # Format for TradingView
            ma_data = []
            for idx, row in df.iterrows():
                if idx >= period - 1:  # Only show after enough data
                    timestamp = int(row.name.timestamp()) if hasattr(row.name, 'timestamp') else int(row.name)
                    ma_data.append({
                        "time": timestamp,
                        "value": float(ma_values.iloc[idx])
                    })
            
            # Add to lines
            if 'lines' not in chart_config:
                chart_config['lines'] = []
            
            chart_config['lines'].append({
                "data": ma_data,
                "color": colors.get(period, "#38BDF8"),
                "width": 2,
                "style": 0,  # Solid line
                "title": f"EMA {period}"
            })
            
            logger.debug(f"Added EMA {period} with {len(ma_data)} points")
    
    return chart_config


def add_volume_profile(chart_config: Dict, df: pd.DataFrame, bins: int = 20) -> Dict:
    """
    Add volume profile to chart.
    
    Args:
        chart_config: Existing chart configuration
        df: OHLCV DataFrame  
        bins: Number of price bins
        
    Returns:
        Updated chart config with volume profile
    """
    if 'volume' not in df.columns:
        return chart_config
    
    # Calculate price range
    price_min = df['low'].min()
    price_max = df['high'].max()
    
    # Create bins
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    
    # Aggregate volume by price
    volumes_by_price = np.zeros(bins)
    
    for _, row in df.iterrows():
        # Find which bin this candle's volume belongs to
        avg_price = (row['high'] + row['low']) / 2
        bin_idx = np.digitize(avg_price, bin_edges) - 1
        bin_idx = max(0, min(bins - 1, bin_idx))  # Clamp to valid range
        
        volumes_by_price[bin_idx] += row['volume']
    
    # Find POC (Point of Control - highest volume bin)
    poc_idx = np.argmax(volumes_by_price)
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
    
    # Add POC line
    if 'zones' not in chart_config:
        chart_config['zones'] = []
    
    chart_config['zones'].append({
        "high": float(poc_price + (bin_edges[1] - bin_edges[0]) / 2),
        "low": float(poc_price - (bin_edges[1] - bin_edges[0]) / 2),
        "color": "#8b5cf6",  # Purple for POC
        "title": "POC"
    })
    
    logger.debug(f"Added volume profile with POC at ${poc_price:.2f}")
    
    return chart_config
