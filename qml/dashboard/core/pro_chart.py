"""
QML Trading Dashboard - Professional Chart Component v3
========================================================
Premium TradingView-grade chart visualization with precise pattern annotations.

Features:
- Numbered swing points with clear labels
- Trend line showing initial detection
- Position boxes (TP/SL zones) with proper styling
- Professional dark theme
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger


def _to_timestamp(value) -> int:
    """Convert various time formats to Unix timestamp (seconds)."""
    if value is None:
        return 0
    if isinstance(value, (pd.Timestamp, datetime)):
        return int(value.timestamp())
    elif isinstance(value, (int, float, np.integer, np.floating)):
        if value > 1e12:
            return int(value / 1000)
        return int(value)
    elif isinstance(value, str):
        try:
            return int(pd.to_datetime(value).timestamp())
        except:
            return 0
    return 0


def _prepare_candlestick_data(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to lightweight-charts candlestick format."""
    candles = []
    for _, row in df.iterrows():
        if 'time' in df.columns:
            time_val = row['time']
        elif 'timestamp' in df.columns:
            time_val = row['timestamp']
        else:
            time_val = row.name
        timestamp = _to_timestamp(time_val)
        if timestamp > 0:
            candles.append({
                "time": timestamp,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            })
    return candles


def _prepare_volume_data(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to lightweight-charts volume format."""
    if 'volume' not in df.columns:
        return []
    volume = []
    for _, row in df.iterrows():
        if 'time' in df.columns:
            time_val = row['time']
        elif 'timestamp' in df.columns:
            time_val = row['timestamp']
        else:
            time_val = row.name
        timestamp = _to_timestamp(time_val)
        if timestamp > 0:
            is_bullish = row['close'] >= row['open']
            volume.append({
                "time": timestamp,
                "value": float(row["volume"]),
                "color": 'rgba(38, 166, 154, 0.5)' if is_bullish else 'rgba(239, 83, 80, 0.5)'
            })
    return volume


def _prepare_pattern_data(pattern: Dict, df: pd.DataFrame) -> Dict:
    """Prepare all pattern data for the chart."""
    if not pattern:
        return {
            "swingPoints": [],
            "lineData": [],
            "positionBox": None,
            "trendLine": None,
            "bosLine": None,
            "isLong": True
        }

    is_long = "bullish" in pattern.get("pattern_type", "bullish").lower()
    swing_points = []
    line_data = []

    # Extract swing points (1-5)
    for i in range(1, 6):
        time_key = f"p{i}_timestamp"
        price_key = f"p{i}_price"
        time_val = pattern.get(time_key)
        price_val = pattern.get(price_key)

        if time_val and price_val:
            timestamp = _to_timestamp(time_val)
            price = float(price_val)

            swing_points.append({
                "time": timestamp,
                "price": price,
                "label": str(i),
                "index": i
            })

            line_data.append({
                "time": timestamp,
                "value": price
            })

    # Trend line (from trend start to P1)
    trend_line = None
    trend_start_time = pattern.get("trend_start_time")
    trend_start_price = pattern.get("trend_start_price")
    if trend_start_time and trend_start_price and swing_points:
        trend_line = {
            "start": {
                "time": _to_timestamp(trend_start_time),
                "price": float(trend_start_price)
            },
            "end": {
                "time": swing_points[0]["time"],
                "price": swing_points[0]["price"]
            }
        }

    # Break of Structure (BOS) line
    # For bullish: P4 breaks above P2 (previous high)
    # For bearish: P4 breaks below P2 (previous low)
    bos_line = None
    if len(swing_points) >= 4:
        p2 = swing_points[1]  # P2 is the level that gets broken
        p4 = swing_points[3]  # P4 is where it breaks
        bos_line = {
            "price": p2["price"],
            "startTime": p2["time"],
            "breakTime": p4["time"],
            "isLong": is_long
        }

    # Position box data
    entry = pattern.get("entry_price")
    sl = pattern.get("stop_loss") or pattern.get("stop_loss_price")
    tp1 = pattern.get("take_profit") or pattern.get("take_profit_price") or pattern.get("take_profit_1")
    tp2 = pattern.get("take_profit_2")

    # Use P5 time as entry time, or explicit entry_time
    entry_time = pattern.get("entry_time") or pattern.get("p5_timestamp")

    position_box = None
    if entry and sl and entry_time:
        position_box = {
            "direction": "LONG" if is_long else "SHORT",
            "entryPrice": float(entry),
            "entryTime": _to_timestamp(entry_time),
            "stopLoss": float(sl),
            "takeProfit1": float(tp1) if tp1 else None,
            "takeProfit2": float(tp2) if tp2 else None
        }

    return {
        "swingPoints": swing_points,
        "lineData": line_data,
        "positionBox": position_box,
        "trendLine": trend_line,
        "bosLine": bos_line,
        "isLong": is_long
    }


def _generate_chart_html(
    candles: List[Dict],
    volume: List[Dict],
    pattern_data: Dict,
    height: int,
    title: str
) -> str:
    """Generate complete HTML with premium chart and annotations.

    Uses the exact same pattern as the working tradingview_chart.py implementation.
    """

    # Build trade zones JS if pattern has position data
    trade_zones_js = ""
    position_box = pattern_data.get("positionBox")
    if position_box:
        entry = position_box.get("entryPrice", 0)
        sl = position_box.get("stopLoss", 0)
        tp1 = position_box.get("takeProfit1", 0)
        tp2 = position_box.get("takeProfit2", 0)

        trade_zones_js = f"""
            // Entry line
            series.createPriceLine({{
                price: {entry},
                color: '#2962FF',
                lineWidth: 2,
                lineStyle: 0,
                axisLabelVisible: true,
                title: 'Entry'
            }});

            // Stop Loss line
            series.createPriceLine({{
                price: {sl},
                color: '#ef4444',
                lineWidth: 2,
                lineStyle: 2,
                axisLabelVisible: true,
                title: 'SL'
            }});
        """

        if tp1:
            trade_zones_js += f"""
            // Take Profit 1
            series.createPriceLine({{
                price: {tp1},
                color: '#22c55e',
                lineWidth: 2,
                lineStyle: 1,
                axisLabelVisible: true,
                title: 'TP1'
            }});
            """

        if tp2:
            trade_zones_js += f"""
            // Take Profit 2
            series.createPriceLine({{
                price: {tp2},
                color: '#10b981',
                lineWidth: 2,
                lineStyle: 1,
                axisLabelVisible: true,
                title: 'TP2'
            }});
            """

    # Pattern line data for connecting swing points
    line_data = pattern_data.get("lineData", [])
    pattern_line_js = ""
    if line_data and len(line_data) > 1:
        pattern_line_js = f"""
            // Pattern connection line
            const patternLine = chart.addLineSeries({{
                color: '#2962FF',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            patternLine.setData({json.dumps(line_data)});
        """

    # Volume data string
    volume_js = ""
    if volume:
        volume_js = f"""
            const volumeSeries = chart.addHistogramSeries({{
                color: '#26a69a',
                priceFormat: {{ type: 'volume' }},
                priceScaleId: ''
            }});
            volumeSeries.priceScale().applyOptions({{
                scaleMargins: {{ top: 0.8, bottom: 0 }}
            }});
            volumeSeries.setData({json.dumps(volume)});
        """

    # Use exact same HTML structure as working tradingview_chart.py
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
                background: #0f172a;
            }}
            #chart-container {{
                width: 100%;
                height: {height}px;
                position: relative;
            }}
        </style>
    </head>
    <body>
        <div id="chart-container"></div>
        <script>
            const chartContainer = document.getElementById('chart-container');

            // Use fixed width if clientWidth is 0 (common in iframes)
            const chartWidth = chartContainer.clientWidth > 0 ? chartContainer.clientWidth : 1200;

            const chart = LightweightCharts.createChart(chartContainer, {{
                width: chartWidth,
                height: {height},
                layout: {{
                    background: {{ color: '#0f172a' }},
                    textColor: '#cbd5e1',
                }},
                grid: {{
                    vertLines: {{ color: '#1e293b' }},
                    horzLines: {{ color: '#1e293b' }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                }},
                rightPriceScale: {{
                    borderColor: '#1e293b',
                }},
                timeScale: {{
                    borderColor: '#1e293b',
                    timeVisible: true,
                    secondsVisible: false,
                }},
            }});

            // Candlestick series
            const series = chart.addCandlestickSeries({{
                upColor: '#22c55e',
                downColor: '#ef4444',
                borderUpColor: '#22c55e',
                borderDownColor: '#ef4444',
                wickUpColor: '#22c55e',
                wickDownColor: '#ef4444',
            }});

            const candleData = {json.dumps(candles)};
            series.setData(candleData);

            {trade_zones_js}

            {pattern_line_js}

            {volume_js}

            // Auto-fit content
            chart.timeScale().fitContent();

            // Handle window resize
            window.addEventListener('resize', () => {{
                chart.applyOptions({{
                    width: chartContainer.clientWidth,
                }});
            }});
        </script>
    </body>
    </html>
    """

    return html


def render_professional_chart(
    df: pd.DataFrame,
    pattern: Optional[Dict] = None,
    height: int = 600,
    title: str = "QML Pattern",
    key: Optional[str] = None
) -> None:
    """
    Render a premium-grade trading chart with pattern visualization.

    Args:
        df: OHLCV DataFrame
        pattern: Pattern data dict with keys:
            - pattern_type: "bullish_qml" or "bearish_qml"
            - p1_timestamp, p1_price through p5_timestamp, p5_price
            - entry_price, stop_loss, take_profit, take_profit_2
            - trend_start_time, trend_start_price (optional)
        height: Chart height in pixels
        title: Chart title
        key: Streamlit component key
    """
    logger.info(f"Rendering chart: {title}")
    logger.info(f"DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")

    # Normalize columns
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Prepare data
    candles = _prepare_candlestick_data(df)
    volume = _prepare_volume_data(df)
    pattern_data = _prepare_pattern_data(pattern, df)

    logger.info(f"Prepared: {len(candles)} candles, {len(volume)} volume bars, {len(pattern_data['swingPoints'])} swing points")
    if candles:
        logger.info(f"First candle: {candles[0]}")
        logger.info(f"Last candle: {candles[-1]}")

    # Generate and render
    html = _generate_chart_html(candles, volume, pattern_data, height, title)
    components.html(html, height=height + 20, scrolling=False)


def render_simple_chart(df: pd.DataFrame, height: int = 400, title: str = "Price Chart") -> None:
    """Render simple chart without pattern annotations."""
    render_professional_chart(df, pattern=None, height=height, title=title)


def render_mini_chart(df: pd.DataFrame, height: int = 200, title: str = "") -> None:
    """Render a compact mini chart for thumbnails/previews."""
    render_professional_chart(df, pattern=None, height=height, title=title)
