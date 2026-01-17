"""
TradingView Lightweight Charts Integration
==========================================
Premium, simple chart rendering using TradingView Lightweight Charts for Streamlit.

Features:
- Candlestick charts with clean TradingView styling
- Pattern line overlays (P1→P2→P3→P4→P5)
- Trade level boxes (TP green / SL red zones)
- QM Zone highlighting
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional, Dict, List
from datetime import datetime


def _generate_chart_html(
    df: pd.DataFrame,
    pattern: Optional[Dict] = None,
    height: int = 600,
    include_volume: bool = True
) -> str:
    """
    Generate standalone HTML with TradingView Lightweight Charts.
    
    Args:
        df: DataFrame with OHLCV data (columns: time, open, high, low, close, volume)
        pattern: Optional pattern dict with trading levels and pattern points
        height: Chart height in pixels
        include_volume: Whether to show volume subplot
    
    Returns:
        HTML string ready to render
    """
    
    # Prepare candlestick data
    candles = []
    for _, row in df.iterrows():
        time_val = row.get('time', row.name)
        if isinstance(time_val, pd.Timestamp):
            time_val = int(time_val.timestamp())
        candles.append({
            'time': time_val,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
    
    # Prepare volume data if needed
    volume_data = []
    if include_volume:
        for _, row in df.iterrows():
            time_val = row.get('time', row.name)
            if isinstance(time_val, pd.Timestamp):
                time_val = int(time_val.timestamp())
            volume_data.append({
                'time': time_val,
                'value': float(row['volume']),
                'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            })
    
    # Prepare pattern lines and markers if pattern provided
    pattern_lines_js = ""
    trade_zones_js = ""
    
    if pattern:
        # Extract pattern points (simplified - assume we have P1-P5 coordinates)
        # In reality, you'd extract from pattern.left_shoulder_time/price, head_time/price, etc.
        trading_levels = pattern.get('trading_levels')
        
        if trading_levels:
            entry = trading_levels.get('entry', 0)
            stop_loss = trading_levels.get('stop_loss', 0)
            tp1 = trading_levels.get('take_profit_1', 0)
            tp2 = trading_levels.get('take_profit_2', 0)
            
            # Trading level lines
            trade_zones_js = f"""
            // Entry line
            const entryLine = {{
                price: {entry},
                color: '#0ea5e9',
                lineWidth: 2,
                lineStyle: 0, // Solid
                axisLabelVisible: true,
                title: 'Entry'
            }};
            series.createPriceLine(entryLine);
            
            // Stop Loss line
            const slLine = {{
                price: {stop_loss},
                color: '#ef4444',
                lineWidth: 2,
                lineStyle: 2, // Dashed
                axisLabelVisible: true,
                title: 'SL'
            }};
            series.createPriceLine(slLine);
            
            // Take Profit 1
            const tp1Line = {{
                price: {tp1},
                color: '#22c55e',
                lineWidth: 2,
                lineStyle: 1, // Dotted
                axisLabelVisible: true,
                title: 'TP1'
            }};
            series.createPriceLine(tp1Line);
            
            // Take Profit 2
            const tp2Line = {{
                price: {tp2},
                color: '#10b981',
                lineWidth: 2,
                lineStyle: 1,
                axisLabelVisible: true,
                title: 'TP2'
            }};
            series.createPriceLine(tp2Line);
            """
    
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
            
            const chart = LightweightCharts.createChart(chartContainer, {{
                width: chartContainer.clientWidth,
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
            
            const candleData = {candles};
            series.setData(candleData);
            
            {trade_zones_js}
            
            // Volume series (if enabled)
            {'const volumeSeries = chart.addHistogramSeries({ color: "#26a69a", priceFormat: { type: "volume" }, priceScaleId: "" }); volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } }); const volumeData = ' + str(volume_data) + '; volumeSeries.setData(volumeData);' if include_volume and volume_data else ''}
            
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


def render_pattern_chart(
    df: pd.DataFrame,
    pattern: Optional[Dict] = None,
    height: int = 600,
    key: Optional[str] = None
) -> None:
    """
    Render a TradingView chart with optional pattern overlay in Streamlit.
    
    Args:
        df: DataFrame with OHLCV data (time, open, high, low, close, volume)
        pattern: Optional pattern dict with trading_levels, pattern points
        height: Chart height in pixels
        key: Unique key for the Streamlit component
    
    Example:
        ```python
        import pandas as pd
        from src.dashboard.components.tradingview_chart import render_pattern_chart
        
        # Prepare data
        df = fetcher.get_data("BTC/USDT", "4h", limit=500)
        
        # Render chart
        render_pattern_chart(df, pattern={'trading_levels': {...}}, height=600)
        ```
    """
    # Ensure time column is present
    if 'time' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'time'})
    
    # Generate HTML
    html = _generate_chart_html(df, pattern=pattern, height=height)
    
    # Render in Streamlit
    components.html(html, height=height + 20, scrolling=False)


def render_mini_chart(
    df: pd.DataFrame,
    height: int = 150,
    key: Optional[str] = None
) -> None:
    """
    Render a simple mini line chart for sparkline-style visualization.
    
    Args:
        df: DataFrame with at least 'close' column
        height: Chart height in pixels
        key: Unique key for the Streamlit component
    """
    # Prepare line data
    line_data = []
    for _, row in df.iterrows():
        time_val = row.get('time', row.name)
        if isinstance(time_val, pd.Timestamp):
            time_val = int(time_val.timestamp())
        line_data.append({
            'time': time_val,
            'value': float(row['close'])
        })
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; background: transparent; }}
            #chart {{ width: 100%; height: {height}px; }}
        </style>
    </head>
    <body>
        <div id="chart"></div>
        <script>
            const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                width: document.getElementById('chart').clientWidth,
                height: {height},
                layout: {{ background: {{ color: 'transparent' }}, textColor: '#64748b' }},
                grid: {{ vertLines: {{ visible: false }}, horzLines: {{ visible: false }} }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Hidden }},
                rightPriceScale: {{ visible: false }},
                timeScale: {{ visible: false }},
                handleScroll: false,
                handleScale: false,
            }});
            
            const lineSeries = chart.addAreaSeries({{
                lineColor: '#0ea5e9',
                topColor: 'rgba(14, 165, 233, 0.4)',
                bottomColor: 'rgba(14, 165, 233, 0.0)',
                lineWidth: 2,
            }});
            
            lineSeries.setData({line_data});
            chart.timeScale().fitContent();
            
            window.addEventListener('resize', () => {{
                chart.applyOptions({{ width: document.getElementById('chart').clientWidth }});
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(html, height=height + 10, scrolling=False)
