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

import json
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
    swing_points_js = ""
    trend_line_js = ""

    if pattern:
        # Extract trend line data (swings before the pattern showing prior trend)
        trend_swings = pattern.get('trend_swings', [])
        if trend_swings and len(trend_swings) >= 2:
            trend_line_data = []
            trend_markers = []

            for ts in trend_swings:
                time_val = ts.get('time')
                if hasattr(time_val, 'timestamp'):
                    time_val = int(time_val.timestamp())
                elif isinstance(time_val, str):
                    time_val = int(pd.to_datetime(time_val).timestamp())

                trend_line_data.append({
                    'time': time_val,
                    'value': float(ts.get('price', 0))
                })

                # Add marker with HH/HL/LH/LL label
                label = ts.get('label', '')
                trend_markers.append({
                    'time': time_val,
                    'position': 'aboveBar' if ts.get('type') == 'high' else 'belowBar',
                    'color': '#f59e0b',  # Amber color for trend
                    'shape': 'arrowDown' if ts.get('type') == 'high' else 'arrowUp',
                    'text': label
                })

            # Sort by time
            trend_line_data.sort(key=lambda x: x['time'])
            trend_markers.sort(key=lambda x: x['time'])

            trend_line_js = f"""
            // Prior trend line (amber/orange showing trend before pattern)
            const trendLine = chart.addLineSeries({{
                color: '#f59e0b',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            trendLine.setData({json.dumps(trend_line_data)});
            """
        # Extract trading levels - support both nested and flat formats
        trading_levels = pattern.get('trading_levels', {})

        # Flat format (from app.py): entry_price, stop_loss_price, take_profit_price
        entry = trading_levels.get('entry') or pattern.get('entry_price') or pattern.get('entry') or 0
        stop_loss = trading_levels.get('stop_loss') or pattern.get('stop_loss_price') or pattern.get('stop_loss') or 0
        tp1 = trading_levels.get('take_profit_1') or pattern.get('take_profit_price') or pattern.get('take_profit') or 0
        tp2 = trading_levels.get('take_profit_2') or pattern.get('take_profit_2') or 0

        if entry and stop_loss:
            # Trading level lines
            trade_zones_js = f"""
            // Entry line
            series.createPriceLine({{
                price: {entry},
                color: '#0ea5e9',
                lineWidth: 2,
                lineStyle: 0,
                axisLabelVisible: true,
                title: 'Entry'
            }});

            // Stop Loss line
            series.createPriceLine({{
                price: {stop_loss},
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

        # === POSITION BOX (Trade outcome visualization) ===
        position_box_js = ""
        position_box = pattern.get('position_box')
        if position_box:
            entry_time = position_box.get('entry_time')
            exit_time = position_box.get('exit_time')
            entry_price = position_box.get('entry_price', 0)
            exit_price = position_box.get('exit_price', 0)
            outcome = position_box.get('outcome', '')  # 'tp' or 'sl'
            is_long = position_box.get('is_long', True)
            sl_price = position_box.get('stop_loss', 0)
            tp_price = position_box.get('take_profit', 0)

            # Convert times to timestamps
            if hasattr(entry_time, 'timestamp'):
                entry_ts = int(entry_time.timestamp())
            else:
                entry_ts = int(pd.to_datetime(entry_time).timestamp())

            if hasattr(exit_time, 'timestamp'):
                exit_ts = int(exit_time.timestamp())
            else:
                exit_ts = int(pd.to_datetime(exit_time).timestamp())

            # Colors based on outcome
            if outcome == 'tp':
                border_color = '#22c55e'
                result_text = 'WIN'
            else:
                border_color = '#ef4444'
                result_text = 'LOSS'

            # Use baseline series for proper bounded zones
            # For LONG: Green above entry (profit), Red below entry (drawdown/risk)
            # For SHORT: Red above entry (drawdown/risk), Green below entry (profit)

            position_box_js = f"""
            // Position Box - Profit zone (green) and Risk zone (red)
            // Using baseline series with entry as the baseline

            const positionZone = chart.addBaselineSeries({{
                baseValue: {{ type: 'price', price: {entry_price} }},
                topLineColor: 'rgba(34, 197, 94, 0.8)',
                topFillColor1: 'rgba(34, 197, 94, 0.3)',
                topFillColor2: 'rgba(34, 197, 94, 0.1)',
                bottomLineColor: 'rgba(239, 68, 68, 0.8)',
                bottomFillColor1: 'rgba(239, 68, 68, 0.1)',
                bottomFillColor2: 'rgba(239, 68, 68, 0.3)',
                lastValueVisible: false,
                priceLineVisible: false,
            }});

            // Create position zone data - goes from entry to exit showing price path
            // We'll create intermediate points to show the actual price movement
            positionZone.setData([
                {{ time: {entry_ts}, value: {entry_price} }},
                {{ time: {exit_ts}, value: {exit_price} }}
            ]);

            // TP level line (bounded)
            const tpLevelLine = chart.addLineSeries({{
                color: 'rgba(34, 197, 94, 0.6)',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            tpLevelLine.setData([
                {{ time: {entry_ts}, value: {tp_price} }},
                {{ time: {exit_ts}, value: {tp_price} }}
            ]);

            // SL level line (bounded)
            const slLevelLine = chart.addLineSeries({{
                color: 'rgba(239, 68, 68, 0.6)',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            slLevelLine.setData([
                {{ time: {entry_ts}, value: {sl_price} }},
                {{ time: {exit_ts}, value: {sl_price} }}
            ]);

            // Entry level line
            const entryLevelLine = chart.addLineSeries({{
                color: 'rgba(14, 165, 233, 0.6)',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            entryLevelLine.setData([
                {{ time: {entry_ts}, value: {entry_price} }},
                {{ time: {exit_ts}, value: {entry_price} }}
            ]);

            // Entry to Exit line (shows the actual trade path)
            const tradePath = chart.addLineSeries({{
                color: '{border_color}',
                lineWidth: 3,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            tradePath.setData([
                {{ time: {entry_ts}, value: {entry_price} }},
                {{ time: {exit_ts}, value: {exit_price} }}
            ]);

            // Add exit marker
            series.setMarkers([
                ...series.markers() || [],
                {{
                    time: {exit_ts},
                    position: '{"aboveBar" if (outcome == "tp" and is_long) or (outcome == "sl" and not is_long) else "belowBar"}',
                    color: '{border_color}',
                    shape: '{"arrowUp" if outcome == "tp" else "arrowDown"}',
                    text: '{result_text}'
                }}
            ]);
            """

        # Extract swing points P1-P5 for pattern line
        swing_points = []
        for i in range(1, 6):
            time_key = f'p{i}_timestamp'
            price_key = f'p{i}_price'
            if time_key in pattern and price_key in pattern:
                time_val = pattern[time_key]
                # Convert timestamp
                if hasattr(time_val, 'timestamp'):
                    time_val = int(time_val.timestamp())
                elif isinstance(time_val, str):
                    time_val = int(pd.to_datetime(time_val).timestamp())
                swing_points.append({'time': time_val, 'value': float(pattern[price_key]), 'label': str(i)})

        if len(swing_points) >= 2:
            # Sort swing points by time - this is the chronological order
            swing_points_sorted = sorted(swing_points, key=lambda x: x['time'])

            # Line data for the pattern connection
            line_data = [{'time': p['time'], 'value': p['value']} for p in swing_points_sorted]

            # Create markers with sequential labels 1-5 in time order
            markers = []
            for i, p in enumerate(swing_points_sorted):
                # Determine if this is a local high or low
                is_high = False
                if i > 0 and i < len(swing_points_sorted) - 1:
                    is_high = p['value'] > swing_points_sorted[i-1]['value'] and p['value'] > swing_points_sorted[i+1]['value']
                elif i == 0:
                    is_high = len(swing_points_sorted) > 1 and p['value'] > swing_points_sorted[1]['value']
                else:
                    is_high = p['value'] > swing_points_sorted[-2]['value']

                markers.append({
                    'time': p['time'],
                    'position': 'aboveBar' if is_high else 'belowBar',
                    'color': '#2962FF',
                    'shape': 'circle',
                    'text': str(i + 1)  # Sequential 1, 2, 3, 4, 5 in time order
                })

            swing_points_js = f"""
            // Pattern connection line (blue dashed zigzag)
            const patternLine = chart.addLineSeries({{
                color: '#2962FF',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            patternLine.setData({json.dumps(line_data)});

            // Add numbered markers (1-5 in chronological order)
            series.setMarkers({json.dumps(markers)});
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

            {position_box_js}

            {trend_line_js}

            {swing_points_js}

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
