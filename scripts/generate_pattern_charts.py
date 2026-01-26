"""
Pattern Chart Generator
========================
Generates visual verification charts for all detected patterns.

Creates standalone HTML charts using TradingView Lightweight Charts.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.backtest_adapter import BacktestAdapter
from src.detection.pattern_scorer import PatternTier


def load_price_data(symbol: str, timeframe: str = "4h") -> Optional[pd.DataFrame]:
    """Load price data for a symbol."""
    data_path = PROJECT_ROOT / "data" / "processed" / symbol / f"{timeframe}_master.parquet"
    if not data_path.exists():
        print(f"  Warning: Data not found for {symbol}")
        return None

    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]

    # Ensure time column is tz-naive for comparison
    if 'time' in df.columns and df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_localize(None)

    return df


def simulate_trade_outcome(
    df: pd.DataFrame,
    entry_bar: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    is_long: bool,
    max_bars: int = 100
) -> Dict:
    """
    Simulate trade outcome by walking forward through price data.

    Returns dict with: outcome ('tp', 'sl', 'open'), exit_bar, exit_price, exit_time
    """
    for i in range(entry_bar, min(entry_bar + max_bars, len(df))):
        row = df.iloc[i]
        high = row['high']
        low = row['low']

        if is_long:
            # Check SL first (more conservative)
            if low <= stop_loss:
                return {
                    'outcome': 'sl',
                    'exit_bar': i,
                    'exit_price': stop_loss,
                    'exit_time': row['time'],
                }
            if high >= take_profit:
                return {
                    'outcome': 'tp',
                    'exit_bar': i,
                    'exit_price': take_profit,
                    'exit_time': row['time'],
                }
        else:  # Short
            # Check SL first
            if high >= stop_loss:
                return {
                    'outcome': 'sl',
                    'exit_bar': i,
                    'exit_price': stop_loss,
                    'exit_time': row['time'],
                }
            if low <= take_profit:
                return {
                    'outcome': 'tp',
                    'exit_bar': i,
                    'exit_price': take_profit,
                    'exit_time': row['time'],
                }

    # Trade still open
    return {
        'outcome': 'open',
        'exit_bar': min(entry_bar + max_bars, len(df) - 1),
        'exit_price': df.iloc[min(entry_bar + max_bars, len(df) - 1)]['close'],
        'exit_time': df.iloc[min(entry_bar + max_bars, len(df) - 1)]['time'],
    }


def find_prior_trend(df: pd.DataFrame, p1_bar: int, direction: str, lookback: int = 30) -> List[Dict]:
    """
    Find swing points showing prior trend before the pattern.

    For BULLISH pattern: prior downtrend (LH, LL sequence)
    For BEARISH pattern: prior uptrend (HH, HL sequence)
    """
    trend_swings = []
    start_bar = max(0, p1_bar - lookback)
    window_df = df.iloc[start_bar:p1_bar]

    if len(window_df) < 10:
        return trend_swings

    # Simple swing detection using local extrema
    highs = []
    lows = []

    for i in range(2, len(window_df) - 2):
        idx = start_bar + i
        row = window_df.iloc[i]

        # Local high
        if (row['high'] > window_df.iloc[i-1]['high'] and
            row['high'] > window_df.iloc[i-2]['high'] and
            row['high'] > window_df.iloc[i+1]['high'] and
            row['high'] > window_df.iloc[i+2]['high']):
            highs.append({
                'bar': idx,
                'price': row['high'],
                'time': row['time'],
                'type': 'high'
            })

        # Local low
        if (row['low'] < window_df.iloc[i-1]['low'] and
            row['low'] < window_df.iloc[i-2]['low'] and
            row['low'] < window_df.iloc[i+1]['low'] and
            row['low'] < window_df.iloc[i+2]['low']):
            lows.append({
                'bar': idx,
                'price': row['low'],
                'time': row['time'],
                'type': 'low'
            })

    # Combine and sort by time
    all_swings = highs + lows
    all_swings.sort(key=lambda x: x['bar'])

    # Take last 4-6 swings to show trend
    recent_swings = all_swings[-6:] if len(all_swings) >= 6 else all_swings

    # Label swings based on pattern direction
    if direction == 'BULLISH':
        # Prior downtrend: LH, LL
        for i, swing in enumerate(recent_swings):
            if swing['type'] == 'high':
                swing['label'] = 'LH'
            else:
                swing['label'] = 'LL'
    else:
        # Prior uptrend: HH, HL
        for i, swing in enumerate(recent_swings):
            if swing['type'] == 'high':
                swing['label'] = 'HH'
            else:
                swing['label'] = 'HL'

    return recent_swings


def generate_chart_html(
    df: pd.DataFrame,
    pattern: Dict,
    title: str,
    height: int = 600
) -> str:
    """Generate standalone HTML chart for a pattern."""

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

    # Prepare volume data
    volume_data = []
    for _, row in df.iterrows():
        time_val = row.get('time', row.name)
        if isinstance(time_val, pd.Timestamp):
            time_val = int(time_val.timestamp())
        volume_data.append({
            'time': time_val,
            'value': float(row['volume']),
            'color': '#26a69a80' if row['close'] >= row['open'] else '#ef535080'
        })

    # Pattern swing points P1-P5
    swing_points = []
    for i in range(1, 6):
        p_time = pattern.get(f'p{i}_time')
        p_price = pattern.get(f'p{i}_price')
        if p_time is not None and p_price is not None:
            if isinstance(p_time, pd.Timestamp):
                p_time = int(p_time.timestamp())
            swing_points.append({
                'time': p_time,
                'value': float(p_price),
                'label': str(i)
            })

    # Sort by time for correct display
    swing_points.sort(key=lambda x: x['time'])

    # Pattern line data
    pattern_line_data = [{'time': p['time'], 'value': p['value']} for p in swing_points]

    # Pattern markers
    pattern_markers = []
    for i, p in enumerate(swing_points):
        # Determine if high or low
        is_high = False
        if i > 0 and i < len(swing_points) - 1:
            is_high = p['value'] > swing_points[i-1]['value'] and p['value'] > swing_points[i+1]['value']
        elif i == 0:
            is_high = len(swing_points) > 1 and p['value'] > swing_points[1]['value']
        else:
            is_high = p['value'] > swing_points[-2]['value']

        pattern_markers.append({
            'time': p['time'],
            'position': 'aboveBar' if is_high else 'belowBar',
            'color': '#2962FF',
            'shape': 'circle',
            'text': str(i + 1)
        })

    # Prior trend line
    trend_line_data = []
    trend_markers = []
    trend_swings = pattern.get('trend_swings', [])
    for ts in trend_swings:
        t_time = ts.get('time')
        if isinstance(t_time, pd.Timestamp):
            t_time = int(t_time.timestamp())
        trend_line_data.append({
            'time': t_time,
            'value': float(ts.get('price', 0))
        })
        trend_markers.append({
            'time': t_time,
            'position': 'aboveBar' if ts.get('type') == 'high' else 'belowBar',
            'color': '#f59e0b',
            'shape': 'arrowDown' if ts.get('type') == 'high' else 'arrowUp',
            'text': ts.get('label', '')
        })

    # Trading levels
    entry = pattern.get('entry_price', 0)
    stop_loss = pattern.get('stop_loss', 0)
    take_profit = pattern.get('take_profit', 0)

    # Position box data
    entry_time = pattern.get('entry_time')
    exit_time = pattern.get('exit_time')
    exit_price = pattern.get('exit_price', 0)
    outcome = pattern.get('outcome', '')
    is_long = pattern.get('is_long', True)

    if isinstance(entry_time, pd.Timestamp):
        entry_ts = int(entry_time.timestamp())
    else:
        entry_ts = 0

    if isinstance(exit_time, pd.Timestamp):
        exit_ts = int(exit_time.timestamp())
    else:
        exit_ts = 0

    # Outcome color
    outcome_color = '#22c55e' if outcome == 'tp' else '#ef4444' if outcome == 'sl' else '#888888'
    outcome_text = 'WIN' if outcome == 'tp' else 'LOSS' if outcome == 'sl' else 'OPEN'

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 8px;
            border: 1px solid #2a2a4e;
        }}
        .title {{
            font-size: 1.4em;
            font-weight: 600;
            color: #00d4ff;
        }}
        .stats {{
            display: flex;
            gap: 30px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-label {{
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-value {{
            font-size: 1.2em;
            font-weight: 600;
        }}
        .stat-value.win {{ color: #22c55e; }}
        .stat-value.loss {{ color: #ef4444; }}
        .stat-value.tierA {{ color: #ffd700; }}
        .stat-value.tierB {{ color: #c0c0c0; }}
        .stat-value.tierC {{ color: #cd7f32; }}
        .chart-container {{
            background: #131722;
            border-radius: 8px;
            border: 1px solid #2a2a4e;
            overflow: hidden;
        }}
        #chart {{
            width: 100%;
            height: {height}px;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            padding: 15px 20px;
            background: #1a1a2e;
            border-top: 1px solid #2a2a4e;
            font-size: 0.85em;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 3px;
            border-radius: 2px;
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 15px;
            color: #00d4ff;
            text-decoration: none;
            font-size: 0.9em;
        }}
        .back-link:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <a href="index.html" class="back-link">&larr; Back to Gallery</a>

    <div class="header">
        <div class="title">{title}</div>
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Direction</div>
                <div class="stat-value">{pattern.get('direction', 'N/A')}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Tier</div>
                <div class="stat-value tier{pattern.get('tier', 'C')}">{pattern.get('tier', 'N/A')}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Score</div>
                <div class="stat-value">{pattern.get('total_score', 0):.3f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Outcome</div>
                <div class="stat-value {'win' if outcome == 'tp' else 'loss' if outcome == 'sl' else ''}">{outcome_text}</div>
            </div>
        </div>
    </div>

    <div class="chart-container">
        <div id="chart"></div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #f59e0b;"></div>
                <span>Prior Trend</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2962FF;"></div>
                <span>Pattern (P1-P5)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #0ea5e9;"></div>
                <span>Entry</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #22c55e;"></div>
                <span>Take Profit</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ef4444;"></div>
                <span>Stop Loss</span>
            </div>
        </div>
    </div>

    <script>
        const container = document.getElementById('chart');

        const chart = LightweightCharts.createChart(container, {{
            width: container.clientWidth,
            height: {height},
            layout: {{
                background: {{ type: 'solid', color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            rightPriceScale: {{
                borderColor: '#2a2a4e',
            }},
            timeScale: {{
                borderColor: '#2a2a4e',
                timeVisible: true,
                secondsVisible: false,
            }},
        }});

        // Candlestick series
        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderDownColor: '#ef5350',
            borderUpColor: '#26a69a',
            wickDownColor: '#ef5350',
            wickUpColor: '#26a69a',
        }});
        candleSeries.setData({json.dumps(candles)});

        // Volume series
        const volumeSeries = chart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: '',
            scaleMargins: {{ top: 0.85, bottom: 0 }},
        }});
        volumeSeries.setData({json.dumps(volume_data)});

        // Prior trend line (orange)
        {"" if not trend_line_data else f'''
        const trendLine = chart.addLineSeries({{
            color: '#f59e0b',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        trendLine.setData({json.dumps(sorted(trend_line_data, key=lambda x: x['time']))});
        '''}

        // Pattern connection line (blue dashed)
        {"" if not pattern_line_data else f'''
        const patternLine = chart.addLineSeries({{
            color: '#2962FF',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        patternLine.setData({json.dumps(pattern_line_data)});
        '''}

        // Trading level lines
        {f'''
        // Entry line
        candleSeries.createPriceLine({{
            price: {entry},
            color: '#0ea5e9',
            lineWidth: 2,
            lineStyle: 0,
            axisLabelVisible: true,
            title: 'Entry'
        }});

        // Stop Loss line
        candleSeries.createPriceLine({{
            price: {stop_loss},
            color: '#ef4444',
            lineWidth: 2,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'SL'
        }});

        // Take Profit line
        candleSeries.createPriceLine({{
            price: {take_profit},
            color: '#22c55e',
            lineWidth: 2,
            lineStyle: 1,
            axisLabelVisible: true,
            title: 'TP'
        }});
        ''' if entry and stop_loss and take_profit else ''}

        // Position box (trade path)
        {f'''
        const positionZone = chart.addBaselineSeries({{
            baseValue: {{ type: 'price', price: {entry} }},
            topLineColor: 'rgba(34, 197, 94, 0.8)',
            topFillColor1: 'rgba(34, 197, 94, 0.3)',
            topFillColor2: 'rgba(34, 197, 94, 0.1)',
            bottomLineColor: 'rgba(239, 68, 68, 0.8)',
            bottomFillColor1: 'rgba(239, 68, 68, 0.1)',
            bottomFillColor2: 'rgba(239, 68, 68, 0.3)',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        positionZone.setData([
            {{ time: {entry_ts}, value: {entry} }},
            {{ time: {exit_ts}, value: {exit_price} }}
        ]);
        ''' if entry_ts and exit_ts else ''}

        // Set markers for pattern points and trend
        const allMarkers = [
            ...{json.dumps(pattern_markers)},
            ...{json.dumps(trend_markers)}
        ].sort((a, b) => a.time - b.time);

        candleSeries.setMarkers(allMarkers);

        // Auto-fit
        chart.timeScale().fitContent();

        // Resize handler
        window.addEventListener('resize', () => {{
            chart.applyOptions({{ width: container.clientWidth }});
        }});
    </script>
</body>
</html>"""

    return html


def generate_index_html(patterns: List[Dict], output_dir: Path) -> str:
    """Generate gallery index page."""

    # Calculate stats
    total = len(patterns)
    wins = sum(1 for p in patterns if p.get('outcome') == 'tp')
    losses = sum(1 for p in patterns if p.get('outcome') == 'sl')
    open_trades = sum(1 for p in patterns if p.get('outcome') == 'open')
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    tier_counts = {'A': 0, 'B': 0, 'C': 0}
    for p in patterns:
        tier = p.get('tier', 'C')
        if tier in tier_counts:
            tier_counts[tier] += 1

    # Build pattern cards
    cards_html = ""
    for i, p in enumerate(patterns):
        outcome = p.get('outcome', 'open')
        outcome_class = 'win' if outcome == 'tp' else 'loss' if outcome == 'sl' else 'open'
        outcome_text = 'WIN' if outcome == 'tp' else 'LOSS' if outcome == 'sl' else 'OPEN'
        tier = p.get('tier', 'C')

        cards_html += f"""
        <a href="{p['filename']}" class="pattern-card {outcome_class}">
            <div class="card-header">
                <span class="symbol">{p['symbol']}</span>
                <span class="tier tier{tier}">{tier}</span>
            </div>
            <div class="card-body">
                <div class="direction">{p['direction']}</div>
                <div class="score">Score: {p['total_score']:.3f}</div>
                <div class="date">{p['detection_time'].strftime('%Y-%m-%d') if hasattr(p['detection_time'], 'strftime') else p['detection_time']}</div>
            </div>
            <div class="card-footer">
                <span class="outcome {outcome_class}">{outcome_text}</span>
            </div>
        </a>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QML Pattern Gallery - Phase 7.5 Visual Verification</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            padding: 30px;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            border: 1px solid #2a2a4e;
        }}
        h1 {{
            font-size: 2em;
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #888;
            margin-bottom: 25px;
        }}
        .stats-row {{
            display: flex;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
        }}
        .stat-box {{
            text-align: center;
            padding: 15px 25px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(0, 212, 255, 0.3);
        }}
        .stat-box.win {{ border-color: #22c55e; background: rgba(34, 197, 94, 0.1); }}
        .stat-box.loss {{ border-color: #ef4444; background: rgba(239, 68, 68, 0.1); }}
        .stat-value {{
            font-size: 2em;
            font-weight: 700;
        }}
        .stat-value.green {{ color: #22c55e; }}
        .stat-value.red {{ color: #ef4444; }}
        .stat-value.blue {{ color: #00d4ff; }}
        .stat-label {{
            font-size: 0.8em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 5px;
        }}
        .filters {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            padding: 8px 20px;
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 20px;
            color: #888;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter-btn:hover, .filter-btn.active {{
            border-color: #00d4ff;
            color: #00d4ff;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        .pattern-card {{
            background: #131722;
            border-radius: 10px;
            border: 2px solid #2a2a4e;
            text-decoration: none;
            color: inherit;
            transition: all 0.2s;
            overflow: hidden;
        }}
        .pattern-card:hover {{
            transform: translateY(-3px);
            border-color: #00d4ff;
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
        }}
        .pattern-card.win {{ border-color: #22c55e40; }}
        .pattern-card.win:hover {{ border-color: #22c55e; }}
        .pattern-card.loss {{ border-color: #ef444440; }}
        .pattern-card.loss:hover {{ border-color: #ef4444; }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #1a1a2e;
            border-bottom: 1px solid #2a2a4e;
        }}
        .symbol {{
            font-weight: 600;
            font-size: 1.1em;
            color: #00d4ff;
        }}
        .tier {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        .tierA {{ background: #ffd70030; color: #ffd700; }}
        .tierB {{ background: #c0c0c030; color: #c0c0c0; }}
        .tierC {{ background: #cd7f3230; color: #cd7f32; }}
        .card-body {{
            padding: 15px;
        }}
        .direction {{
            font-size: 0.9em;
            color: #888;
            margin-bottom: 8px;
        }}
        .score {{
            font-size: 1.1em;
            margin-bottom: 8px;
        }}
        .date {{
            font-size: 0.85em;
            color: #666;
        }}
        .card-footer {{
            padding: 12px 15px;
            background: #0a0a0f;
            border-top: 1px solid #2a2a4e;
        }}
        .outcome {{
            font-weight: 600;
            font-size: 0.9em;
        }}
        .outcome.win {{ color: #22c55e; }}
        .outcome.loss {{ color: #ef4444; }}
        .outcome.open {{ color: #888; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>QML Pattern Gallery</h1>
        <p class="subtitle">Phase 7.5 Visual Verification - {total} Patterns Detected</p>

        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-value blue">{total}</div>
                <div class="stat-label">Total Patterns</div>
            </div>
            <div class="stat-box win">
                <div class="stat-value green">{wins}</div>
                <div class="stat-label">Wins</div>
            </div>
            <div class="stat-box loss">
                <div class="stat-value red">{losses}</div>
                <div class="stat-label">Losses</div>
            </div>
            <div class="stat-box">
                <div class="stat-value green">{win_rate:.1%}</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">A:{tier_counts['A']} B:{tier_counts['B']} C:{tier_counts['C']}</div>
                <div class="stat-label">Tier Distribution</div>
            </div>
        </div>
    </div>

    <div class="filters">
        <button class="filter-btn active" onclick="filterPatterns('all')">All ({total})</button>
        <button class="filter-btn" onclick="filterPatterns('win')">Wins ({wins})</button>
        <button class="filter-btn" onclick="filterPatterns('loss')">Losses ({losses})</button>
        <button class="filter-btn" onclick="filterPatterns('tierA')">Tier A ({tier_counts['A']})</button>
        <button class="filter-btn" onclick="filterPatterns('tierB')">Tier B ({tier_counts['B']})</button>
        <button class="filter-btn" onclick="filterPatterns('tierC')">Tier C ({tier_counts['C']})</button>
    </div>

    <div class="grid">
        {cards_html}
    </div>

    <script>
        function filterPatterns(filter) {{
            const cards = document.querySelectorAll('.pattern-card');
            const buttons = document.querySelectorAll('.filter-btn');

            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            cards.forEach(card => {{
                if (filter === 'all') {{
                    card.style.display = 'block';
                }} else if (filter === 'win') {{
                    card.style.display = card.classList.contains('win') ? 'block' : 'none';
                }} else if (filter === 'loss') {{
                    card.style.display = card.classList.contains('loss') ? 'block' : 'none';
                }} else if (filter.startsWith('tier')) {{
                    const tier = filter.replace('tier', '');
                    const cardTier = card.querySelector('.tier').textContent;
                    card.style.display = cardTier === tier ? 'block' : 'none';
                }}
            }});
        }}
    </script>
</body>
</html>"""

    return html


def main():
    """Generate all pattern charts."""
    print("=" * 70)
    print("PATTERN CHART GENERATOR")
    print("=" * 70)

    # Load patterns
    patterns_df = pd.read_parquet(PROJECT_ROOT / "results" / "ml_training_patterns.parquet")
    print(f"Loaded {len(patterns_df)} patterns")

    # Create output directory
    output_dir = PROJECT_ROOT / "results" / "pattern_charts"
    output_dir.mkdir(exist_ok=True)

    # Load adapter for trading level calculation
    adapter = BacktestAdapter()

    # Process each pattern
    all_patterns = []
    symbol_counts = {}

    for idx, row in patterns_df.iterrows():
        symbol = row['symbol']
        print(f"\nProcessing {symbol} pattern {idx + 1}/{len(patterns_df)}...")

        # Load price data
        df = load_price_data(symbol)
        if df is None:
            continue

        # Get pattern details
        p5_bar = row['p5_bar']
        direction = row['direction']
        is_long = direction == 'BEARISH'  # BEARISH pattern = LONG trade

        # Calculate trading levels
        atr = row['atr_p5']
        p5_price = row['p5_price']
        p3_price = row['p3_price']

        if is_long:
            entry = p5_price - (0.1 * atr)
            stop_loss = p3_price - (0.5 * atr)
            take_profit = entry + (1.5 * abs(entry - stop_loss))
        else:
            entry = p5_price + (0.1 * atr)
            stop_loss = p3_price + (0.5 * atr)
            take_profit = entry - (1.5 * abs(entry - stop_loss))

        # Simulate trade outcome
        trade_result = simulate_trade_outcome(
            df=df,
            entry_bar=p5_bar + 1,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            is_long=is_long
        )

        # Get time values for swing points
        p1_time = df.iloc[row['p1_bar']]['time'] if row['p1_bar'] < len(df) else None
        p2_time = df.iloc[row['p2_bar']]['time'] if row['p2_bar'] < len(df) else None
        p3_time = df.iloc[row['p3_bar']]['time'] if row['p3_bar'] < len(df) else None
        p4_time = df.iloc[row['p4_bar']]['time'] if row['p4_bar'] < len(df) else None
        p5_time = df.iloc[row['p5_bar']]['time'] if row['p5_bar'] < len(df) else None

        # Find prior trend swings
        trend_swings = find_prior_trend(df, row['p1_bar'], direction)

        # Prepare chart window
        start_bar = max(0, row['p1_bar'] - 50)
        end_bar = min(len(df), trade_result['exit_bar'] + 20)
        chart_df = df.iloc[start_bar:end_bar].copy()

        # Build pattern dict for chart
        pattern_dict = {
            'symbol': symbol,
            'direction': direction,
            'tier': row['tier'],
            'total_score': row['total_score'],
            'detection_time': row['detection_time'],
            'p1_time': p1_time,
            'p1_price': row['p1_price'],
            'p2_time': p2_time,
            'p2_price': row['p2_price'],
            'p3_time': p3_time,
            'p3_price': row['p3_price'],
            'p4_time': p4_time,
            'p4_price': row['p4_price'],
            'p5_time': p5_time,
            'p5_price': row['p5_price'],
            'entry_price': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': df.iloc[p5_bar + 1]['time'] if p5_bar + 1 < len(df) else p5_time,
            'exit_time': trade_result['exit_time'],
            'exit_price': trade_result['exit_price'],
            'outcome': trade_result['outcome'],
            'is_long': is_long,
            'trend_swings': trend_swings,
        }

        # Generate filename
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        filename = f"{symbol}_{symbol_counts[symbol]:03d}.html"
        pattern_dict['filename'] = filename

        # Generate chart HTML
        title = f"{symbol} - {direction} - Tier {row['tier']}"
        chart_html = generate_chart_html(chart_df, pattern_dict, title)

        # Save chart
        chart_path = output_dir / filename
        with open(chart_path, 'w') as f:
            f.write(chart_html)

        all_patterns.append(pattern_dict)

        outcome_str = "WIN" if trade_result['outcome'] == 'tp' else "LOSS" if trade_result['outcome'] == 'sl' else "OPEN"
        print(f"  ✓ Saved {filename} - {outcome_str}")

    # Generate index page
    print(f"\nGenerating index page...")
    index_html = generate_index_html(all_patterns, output_dir)
    with open(output_dir / "index.html", 'w') as f:
        f.write(index_html)

    # Summary
    wins = sum(1 for p in all_patterns if p['outcome'] == 'tp')
    losses = sum(1 for p in all_patterns if p['outcome'] == 'sl')

    print(f"\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total Charts: {len(all_patterns)}")
    print(f"Wins: {wins}, Losses: {losses}")
    print(f"Win Rate: {wins/(wins+losses)*100:.1f}%" if wins + losses > 0 else "N/A")
    print(f"\nOutput: {output_dir}")
    print(f"Open: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
