#!/usr/bin/env python3
"""
Generate Visual Verification Charts for Phase 7.6 Patterns
==========================================================
Randomly samples patterns and generates HTML charts for visual inspection.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Project imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import load_master_data
from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
from src.detection.pattern_validator import PatternValidator, PatternValidationConfig, PatternDirection
from src.detection.trend_validator import TrendValidator, TrendValidationConfig
from src.detection.pattern_scorer import PatternScorer, PatternScoringConfig


@dataclass
class DetectedPattern:
    """Full pattern data for visualization."""
    symbol: str
    timeframe: str
    direction: str  # 'BULLISH' or 'BEARISH'
    tier: str
    score: float

    # Swing points (P1-P5)
    p1_time: datetime
    p1_price: float
    p1_idx: int
    p2_time: datetime
    p2_price: float
    p2_idx: int
    p3_time: datetime
    p3_price: float
    p3_idx: int
    p4_time: datetime
    p4_price: float
    p4_idx: int
    p5_time: datetime
    p5_price: float
    p5_idx: int

    # Prior trend swings
    trend_swings: List[Dict]
    trend_direction: str

    # Trade levels
    entry_price: float
    stop_loss: float
    take_profit: float

    # Context
    adx_at_pattern: float
    atr_at_pattern: float


def load_optimized_params() -> Dict[str, Any]:
    """Load optimized parameters from Phase 7.6."""
    params_path = PROJECT_ROOT / "results" / "optimization" / "best_params.json"
    with open(params_path) as f:
        data = json.load(f)
    return data['params']


def create_configs(params: Dict[str, Any]):
    """Create detector configs from parameter dict."""
    swing_config = HierarchicalSwingConfig(
        min_bar_separation=params.get('min_bar_separation', 5),
        min_move_atr=params.get('min_move_atr', 1.0),
        forward_confirm_pct=params.get('forward_confirm_pct', 0.3),
        lookback=params.get('lookback', 5),
        lookforward=params.get('lookforward', 5),
    )

    validation_config = PatternValidationConfig(
        p3_min_extension_atr=params.get('p3_min_extension_atr', 0.5),
        p3_max_extension_atr=params.get('p3_max_extension_atr', 5.0),
        p4_min_break_atr=params.get('p4_min_break_atr', 0.1),
        p5_max_symmetry_atr=params.get('p5_max_symmetry_atr', 2.0),
        min_pattern_bars=params.get('min_pattern_bars', 10),
    )

    trend_config = TrendValidationConfig(
        min_adx=params.get('min_adx', 20.0),
        min_trend_move_atr=params.get('min_trend_move_atr', 3.0),
        min_trend_swings=params.get('min_trend_swings', 3),
    )

    # Calculate weights to sum to 1.0
    head_weight = params.get('head_extension_weight', 0.25)
    bos_weight = params.get('bos_efficiency_weight', 0.20)
    shoulder_weight = 0.15
    volume_weight = 0.10
    path_weight = 0.10
    trend_weight = 0.10
    swing_weight = 1.0 - head_weight - bos_weight - shoulder_weight - volume_weight - path_weight - trend_weight

    scoring_config = PatternScoringConfig(
        head_extension_weight=head_weight,
        bos_efficiency_weight=bos_weight,
        shoulder_symmetry_weight=shoulder_weight,
        swing_significance_weight=swing_weight,
        volume_spike_weight=volume_weight,
        path_efficiency_weight=path_weight,
        trend_strength_weight=trend_weight,
    )

    return swing_config, validation_config, trend_config, scoring_config


def detect_patterns_for_symbol(
    symbol: str,
    timeframe: str,
    swing_config: HierarchicalSwingConfig,
    validation_config: PatternValidationConfig,
    trend_config: TrendValidationConfig,
    scoring_config: PatternScoringConfig,
) -> List[DetectedPattern]:
    """Detect patterns for a single symbol/timeframe."""

    patterns = []

    try:
        # Load data
        df = load_master_data(timeframe, symbol=symbol)
        df.columns = [c.lower() for c in df.columns]

        if len(df) < 100:
            return patterns

        # Initialize components
        swing_detector = HierarchicalSwingDetector(
            config=swing_config,
            symbol=symbol,
            timeframe=timeframe,
        )
        validator = PatternValidator(config=validation_config)
        trend_validator = TrendValidator(config=trend_config)
        scorer = PatternScorer(config=scoring_config)

        # Detect swings
        swings = swing_detector.detect(df)

        if len(swings) < 5:
            return patterns

        # Find patterns
        price_data = df['close'].values
        candidates = validator.find_patterns(swings, price_data)

        for pattern in candidates:
            if not pattern.is_valid:
                continue

            # Validate prior trend
            trend_result = trend_validator.validate(
                swings, pattern.p1.bar_index, df, pattern.direction
            )

            if not trend_result.is_valid:
                continue

            # Score the pattern
            score_result = scorer.score(pattern)

            # Calculate trade levels
            if pattern.direction == PatternDirection.BULLISH:
                entry = pattern.p5.price
                stop_loss = pattern.p3.price - (pattern.p3.price - pattern.p4.price) * 0.1
                take_profit = entry + (entry - stop_loss) * 2.0
            else:
                entry = pattern.p5.price
                stop_loss = pattern.p3.price + (pattern.p4.price - pattern.p3.price) * 0.1
                take_profit = entry - (stop_loss - entry) * 2.0

            # Get ADX at pattern
            adx_col = 'adx' if 'adx' in df.columns else 'ADX'
            atr_col = 'atr' if 'atr' in df.columns else 'ATR'
            adx_at_pattern = df.iloc[pattern.p5.bar_index][adx_col] if adx_col in df.columns else 0
            atr_at_pattern = df.iloc[pattern.p5.bar_index][atr_col] if atr_col in df.columns else 0

            # Build trend swing markers using find_trend_sequence
            trend_swings = []
            trend_swing_objects = trend_validator.find_trend_sequence(
                swings, pattern.p1.bar_index, trend_result.trend_direction or 'UP'
            )
            if trend_swing_objects:
                for swing in trend_swing_objects[-6:]:  # Last 6 swings
                    label = "HH" if swing.swing_type == 'HIGH' else "HL"
                    if trend_result.trend_direction == 'DOWN':
                        label = "LH" if swing.swing_type == 'HIGH' else "LL"
                    trend_swings.append({
                        'time': swing.timestamp,
                        'price': swing.price,
                        'idx': swing.bar_index,
                        'label': label,
                    })

            detected = DetectedPattern(
                symbol=symbol,
                timeframe=timeframe,
                direction=pattern.direction.value,
                tier=score_result.tier.value,
                score=score_result.total_score,
                p1_time=pattern.p1.timestamp,
                p1_price=pattern.p1.price,
                p1_idx=pattern.p1.bar_index,
                p2_time=pattern.p2.timestamp,
                p2_price=pattern.p2.price,
                p2_idx=pattern.p2.bar_index,
                p3_time=pattern.p3.timestamp,
                p3_price=pattern.p3.price,
                p3_idx=pattern.p3.bar_index,
                p4_time=pattern.p4.timestamp,
                p4_price=pattern.p4.price,
                p4_idx=pattern.p4.bar_index,
                p5_time=pattern.p5.timestamp,
                p5_price=pattern.p5.price,
                p5_idx=pattern.p5.bar_index,
                trend_swings=trend_swings,
                trend_direction=trend_result.trend_direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                adx_at_pattern=float(adx_at_pattern) if pd.notna(adx_at_pattern) else 0,
                atr_at_pattern=float(atr_at_pattern) if pd.notna(atr_at_pattern) else 0,
            )
            patterns.append(detected)

    except Exception as e:
        print(f"  Error processing {symbol} {timeframe}: {e}")

    return patterns


def generate_chart_html(pattern: DetectedPattern, df: pd.DataFrame, chart_id: int) -> str:
    """Generate HTML chart for a single pattern."""

    # Calculate display window
    start_idx = max(0, pattern.p1_idx - 50)
    end_idx = min(len(df) - 1, pattern.p5_idx + 20)

    window_df = df.iloc[start_idx:end_idx + 1].copy()

    # Format candlestick data
    candles = []
    for _, row in window_df.iterrows():
        time_val = row['time']
        if hasattr(time_val, 'timestamp'):
            time_unix = int(time_val.timestamp())
        else:
            time_unix = int(pd.Timestamp(time_val).timestamp())
        candles.append({
            'time': time_unix,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })

    # Pattern points
    pattern_points = [
        {'time': int(pd.Timestamp(pattern.p1_time).timestamp()), 'price': pattern.p1_price, 'label': '1'},
        {'time': int(pd.Timestamp(pattern.p2_time).timestamp()), 'price': pattern.p2_price, 'label': '2'},
        {'time': int(pd.Timestamp(pattern.p3_time).timestamp()), 'price': pattern.p3_price, 'label': '3'},
        {'time': int(pd.Timestamp(pattern.p4_time).timestamp()), 'price': pattern.p4_price, 'label': '4'},
        {'time': int(pd.Timestamp(pattern.p5_time).timestamp()), 'price': pattern.p5_price, 'label': '5'},
    ]

    # Trend swings
    trend_markers = []
    for swing in pattern.trend_swings:
        time_val = swing['time']
        if hasattr(time_val, 'timestamp'):
            time_unix = int(time_val.timestamp())
        else:
            time_unix = int(pd.Timestamp(time_val).timestamp())
        trend_markers.append({
            'time': time_unix,
            'price': float(swing['price']),
            'label': swing['label'],
        })

    # Colors based on direction
    is_bullish = pattern.direction == 'BULLISH'
    direction_color = '#22c55e' if is_bullish else '#ef4444'

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{pattern.symbol} {pattern.timeframe} - Pattern Verification</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0B1426;
            font-family: 'Inter', -apple-system, sans-serif;
            color: #e2e8f0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 12px;
            border: 1px solid #334155;
        }}
        .title {{
            font-size: 24px;
            font-weight: 700;
        }}
        .symbol {{ color: #3B82F6; }}
        .direction {{ color: {direction_color}; margin-left: 10px; }}
        .metadata {{
            display: flex;
            gap: 20px;
        }}
        .meta-item {{
            text-align: center;
            padding: 10px 20px;
            background: #0f172a;
            border-radius: 8px;
        }}
        .meta-label {{ font-size: 12px; color: #94a3b8; }}
        .meta-value {{ font-size: 18px; font-weight: 600; margin-top: 4px; }}
        .tier-A {{ color: #22c55e; }}
        .tier-B {{ color: #3B82F6; }}
        .tier-C {{ color: #f59e0b; }}
        .chart-container {{
            background: #0f172a;
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 20px;
        }}
        #chart {{
            width: 100%;
            height: 600px;
        }}
        .legend {{
            display: flex;
            gap: 30px;
            margin-top: 20px;
            padding: 15px;
            background: #1e293b;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }}
        .pattern-line {{ background: #2962FF; }}
        .trend-line {{ background: #f59e0b; }}
        .tp-zone {{ background: rgba(34, 197, 94, 0.5); }}
        .sl-zone {{ background: rgba(239, 68, 68, 0.5); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">
                <span class="symbol">{pattern.symbol}</span>
                <span style="color: #64748b;">{pattern.timeframe}</span>
                <span class="direction">{'BULLISH' if is_bullish else 'BEARISH'} QML</span>
            </div>
            <div class="metadata">
                <div class="meta-item">
                    <div class="meta-label">TIER</div>
                    <div class="meta-value tier-{pattern.tier}">{pattern.tier}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">SCORE</div>
                    <div class="meta-value">{pattern.score:.1%}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">ADX</div>
                    <div class="meta-value">{pattern.adx_at_pattern:.1f}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">R:R</div>
                    <div class="meta-value">1:2</div>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div id="chart"></div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color pattern-line"></div>
                <span>Pattern (P1→P5)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color trend-line"></div>
                <span>Prior Trend</span>
            </div>
            <div class="legend-item">
                <div class="legend-color tp-zone"></div>
                <span>Take Profit Zone</span>
            </div>
            <div class="legend-item">
                <div class="legend-color sl-zone"></div>
                <span>Stop Loss Zone</span>
            </div>
        </div>
    </div>

    <script>
        const candles = {json.dumps(candles)};
        const patternPoints = {json.dumps(pattern_points)};
        const trendMarkers = {json.dumps(trend_markers)};
        const entry = {pattern.entry_price};
        const stopLoss = {pattern.stop_loss};
        const takeProfit = {pattern.take_profit};
        const isBullish = {'true' if is_bullish else 'false'};

        // Create chart
        const chartContainer = document.getElementById('chart');
        const chart = LightweightCharts.createChart(chartContainer, {{
            width: chartContainer.clientWidth,
            height: 600,
            layout: {{
                background: {{ type: 'solid', color: '#0f172a' }},
                textColor: '#94a3b8',
            }},
            grid: {{
                vertLines: {{ color: '#1e293b' }},
                horzLines: {{ color: '#1e293b' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            rightPriceScale: {{
                borderColor: '#334155',
            }},
            timeScale: {{
                borderColor: '#334155',
                timeVisible: true,
            }},
        }});

        // Candlestick series
        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderUpColor: '#22c55e',
            borderDownColor: '#ef4444',
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
        }});
        candleSeries.setData(candles);

        // Position zones using baseline series
        const lastTime = candles[candles.length - 1].time;

        // TP Zone (green)
        const tpData = candles.map(c => ({{
            time: c.time,
            value: c.close >= Math.min(entry, takeProfit) && c.close <= Math.max(entry, takeProfit) ? c.close : entry
        }}));

        // Entry line
        const entryLine = chart.addLineSeries({{
            color: '#0ea5e9',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            priceLineVisible: false,
        }});
        entryLine.setData([
            {{ time: patternPoints[4].time, value: entry }},
            {{ time: lastTime, value: entry }}
        ]);

        // SL line
        const slLine = chart.addLineSeries({{
            color: '#ef4444',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            priceLineVisible: false,
        }});
        slLine.setData([
            {{ time: patternPoints[4].time, value: stopLoss }},
            {{ time: lastTime, value: stopLoss }}
        ]);

        // TP line
        const tpLine = chart.addLineSeries({{
            color: '#22c55e',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            priceLineVisible: false,
        }});
        tpLine.setData([
            {{ time: patternPoints[4].time, value: takeProfit }},
            {{ time: lastTime, value: takeProfit }}
        ]);

        // Prior trend line (orange)
        if (trendMarkers.length >= 2) {{
            const trendLine = chart.addLineSeries({{
                color: '#f59e0b',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                priceLineVisible: false,
            }});
            trendLine.setData(trendMarkers.map(m => ({{ time: m.time, value: m.price }})));

            // Trend markers
            const trendMarkerData = trendMarkers.map(m => ({{
                time: m.time,
                position: m.label.includes('H') ? 'aboveBar' : 'belowBar',
                color: '#f59e0b',
                shape: 'circle',
                text: m.label,
            }}));
            candleSeries.setMarkers(trendMarkerData);
        }}

        // Pattern line (blue dashed)
        const patternLine = chart.addLineSeries({{
            color: '#2962FF',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            priceLineVisible: false,
        }});
        patternLine.setData(patternPoints.map(p => ({{ time: p.time, value: p.price }})));

        // Pattern point markers
        const patternMarkers = patternPoints.map((p, i) => ({{
            time: p.time,
            position: (i === 1 || i === 3) ? (isBullish ? 'belowBar' : 'aboveBar') : (isBullish ? 'aboveBar' : 'belowBar'),
            color: '#2962FF',
            shape: 'circle',
            text: p.label,
        }}));

        // Combine markers (trend + pattern)
        const allMarkers = [
            ...trendMarkers.map(m => ({{
                time: m.time,
                position: m.label.includes('H') ? 'aboveBar' : 'belowBar',
                color: '#f59e0b',
                shape: 'circle',
                text: m.label,
            }})),
            ...patternMarkers
        ].sort((a, b) => a.time - b.time);

        candleSeries.setMarkers(allMarkers);

        // Fit content
        chart.timeScale().fitContent();

        // Handle resize
        window.addEventListener('resize', () => {{
            chart.applyOptions({{ width: chartContainer.clientWidth }});
        }});
    </script>
</body>
</html>'''

    return html


def generate_index_html(patterns: List[DetectedPattern]) -> str:
    """Generate gallery index page."""

    cards_html = ""
    for i, p in enumerate(patterns, 1):
        direction_color = '#22c55e' if p.direction == 'BULLISH' else '#ef4444'
        tier_color = {'A': '#22c55e', 'B': '#3B82F6', 'C': '#f59e0b'}.get(p.tier, '#94a3b8')

        cards_html += f'''
        <a href="random_sample_{i:03d}.html" class="card">
            <div class="card-header">
                <span class="symbol">{p.symbol}</span>
                <span class="timeframe">{p.timeframe}</span>
            </div>
            <div class="card-body">
                <div class="direction" style="color: {direction_color};">{p.direction}</div>
                <div class="score">{p.score:.1%}</div>
            </div>
            <div class="card-footer">
                <span class="tier" style="background: {tier_color};">Tier {p.tier}</span>
                <span class="adx">ADX: {p.adx_at_pattern:.1f}</span>
            </div>
        </a>
        '''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Phase 7.6 Visual Verification - 10 Random Patterns</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0B1426;
            font-family: 'Inter', -apple-system, sans-serif;
            color: #e2e8f0;
            padding: 40px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #3B82F6;
        }}
        .subtitle {{
            text-align: center;
            color: #94a3b8;
            margin-bottom: 40px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .card {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            text-decoration: none;
            color: inherit;
            transition: all 0.2s;
        }}
        .card:hover {{
            border-color: #3B82F6;
            transform: translateY(-4px);
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }}
        .symbol {{
            font-size: 20px;
            font-weight: 700;
            color: #3B82F6;
        }}
        .timeframe {{
            color: #64748b;
            font-size: 14px;
        }}
        .card-body {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .direction {{
            font-weight: 600;
        }}
        .score {{
            font-size: 24px;
            font-weight: 700;
        }}
        .card-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .tier {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            color: white;
        }}
        .adx {{
            color: #94a3b8;
            font-size: 14px;
        }}
        .summary {{
            max-width: 1200px;
            margin: 0 auto 40px;
            padding: 20px;
            background: #1e293b;
            border-radius: 12px;
            display: flex;
            justify-content: space-around;
        }}
        .summary-item {{
            text-align: center;
        }}
        .summary-label {{
            color: #94a3b8;
            font-size: 14px;
        }}
        .summary-value {{
            font-size: 24px;
            font-weight: 700;
            color: #3B82F6;
        }}
    </style>
</head>
<body>
    <h1>Phase 7.6 Visual Verification</h1>
    <p class="subtitle">10 Randomly Sampled Patterns - Click to view full chart</p>

    <div class="summary">
        <div class="summary-item">
            <div class="summary-label">TIER A</div>
            <div class="summary-value">{len([p for p in patterns if p.tier == 'A'])}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">TIER B</div>
            <div class="summary-value">{len([p for p in patterns if p.tier == 'B'])}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">TIER C</div>
            <div class="summary-value">{len([p for p in patterns if p.tier == 'C'])}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">AVG SCORE</div>
            <div class="summary-value">{np.mean([p.score for p in patterns]):.1%}</div>
        </div>
    </div>

    <div class="grid">
        {cards_html}
    </div>
</body>
</html>'''

    return html


def main():
    print("=" * 70)
    print("PHASE 7.6 VISUAL VERIFICATION")
    print("=" * 70)

    # Load optimized parameters
    print("\n1. Loading optimized parameters...")
    params = load_optimized_params()
    print(f"   Loaded {len(params)} parameters")

    # Create configs
    swing_config, validation_config, trend_config, scoring_config = create_configs(params)

    # Get available symbols
    data_dir = PROJECT_ROOT / "data" / "processed"
    symbols = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    timeframes = ['1h', '4h', '1d']

    print(f"\n2. Detecting patterns across {len(symbols)} symbols...")

    all_patterns = []

    for symbol in symbols:
        for tf in timeframes:
            # Check if data exists
            data_path = data_dir / symbol / f"{tf}_master.parquet"
            if not data_path.exists():
                continue

            patterns = detect_patterns_for_symbol(
                symbol, tf,
                swing_config, validation_config, trend_config, scoring_config
            )

            if patterns:
                print(f"   {symbol} {tf}: {len(patterns)} patterns")
                all_patterns.extend(patterns)

    print(f"\n   Total patterns found: {len(all_patterns)}")

    # Count by tier
    tier_counts = {}
    for p in all_patterns:
        tier_counts[p.tier] = tier_counts.get(p.tier, 0) + 1
    print(f"   Tier breakdown: {tier_counts}")

    # Sample 10 patterns: 3 Tier A, 4 Tier B, 3 Tier C
    print("\n3. Sampling 10 patterns (3 A, 4 B, 3 C)...")

    tier_a = [p for p in all_patterns if p.tier == 'A']
    tier_b = [p for p in all_patterns if p.tier == 'B']
    tier_c = [p for p in all_patterns if p.tier == 'C']

    random.seed(42)  # Reproducible

    selected = []
    used_symbols = set()

    def select_from_tier(tier_patterns, count, selected, used_symbols):
        """Select patterns ensuring symbol diversity."""
        random.shuffle(tier_patterns)
        for p in tier_patterns:
            if len([s for s in selected if s.tier == p.tier]) >= count:
                break
            # Allow max 2 from same symbol
            symbol_count = len([s for s in selected if s.symbol == p.symbol])
            if symbol_count < 2:
                selected.append(p)
                used_symbols.add(p.symbol)
        return selected

    selected = select_from_tier(tier_a, 3, selected, used_symbols)
    selected = select_from_tier(tier_b, 4, selected, used_symbols)
    selected = select_from_tier(tier_c, 3, selected, used_symbols)

    print(f"   Selected {len(selected)} patterns")

    # Print selection
    print("\n   Selected patterns:")
    for i, p in enumerate(selected, 1):
        print(f"   {i:2}. {p.symbol:10} {p.timeframe:3} {p.direction:8} Tier {p.tier} Score={p.score:.1%}")

    # Create output directory
    output_dir = PROJECT_ROOT / "results" / "visual_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate charts
    print("\n4. Generating HTML charts...")

    for i, pattern in enumerate(selected, 1):
        # Load data for chart
        df = load_master_data(pattern.timeframe, symbol=pattern.symbol)
        df.columns = [c.lower() for c in df.columns]

        html = generate_chart_html(pattern, df, i)

        chart_path = output_dir / f"random_sample_{i:03d}.html"
        with open(chart_path, 'w') as f:
            f.write(html)
        print(f"   Saved: {chart_path.name}")

    # Generate index
    print("\n5. Generating index page...")
    index_html = generate_index_html(selected)
    index_path = output_dir / "verification_index.html"
    with open(index_path, 'w') as f:
        f.write(index_html)
    print(f"   Saved: {index_path}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nOpen: file://{index_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
