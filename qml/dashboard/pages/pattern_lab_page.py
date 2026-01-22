"""
Pattern Lab Page - Interactive pattern analysis and visualization.

IMPORTANT: The chart visualization functions (find_swing_points, map_to_geometry)
are LOCKED and must NOT be modified without explicit user approval.
See CLAUDE.md "Pattern Visualization - FINAL SPECIFICATION" for details.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from theme import ARCTIC_PRO, TYPOGRAPHY

# Phase 8 detection integration
from src.detection import (
    QMLPatternDetector, QMLConfig, QMLPattern, PatternDirection,
    SwingAlgorithm, MarketRegimeDetector, RegimeResult
)

# Phase 8.6: Real data loading
from src.data.loader import load_ohlcv, get_available_symbols, get_available_timeframes


# =============================================================================
# LOCKED FUNCTIONS - DO NOT MODIFY
# =============================================================================

def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> List[Dict]:
    """Find swing highs and lows in price data.

    LOCKED - DO NOT MODIFY without explicit user approval.

    Args:
        df: OHLCV DataFrame with high/low columns
        lookback: Number of bars to look back/forward for swing detection

    Returns:
        List of swing point dicts sorted by time: {time, price, type, idx}
    """
    swings = []
    time_col = 'time' if 'time' in df.columns else 'timestamp' if 'timestamp' in df.columns else None

    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if df.iloc[i]['high'] <= df.iloc[i-j]['high'] or df.iloc[i]['high'] <= df.iloc[i+j]['high']:
                is_swing_high = False
                break

        if is_swing_high:
            t = df.iloc[i][time_col] if time_col else df.index[i]
            swings.append({'time': pd.to_datetime(t), 'price': float(df.iloc[i]['high']), 'type': 'high', 'idx': i})

        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if df.iloc[i]['low'] >= df.iloc[i-j]['low'] or df.iloc[i]['low'] >= df.iloc[i+j]['low']:
                is_swing_low = False
                break

        if is_swing_low:
            t = df.iloc[i][time_col] if time_col else df.index[i]
            swings.append({'time': pd.to_datetime(t), 'price': float(df.iloc[i]['low']), 'type': 'low', 'idx': i})

    return sorted(swings, key=lambda x: x['time'])


def map_to_geometry(pattern: Dict, df: pd.DataFrame) -> Dict:
    """Find actual swing points from price data for pattern visualization.

    LOCKED - DO NOT MODIFY without explicit user approval.

    Args:
        pattern: Pattern dict with head_time, detection_time, entry_price, etc.
        df: OHLCV DataFrame

    Returns:
        Geometry dict with p1-p5 timestamps/prices, trade levels, trend_swings
    """
    geo = {}

    # Get the head time as anchor point
    head_time = pattern.get('head_time')
    detection_time = pattern.get('detection_time')

    if not head_time or not detection_time:
        # Fallback: just use what we have
        geo['entry_price'] = pattern.get('entry_price')
        geo['stop_loss_price'] = pattern.get('stop_loss')
        geo['take_profit_price'] = pattern.get('take_profit')
        return geo

    head_ts = pd.to_datetime(head_time)
    det_ts = pd.to_datetime(detection_time)

    # Make timezone-naive
    if head_ts.tzinfo:
        head_ts = head_ts.tz_localize(None)
    if det_ts.tzinfo:
        det_ts = det_ts.tz_localize(None)

    # Find all swing points in the data
    all_swings = find_swing_points(df, lookback=2)

    # Find swing closest to head time (this is P3 - the QM point)
    p3 = None
    min_diff = float('inf')
    for s in all_swings:
        s_time = s['time']
        if s_time.tzinfo:
            s_time = s_time.tz_localize(None)
        diff = abs((s_time - head_ts).total_seconds())
        if diff < min_diff:
            min_diff = diff
            p3 = s

    if not p3:
        geo['entry_price'] = pattern.get('entry_price')
        geo['stop_loss_price'] = pattern.get('stop_loss')
        geo['take_profit_price'] = pattern.get('take_profit')
        return geo

    is_bullish = 'bullish' in pattern.get('type', '').lower()

    # For bullish: P3 is a LOW, P1 is LOW before, P2/P4 are HIGHs
    # For bearish: P3 is a HIGH, P1 is HIGH before, P2/P4 are LOWs

    p3_time = p3['time']
    if p3_time.tzinfo:
        p3_time = p3_time.tz_localize(None)

    # Find swings BEFORE P3
    before_p3 = [s for s in all_swings if s['idx'] < p3['idx']]
    # Find swings AFTER P3
    after_p3 = [s for s in all_swings if s['idx'] > p3['idx']]

    if is_bullish:
        # P1 = swing LOW before P3, P2 = swing HIGH between P1 and P3
        # P4 = swing HIGH after P3, P5 = detection/entry
        lows_before = [s for s in before_p3 if s['type'] == 'low'][-2:] if before_p3 else []
        highs_before = [s for s in before_p3 if s['type'] == 'high'][-2:] if before_p3 else []
        highs_after = [s for s in after_p3 if s['type'] == 'high'][:2] if after_p3 else []

        # P1 = first significant low before head
        p1 = lows_before[-1] if lows_before else None
        # P2 = high between P1 and P3
        p2 = highs_before[-1] if highs_before else None
        # P4 = first high after P3
        p4 = highs_after[0] if highs_after else None
    else:
        # Bearish: P1 = HIGH, P2 = LOW, P3 = higher HIGH, P4 = LOW, P5 = entry
        highs_before = [s for s in before_p3 if s['type'] == 'high'][-2:] if before_p3 else []
        lows_before = [s for s in before_p3 if s['type'] == 'low'][-2:] if before_p3 else []
        lows_after = [s for s in after_p3 if s['type'] == 'low'][:2] if after_p3 else []

        p1 = highs_before[-1] if highs_before else None
        p2 = lows_before[-1] if lows_before else None
        p4 = lows_after[0] if lows_after else None

    # P5 = detection point
    p5_time = det_ts
    p5_price = float(pattern.get('entry_price', 0))

    # Build geometry - only include points we actually found
    if p1:
        geo['p1_timestamp'] = p1['time']
        geo['p1_price'] = p1['price']
    if p2:
        geo['p2_timestamp'] = p2['time']
        geo['p2_price'] = p2['price']
    if p3:
        geo['p3_timestamp'] = p3['time']
        geo['p3_price'] = p3['price']
    if p4:
        geo['p4_timestamp'] = p4['time']
        geo['p4_price'] = p4['price']
    if p5_time and p5_price:
        geo['p5_timestamp'] = p5_time
        geo['p5_price'] = p5_price

    # Add trade levels
    geo['entry_price'] = pattern.get('entry_price')
    geo['stop_loss_price'] = pattern.get('stop_loss')
    geo['take_profit_price'] = pattern.get('take_profit')

    # === TREND VISUALIZATION ===
    # Find swing points BEFORE P1 to show the prior trend
    # For bullish QML: prior trend is DOWNTREND (LH/LL)
    # For bearish QML: prior trend is UPTREND (HH/HL)
    trend_swings = []

    if p1:
        # Get swings before P1
        swings_before_p1 = [s for s in all_swings if s['idx'] < p1['idx']]

        # Get last few highs and lows before P1
        highs_trend = [s for s in swings_before_p1 if s['type'] == 'high'][-4:]
        lows_trend = [s for s in swings_before_p1 if s['type'] == 'low'][-4:]

        if is_bullish:
            # Prior trend is DOWNTREND - look for Lower Highs (LH) and Lower Lows (LL)
            # Label highs
            for i, h in enumerate(highs_trend):
                if i == 0:
                    label = "H"  # First high, reference
                elif h['price'] < highs_trend[i-1]['price']:
                    label = "LH"  # Lower High
                else:
                    label = "HH"  # Higher High (not expected in downtrend)
                trend_swings.append({
                    'time': h['time'],
                    'price': h['price'],
                    'type': 'high',
                    'label': label
                })
            # Label lows
            for i, l in enumerate(lows_trend):
                if i == 0:
                    label = "L"  # First low, reference
                elif l['price'] < lows_trend[i-1]['price']:
                    label = "LL"  # Lower Low
                else:
                    label = "HL"  # Higher Low (not expected in downtrend)
                trend_swings.append({
                    'time': l['time'],
                    'price': l['price'],
                    'type': 'low',
                    'label': label
                })
        else:
            # Prior trend is UPTREND - look for Higher Highs (HH) and Higher Lows (HL)
            # Label highs
            for i, h in enumerate(highs_trend):
                if i == 0:
                    label = "H"
                elif h['price'] > highs_trend[i-1]['price']:
                    label = "HH"  # Higher High
                else:
                    label = "LH"  # Lower High (not expected in uptrend)
                trend_swings.append({
                    'time': h['time'],
                    'price': h['price'],
                    'type': 'high',
                    'label': label
                })
            # Label lows
            for i, l in enumerate(lows_trend):
                if i == 0:
                    label = "L"
                elif l['price'] > lows_trend[i-1]['price']:
                    label = "HL"  # Higher Low
                else:
                    label = "LL"  # Lower Low (not expected in uptrend)
                trend_swings.append({
                    'time': l['time'],
                    'price': l['price'],
                    'type': 'low',
                    'label': label
                })

        # Sort by time
        trend_swings.sort(key=lambda x: x['time'])

    geo['trend_swings'] = trend_swings

    return geo


# =============================================================================
# END LOCKED FUNCTIONS
# =============================================================================


# =============================================================================
# DETECTION INTEGRATION
# =============================================================================

def render_detection_sidebar() -> dict:
    """Render sidebar controls for detection settings."""
    with st.sidebar:
        st.markdown(f'<div style="color: {ARCTIC_PRO["text_secondary"]}; font-weight: 600; margin-bottom: 8px;">Detection Settings</div>', unsafe_allow_html=True)

        algorithm = st.selectbox(
            "Swing Algorithm",
            options=["rolling", "savgol", "fractal", "wavelet"],
            index=0,
            key="detection_algorithm"
        )

        swing_lookback = st.slider("Swing Lookback", 3, 15, 5, key="swing_lookback")
        min_head_ext = st.slider("Min Head Extension (ATR)", 0.1, 2.0, 0.5, 0.1, key="min_head_ext")
        bos_requirement = st.selectbox("BOS Requirement", [1, 2], index=0, key="bos_req")
        regime_filter = st.checkbox("Filter Trending Markets", value=True, key="regime_filter")

        return {
            'algorithm': algorithm,
            'swing_lookback': swing_lookback,
            'min_head_ext': min_head_ext,
            'bos_requirement': bos_requirement,
            'regime_filter': regime_filter
        }


def run_pattern_detection(df: pd.DataFrame, settings: dict) -> tuple:
    """Run detection with current settings."""
    algo_map = {
        'rolling': SwingAlgorithm.ROLLING,
        'savgol': SwingAlgorithm.SAVGOL,
        'fractal': SwingAlgorithm.FRACTAL,
        'wavelet': SwingAlgorithm.WAVELET
    }

    config = QMLConfig(
        swing_algorithm=algo_map[settings['algorithm']],
        swing_lookback=settings['swing_lookback'],
        min_head_extension_atr=settings['min_head_ext'],
        bos_requirement=settings['bos_requirement'],
        require_trend_alignment=settings['regime_filter']
    )

    detector = QMLPatternDetector(config)
    patterns = detector.detect(df)

    regime_detector = MarketRegimeDetector()
    regime = regime_detector.get_regime(df)

    return patterns, regime


def render_regime_indicator(regime: RegimeResult) -> None:
    """Display market regime as colored indicator."""
    regime_styles = {
        'RANGING': (ARCTIC_PRO['success'], '🟢', 'Ideal for QML'),
        'TRENDING': (ARCTIC_PRO['danger'], '🔴', 'Avoid QML'),
        'VOLATILE': (ARCTIC_PRO['warning'], '🟡', 'Caution'),
        'EXTREME': (ARCTIC_PRO['warning'], '🟠', 'Overbought/Oversold')
    }
    color, icon, desc = regime_styles.get(regime.regime.value, (ARCTIC_PRO['text_muted'], '⚪', 'Unknown'))

    html = f'<div style="background: {ARCTIC_PRO["bg_card"]}; border: 1px solid {ARCTIC_PRO["border"]}; '
    html += f'border-radius: 8px; padding: 12px; margin-bottom: 12px;">'
    html += f'<div style="display: flex; align-items: center; gap: 8px;">'
    html += f'<span style="font-size: 1.5rem;">{icon}</span>'
    html += f'<div>'
    html += f'<div style="color: {color}; font-weight: 600; font-size: 1rem;">{regime.regime.value}</div>'
    html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: 0.75rem;">{desc}</div>'
    html += f'</div></div>'
    html += f'<div style="display: flex; gap: 16px; margin-top: 8px; font-size: 0.75rem; color: {ARCTIC_PRO["text_muted"]};">'
    html += f'<span>ADX: <span style="color: {ARCTIC_PRO["text_secondary"]};">{regime.adx:.1f}</span></span>'
    html += f'<span>RSI: <span style="color: {ARCTIC_PRO["text_secondary"]};">{regime.rsi:.1f}</span></span>'
    html += f'<span>Vol: <span style="color: {ARCTIC_PRO["text_secondary"]};">{regime.volatility_percentile:.0%}</span></span>'
    html += f'</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_pattern_metrics(patterns: List) -> None:
    """Display pattern count metrics."""
    bullish = len([p for p in patterns if p.direction == PatternDirection.BULLISH])
    bearish = len([p for p in patterns if p.direction == PatternDirection.BEARISH])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patterns", len(patterns))
    with col2:
        st.metric("Bullish (Short)", bullish)
    with col3:
        st.metric("Bearish (Long)", bearish)


def render_pattern_list(patterns: List) -> Optional[QMLPattern]:
    """Display selectable pattern list. Returns selected pattern."""
    selected = None

    for i, p in enumerate(patterns):
        direction_color = ARCTIC_PRO['danger'] if p.direction == PatternDirection.BULLISH else ARCTIC_PRO['success']
        direction_label = "SHORT" if p.direction == PatternDirection.BULLISH else "LONG"

        with st.expander(f"{p.id} | {direction_label} | Strength: {p.pattern_strength:.2f}", expanded=(i == 0)):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Direction:** <span style='color: {direction_color};'>{direction_label}</span>", unsafe_allow_html=True)
                st.write(f"**Head Extension:** {p.head_extension_atr:.2f} ATR")
                st.write(f"**Shoulder Symmetry:** {p.shoulder_symmetry:.2f} ATR")
                st.write(f"**BOS Count:** {p.bos_count}")
                st.write(f"**Market Regime:** {p.market_regime}")

            with col2:
                st.write("**Trading Levels:**")
                st.write(f"Entry: ${p.entry_price:,.2f}")
                st.write(f"Stop Loss: ${p.stop_loss:,.2f}")
                st.write(f"TP1: ${p.take_profit_1:,.2f}")
                if p.take_profit_2:
                    st.write(f"TP2: ${p.take_profit_2:,.2f}")

                risk = abs(p.stop_loss - p.entry_price)
                reward = abs(p.take_profit_1 - p.entry_price)
                rr = reward / risk if risk > 0 else 0
                st.write(f"**R:R:** 1:{rr:.1f}")

            if st.button(f"View on Chart", key=f"view_{p.id}"):
                selected = p

    return selected


def generate_synthetic_ohlcv(symbol: str, bars: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demonstration.

    Creates realistic-looking price data with proper candle structure.

    Args:
        symbol: Trading pair for setting base price
        bars: Number of candles to generate

    Returns:
        DataFrame with time, open, high, low, close, volume columns
    """
    import random
    from datetime import datetime, timedelta

    # Set base price based on symbol
    if 'BTC' in symbol:
        base_price = 97000
        volatility = 0.015
    elif 'ETH' in symbol:
        base_price = 3300
        volatility = 0.02
    elif 'SOL' in symbol:
        base_price = 200
        volatility = 0.025
    else:
        base_price = 100
        volatility = 0.02

    data = []
    current_price = base_price
    now = datetime.now()

    for i in range(bars):
        # Time going backwards
        time = now - timedelta(hours=(bars - i) * 4)

        # Generate realistic candle
        change = random.gauss(0, volatility)
        open_price = current_price
        close_price = current_price * (1 + change)

        # High and low
        wick_up = abs(random.gauss(0, volatility * 0.5))
        wick_down = abs(random.gauss(0, volatility * 0.5))

        high_price = max(open_price, close_price) * (1 + wick_up)
        low_price = min(open_price, close_price) * (1 - wick_down)

        # Volume
        volume = random.uniform(100, 1000) * (base_price / 100)

        data.append({
            'time': time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

        current_price = close_price

    return pd.DataFrame(data)


def load_ohlcv_data(symbol: str, timeframe: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Load OHLCV data from parquet files, or generate synthetic if not found.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Candle timeframe (e.g., "4h")
        days: Number of days of data to load

    Returns:
        DataFrame with OHLCV data (real or synthetic)
    """
    # Try loading from data directory
    data_dir = Path("data")
    parquet_pattern = f"*{symbol}*{timeframe}*.parquet"

    parquet_files = list(data_dir.glob(parquet_pattern))
    if parquet_files:
        try:
            df = pd.read_parquet(parquet_files[0])
            df.columns = df.columns.str.lower()

            # Filter to requested days
            if 'time' in df.columns:
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
                df = df[df['time'] >= cutoff]

            if len(df) > 0:
                return df
        except Exception:
            pass

    # Generate synthetic data for demonstration
    bars = days * 6 if timeframe == '4h' else days * 24 if timeframe == '1h' else days
    return generate_synthetic_ohlcv(symbol, min(bars, 500))


def render_pattern_lab_page() -> None:
    """Render the Pattern Lab analysis page with chart visualization."""

    # Detection settings sidebar
    settings = render_detection_sidebar()

    # Page header with status
    header_html = '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">'
    header_html += f'<div style="font-family: {TYPOGRAPHY["font_display"]}; font-size: {TYPOGRAPHY["size_xl"]}; font-weight: {TYPOGRAPHY["weight_semibold"]}; color: {ARCTIC_PRO["text_primary"]};">Pattern Lab</div>'
    header_html += '<div class="status-badge neutral"><span class="status-dot"></span>Ready to Scan</div>'
    header_html += '</div>'
    st.markdown(header_html, unsafe_allow_html=True)

    # Control panel in premium style
    control_html = '<div class="panel">'
    control_html += '<div class="panel-header">'
    control_html += '<span class="panel-title">Chart Settings</span>'
    control_html += '</div></div>'
    st.markdown(control_html, unsafe_allow_html=True)

    # Data source selection row
    src_col1, src_col2 = st.columns([1, 3])
    with src_col1:
        data_source = st.selectbox(
            "Data Source",
            ["Real Data (Binance)", "Sample Data"],
            key="data_source"
        )

    # Symbol/timeframe row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Show more symbols for real data
        if "Real Data" in data_source:
            symbols = [s.replace("/", "") for s in get_available_symbols()[:6]]
        else:
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        symbol = st.selectbox(
            "Symbol",
            symbols,
            key="pattern_symbol"
        )

    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            ["1h", "4h", "1d"],
            index=1,
            key="pattern_timeframe"
        )

    with col3:
        days = st.slider(
            "Days",
            min_value=7,
            max_value=180,
            value=90,
            key="pattern_days"
        )

    with col4:
        st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        scan_clicked = st.button("🔍 Scan Patterns", type="primary", use_container_width=True)

    # Chart container with premium header
    chart_panel = '<div class="panel" style="margin-top: 16px;">'
    chart_panel += '<div class="panel-header">'
    chart_panel += f'<span class="panel-title">{symbol} · {timeframe.upper()}</span>'
    chart_panel += '<div class="panel-actions">'
    chart_panel += '<span class="panel-action-btn">Fullscreen</span>'
    chart_panel += '</div></div>'
    chart_panel += '</div>'
    st.markdown(chart_panel, unsafe_allow_html=True)

    # Load data based on source selection
    use_real_data = "Real Data" in data_source

    if use_real_data:
        # Use the new loader for real data
        with st.spinner("Loading market data..."):
            df, source_info = load_ohlcv(symbol, timeframe, days, source="auto")
        if df is None:
            df = generate_synthetic_ohlcv(symbol, days * 6 if timeframe == '4h' else days * 24)
            source_info = "Fallback: Sample Data"
    else:
        df = load_ohlcv_data(symbol, timeframe, days)
        source_info = "Sample Data"

    if df is not None and len(df) > 0:
        # Show data info with source
        data_info = f'<div style="display: flex; gap: 24px; margin-bottom: 8px; font-size: {TYPOGRAPHY["size_xs"]}; color: {ARCTIC_PRO["text_muted"]};">'
        data_info += f'<span>Candles: <span style="color: {ARCTIC_PRO["text_secondary"]}; font-family: {TYPOGRAPHY["font_mono"]};">{len(df)}</span></span>'
        if 'close' in df.columns:
            latest_price = df['close'].iloc[-1]
            data_info += f'<span>Latest: <span style="color: {ARCTIC_PRO["text_secondary"]}; font-family: {TYPOGRAPHY["font_mono"]};">${latest_price:,.2f}</span></span>'
        # Color based on data source
        source_color = ARCTIC_PRO["success"] if "Binance" in source_info or "Local" in source_info else ARCTIC_PRO["warning"]
        data_info += f'<span style="color: {source_color};">{source_info}</span>'
        data_info += '</div>'
        st.markdown(data_info, unsafe_allow_html=True)

        # Import and use the chart component
        try:
            from src.dashboard.components.tradingview_chart import render_pattern_chart
            render_pattern_chart(df, pattern=None, height=500, key="pattern_lab_chart")
        except ImportError:
            # Fallback: show simple line chart with Plotly
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['time'] if 'time' in df.columns else df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing_line_color=ARCTIC_PRO['success'],
                decreasing_line_color=ARCTIC_PRO['danger'],
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=450,
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    showline=False,
                    rangeslider=dict(visible=False),
                    tickfont=dict(color=ARCTIC_PRO['text_muted'], size=10),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=ARCTIC_PRO['border'],
                    showline=False,
                    tickfont=dict(color=ARCTIC_PRO['text_muted'], size=10),
                    tickprefix='$',
                ),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        placeholder = f'<div style="height: 400px; display: flex; align-items: center; justify-content: center; '
        placeholder += f'background: {ARCTIC_PRO["bg_secondary"]}; border-radius: 8px; color: {ARCTIC_PRO["text_muted"]};">'
        placeholder += 'Loading chart data...'
        placeholder += '</div>'
        st.markdown(placeholder, unsafe_allow_html=True)

    # Run detection when scan clicked
    if scan_clicked:
        with st.spinner("Scanning for patterns..."):
            patterns, regime = run_pattern_detection(df, settings)
            st.session_state['detected_patterns'] = patterns
            st.session_state['market_regime'] = regime
            if patterns:
                st.success(f"Found {len(patterns)} pattern(s)")
            else:
                st.warning("No patterns detected with current settings")

    # Pattern Results Panel
    patterns_panel = '<div class="panel" style="margin-top: 16px;">'
    patterns_panel += '<div class="panel-header">'
    patterns_panel += '<span class="panel-title">Detected Patterns</span>'

    # Get stored results
    patterns = st.session_state.get('detected_patterns', [])
    regime = st.session_state.get('market_regime', None)

    patterns_panel += f'<span style="font-size: 0.75rem; color: {ARCTIC_PRO["text_muted"]};">{len(patterns)} found</span>'
    patterns_panel += '</div></div>'
    st.markdown(patterns_panel, unsafe_allow_html=True)

    if regime:
        render_regime_indicator(regime)

    if patterns:
        render_pattern_metrics(patterns)
        selected_pattern = render_pattern_list(patterns)

        # If pattern selected, update chart (future integration)
        if selected_pattern:
            st.session_state['selected_pattern'] = selected_pattern
    else:
        placeholder = f'<div style="padding: 32px; text-align: center; color: {ARCTIC_PRO["text_muted"]};">'
        placeholder += 'Click "Scan Patterns" to detect QML patterns in the current view'
        placeholder += '</div>'
        st.markdown(placeholder, unsafe_allow_html=True)
