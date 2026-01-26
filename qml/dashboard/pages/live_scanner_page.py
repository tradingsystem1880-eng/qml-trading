"""
Live Scanner Page - Multi-symbol pattern scanner with auto-refresh.

Scans all 32 symbols across 1h, 4h, 1d timeframes for QML patterns.
Features:
- Manual and auto-refresh (15 min interval)
- Multi-timeframe alignment detection
- Sortable results table with full pattern details
- Click-to-view pattern chart visualization
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time

# Theme imports
from theme import ARCTIC_PRO, TYPOGRAPHY

# Detection imports
from src.detection import (
    QMLPatternDetector,
    QMLConfig,
    QMLPattern,
    PatternDirection,
    SwingAlgorithm,
    MarketRegimeDetector,
)

# Data loading
from src.data.loader import load_ohlcv

# Chart visualization
from src.dashboard.components.tradingview_chart import render_pattern_chart
from components.pattern_display import pattern_to_chart_dict

# MT5 Export
from src.export.mt5_exporter import export_pattern_to_mt5, check_mt5_installed


# =============================================================================
# CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TIMEFRAMES = ["1h", "4h", "1d"]
AUTO_REFRESH_INTERVAL_MS = 15 * 60 * 1000  # 15 minutes

# Detection config (Phase 7.9 baseline)
DEFAULT_CONFIG = QMLConfig(
    swing_algorithm=SwingAlgorithm.ROLLING,
    swing_lookback=5,
    min_head_extension_atr=0.5,
    bos_requirement=1,
    require_trend_alignment=True,
)


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def get_all_symbols() -> List[str]:
    """Get list of all symbols with data available."""
    if not DATA_DIR.exists():
        return []

    symbols = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            symbols.append(d.name)

    return sorted(symbols)


def load_symbol_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a symbol/timeframe from parquet."""
    data_path = DATA_DIR / symbol / f"{timeframe}_master.parquet"

    if not data_path.exists():
        return None

    try:
        df = pd.read_parquet(data_path)
        df.columns = df.columns.str.lower()

        # Rename timestamp to time if needed
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'time'})

        # Filter to last 180 days for efficiency
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
            if df['time'].dt.tz is not None:
                cutoff = cutoff.tz_localize('UTC')
            df = df[df['time'] >= cutoff]

        return df
    except Exception as e:
        return None


def scan_symbol(
    symbol: str,
    timeframe: str,
    config: QMLConfig,
    min_quality: float = 0.0
) -> List[Dict]:
    """Scan a single symbol/timeframe for patterns."""
    df = load_symbol_data(symbol, timeframe)
    if df is None or len(df) < 100:
        return []

    try:
        detector = QMLPatternDetector(config)
        patterns = detector.detect(df)

        # Get market regime
        regime_detector = MarketRegimeDetector()
        regime = regime_detector.get_regime(df)

        results = []
        for p in patterns:
            if p.pattern_strength < min_quality:
                continue

            # Calculate R:R
            risk = abs(p.stop_loss - p.entry_price)
            reward = abs(p.take_profit_1 - p.entry_price)
            rr = reward / risk if risk > 0 else 0

            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': 'LONG' if p.direction == PatternDirection.BEARISH else 'SHORT',
                'quality': p.pattern_strength,
                'regime': regime.regime.value if regime else 'UNKNOWN',
                'entry': p.entry_price,
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit_1,
                'rr_ratio': rr,
                'detected_at': p.detection_time if hasattr(p, 'detection_time') else datetime.now(),
                'pattern_id': p.id,
                'head_extension': p.head_extension_atr,
                'bos_count': p.bos_count,
                'pattern_obj': p,  # Store full pattern for chart visualization
            })

        return results
    except Exception as e:
        return []


def scan_all_symbols(
    timeframes: List[str],
    min_quality: float = 0.0,
    progress_callback=None
) -> Tuple[List[Dict], Dict]:
    """Scan all symbols across specified timeframes."""
    symbols = get_all_symbols()
    all_results = []
    stats = {
        'symbols_scanned': 0,
        'total_symbols': len(symbols),
        'patterns_found': 0,
        'scan_time': 0,
    }

    start_time = time.time()
    total_tasks = len(symbols) * len(timeframes)
    completed = 0

    for symbol in symbols:
        for tf in timeframes:
            patterns = scan_symbol(symbol, tf, DEFAULT_CONFIG, min_quality)
            all_results.extend(patterns)
            completed += 1

            if progress_callback:
                progress_callback(completed / total_tasks)

        stats['symbols_scanned'] += 1

    stats['patterns_found'] = len(all_results)
    stats['scan_time'] = time.time() - start_time

    return all_results, stats


def find_multi_tf_aligned(patterns: List[Dict]) -> Dict[str, List[Dict]]:
    """Find symbols with patterns on multiple timeframes (same direction)."""
    # Group by symbol and direction
    by_symbol_direction = defaultdict(list)
    for p in patterns:
        key = (p['symbol'], p['direction'])
        by_symbol_direction[key].append(p)

    # Filter to those with 2+ timeframes
    aligned = {}
    for (symbol, direction), pats in by_symbol_direction.items():
        timeframes = set(p['timeframe'] for p in pats)
        if len(timeframes) >= 2:
            aligned[symbol] = pats

    return aligned


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_scanner_controls() -> Tuple[bool, List[str], float, bool]:
    """Render scanner control panel. Returns (scan_clicked, timeframes, min_quality, auto_refresh)."""

    # Control panel header
    html = '<div class="panel">'
    html += '<div class="panel-header">'
    html += '<span class="panel-title">Scanner Controls</span>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        # Timeframe selection
        selected_tfs = st.multiselect(
            "Timeframes",
            options=TIMEFRAMES,
            default=TIMEFRAMES,
            key="scanner_timeframes"
        )

    with col2:
        # Quality threshold
        min_quality = st.slider(
            "Min Quality",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            key="scanner_min_quality"
        )

    with col3:
        # Auto-refresh toggle
        auto_refresh = st.checkbox(
            "Auto-refresh (15 min)",
            value=False,
            key="scanner_auto_refresh"
        )

    with col4:
        st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        scan_clicked = st.button(
            "🔍 Scan Now",
            type="primary",
            use_container_width=True,
            key="scanner_scan_btn"
        )

    return scan_clicked, selected_tfs or TIMEFRAMES, min_quality, auto_refresh


def render_status_bar(stats: Optional[Dict], last_scan: Optional[datetime]) -> None:
    """Render scanner status bar."""
    if stats is None:
        status_text = "Ready to scan"
        color = ARCTIC_PRO['text_muted']
    else:
        status_text = f"Found {stats['patterns_found']} patterns across {stats['symbols_scanned']} symbols"
        color = ARCTIC_PRO['success'] if stats['patterns_found'] > 0 else ARCTIC_PRO['text_secondary']

    html = f'<div style="display: flex; justify-content: space-between; align-items: center; '
    html += f'padding: 8px 12px; background: {ARCTIC_PRO["bg_card"]}; border-radius: 6px; '
    html += f'border: 1px solid {ARCTIC_PRO["border"]}; margin-bottom: 16px;">'

    # Status text
    html += f'<div style="display: flex; align-items: center; gap: 8px;">'
    html += f'<span style="color: {color}; font-weight: 500;">{status_text}</span>'
    if stats and stats.get('scan_time'):
        html += f'<span style="color: {ARCTIC_PRO["text_muted"]}; font-size: 0.75rem;">'
        html += f'({stats["scan_time"]:.1f}s)</span>'
    html += '</div>'

    # Last scan time
    if last_scan:
        time_ago = datetime.now() - last_scan
        if time_ago.seconds < 60:
            time_str = "just now"
        elif time_ago.seconds < 3600:
            time_str = f"{time_ago.seconds // 60}m ago"
        else:
            time_str = last_scan.strftime("%H:%M")

        html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: 0.75rem;">'
        html += f'Last scan: {time_str}'
        html += '</div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_results_table(patterns: List[Dict]) -> Optional[Dict]:
    """Render pattern cards with View Chart buttons. Returns selected pattern if clicked."""
    if not patterns:
        html = f'<div style="padding: 48px; text-align: center; color: {ARCTIC_PRO["text_muted"]};">'
        html += 'No patterns detected. Click "Scan Now" to search for patterns.'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
        return None

    selected_pattern = None

    # Render each pattern as an expandable card with View Chart button
    for i, p in enumerate(patterns):
        dir_color = ARCTIC_PRO['success'] if p['direction'] == 'LONG' else ARCTIC_PRO['danger']
        regime_colors = {
            'RANGING': ARCTIC_PRO['success'],
            'TRENDING': ARCTIC_PRO['danger'],
            'VOLATILE': ARCTIC_PRO['warning'],
            'EXTREME': ARCTIC_PRO['warning'],
        }
        regime_color = regime_colors.get(p['regime'], ARCTIC_PRO['text_muted'])

        with st.expander(
            f"**{p['symbol']}** · {p['timeframe']} · {p['direction']} · Quality: {p['quality']:.0%}",
            expanded=(i == 0)
        ):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**Direction:** <span style='color: {dir_color};'>{p['direction']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Quality:** {p['quality']:.0%}")
                st.markdown(f"**Regime:** <span style='color: {regime_color};'>{p['regime']}</span>", unsafe_allow_html=True)
                st.markdown(f"**R:R:** 1:{p['rr_ratio']:.1f}")

            with col2:
                st.markdown("**Trading Levels:**")
                st.markdown(f"Entry: ${p['entry']:,.2f}")
                st.markdown(f"Stop Loss: ${p['stop_loss']:,.2f}")
                st.markdown(f"Take Profit: ${p['take_profit']:,.2f}")

            with col3:
                st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
                if st.button("📈 View Chart", key=f"view_chart_{i}", use_container_width=True):
                    selected_pattern = p
                if st.button("📤 Send to MT5", key=f"send_mt5_{i}", use_container_width=True):
                    # Export to MT5
                    pattern_obj = p.get('pattern_obj')
                    if pattern_obj:
                        result = export_pattern_to_mt5(pattern_obj, p['symbol'], p['timeframe'])
                        if result['success']:
                            st.success(f"✓ Sent to MT5! Open {p['symbol']} chart.")
                        else:
                            st.error(result['error'])
                    else:
                        st.warning("Pattern object not available")

    return selected_pattern


def render_pattern_chart_panel(pattern: Dict) -> None:
    """Render the chart panel for a selected pattern."""
    symbol = pattern['symbol']
    timeframe = pattern['timeframe']
    pattern_obj = pattern.get('pattern_obj')

    # Load OHLCV data
    df = load_symbol_data(symbol, timeframe)
    if df is None:
        st.error(f"Could not load data for {symbol}")
        return

    # Chart panel header
    html = '<div class="panel" style="margin-top: 16px;">'
    html += '<div class="panel-header">'
    html += f'<span class="panel-title">{symbol} · {timeframe.upper()} · Pattern Chart</span>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

    # Convert pattern to chart dict format
    if pattern_obj:
        chart_pattern = pattern_to_chart_dict(pattern_obj)
    else:
        # Fallback: use basic pattern data
        chart_pattern = {
            'entry_price': pattern['entry'],
            'stop_loss_price': pattern['stop_loss'],
            'take_profit_price': pattern['take_profit'],
        }

    # Filter data to show pattern context (100 bars before detection, 50 after)
    if pattern_obj and hasattr(pattern_obj, 'p5'):
        p5_time = pattern_obj.p5.timestamp
        if hasattr(p5_time, 'tz_localize') and p5_time.tzinfo is not None:
            p5_time = p5_time.tz_localize(None)

        # Make df time tz-naive for comparison
        df_display = df.copy()
        if df_display['time'].dt.tz is not None:
            df_display['time'] = df_display['time'].dt.tz_localize(None)

        # Find P5 index
        mask = df_display['time'] <= p5_time
        if mask.any():
            p5_idx = mask.sum() - 1
            start_idx = max(0, p5_idx - 100)
            end_idx = min(len(df_display), p5_idx + 50)
            df_display = df_display.iloc[start_idx:end_idx].reset_index(drop=True)
    else:
        df_display = df.tail(200)

    # Render the chart
    try:
        render_pattern_chart(df_display, pattern=chart_pattern, height=500, key=f"scanner_chart_{symbol}_{timeframe}")
    except Exception as e:
        st.error(f"Chart rendering error: {e}")

        # Fallback: show simple plotly chart
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_display['time'],
            open=df_display['open'],
            high=df_display['high'],
            low=df_display['low'],
            close=df_display['close'],
        ))

        # Add entry/SL/TP lines
        fig.add_hline(y=pattern['entry'], line_color='cyan', annotation_text='Entry')
        fig.add_hline(y=pattern['stop_loss'], line_color='red', line_dash='dash', annotation_text='SL')
        fig.add_hline(y=pattern['take_profit'], line_color='green', line_dash='dot', annotation_text='TP')

        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
        )
        st.plotly_chart(fig, use_container_width=True)


def render_alignment_panel(aligned: Dict[str, List[Dict]]) -> None:
    """Render multi-timeframe alignment panel."""
    html = '<div class="panel" style="margin-top: 16px;">'
    html += '<div class="panel-header">'
    html += '<span class="panel-title">Multi-TF Alignment</span>'

    if aligned:
        html += f'<span style="background: {ARCTIC_PRO["success"]}; color: white; '
        html += f'padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">'
        html += f'{len(aligned)} ALIGNED</span>'

    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

    if not aligned:
        html = f'<div style="padding: 24px; text-align: center; color: {ARCTIC_PRO["text_muted"]}; '
        html += f'background: {ARCTIC_PRO["bg_secondary"]}; border-radius: 8px;">'
        html += 'No multi-timeframe alignments detected'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
        return

    # Display aligned symbols
    for symbol, patterns in aligned.items():
        direction = patterns[0]['direction']
        timeframes = sorted(set(p['timeframe'] for p in patterns))
        best_quality = max(p['quality'] for p in patterns)

        dir_color = ARCTIC_PRO['success'] if direction == 'LONG' else ARCTIC_PRO['danger']

        html = f'<div style="display: flex; align-items: center; justify-content: space-between; '
        html += f'padding: 12px; background: {ARCTIC_PRO["bg_card"]}; border-radius: 8px; '
        html += f'border: 1px solid {ARCTIC_PRO["border"]}; margin-bottom: 8px;">'

        # Symbol and direction
        html += f'<div style="display: flex; align-items: center; gap: 12px;">'
        html += f'<span style="font-weight: 600; color: {ARCTIC_PRO["text_primary"]}; font-size: 1rem;">{symbol}</span>'
        html += f'<span style="color: {dir_color}; font-weight: 600;">{direction}</span>'
        html += '</div>'

        # Timeframes and quality
        html += f'<div style="display: flex; align-items: center; gap: 16px;">'
        for tf in timeframes:
            html += f'<span style="background: {ARCTIC_PRO["accent"]}20; color: {ARCTIC_PRO["accent"]}; '
            html += f'padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 500;">{tf}</span>'
        html += f'<span style="color: {ARCTIC_PRO["text_muted"]};">Quality: {best_quality:.0%}</span>'
        html += '</div>'

        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)


def render_pattern_details(pattern: Dict) -> None:
    """Render expanded pattern details."""
    html = f'<div style="padding: 16px; background: {ARCTIC_PRO["bg_secondary"]}; '
    html += f'border-radius: 8px; margin: 8px 0;">'

    html += f'<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">'

    # Pattern metrics
    metrics = [
        ('Head Extension', f"{pattern.get('head_extension', 0):.2f} ATR"),
        ('BOS Count', pattern.get('bos_count', 'N/A')),
        ('Pattern ID', pattern.get('pattern_id', 'N/A')[:8]),
        ('Detected', pattern.get('detected_at', 'N/A')),
    ]

    for label, value in metrics:
        html += f'<div>'
        html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: 0.75rem;">{label}</div>'
        html += f'<div style="color: {ARCTIC_PRO["text_primary"]}; font-weight: 500;">{value}</div>'
        html += '</div>'

    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# MAIN PAGE RENDER
# =============================================================================

def render_live_scanner_page() -> None:
    """Render the Live Scanner page."""

    # Page header
    header_html = '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">'
    header_html += f'<div style="font-family: {TYPOGRAPHY["font_display"]}; font-size: {TYPOGRAPHY["size_xl"]}; '
    header_html += f'font-weight: {TYPOGRAPHY["weight_semibold"]}; color: {ARCTIC_PRO["text_primary"]};">'
    header_html += 'Live Scanner</div>'
    header_html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: 0.875rem;">'
    header_html += f'{len(get_all_symbols())} symbols · {len(TIMEFRAMES)} timeframes'
    header_html += '</div></div>'
    st.markdown(header_html, unsafe_allow_html=True)

    # Scanner controls
    scan_clicked, selected_tfs, min_quality, auto_refresh = render_scanner_controls()

    # Auto-refresh logic using timestamp tracking
    if auto_refresh:
        last_auto_scan = st.session_state.get('scanner_last_auto_scan')
        now = datetime.now()

        if last_auto_scan is None:
            # First time - trigger scan
            scan_clicked = True
            st.session_state['scanner_last_auto_scan'] = now
        elif (now - last_auto_scan).total_seconds() >= 900:  # 15 minutes
            # Time for auto-refresh
            scan_clicked = True
            st.session_state['scanner_last_auto_scan'] = now
            st.toast("Auto-refreshing scanner...", icon="🔄")

    # Get stored results
    patterns = st.session_state.get('scanner_results', [])
    stats = st.session_state.get('scanner_stats', None)
    last_scan = st.session_state.get('scanner_last_scan', None)

    # Run scan if triggered
    if scan_clicked:
        with st.spinner(f"Scanning {len(get_all_symbols())} symbols..."):
            progress_bar = st.progress(0)

            def update_progress(pct):
                progress_bar.progress(pct)

            patterns, stats = scan_all_symbols(
                timeframes=selected_tfs,
                min_quality=min_quality,
                progress_callback=update_progress
            )

            progress_bar.empty()

            # Store results
            st.session_state['scanner_results'] = patterns
            st.session_state['scanner_stats'] = stats
            st.session_state['scanner_last_scan'] = datetime.now()

            if patterns:
                st.success(f"Found {len(patterns)} patterns!")
            else:
                st.info("No patterns found with current filters")

    # Status bar
    render_status_bar(stats, last_scan)

    # Results panel
    results_html = '<div class="panel">'
    results_html += '<div class="panel-header">'
    results_html += '<span class="panel-title">Detected Patterns</span>'
    results_html += f'<span style="color: {ARCTIC_PRO["text_muted"]}; font-size: 0.75rem;">'
    results_html += f'{len(patterns)} total</span>'
    results_html += '</div></div>'
    st.markdown(results_html, unsafe_allow_html=True)

    # Results table with chart view buttons
    selected_pattern = render_results_table(patterns)

    # Show chart if pattern selected
    if selected_pattern:
        st.session_state['scanner_selected_pattern'] = selected_pattern
        st.rerun()

    # Render chart for previously selected pattern
    if 'scanner_selected_pattern' in st.session_state:
        selected = st.session_state['scanner_selected_pattern']
        render_pattern_chart_panel(selected)

        # Clear selection button
        if st.button("✕ Close Chart", key="close_chart"):
            del st.session_state['scanner_selected_pattern']
            st.rerun()

    # Multi-TF alignment
    if patterns:
        aligned = find_multi_tf_aligned(patterns)
        render_alignment_panel(aligned)

    # Auto-refresh reminder
    if auto_refresh:
        next_refresh = st.session_state.get('scanner_last_auto_scan', datetime.now()) + timedelta(minutes=15)
        time_until = (next_refresh - datetime.now()).total_seconds()
        if time_until > 0:
            mins = int(time_until // 60)
            secs = int(time_until % 60)
            st.caption(f"Next auto-refresh in {mins}m {secs}s")
