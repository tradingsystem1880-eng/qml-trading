"""
Chart Component Test v2
=======================
Premium chart visualization test with pattern-aligned data.

Usage:
    cd /Users/hunternovotny/Desktop/QML_SYSTEM
    streamlit run qml/dashboard/test_chart.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Chart Test",
    page_icon="📊",
    layout="wide"
)

# Import our components
from qml.dashboard.core import apply_design_system, render_professional_chart

# Apply design system
apply_design_system()

st.title("Professional Chart Component Test v2")
st.markdown("Validating premium TradingView-grade visualization.")

# =============================================================================
# GENERATE PATTERN-ALIGNED TEST DATA
# =============================================================================

def generate_bullish_qml_data(n_bars: int = 350) -> tuple[pd.DataFrame, dict]:
    """
    Generate OHLCV data that forms a bullish QML pattern.
    Returns both the DataFrame and pattern dict with matching swing points.
    """
    np.random.seed(42)
    base_price = 42000
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    # Define the swing points we want (indices within the data)
    trend_start_idx = 20   # Trend starts here
    p1_idx = 80            # Point 1 - first major low (higher low in uptrend)
    p2_idx = 120           # Point 2 - swing high
    p3_idx = 160           # Point 3 - head low (lower low / QM point)
    p4_idx = 200           # Point 4 - higher high
    p5_idx = 240           # Point 5 - entry (retest of QM zone)

    # Define price levels
    trend_start_price = base_price * 0.95   # ~39,900
    p1_price = base_price * 1.02            # ~42,840 (first low)
    p2_price = base_price * 1.10            # ~46,200 (first high)
    p3_price = base_price * 0.98            # ~41,160 (head - lower low)
    p4_price = base_price * 1.14            # ~47,880 (higher high)
    p5_price = base_price * 1.08            # ~45,360 (entry - retest)

    # Generate smooth price path through these points
    prices = np.zeros(n_bars)

    # Segment 1: Trend start to P1 (uptrend establishing)
    for i in range(trend_start_idx, p1_idx + 1):
        t = (i - trend_start_idx) / (p1_idx - trend_start_idx)
        prices[i] = trend_start_price + (p1_price - trend_start_price) * t

    # Fill in before trend start
    for i in range(trend_start_idx):
        t = i / trend_start_idx
        prices[i] = base_price * 0.92 + (trend_start_price - base_price * 0.92) * t

    # Segment 2: P1 to P2 (rally to first high)
    for i in range(p1_idx, p2_idx + 1):
        t = (i - p1_idx) / (p2_idx - p1_idx)
        # Smooth sine curve for natural movement
        prices[i] = p1_price + (p2_price - p1_price) * (0.5 - 0.5 * np.cos(np.pi * t))

    # Segment 3: P2 to P3 (pullback to head)
    for i in range(p2_idx, p3_idx + 1):
        t = (i - p2_idx) / (p3_idx - p2_idx)
        prices[i] = p2_price + (p3_price - p2_price) * (0.5 - 0.5 * np.cos(np.pi * t))

    # Segment 4: P3 to P4 (rally to higher high)
    for i in range(p3_idx, p4_idx + 1):
        t = (i - p3_idx) / (p4_idx - p3_idx)
        prices[i] = p3_price + (p4_price - p3_price) * (0.5 - 0.5 * np.cos(np.pi * t))

    # Segment 5: P4 to P5 (pullback to entry)
    for i in range(p4_idx, p5_idx + 1):
        t = (i - p4_idx) / (p5_idx - p4_idx)
        prices[i] = p4_price + (p5_price - p4_price) * (0.5 - 0.5 * np.cos(np.pi * t))

    # Segment 6: P5 onwards (continuation up)
    final_price = base_price * 1.25  # Rally after entry
    for i in range(p5_idx, n_bars):
        t = (i - p5_idx) / (n_bars - p5_idx)
        prices[i] = p5_price + (final_price - p5_price) * t

    # Add realistic noise
    noise = np.random.randn(n_bars) * base_price * 0.003
    prices = prices + noise

    # Generate OHLC from close prices
    data = []
    for i in range(n_bars):
        close = prices[i]
        volatility = close * 0.006  # 0.6% typical range

        open_price = close + np.random.randn() * volatility * 0.3
        high = max(open_price, close) + abs(np.random.randn()) * volatility
        low = min(open_price, close) - abs(np.random.randn()) * volatility
        volume = np.random.randint(100, 1000) * 1000000

        data.append({
            'time': start_time + timedelta(hours=4 * i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)

    # Create pattern dict with exact timestamps
    pattern = {
        "pattern_type": "bullish_qml",

        # Trend line
        "trend_start_time": start_time + timedelta(hours=4 * trend_start_idx),
        "trend_start_price": prices[trend_start_idx],

        # Swing points
        "p1_timestamp": start_time + timedelta(hours=4 * p1_idx),
        "p1_price": prices[p1_idx],

        "p2_timestamp": start_time + timedelta(hours=4 * p2_idx),
        "p2_price": prices[p2_idx],

        "p3_timestamp": start_time + timedelta(hours=4 * p3_idx),
        "p3_price": prices[p3_idx],

        "p4_timestamp": start_time + timedelta(hours=4 * p4_idx),
        "p4_price": prices[p4_idx],

        "p5_timestamp": start_time + timedelta(hours=4 * p5_idx),
        "p5_price": prices[p5_idx],

        # Trade levels
        "entry_price": prices[p5_idx],
        "stop_loss": prices[p3_idx] - (base_price * 0.02),  # SL below P3
        "take_profit": prices[p4_idx] + (base_price * 0.05),  # TP1 above P4
        "take_profit_2": prices[p4_idx] + (base_price * 0.12),  # TP2 further up
    }

    return df, pattern


def generate_bearish_qml_data(n_bars: int = 350) -> tuple[pd.DataFrame, dict]:
    """
    Generate OHLCV data that forms a bearish QML pattern.
    Returns both the DataFrame and pattern dict with matching swing points.
    """
    np.random.seed(123)
    base_price = 48000
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    # Define the swing points (inverted from bullish)
    trend_start_idx = 20   # Downtrend starts here
    p1_idx = 80            # Point 1 - first major high (lower high in downtrend)
    p2_idx = 120           # Point 2 - swing low
    p3_idx = 160           # Point 3 - head high (higher high / QM point)
    p4_idx = 200           # Point 4 - lower low
    p5_idx = 240           # Point 5 - entry (retest of QM zone)

    # Define price levels (inverted)
    trend_start_price = base_price * 1.05   # ~50,400
    p1_price = base_price * 0.98            # ~47,040 (first high - lower than trend)
    p2_price = base_price * 0.90            # ~43,200 (first low)
    p3_price = base_price * 1.02            # ~48,960 (head - higher high)
    p4_price = base_price * 0.86            # ~41,280 (lower low)
    p5_price = base_price * 0.96            # ~46,080 (entry - retest)

    # Generate smooth price path
    prices = np.zeros(n_bars)

    # Segment 1: Before trend start
    for i in range(trend_start_idx):
        t = i / trend_start_idx
        prices[i] = base_price * 1.08 - (base_price * 0.03) * t

    # Segment 2: Trend start to P1
    for i in range(trend_start_idx, p1_idx + 1):
        t = (i - trend_start_idx) / (p1_idx - trend_start_idx)
        prices[i] = trend_start_price + (p1_price - trend_start_price) * t

    # Segment 3: P1 to P2
    for i in range(p1_idx, p2_idx + 1):
        t = (i - p1_idx) / (p2_idx - p1_idx)
        prices[i] = p1_price + (p2_price - p1_price) * (0.5 - 0.5 * np.cos(np.pi * t))

    # Segment 4: P2 to P3
    for i in range(p2_idx, p3_idx + 1):
        t = (i - p2_idx) / (p3_idx - p2_idx)
        prices[i] = p2_price + (p3_price - p2_price) * (0.5 - 0.5 * np.cos(np.pi * t))

    # Segment 5: P3 to P4
    for i in range(p3_idx, p4_idx + 1):
        t = (i - p3_idx) / (p4_idx - p3_idx)
        prices[i] = p3_price + (p4_price - p3_price) * (0.5 - 0.5 * np.cos(np.pi * t))

    # Segment 6: P4 to P5
    for i in range(p4_idx, p5_idx + 1):
        t = (i - p4_idx) / (p5_idx - p4_idx)
        prices[i] = p4_price + (p5_price - p4_price) * (0.5 - 0.5 * np.cos(np.pi * t))

    # Segment 7: P5 onwards (continuation down)
    final_price = base_price * 0.75
    for i in range(p5_idx, n_bars):
        t = (i - p5_idx) / (n_bars - p5_idx)
        prices[i] = p5_price + (final_price - p5_price) * t

    # Add noise
    noise = np.random.randn(n_bars) * base_price * 0.003
    prices = prices + noise

    # Generate OHLC
    data = []
    for i in range(n_bars):
        close = prices[i]
        volatility = close * 0.006

        open_price = close + np.random.randn() * volatility * 0.3
        high = max(open_price, close) + abs(np.random.randn()) * volatility
        low = min(open_price, close) - abs(np.random.randn()) * volatility
        volume = np.random.randint(100, 1000) * 1000000

        data.append({
            'time': start_time + timedelta(hours=4 * i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)

    # Create pattern dict
    pattern = {
        "pattern_type": "bearish_qml",

        # Trend line
        "trend_start_time": start_time + timedelta(hours=4 * trend_start_idx),
        "trend_start_price": prices[trend_start_idx],

        # Swing points
        "p1_timestamp": start_time + timedelta(hours=4 * p1_idx),
        "p1_price": prices[p1_idx],

        "p2_timestamp": start_time + timedelta(hours=4 * p2_idx),
        "p2_price": prices[p2_idx],

        "p3_timestamp": start_time + timedelta(hours=4 * p3_idx),
        "p3_price": prices[p3_idx],

        "p4_timestamp": start_time + timedelta(hours=4 * p4_idx),
        "p4_price": prices[p4_idx],

        "p5_timestamp": start_time + timedelta(hours=4 * p5_idx),
        "p5_price": prices[p5_idx],

        # Trade levels (inverted for SHORT)
        "entry_price": prices[p5_idx],
        "stop_loss": prices[p3_idx] + (base_price * 0.02),  # SL above P3
        "take_profit": prices[p4_idx] - (base_price * 0.05),  # TP1 below P4
        "take_profit_2": prices[p4_idx] - (base_price * 0.12),  # TP2 further down
    }

    return df, pattern


# =============================================================================
# DISPLAY BULLISH PATTERN
# =============================================================================

st.markdown("---")
st.markdown("### Bullish QML Pattern (LONG)")
st.markdown("""
**Pattern Structure:**
- **Trend Line**: Initial uptrend (green) leading to Point 1
- **Point 1**: First significant low in the uptrend
- **Point 2**: Swing high (first peak)
- **Point 3**: Head low (QM point - lower low)
- **Point 4**: Higher high (breaks above P2)
- **Point 5**: Entry point (retest of QM zone)
- **SL Zone**: Red box below entry
- **TP Zones**: Green boxes above entry
""")

df_bull, pattern_bull = generate_bullish_qml_data(350)

render_professional_chart(
    df=df_bull,
    pattern=pattern_bull,
    height=650,
    title="BTC/USDT 4H - Bullish QML Pattern (LONG)"
)

# Pattern metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Entry", f"${pattern_bull['entry_price']:,.0f}")
with col2:
    sl_dist = abs(pattern_bull['entry_price'] - pattern_bull['stop_loss'])
    st.metric("Stop Loss", f"${pattern_bull['stop_loss']:,.0f}", f"-${sl_dist:,.0f}")
with col3:
    tp1_dist = abs(pattern_bull['take_profit'] - pattern_bull['entry_price'])
    st.metric("TP1", f"${pattern_bull['take_profit']:,.0f}", f"+${tp1_dist:,.0f}")
with col4:
    tp2_dist = abs(pattern_bull['take_profit_2'] - pattern_bull['entry_price'])
    st.metric("TP2", f"${pattern_bull['take_profit_2']:,.0f}", f"+${tp2_dist:,.0f}")
with col5:
    rr = tp1_dist / sl_dist if sl_dist > 0 else 0
    st.metric("Risk:Reward", f"{rr:.2f}:1")

# =============================================================================
# DISPLAY BEARISH PATTERN
# =============================================================================

st.markdown("---")
st.markdown("### Bearish QML Pattern (SHORT)")
st.markdown("""
**Pattern Structure (Inverted):**
- **Trend Line**: Initial downtrend (red) leading to Point 1
- **Point 1**: First significant high in the downtrend
- **Point 2**: Swing low (first trough)
- **Point 3**: Head high (QM point - higher high)
- **Point 4**: Lower low (breaks below P2)
- **Point 5**: Entry point (retest of QM zone)
- **SL Zone**: Red box ABOVE entry
- **TP Zones**: Green boxes BELOW entry
""")

df_bear, pattern_bear = generate_bearish_qml_data(350)

render_professional_chart(
    df=df_bear,
    pattern=pattern_bear,
    height=650,
    title="BTC/USDT 4H - Bearish QML Pattern (SHORT)"
)

# Pattern metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Entry", f"${pattern_bear['entry_price']:,.0f}")
with col2:
    sl_dist = abs(pattern_bear['stop_loss'] - pattern_bear['entry_price'])
    st.metric("Stop Loss", f"${pattern_bear['stop_loss']:,.0f}", f"+${sl_dist:,.0f} (above)")
with col3:
    tp1_dist = abs(pattern_bear['entry_price'] - pattern_bear['take_profit'])
    st.metric("TP1", f"${pattern_bear['take_profit']:,.0f}", f"-${tp1_dist:,.0f} (below)")
with col4:
    tp2_dist = abs(pattern_bear['entry_price'] - pattern_bear['take_profit_2'])
    st.metric("TP2", f"${pattern_bear['take_profit_2']:,.0f}", f"-${tp2_dist:,.0f} (below)")
with col5:
    rr = tp1_dist / sl_dist if sl_dist > 0 else 0
    st.metric("Risk:Reward", f"{rr:.2f}:1")

# =============================================================================
# SIMPLE CHART TEST
# =============================================================================

st.markdown("---")
st.markdown("### Simple Chart (No Pattern Overlay)")

from qml.dashboard.core import render_simple_chart
render_simple_chart(df_bull, height=400, title="BTC/USDT 4H - Price Only")

# =============================================================================
# CHECKLIST
# =============================================================================

st.markdown("---")
st.markdown("### Visual Checklist")

st.markdown("""
| Element | Expected | Check |
|---------|----------|-------|
| **Trend Line** | Solid line showing initial trend (green for bull, red for bear) | |
| **Swing Points 1-5** | Blue numbered circles at exact swing highs/lows | |
| **Connection Lines** | Blue dashed zigzag connecting points 1→2→3→4→5 | |
| **Entry Line** | Blue solid horizontal line at entry price | |
| **SL Zone** | Red shaded box from entry to SL | |
| **TP1 Zone** | Green shaded box from entry to TP1 | |
| **TP2 Zone** | Lighter green box from TP1 to TP2 | |
| **Level Badges** | ENTRY (blue), SL (red), TP1/TP2 (green) badges on right | |
| **LONG**: SL below, TP above | | |
| **SHORT**: SL above, TP below | | |
""")

st.success("✅ Chart component test v2 complete!")
