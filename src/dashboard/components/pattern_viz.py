"""
Core Pattern Visualization Engine
==================================
Professional, interactive Plotly visualizations for QML patterns.

Integrates with the ML Pattern Registry database to render patterns
with confidence scoring, trade levels, and publication-ready styling.

Usage:
    from src.dashboard.components.pattern_viz import add_pattern_to_figure
    
    fig = go.Figure(...)  # Base candlestick chart
    fig = add_pattern_to_figure(fig, pattern_record, ohlcv_df)
    fig.show()
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger


# =============================================================================
# COLOR CONSTANTS - TradingView Style
# =============================================================================

# QML Pattern line color (Blue dashed like TradingView)
QML_LINE_COLOR = "#2962FF"  # TradingView blue

# Bullish pattern colors (green scale)
BULLISH_COLORS = {
    "primary": "#2962FF",      # TradingView blue (pattern line)
    "secondary": "#1E88E5",    # Darker blue
    "line": "#2962FF",         # Blue dashed line
    "fill": "rgba(41, 98, 255, 0.1)",
    "marker": "#00E676",       # Green for bullish markers
}

# Bearish pattern colors (orange/red scale)
BEARISH_COLORS = {
    "primary": "#2962FF",      # Same blue line
    "secondary": "#1E88E5",    # Darker blue
    "line": "#2962FF",         # Blue dashed line
    "fill": "rgba(41, 98, 255, 0.1)",
    "marker": "#FF5252",       # Red for bearish markers
}

# Trade level colors (TradingView style)
TRADE_COLORS = {
    "entry": "#2962FF",        # Blue
    "stop_loss": "#FF5252",    # Red
    "stop_loss_fill": "rgba(255, 82, 82, 0.2)",
    "take_profit": "#00E676",  # Green
    "take_profit_fill": "rgba(0, 230, 118, 0.2)",
}

# Structure labels
STRUCTURE_COLORS = {
    "choch": "#FF9800",        # Orange
    "bos": "#9C27B0",          # Purple
    "qm_zone": "rgba(41, 98, 255, 0.15)",
}

# Styling constants
FONT_FAMILY = "Inter, Arial, sans-serif"
ANNOTATION_FONT_SIZE = 11
MARKER_SIZE = 10
LABEL_FONT_SIZE = 12


# =============================================================================
# CORE VISUALIZATION FUNCTION
# =============================================================================

def add_pattern_to_figure(
    fig: go.Figure,
    pattern_record: dict,
    ohlcv_df: pd.DataFrame
) -> go.Figure:
    """
    Add TradingView-style QML pattern visualization to a Plotly figure.
    
    This modular function takes a base candlestick chart and overlays:
    1. Dashed blue pattern outline (P1 → P2 → P3 → P4 → P5) with numbered labels
    2. Structure labels (CHoCH, BoS, QM-Zone)
    3. TradingView-style position boxes (green TP, red SL)
    4. ML Confidence shading (optional)
    
    Args:
        fig: Existing Plotly Figure (assumed to have a Candlestick trace)
        pattern_record: Dictionary from ml_pattern_registry table
        ohlcv_df: DataFrame with OHLCV data for context
        
    Returns:
        Modified Plotly Figure with pattern visualization layers
    """
    
    # Extract pattern metadata
    pattern_type = pattern_record.get("pattern_type", "bullish_qml")
    ml_confidence = pattern_record.get("ml_confidence")
    
    # Extract geometry from features_json
    geometry = _extract_geometry(pattern_record)
    
    if geometry is None:
        logger.warning("Could not extract pattern geometry, skipping visualization")
        return fig
    
    # Determine pattern direction
    is_bullish = "bullish" in pattern_type.lower()
    colors = BULLISH_COLORS if is_bullish else BEARISH_COLORS
    
    # Get time range
    time_min = geometry["p1_timestamp"]
    time_max = geometry["p5_timestamp"]
    
    # 1. Add ML Confidence Background (behind everything)
    if ml_confidence is not None:
        fig = _add_confidence_shading(fig, geometry, ml_confidence, time_min, time_max)
    
    # 2. Add Structure Labels (CHoCH, BoS, QM-Zone)
    fig = _add_structure_labels(fig, geometry, is_bullish)
    
    # 3. Add Pattern Outline (dashed blue with numbered labels)
    fig = _add_pattern_outline(fig, geometry, colors, is_bullish)
    
    # 4. Add Position Boxes (green TP, red SL)
    fig = _add_trade_levels(fig, geometry, time_min, time_max, is_bullish)
    
    # 5. Apply Professional Styling
    fig = _apply_professional_styling(fig)
    
    return fig


# =============================================================================
# GEOMETRY EXTRACTION
# =============================================================================

def _extract_geometry(pattern_record: dict) -> Optional[Dict[str, Any]]:
    """
    Extract P1-P5 geometry from pattern_record's features_json.
    
    CRITICAL: This function EXTRACTS validated data from the database.
    It does NOT infer pattern points from OHLCV data.
    
    The features_json stores raw geometry coordinates in a flat structure
    with keys like p1_timestamp, p1_price, p2_timestamp, p2_price, etc.
    """
    features_json = pattern_record.get("features_json", "{}")
    
    # Parse JSON if string
    try:
        if isinstance(features_json, str):
            features = json.loads(features_json)
        else:
            features = features_json
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse features_json: {e}")
        return None
    
    # Debug: Print available keys for visibility
    logger.debug(f"Features JSON keys: {list(features.keys())}")
    
    # Check multiple possible locations for geometry
    geo = None
    
    if "geometry" in features:
        geo = features["geometry"]
    elif "pattern_geometry" in features:
        geo = features["pattern_geometry"]
    elif "points" in features:
        geo = features["points"]
    else:
        # Try flat structure - P1-P5 coordinates are directly in features
        geo = _extract_flat_geometry(features)
    
    # Validate we have required points
    if geo is None:
        logger.warning("No geometry found in features_json")
        return None
    
    required_keys = ["p1_timestamp", "p1_price", "p3_timestamp", "p3_price", 
                     "p5_timestamp", "p5_price"]
    
    missing = [k for k in required_keys if k not in geo or geo[k] is None]
    if missing:
        logger.warning(f"Missing required geometry keys: {missing}")
        logger.warning(f"Available keys: {list(geo.keys())}")
        return None
    
    # Convert timestamps to datetime if they're strings
    for i in range(1, 6):
        ts_key = f"p{i}_timestamp"
        if ts_key in geo and geo[ts_key] is not None:
            if isinstance(geo[ts_key], str):
                try:
                    geo[ts_key] = pd.to_datetime(geo[ts_key])
                except:
                    pass
    
    # Extract trade levels from geometry or pattern_record
    if "entry_price" not in geo:
        geo["entry_price"] = pattern_record.get("entry_price") or features.get("entry_price")
    if "stop_loss_price" not in geo:
        geo["stop_loss_price"] = pattern_record.get("stop_loss_price") or features.get("stop_loss_price")
    if "take_profit_price" not in geo:
        geo["take_profit_price"] = pattern_record.get("take_profit_price") or features.get("take_profit_price")
    
    return geo


def _extract_flat_geometry(features: dict) -> Optional[dict]:
    """
    Extract geometry when P1-P5 coordinates are flat keys in features dict.
    
    This handles the case where coordinates are stored directly in features_json
    rather than nested under a 'geometry' key.
    """
    geo = {}
    has_any = False
    
    # Extract P1-P5 coordinates
    for i in range(1, 6):
        for field in ["timestamp", "price", "idx"]:
            key = f"p{i}_{field}"
            if key in features:
                geo[key] = features[key]
                if field in ["timestamp", "price"] and features[key] is not None:
                    has_any = True
    
    # Extract trade levels
    for key in ["entry_price", "stop_loss_price", "take_profit_price"]:
        if key in features:
            geo[key] = features[key]
    
    # Also check pattern_type
    if "pattern_type" in features:
        geo["pattern_type"] = features["pattern_type"]
    
    return geo if has_any else None


# =============================================================================
# VISUALIZATION LAYER FUNCTIONS
# =============================================================================

def _add_confidence_shading(
    fig: go.Figure,
    geometry: dict,
    confidence: float,
    time_min: Any,
    time_max: Any
) -> go.Figure:
    """Add ML confidence background shading."""
    
    # Map confidence to color: 0.0 = red, 1.0 = green
    if confidence >= 0.5:
        # Green gradient for higher confidence
        intensity = (confidence - 0.5) * 2  # 0 to 1
        r, g, b = 34, int(197 + intensity * 58), int(94 + intensity * 60)
    else:
        # Red gradient for lower confidence
        intensity = (0.5 - confidence) * 2  # 0 to 1
        r, g, b = int(239 - intensity * 100), int(68 - intensity * 30), int(68 - intensity * 30)
    
    fill_color = f"rgba({r}, {g}, {b}, 0.15)"
    
    # Get price range from geometry
    prices = [geometry[f"p{i}_price"] for i in range(1, 6)]
    y_min = min(prices) * 0.995
    y_max = max(prices) * 1.005
    
    # Add shaded rectangle
    fig.add_shape(
        type="rect",
        x0=time_min,
        x1=time_max,
        y0=y_min,
        y1=y_max,
        fillcolor=fill_color,
        line=dict(width=0),
        layer="below",
    )
    
    # Add confidence annotation
    score_pct = int(confidence * 100)
    fig.add_annotation(
        x=time_min,
        y=y_max,
        text=f"<b>ML Score: {score_pct}%</b>",
        font=dict(size=ANNOTATION_FONT_SIZE, color="#94A3B8", family=FONT_FAMILY),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(15, 23, 42, 0.8)",
        borderpad=4,
    )
    
    return fig


def _add_pattern_outline(
    fig: go.Figure, 
    geometry: dict, 
    colors: dict,
    is_bullish: bool = True
) -> go.Figure:
    """Add TradingView-style dashed blue pattern outline P1→P2→P3→P4→P5."""
    
    timestamps = [geometry[f"p{i}_timestamp"] for i in range(1, 6)]
    prices = [geometry[f"p{i}_price"] for i in range(1, 6)]
    
    # Main pattern line - dashed blue like TradingView
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=prices,
        mode="lines",
        name="QML Pattern",
        line=dict(
            color=QML_LINE_COLOR,
            width=2.5,
            dash="dash",  # Dashed line like TradingView
        ),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Add numbered labels at each point (1, 2, 3, 4, 5)
    point_labels = [
        ("1", "Left Shoulder"),
        ("2", "Lower High" if not is_bullish else "Higher Low"),
        ("3", "Head"),
        ("4", "CHoCH"),
        ("5", "Entry"),
    ]
    
    for i, (num, name) in enumerate(point_labels, 1):
        ts = geometry[f"p{i}_timestamp"]
        price = geometry[f"p{i}_price"]
        
        # Determine label position
        if i in [1, 3]:  # Swing highs (for bearish) or lows (for bullish)
            yanchor = "bottom" if is_bullish else "top"
            yshift = 10 if is_bullish else -10
        else:
            yanchor = "top" if is_bullish else "bottom"
            yshift = -10 if is_bullish else 10
        
        # Add number label
        fig.add_annotation(
            x=ts,
            y=price,
            text=f"<b>{num}</b>",
            font=dict(size=LABEL_FONT_SIZE, color=QML_LINE_COLOR, family=FONT_FAMILY),
            showarrow=False,
            yanchor=yanchor,
            yshift=yshift,
        )
    
    return fig


def _add_critical_points(
    fig: go.Figure,
    geometry: dict,
    colors: dict,
    is_bullish: bool
) -> go.Figure:
    """Add markers for P1 (Left Shoulder), P3 (Head), P5 (Entry)."""
    
    # Point configurations
    critical_points = [
        {
            "key": "p1",
            "name": "P1 (Left Shoulder)",
            "symbol": "circle",
            "size": MARKER_SIZE,
        },
        {
            "key": "p3",
            "name": "P3 (Head)",
            "symbol": "triangle-up" if is_bullish else "triangle-down",
            "size": MARKER_SIZE + 2,
        },
        {
            "key": "p5",
            "name": "P5 (Entry)",
            "symbol": "square",
            "size": MARKER_SIZE,
        },
    ]
    
    for point in critical_points:
        key = point["key"]
        timestamp = geometry[f"{key}_timestamp"]
        price = geometry[f"{key}_price"]
        
        # Add marker
        fig.add_trace(go.Scatter(
            x=[timestamp],
            y=[price],
            mode="markers+text",
            name=point["name"],
            marker=dict(
                symbol=point["symbol"],
                size=point["size"],
                color=colors["primary"],
                line=dict(color=colors["secondary"], width=2),
            ),
            text=[key.upper()],
            textposition="top center" if key != "p3" else ("bottom center" if is_bullish else "top center"),
            textfont=dict(
                size=ANNOTATION_FONT_SIZE,
                color=colors["primary"],
                family=FONT_FAMILY,
            ),
            hovertemplate=f"<b>{point['name']}</b><br>Price: $%{{y:,.2f}}<br>Time: %{{x}}<extra></extra>",
        ))
    
    return fig


def _add_trade_levels(
    fig: go.Figure,
    geometry: dict,
    time_min: Any,
    time_max: Any,
    is_bullish: bool = True
) -> go.Figure:
    """Add TradingView-style position boxes (green TP, red SL) starting from entry."""
    
    entry_price = geometry.get("entry_price")
    sl_price = geometry.get("stop_loss_price")
    tp_price = geometry.get("take_profit_price")
    
    if entry_price is None:
        return fig
    
    # Position box should extend from P5 (entry) to the right
    entry_time = geometry.get("p5_timestamp", time_max)
    
    # Extend time range for position box (extend further right)
    if isinstance(time_max, pd.Timestamp):
        box_end = time_max + pd.Timedelta(days=3)
    else:
        box_end = time_max
    
    # Add TP Box (green semi-transparent) - above entry for bullish, below for bearish
    if tp_price is not None:
        fig.add_shape(
            type="rect",
            x0=entry_time,
            x1=box_end,
            y0=entry_price,
            y1=tp_price,
            fillcolor=TRADE_COLORS["take_profit_fill"],
            line=dict(color=TRADE_COLORS["take_profit"], width=1),
            layer="below",
        )
        
        # TP Label
        fig.add_annotation(
            x=box_end,
            y=tp_price,
            text=f"<b>TP</b> ${tp_price:,.0f}",
            font=dict(size=10, color=TRADE_COLORS["take_profit"], family=FONT_FAMILY),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(0, 0, 0, 0.7)",
            borderpad=3,
        )
    
    # Add SL Box (red semi-transparent) - below entry for bullish, above for bearish
    if sl_price is not None:
        fig.add_shape(
            type="rect",
            x0=entry_time,
            x1=box_end,
            y0=sl_price,
            y1=entry_price,
            fillcolor=TRADE_COLORS["stop_loss_fill"],
            line=dict(color=TRADE_COLORS["stop_loss"], width=1),
            layer="below",
        )
        
        # SL Label
        fig.add_annotation(
            x=box_end,
            y=sl_price,
            text=f"<b>SL</b> ${sl_price:,.0f}",
            font=dict(size=10, color=TRADE_COLORS["stop_loss"], family=FONT_FAMILY),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(0, 0, 0, 0.7)",
            borderpad=3,
        )
    
    # Entry line (horizontal dashed)
    fig.add_trace(go.Scatter(
        x=[entry_time, box_end],
        y=[entry_price, entry_price],
        mode="lines",
        name="Entry",
        line=dict(
            color=TRADE_COLORS["entry"],
            width=1.5,
            dash="dot",
        ),
        showlegend=False,
        hoverinfo="skip",
    ))
    
    # Entry marker (arrow pointing at entry)
    direction = "▲" if is_bullish else "▼"
    fig.add_annotation(
        x=entry_time,
        y=entry_price,
        text=f"<b>{direction} ENTRY</b>",
        font=dict(size=10, color=TRADE_COLORS["entry"], family=FONT_FAMILY),
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor=TRADE_COLORS["entry"],
        ax=-50,
        ay=0,
        bgcolor="rgba(0, 0, 0, 0.7)",
        borderpad=3,
    )
    
    return fig


def _add_structure_labels(
    fig: go.Figure,
    geometry: dict,
    is_bullish: bool
) -> go.Figure:
    """Add CHoCH, BoS, and QM-Zone labels."""
    
    # CHoCH is typically at P4 level (where structure changes)
    if geometry.get("p4_timestamp") and geometry.get("p4_price"):
        choch_time = geometry["p4_timestamp"]
        choch_price = geometry["p4_price"]
        
        # CHoCH horizontal line
        fig.add_trace(go.Scatter(
            x=[geometry["p3_timestamp"], geometry["p5_timestamp"]],
            y=[choch_price, choch_price],
            mode="lines",
            name="CHoCH",
            line=dict(color=STRUCTURE_COLORS["choch"], width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        
        # CHoCH label
        fig.add_annotation(
            x=choch_time,
            y=choch_price,
            text="<b>CHoCH</b>",
            font=dict(size=9, color=STRUCTURE_COLORS["choch"], family=FONT_FAMILY),
            showarrow=False,
            yshift=-15 if is_bullish else 15,
            bgcolor="rgba(0, 0, 0, 0.6)",
            borderpad=2,
        )
    
    # BoS (Break of Structure) - typically after P3
    if geometry.get("p2_price"):
        bos_price = geometry["p2_price"]
        
        # BoS horizontal line
        fig.add_trace(go.Scatter(
            x=[geometry["p1_timestamp"], geometry["p4_timestamp"]],
            y=[bos_price, bos_price],
            mode="lines",
            name="BoS",
            line=dict(color=STRUCTURE_COLORS["bos"], width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        
        # BoS label
        fig.add_annotation(
            x=geometry["p2_timestamp"],
            y=bos_price,
            text="<b>BoS</b>",
            font=dict(size=9, color=STRUCTURE_COLORS["bos"], family=FONT_FAMILY),
            showarrow=False,
            yshift=15 if is_bullish else -15,
            bgcolor="rgba(0, 0, 0, 0.6)",
            borderpad=2,
        )
    
    # QM-Zone (shaded area around P3/P4)
    if geometry.get("p3_price") and geometry.get("p4_price"):
        p3_price = geometry["p3_price"]
        p4_price = geometry["p4_price"]
        
        # Create QM-Zone rectangle
        fig.add_shape(
            type="rect",
            x0=geometry["p3_timestamp"],
            x1=geometry["p5_timestamp"],
            y0=min(p3_price, p4_price),
            y1=max(p3_price, p4_price),
            fillcolor=STRUCTURE_COLORS["qm_zone"],
            line=dict(width=0),
            layer="below",
        )
        
        # QM-Zone label
        mid_time = geometry["p4_timestamp"]
        mid_price = (p3_price + p4_price) / 2
        fig.add_annotation(
            x=mid_time,
            y=mid_price,
            text="<b>QM-ZONE</b>",
            font=dict(size=10, color=QML_LINE_COLOR, family=FONT_FAMILY),
            showarrow=False,
            opacity=0.7,
        )
    
    return fig


def _apply_professional_styling(fig: go.Figure) -> go.Figure:
    """Apply professional dark theme styling."""
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(
            family=FONT_FAMILY,
            color="#E2E8F0",
        ),
        title=dict(
            font=dict(size=18, color="#F1F5F9"),
        ),
        xaxis=dict(
            gridcolor="#1E293B",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#1E293B",
            showgrid=True,
            zeroline=False,
            tickformat="$,.0f",
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=80, t=60, b=40),
        hovermode="x unified",
    )
    
    return fig


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test() -> go.Figure:
    """
    Test the pattern visualization with real data from the database.
    
    Connects to experiments.db, fetches a pattern from ml_pattern_registry,
    loads corresponding OHLCV data, and generates a complete visualization.
    
    Returns:
        The final Plotly Figure object
    """
    print("\n" + "=" * 60)
    print("🎯 Pattern Visualization Engine - Test Mode")
    print("=" * 60)
    
    # Database path
    db_path = Path("results/experiments.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    # Connect and fetch a pattern
    print("\n📊 Fetching pattern from ml_pattern_registry...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    
    # First try to find a pattern WITH geometry data
    cursor.execute("""
        SELECT pattern_id, symbol, timeframe, detection_time, pattern_type,
               features_json, validity_score, ml_confidence
        FROM ml_pattern_registry
        WHERE features_json LIKE '%p1_timestamp%'
        ORDER BY detection_time DESC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    
    # Fallback to any pattern if none with geometry
    if row is None:
        print("   ⚠️ No patterns with full geometry, trying any pattern...")
        cursor.execute("""
            SELECT pattern_id, symbol, timeframe, detection_time, pattern_type,
                   features_json, validity_score, ml_confidence
            FROM ml_pattern_registry
            ORDER BY detection_time DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
    
    conn.close()
    
    if row is None:
        raise ValueError("No patterns found in ml_pattern_registry")
    
    # Convert to dict
    pattern_record = dict(row)
    
    print(f"   ✅ Pattern ID: {pattern_record['pattern_id'][:8]}...")
    print(f"   ✅ Symbol: {pattern_record['symbol']}")
    print(f"   ✅ Timeframe: {pattern_record['timeframe']}")
    print(f"   ✅ Type: {pattern_record['pattern_type']}")
    print(f"   ✅ ML Confidence: {pattern_record['ml_confidence']}")
    
    # Debug: Print features_json structure
    print(f"\n📋 Features JSON structure:")
    try:
        features = json.loads(pattern_record.get('features_json', '{}'))
        for key, val in features.items():
            print(f"   {key}: {val}")
        if not features:
            print("   (empty)")
    except Exception as e:
        print(f"   Error parsing: {e}")
    
    # Load OHLCV data
    symbol_clean = pattern_record["symbol"].replace("/", "")
    timeframe = pattern_record["timeframe"]
    parquet_path = Path(f"data/processed/{symbol_clean}/{timeframe}_master.parquet")
    
    print(f"\n📈 Loading OHLCV data from {parquet_path}...")
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    ohlcv_df = pd.read_parquet(parquet_path)
    print(f"   ✅ Loaded {len(ohlcv_df)} candles")
    
    # Normalize column names to lowercase
    ohlcv_df.columns = ohlcv_df.columns.str.lower()
    
    # Ensure time column is datetime and strip timezone for comparison
    if "time" in ohlcv_df.columns:
        ohlcv_df["time"] = pd.to_datetime(ohlcv_df["time"])
        # Strip timezone if present for consistent comparison
        if ohlcv_df["time"].dt.tz is not None:
            ohlcv_df["time"] = ohlcv_df["time"].dt.tz_localize(None)
    
    # Get detection time and strip timezone
    detection_time = pd.to_datetime(pattern_record["detection_time"])
    if detection_time.tz is not None:
        detection_time = detection_time.tz_localize(None)
    
    # Filter to show MORE context around the pattern (50 candles left, 20 right)
    mask = (ohlcv_df["time"] >= detection_time - pd.Timedelta(days=30)) & \
           (ohlcv_df["time"] <= detection_time + pd.Timedelta(days=10))
    
    display_df = ohlcv_df[mask].copy()
    print(f"   ✅ Filtered to {len(display_df)} candles around detection (30 days before, 10 after)")
    
    # Create base candlestick chart
    print("\n🖼️ Creating base candlestick chart...")
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=display_df["time"],
        open=display_df["open"],
        high=display_df["high"],
        low=display_df["low"],
        close=display_df["close"],
        name="Price",
        increasing_line_color="#22C55E",
        decreasing_line_color="#EF4444",
        increasing_fillcolor="#22C55E",
        decreasing_fillcolor="#EF4444",
    ))
    
    # Add pattern visualization
    print("🎨 Adding pattern visualization layers...")
    fig = add_pattern_to_figure(fig, pattern_record, ohlcv_df)
    
    # Update layout with title
    symbol = pattern_record["symbol"]
    pattern_type = pattern_record["pattern_type"].replace("_", " ").title()
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> - {pattern_type} Pattern Visualization",
            x=0.5,
            xanchor="center",
        ),
        height=700,
        xaxis_rangeslider_visible=False,
    )
    
    # Save and show
    output_path = Path("results/pattern_viz_test.html")
    fig.write_html(output_path)
    print(f"\n✅ Chart saved to: {output_path}")
    
    # Try to open in browser
    try:
        fig.show()
        print("✅ Chart opened in browser")
    except Exception as e:
        print(f"⚠️ Could not open browser: {e}")
        print(f"   Open {output_path} manually to view")
    
    print("\n" + "=" * 60)
    print("🎯 Test Complete!")
    print("=" * 60 + "\n")
    
    return fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    test()
