# =============================================================================
# 🎯 QML PREMIUM PATTERN VISUALIZER - GOOGLE COLAB VERSION
# =============================================================================
# Copy this ENTIRE block into a Google Colab cell and run it.
# You will be prompted to upload 2 files:
#   1. qml_patterns_export.csv (pattern coordinates)
#   2. btc_ohlcv_export.csv (OHLCV price data)
# =============================================================================

# STEP 1: Install dependencies
!pip install mplfinance -q

# STEP 2: Imports
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

print("✅ Dependencies loaded!")

# =============================================================================
# STEP 3: File Upload
# =============================================================================
print("\n📂 Please upload the 2 export files:")
print("   1. qml_patterns_export.csv")
print("   2. btc_ohlcv_export.csv")
print("\n⬇️ Click 'Choose Files' below and select BOTH files:")

uploaded = files.upload()

# Parse uploaded files
patterns_df = None
ohlcv_df = None

for filename, content in uploaded.items():
    if 'pattern' in filename.lower():
        patterns_df = pd.read_csv(filename, parse_dates=[
            'TS_Date', 'P1_Date', 'P2_Date', 'P3_Date', 'P4_Date', 'P5_Date'
        ])
        print(f"✅ Loaded patterns: {len(patterns_df)} patterns")
    elif 'ohlcv' in filename.lower():
        ohlcv_df = pd.read_csv(filename, parse_dates=['time'])
        ohlcv_df = ohlcv_df.set_index('time')
        print(f"✅ Loaded OHLCV: {len(ohlcv_df)} bars")

if patterns_df is None or ohlcv_df is None:
    print("❌ ERROR: Please upload both qml_patterns_export.csv and btc_ohlcv_export.csv")
else:
    print("\n🎉 Data loaded successfully! Run the next cell to visualize patterns.")

# =============================================================================
# STEP 4: PREMIUM VISUALIZATION FUNCTION
# =============================================================================

def plot_premium_qml(pattern_id):
    """
    🎨 PREMIUM QML PATTERN VISUALIZATION
    
    Design Rules:
    - 60+ bars before pattern, 60+ bars after (show trade result)
    - Pattern sits middle-left, future visible on right
    - Clean gray dashed pre-trend line
    - Bold blue QML zig-zag structure
    - Entry marker (blue triangle at P5)
    - Smart labels with dynamic offset (never overlap candles)
    - Trade result boxes (red SL, green TP zones)
    """
    
    if pattern_id < 0 or pattern_id >= len(patterns_df):
        print(f"❌ Invalid pattern_id. Must be 0-{len(patterns_df)-1}")
        return
    
    pattern = patterns_df.iloc[pattern_id]
    is_bullish = 'bullish' in pattern['pattern_type']
    
    # ==========================================================================
    # EXTRACT PATTERN COORDINATES
    # ==========================================================================
    
    points = {
        'TS': (pd.Timestamp(pattern['TS_Date']), float(pattern['TS_Price'])),
        'P1': (pd.Timestamp(pattern['P1_Date']), float(pattern['P1_Price'])),
        'P2': (pd.Timestamp(pattern['P2_Date']), float(pattern['P2_Price'])),
        'P3': (pd.Timestamp(pattern['P3_Date']), float(pattern['P3_Price'])),
        'P4': (pd.Timestamp(pattern['P4_Date']), float(pattern['P4_Price'])),
        'P5': (pd.Timestamp(pattern['P5_Date']), float(pattern['P5_Price'])),
    }
    
    entry = float(pattern['entry_price'])
    sl = float(pattern['stop_loss'])
    tp = float(pattern['take_profit'])
    
    # ==========================================================================
    # SLICE DATA WITH 60+ BARS PADDING (handle out of bounds)
    # ==========================================================================
    
    # Find indices
    ts_idx = ohlcv_df.index.get_indexer([points['TS'][0]], method='nearest')[0]
    p1_idx = ohlcv_df.index.get_indexer([points['P1'][0]], method='nearest')[0]
    p5_idx = ohlcv_df.index.get_indexer([points['P5'][0]], method='nearest')[0]
    
    # Calculate padding (60 bars before TS, 60 bars after P5)
    start_idx = max(0, ts_idx - 60)
    end_idx = min(len(ohlcv_df) - 1, p5_idx + 60)
    
    chart_df = ohlcv_df.iloc[start_idx:end_idx + 1].copy()
    
    if len(chart_df) < 30:
        print(f"⚠️ Warning: Limited data available ({len(chart_df)} bars)")
    
    # ==========================================================================
    # CALCULATE PRICE RANGE FOR DYNAMIC OFFSETS
    # ==========================================================================
    
    price_max = chart_df['High'].max()
    price_min = chart_df['Low'].min()
    price_range = price_max - price_min
    label_offset = price_range * 0.04  # 4% offset for labels
    
    # ==========================================================================
    # BUILD ALINES: Pre-Trend (gray dashed) + QML Structure (blue solid)
    # ==========================================================================
    
    # Pre-trend line (thin, dashed, gray)
    trend_line = [
        (points['TS'][0], points['TS'][1]),
        (points['P1'][0], points['P1'][1]),
    ]
    
    # QML zig-zag (solid, blue, bold)
    qml_line = [
        (points['P1'][0], points['P1'][1]),
        (points['P2'][0], points['P2'][1]),
        (points['P3'][0], points['P3'][1]),
        (points['P4'][0], points['P4'][1]),
        (points['P5'][0], points['P5'][1]),
    ]
    
    alines_spec = dict(
        alines=[trend_line, qml_line],
        colors=['#808080', '#2962ff'],  # Gray, Blue
        linewidths=[1.2, 2.5],
        linestyle=['-', '-'],
    )
    
    # ==========================================================================
    # CREATE YAHOO-STYLE CHART
    # ==========================================================================
    
    mc = mpf.make_marketcolors(
        up='#26a69a',
        down='#ef5350',
        edge='inherit',
        wick='inherit',
    )
    style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridstyle='-',
        gridcolor='#e8e8e8',
        facecolor='white',
        figcolor='white',
    )
    
    title = f"QML {pattern['pattern_type'].upper()} | Pattern #{pattern_id} | Validity: {pattern['validity_score']:.2f}"
    
    fig, axes = mpf.plot(
        chart_df,
        type='candle',
        style=style,
        title=title,
        ylabel='Price (USDT)',
        volume=False,
        figsize=(18, 10),
        alines=alines_spec,
        returnfig=True,
        tight_layout=True,
        scale_padding={'left': 0.1, 'right': 0.1, 'top': 0.3, 'bottom': 0.2},
    )
    
    ax = axes[0]
    
    # ==========================================================================
    # GET X-AXIS POSITIONS FOR ANNOTATIONS
    # ==========================================================================
    
    def get_x_pos(timestamp):
        """Convert timestamp to x-axis position."""
        if timestamp in chart_df.index:
            return list(chart_df.index).index(timestamp)
        else:
            return chart_df.index.get_indexer([timestamp], method='nearest')[0]
    
    p5_x = get_x_pos(points['P5'][0])
    x_max = len(chart_df) - 1
    
    # ==========================================================================
    # DRAW TRADE RESULT BOXES (matplotlib.patches.Rectangle)
    # ==========================================================================
    
    box_width = x_max - p5_x  # Extend to right edge
    
    if is_bullish:
        # STOP LOSS: Red box from entry DOWN to SL
        sl_height = entry - sl
        sl_rect = patches.Rectangle(
            (p5_x, sl), box_width, sl_height,
            linewidth=0, edgecolor='none',
            facecolor='#ef5350', alpha=0.25, zorder=0
        )
        ax.add_patch(sl_rect)
        
        # TAKE PROFIT: Green box from entry UP to TP
        tp_height = tp - entry
        tp_rect = patches.Rectangle(
            (p5_x, entry), box_width, tp_height,
            linewidth=0, edgecolor='none',
            facecolor='#26a69a', alpha=0.25, zorder=0
        )
        ax.add_patch(tp_rect)
    else:
        # STOP LOSS: Red box from entry UP to SL (bearish)
        sl_height = sl - entry
        sl_rect = patches.Rectangle(
            (p5_x, entry), box_width, sl_height,
            linewidth=0, edgecolor='none',
            facecolor='#ef5350', alpha=0.25, zorder=0
        )
        ax.add_patch(sl_rect)
        
        # TAKE PROFIT: Green box from entry DOWN to TP
        tp_height = entry - tp
        tp_rect = patches.Rectangle(
            (p5_x, tp), box_width, tp_height,
            linewidth=0, edgecolor='none',
            facecolor='#26a69a', alpha=0.25, zorder=0
        )
        ax.add_patch(tp_rect)
    
    # ==========================================================================
    # ADD ENTRY MARKER (Blue Triangle at P5)
    # ==========================================================================
    
    ax.plot(
        p5_x, entry,
        marker='^' if is_bullish else 'v',
        markersize=14,
        color='#2962ff',
        markeredgecolor='white',
        markeredgewidth=1.5,
        zorder=15,
    )
    
    # ==========================================================================
    # SMART LABELS: Dynamic offset, clean white boxes
    # ==========================================================================
    
    label_style = dict(
        fontsize=10,
        fontweight='bold',
        color='#333333',
        ha='center',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='#888888',
            alpha=0.92,
            linewidth=0.8,
        ),
        zorder=20,
    )
    
    # Label positions with smart offsets
    labels_config = {
        'P2': ('LS', 'above'),
        'P3': ('H', 'above' if not is_bullish else 'below'),
        'P4': ('LL', 'below' if is_bullish else 'above'),
        'P5': ('RS', 'above'),
    }
    
    for pt_name, (label_text, position) in labels_config.items():
        pt_time, pt_price = points[pt_name]
        x_pos = get_x_pos(pt_time)
        
        if position == 'above':
            y_pos = pt_price + label_offset
            va = 'bottom'
        else:
            y_pos = pt_price - label_offset
            va = 'top'
        
        ax.annotate(
            label_text,
            xy=(x_pos, pt_price),
            xytext=(x_pos, y_pos),
            va=va,
            **label_style
        )
    
    # ==========================================================================
    # ENTRY/SL/TP INFO BOX (top-right)
    # ==========================================================================
    
    outcome = "BULLISH ▲" if is_bullish else "BEARISH ▼"
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else 0
    
    info_text = (
        f"{outcome}\n"
        f"Entry: ${entry:,.2f}\n"
        f"SL: ${sl:,.2f}\n"
        f"TP: ${tp:,.2f}\n"
        f"R:R = 1:{rr:.1f}"
    )
    
    ax.text(
        0.98, 0.97, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='#cccccc',
            alpha=0.95,
            linewidth=1,
        ),
    )
    
    # ==========================================================================
    # HORIZONTAL LINES: Entry, SL, TP (subtle)
    # ==========================================================================
    
    ax.axhline(y=entry, color='#2962ff', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=sl, color='#ef5350', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=tp, color='#26a69a', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.show()
    
    # ==========================================================================
    # CONSOLE OUTPUT
    # ==========================================================================
    
    print(f"\n{'='*70}")
    print(f"📊 Pattern #{pattern_id}: {pattern['pattern_type'].upper()}")
    print(f"   Validity Score: {pattern['validity_score']:.3f}")
    print(f"   Entry: ${entry:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f}")
    print(f"   Risk:Reward = 1:{rr:.2f}")
    print(f"   Chart Range: {len(chart_df)} bars ({chart_df.index[0]} to {chart_df.index[-1]})")
    print(f"{'='*70}")


print("✅ plot_premium_qml() function ready!")
print("   Usage: plot_premium_qml(0)  # Pattern ID 0-39")

# =============================================================================
# STEP 5: Interactive Pattern Selector
# =============================================================================

print("\n🎛️ Use the dropdown below to select a pattern:")

pattern_dropdown = widgets.Dropdown(
    options=[(f"#{i}: {row['pattern_type']} | Validity: {row['validity_score']:.2f}", i) 
             for i, row in patterns_df.iterrows()],
    value=0,
    description='Pattern:',
    style={'description_width': '80px'},
    layout=widgets.Layout(width='400px')
)

plot_button = widgets.Button(
    description='📊 Plot Pattern',
    button_style='primary',
    layout=widgets.Layout(width='150px')
)

def on_button_click(b):
    clear_output(wait=True)
    display(widgets.HBox([pattern_dropdown, plot_button]))
    plot_premium_qml(pattern_dropdown.value)

plot_button.on_click(on_button_click)

display(widgets.HBox([pattern_dropdown, plot_button]))
print("\n👆 Select pattern from dropdown and click 'Plot Pattern'")
print("   Or call directly: plot_premium_qml(8)  # for highest validity pattern")
