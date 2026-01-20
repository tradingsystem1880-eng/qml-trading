"""
QML Trading Dashboard - Professional Design System
===================================================
TradingView-grade design tokens and styling constants.

This module defines the complete visual language for the dashboard:
- Color palette (backgrounds, charts, annotations, positions)
- Typography (fonts, sizes, weights)
- Spacing and layout constants
- Component styling presets
"""

from dataclasses import dataclass
from typing import Dict

# =============================================================================
# COLOR PALETTE
# =============================================================================

@dataclass(frozen=True)
class Colors:
    """Immutable color definitions for the entire application."""

    # -------------------------------------------------------------------------
    # BACKGROUNDS
    # -------------------------------------------------------------------------
    BG_PRIMARY: str = "#0a0a0f"        # Main app background (darkest)
    BG_SECONDARY: str = "#12121a"      # Card/panel backgrounds
    BG_TERTIARY: str = "#1a1a24"       # Hover states, subtle borders
    BG_ELEVATED: str = "#22222e"       # Modals, dropdowns, floating elements
    BG_INPUT: str = "#16161e"          # Input field backgrounds

    # -------------------------------------------------------------------------
    # CHART
    # -------------------------------------------------------------------------
    CHART_BG: str = "#0d0d12"
    CHART_GRID: str = "rgba(255, 255, 255, 0.03)"
    CHART_AXIS: str = "#666666"
    CHART_CROSSHAIR: str = "rgba(255, 255, 255, 0.3)"

    # Candles - TradingView style
    CANDLE_BULLISH: str = "#26a69a"
    CANDLE_BEARISH: str = "#ef5350"

    # Volume
    VOLUME_BULLISH: str = "rgba(38, 166, 154, 0.3)"
    VOLUME_BEARISH: str = "rgba(239, 83, 80, 0.3)"

    # -------------------------------------------------------------------------
    # PATTERN ANNOTATIONS
    # -------------------------------------------------------------------------
    PATTERN_LINE: str = "#3498db"              # Blue dashed lines
    PATTERN_LINE_RGBA: str = "rgba(52, 152, 219, 0.8)"
    SWING_LABEL: str = "#3498db"               # Numbered labels (1-5)
    STRUCTURE_LABEL: str = "#888888"           # HH, HL, LH, LL

    # Structure breaks
    BOS_COLOR: str = "#f39c12"                 # Orange - Break of Structure
    CHOCH_COLOR: str = "#9b59b6"               # Purple - Change of Character

    # Zones
    DEMAND_ZONE: str = "rgba(38, 166, 91, 0.15)"
    DEMAND_BORDER: str = "rgba(38, 166, 91, 0.4)"
    SUPPLY_ZONE: str = "rgba(231, 76, 60, 0.15)"
    SUPPLY_BORDER: str = "rgba(231, 76, 60, 0.4)"
    FVG_ZONE: str = "rgba(155, 89, 182, 0.15)"
    FVG_BORDER: str = "rgba(155, 89, 182, 0.4)"
    QM_ZONE: str = "rgba(52, 152, 219, 0.15)"
    QM_BORDER: str = "rgba(52, 152, 219, 0.4)"

    # Liquidity
    LIQUIDITY: str = "#e74c3c"
    LIQUIDITY_LABEL: str = "#ffeb3b"

    # -------------------------------------------------------------------------
    # POSITION BOX COLORS (CRITICAL - MATCH REFERENCE IMAGES EXACTLY)
    # -------------------------------------------------------------------------
    # Take profit zones
    TP_PRIMARY: str = "rgba(38, 166, 91, 0.25)"        # TP1 - more visible
    TP_SECONDARY: str = "rgba(38, 166, 91, 0.15)"     # TP2, TP3 - lighter
    TP_BORDER: str = "rgba(38, 166, 91, 0.5)"
    TP_TEXT: str = "#26a69a"

    # Stop loss zone
    SL_FILL: str = "rgba(231, 76, 60, 0.25)"
    SL_BORDER: str = "rgba(231, 76, 60, 0.5)"
    SL_TEXT: str = "#ef5350"

    # Entry line
    ENTRY_LINE: str = "#3498db"

    # -------------------------------------------------------------------------
    # TEXT
    # -------------------------------------------------------------------------
    TEXT_PRIMARY: str = "#ffffff"
    TEXT_SECONDARY: str = "#a0a0a0"
    TEXT_TERTIARY: str = "#666666"
    TEXT_DISABLED: str = "#444444"

    # Semantic text
    TEXT_PROFIT: str = "#26a69a"
    TEXT_LOSS: str = "#ef5350"
    TEXT_WARNING: str = "#f39c12"
    TEXT_INFO: str = "#3498db"

    # -------------------------------------------------------------------------
    # INTERACTIVE
    # -------------------------------------------------------------------------
    ACCENT_PRIMARY: str = "#3498db"
    ACCENT_PRIMARY_HOVER: str = "#2980b9"
    ACCENT_SUCCESS: str = "#26a69a"
    ACCENT_SUCCESS_HOVER: str = "#1e8e82"
    ACCENT_DANGER: str = "#ef5350"
    ACCENT_DANGER_HOVER: str = "#d32f2f"
    ACCENT_WARNING: str = "#f39c12"

    # -------------------------------------------------------------------------
    # BORDERS
    # -------------------------------------------------------------------------
    BORDER_SUBTLE: str = "rgba(255, 255, 255, 0.06)"
    BORDER_DEFAULT: str = "rgba(255, 255, 255, 0.1)"
    BORDER_STRONG: str = "rgba(255, 255, 255, 0.2)"
    BORDER_FOCUS: str = "#3498db"


# =============================================================================
# TYPOGRAPHY
# =============================================================================

@dataclass(frozen=True)
class Typography:
    """Typography system with fonts and sizes."""

    # Font families
    FONT_SANS: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    FONT_MONO: str = "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace"

    # Font sizes (in pixels for CSS, rem for responsive)
    SIZE_XXS: str = "9px"      # Micro labels (chart axis ticks)
    SIZE_XS: str = "10px"      # Chart annotations, timestamps
    SIZE_SM: str = "11px"      # Secondary data, table cells
    SIZE_BASE: str = "12px"    # Default body text
    SIZE_MD: str = "13px"      # Primary data display
    SIZE_LG: str = "14px"      # Section headers, important labels
    SIZE_XL: str = "16px"      # Page titles, key metrics
    SIZE_XXL: str = "20px"     # Dashboard hero numbers
    SIZE_XXXL: str = "24px"    # Large display numbers

    # Font weights
    WEIGHT_NORMAL: int = 400
    WEIGHT_MEDIUM: int = 500
    WEIGHT_SEMIBOLD: int = 600
    WEIGHT_BOLD: int = 700

    # Line heights
    LINE_TIGHT: float = 1.1      # Numbers, single-line labels
    LINE_SNUG: float = 1.25      # Compact text
    LINE_NORMAL: float = 1.5     # Body text
    LINE_RELAXED: float = 1.75   # Paragraphs


# =============================================================================
# SPACING & LAYOUT
# =============================================================================

@dataclass(frozen=True)
class Spacing:
    """Spacing system based on 4px grid."""

    PX: str = "1px"
    XS: str = "4px"
    SM: str = "8px"
    MD: str = "12px"
    LG: str = "16px"
    XL: str = "24px"
    XXL: str = "32px"
    XXXL: str = "48px"


@dataclass(frozen=True)
class Layout:
    """Layout constants for panels, sidebar, etc."""

    # Sidebar
    SIDEBAR_COLLAPSED: str = "60px"
    SIDEBAR_EXPANDED: str = "200px"

    # Header
    HEADER_HEIGHT: str = "52px"

    # Panels
    PANEL_MIN_WIDTH: str = "280px"
    PANEL_MAX_WIDTH: str = "400px"
    PANEL_PADDING: str = "16px"

    # Chart
    CHART_MIN_HEIGHT: str = "400px"
    CHART_TOOLBAR_HEIGHT: str = "40px"

    # Tables
    TABLE_HEADER_HEIGHT: str = "36px"
    TABLE_ROW_HEIGHT: str = "40px"

    # Border radius
    RADIUS_SM: str = "4px"
    RADIUS_MD: str = "6px"
    RADIUS_LG: str = "8px"
    RADIUS_XL: str = "12px"


# =============================================================================
# CHART-SPECIFIC STYLING
# =============================================================================

@dataclass(frozen=True)
class ChartStyle:
    """Chart-specific styling constants."""

    # Pattern line styling
    PATTERN_LINE_WIDTH: float = 1.5
    PATTERN_LINE_DASH: str = "6, 4"  # 6px dash, 4px gap

    # Swing point styling
    SWING_POINT_FONT_SIZE: str = "11px"
    SWING_POINT_OFFSET: int = 8  # pixels from candle

    # Position box styling
    POSITION_BOX_BORDER_WIDTH: int = 1
    ENTRY_LINE_WIDTH: int = 2

    # Label styling
    LABEL_FONT_SIZE: str = "10px"
    LABEL_PADDING: str = "2px 6px"
    LABEL_BORDER_RADIUS: str = "2px"

    # Zone opacity
    ZONE_OPACITY: float = 0.15
    ZONE_BORDER_OPACITY: float = 0.4


# =============================================================================
# JARVIS THEME (Alternative - Iron Man inspired)
# =============================================================================

@dataclass(frozen=True)
class JARVISColors:
    """JARVIS/Iron Man inspired color palette with neon accents."""

    # Backgrounds - Deep space
    BG_VOID: str = "#020408"
    BG_PRIMARY: str = "#040810"
    BG_SECONDARY: str = "#0a1018"
    BG_CARD: str = "#0c141e"
    BG_CARD_HOVER: str = "#101a28"
    BG_ELEVATED: str = "#141e2c"

    # Borders with glow potential
    BORDER_DARK: str = "#152030"
    BORDER_COLOR: str = "#1e3048"
    BORDER_GLOW: str = "rgba(0, 212, 170, 0.15)"

    # Text
    TEXT_PRIMARY: str = "#e8f0f8"
    TEXT_SECONDARY: str = "#7890a8"
    TEXT_MUTED: str = "#4a6080"

    # Neon accents
    ACCENT: str = "#00ffcc"
    ACCENT_DIM: str = "#00d4aa"
    ACCENT_SOFT: str = "rgba(0, 255, 204, 0.08)"
    ACCENT_GLOW: str = "rgba(0, 255, 204, 0.4)"

    # Semantic
    PROFIT: str = "#00ffcc"
    PROFIT_SOFT: str = "rgba(0, 255, 204, 0.15)"
    LOSS: str = "#ff4757"
    LOSS_SOFT: str = "rgba(255, 71, 87, 0.15)"
    WARNING: str = "#ffc107"
    INFO: str = "#00b4ff"
    PURPLE: str = "#bf5af2"

    # Chart
    CHART_BG: str = "#0b1018"
    CHART_BULLISH: str = "#00d4aa"
    CHART_BEARISH: str = "#ff5252"

    # Position boxes
    TP_PRIMARY: str = "rgba(0, 212, 170, 0.25)"
    TP_SECONDARY: str = "rgba(0, 212, 170, 0.12)"
    SL_FILL: str = "rgba(255, 82, 82, 0.25)"
    ENTRY_LINE: str = "#4da6ff"


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

COLORS = Colors()
JARVIS = JARVISColors()
TYPOGRAPHY = Typography()
SPACING = Spacing()
LAYOUT = Layout()
CHART_STYLE = ChartStyle()


# =============================================================================
# CSS GENERATION
# =============================================================================

def generate_chart_theme_config() -> Dict:
    """Generate lightweight-charts theme configuration."""
    return {
        "layout": {
            "background": {"color": COLORS.CHART_BG},
            "textColor": COLORS.TEXT_SECONDARY,
        },
        "grid": {
            "vertLines": {"color": COLORS.CHART_GRID},
            "horzLines": {"color": COLORS.CHART_GRID},
        },
        "crosshair": {
            "mode": 1,  # Normal crosshair
            "vertLine": {
                "color": COLORS.CHART_CROSSHAIR,
                "width": 1,
                "style": 1,  # Dashed
                "labelBackgroundColor": COLORS.BG_ELEVATED,
            },
            "horzLine": {
                "color": COLORS.CHART_CROSSHAIR,
                "width": 1,
                "style": 1,
                "labelBackgroundColor": COLORS.BG_ELEVATED,
            },
        },
        "rightPriceScale": {
            "borderColor": COLORS.BORDER_DEFAULT,
            "textColor": COLORS.CHART_AXIS,
        },
        "timeScale": {
            "borderColor": COLORS.BORDER_DEFAULT,
            "textColor": COLORS.CHART_AXIS,
            "timeVisible": True,
            "secondsVisible": False,
        },
    }


def generate_candlestick_config() -> Dict:
    """Generate candlestick series configuration."""
    return {
        "upColor": COLORS.CANDLE_BULLISH,
        "downColor": COLORS.CANDLE_BEARISH,
        "borderUpColor": COLORS.CANDLE_BULLISH,
        "borderDownColor": COLORS.CANDLE_BEARISH,
        "wickUpColor": COLORS.CANDLE_BULLISH,
        "wickDownColor": COLORS.CANDLE_BEARISH,
    }


def generate_dashboard_css() -> str:
    """Generate complete dashboard CSS with design tokens."""
    return f'''
<style>
/* ==========================================================================
   QML Trading Dashboard - Professional Theme
   ========================================================================== */

/* --------------------------------------------------------------------------
   CSS CUSTOM PROPERTIES (Design Tokens)
   -------------------------------------------------------------------------- */
:root {{
    /* Backgrounds */
    --bg-primary: {COLORS.BG_PRIMARY};
    --bg-secondary: {COLORS.BG_SECONDARY};
    --bg-tertiary: {COLORS.BG_TERTIARY};
    --bg-elevated: {COLORS.BG_ELEVATED};
    --bg-input: {COLORS.BG_INPUT};

    /* Chart */
    --chart-bg: {COLORS.CHART_BG};
    --candle-bullish: {COLORS.CANDLE_BULLISH};
    --candle-bearish: {COLORS.CANDLE_BEARISH};

    /* Text */
    --text-primary: {COLORS.TEXT_PRIMARY};
    --text-secondary: {COLORS.TEXT_SECONDARY};
    --text-tertiary: {COLORS.TEXT_TERTIARY};
    --text-profit: {COLORS.TEXT_PROFIT};
    --text-loss: {COLORS.TEXT_LOSS};

    /* Accents */
    --accent-primary: {COLORS.ACCENT_PRIMARY};
    --accent-success: {COLORS.ACCENT_SUCCESS};
    --accent-danger: {COLORS.ACCENT_DANGER};
    --accent-warning: {COLORS.ACCENT_WARNING};

    /* Borders */
    --border-subtle: {COLORS.BORDER_SUBTLE};
    --border-default: {COLORS.BORDER_DEFAULT};
    --border-strong: {COLORS.BORDER_STRONG};

    /* Typography */
    --font-sans: {TYPOGRAPHY.FONT_SANS};
    --font-mono: {TYPOGRAPHY.FONT_MONO};

    /* Spacing */
    --spacing-xs: {SPACING.XS};
    --spacing-sm: {SPACING.SM};
    --spacing-md: {SPACING.MD};
    --spacing-lg: {SPACING.LG};
    --spacing-xl: {SPACING.XL};

    /* Border Radius */
    --radius-sm: {LAYOUT.RADIUS_SM};
    --radius-md: {LAYOUT.RADIUS_MD};
    --radius-lg: {LAYOUT.RADIUS_LG};
}}

/* --------------------------------------------------------------------------
   BASE STYLES
   -------------------------------------------------------------------------- */
.stApp {{
    background-color: var(--bg-primary) !important;
    font-family: var(--font-sans);
}}

/* Hide Streamlit branding */
#MainMenu, footer, header {{visibility: hidden;}}
.stDeployButton {{display: none;}}

/* Main content */
.main .block-container {{
    padding: 1.5rem 2rem;
    max-width: 100%;
}}

/* --------------------------------------------------------------------------
   SIDEBAR
   -------------------------------------------------------------------------- */
section[data-testid="stSidebar"] {{
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
}}

section[data-testid="stSidebar"] .stMarkdown {{
    color: var(--text-secondary);
}}

/* --------------------------------------------------------------------------
   METRICS
   -------------------------------------------------------------------------- */
[data-testid="stMetric"] {{
    background-color: var(--bg-secondary);
    padding: 1rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
}}

[data-testid="stMetric"] label {{
    color: var(--text-tertiary);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-weight: 600;
}}

/* Positive/Negative deltas */
[data-testid="stMetricDelta"] svg {{
    display: none;
}}

[data-testid="stMetricDelta"][data-testid-delta="positive"] {{
    color: var(--text-profit) !important;
}}

[data-testid="stMetricDelta"][data-testid-delta="negative"] {{
    color: var(--text-loss) !important;
}}

/* --------------------------------------------------------------------------
   BUTTONS
   -------------------------------------------------------------------------- */
.stButton > button {{
    background-color: var(--accent-primary);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    font-weight: 500;
    font-family: var(--font-sans);
    padding: 0.5rem 1rem;
    transition: background-color 0.15s ease;
}}

.stButton > button:hover {{
    background-color: {COLORS.ACCENT_PRIMARY_HOVER};
}}

.stButton > button[kind="secondary"] {{
    background-color: var(--bg-tertiary);
    border: 1px solid var(--border-default);
    color: var(--text-primary);
}}

.stButton > button[kind="secondary"]:hover {{
    background-color: var(--bg-elevated);
}}

/* --------------------------------------------------------------------------
   INPUTS
   -------------------------------------------------------------------------- */
.stTextInput input,
.stNumberInput input,
.stSelectbox > div > div {{
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans);
}}

.stTextInput input:focus,
.stNumberInput input:focus {{
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 1px var(--accent-primary) !important;
}}

/* Labels */
.stTextInput label,
.stNumberInput label,
.stSelectbox label {{
    color: var(--text-secondary) !important;
    font-size: 0.8rem;
    font-weight: 500;
}}

/* --------------------------------------------------------------------------
   TABS
   -------------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background-color: var(--bg-secondary);
    border-radius: var(--radius-lg);
    padding: 4px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: var(--text-secondary);
    border-radius: var(--radius-md);
    padding: 8px 16px;
    font-weight: 500;
    border: none;
}}

.stTabs [aria-selected="true"] {{
    background-color: var(--bg-tertiary);
    color: var(--text-primary);
}}

/* --------------------------------------------------------------------------
   EXPANDERS
   -------------------------------------------------------------------------- */
.streamlit-expanderHeader {{
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    color: var(--text-primary);
}}

/* --------------------------------------------------------------------------
   DATA TABLES
   -------------------------------------------------------------------------- */
.stDataFrame {{
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
}}

.stDataFrame thead tr th {{
    background-color: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    border-bottom: 1px solid var(--border-default);
}}

.stDataFrame tbody tr td {{
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    border-bottom: 1px solid var(--border-subtle);
}}

.stDataFrame tbody tr:hover td {{
    background-color: var(--bg-secondary) !important;
}}

/* --------------------------------------------------------------------------
   ALERTS
   -------------------------------------------------------------------------- */
.stSuccess {{
    background-color: rgba(38, 166, 154, 0.1);
    border: 1px solid var(--accent-success);
    border-radius: var(--radius-md);
}}

.stError {{
    background-color: rgba(239, 83, 80, 0.1);
    border: 1px solid var(--accent-danger);
    border-radius: var(--radius-md);
}}

.stWarning {{
    background-color: rgba(243, 156, 18, 0.1);
    border: 1px solid var(--accent-warning);
    border-radius: var(--radius-md);
}}

.stInfo {{
    background-color: rgba(52, 152, 219, 0.1);
    border: 1px solid var(--accent-primary);
    border-radius: var(--radius-md);
}}

/* --------------------------------------------------------------------------
   SCROLLBAR
   -------------------------------------------------------------------------- */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: var(--bg-primary);
}}

::-webkit-scrollbar-thumb {{
    background: var(--border-default);
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: var(--border-strong);
}}

/* --------------------------------------------------------------------------
   CUSTOM COMPONENTS
   -------------------------------------------------------------------------- */

/* Price display - positive */
.price-positive {{
    color: var(--text-profit);
    font-family: var(--font-mono);
    font-weight: 600;
}}

/* Price display - negative */
.price-negative {{
    color: var(--text-loss);
    font-family: var(--font-mono);
    font-weight: 600;
}}

/* Monospace numbers */
.mono {{
    font-family: var(--font-mono);
}}

/* Card component */
.card {{
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
}}

.card-header {{
    color: var(--text-secondary);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: var(--spacing-sm);
}}

.card-value {{
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 700;
}}

/* Status badges */
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.badge-success {{
    background-color: rgba(38, 166, 154, 0.2);
    color: var(--accent-success);
}}

.badge-danger {{
    background-color: rgba(239, 83, 80, 0.2);
    color: var(--accent-danger);
}}

.badge-warning {{
    background-color: rgba(243, 156, 18, 0.2);
    color: var(--accent-warning);
}}

.badge-info {{
    background-color: rgba(52, 152, 219, 0.2);
    color: var(--accent-primary);
}}

/* Chart container */
.chart-container {{
    background-color: var(--chart-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    overflow: hidden;
}}

/* Pattern info panel */
.pattern-info {{
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: var(--spacing-md);
}}

.pattern-info-row {{
    display: flex;
    justify-content: space-between;
    padding: var(--spacing-xs) 0;
    border-bottom: 1px solid var(--border-subtle);
}}

.pattern-info-row:last-child {{
    border-bottom: none;
}}

.pattern-info-label {{
    color: var(--text-tertiary);
    font-size: 0.8rem;
}}

.pattern-info-value {{
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-weight: 500;
}}
</style>
'''


# =============================================================================
# STREAMLIT HELPER
# =============================================================================

def apply_design_system():
    """Apply design system CSS to Streamlit app."""
    import streamlit as st
    st.markdown(generate_dashboard_css(), unsafe_allow_html=True)
