"""
QML Dashboard Theme Module - Bloomberg Terminal Edition
========================================================

Professional trading terminal theme inspired by Bloomberg Terminal.
Features pure black background, amber/orange accents, and monospace fonts.

Color Reference:
- Background: Pure black (#000000)
- Accent: Bloomberg amber (#ff6600)
- Positive: Terminal green (#00ff00)
- Negative: Alert red (#ff3333)
"""

from typing import Dict

# ============================================================================
# BLOOMBERG TERMINAL COLOR PALETTE
# ============================================================================
COLORS = {
    # Backgrounds - Pure black to dark gray
    "bg_terminal": "#000000",      # Pure black (main background)
    "bg_panel": "#0a0a0a",         # Panel background
    "bg_card": "#111111",          # Card background
    "bg_elevated": "#1a1a1a",      # Elevated elements
    "bg_hover": "#222222",         # Hover state
    
    # Borders
    "border_default": "#333333",
    "border_subtle": "#222222",
    "border_accent": "#ff6600",
    
    # Text - High contrast
    "text_primary": "#ffffff",     # Pure white
    "text_secondary": "#888888",   # Gray
    "text_muted": "#555555",       # Muted gray
    "text_amber": "#ff9900",       # Amber (for highlights)
    
    # Bloomberg Accent Colors
    "accent_amber": "#ff6600",     # Primary accent (Bloomberg orange)
    "accent_gold": "#ffaa00",      # Secondary accent
    "accent_yellow": "#ffcc00",    # Tertiary
    
    # Trading Colors
    "bullish": "#00ff00",          # Terminal green
    "bullish_muted": "rgba(0, 255, 0, 0.15)",
    "bearish": "#ff3333",          # Alert red
    "bearish_muted": "rgba(255, 51, 51, 0.15)",
    "neutral": "#888888",          # Gray
    
    # Chart Colors
    "chart_1": "#ff6600",          # Orange
    "chart_2": "#00aaff",          # Blue
    "chart_3": "#ffcc00",          # Yellow
    "chart_4": "#ff3399",          # Pink
    "chart_5": "#00ff88",          # Teal
}

# ============================================================================
# GRADIENTS
# ============================================================================
GRADIENTS = {
    "amber": "linear-gradient(90deg, #ff3300 0%, #ff6600 50%, #ffaa00 100%)",
    "bullish": "linear-gradient(90deg, #00aa00 0%, #00ff00 100%)",
    "bearish": "linear-gradient(90deg, #cc0000 0%, #ff3333 100%)",
    "blue": "linear-gradient(90deg, #0066ff 0%, #00aaff 100%)",
    "gold": "linear-gradient(135deg, #ff8c00 0%, #ffd700 100%)",
}

# ============================================================================
# BLOOMBERG TERMINAL CSS
# ============================================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* ========================================
   ROOT VARIABLES - Bloomberg Terminal
   ======================================== */
:root {
    --bg-terminal: #000000;
    --bg-panel: #0a0a0a;
    --bg-card: #111111;
    --bg-elevated: #1a1a1a;
    --bg-hover: #222222;
    
    --border-default: #333333;
    --border-subtle: #222222;
    --border-accent: #ff6600;
    
    --text-primary: #ffffff;
    --text-secondary: #888888;
    --text-muted: #555555;
    --text-amber: #ff9900;
    
    --accent-amber: #ff6600;
    --accent-gold: #ffaa00;
    
    --bullish: #00ff00;
    --bearish: #ff3333;
    
    --font-mono: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ========================================
   GLOBAL STYLES
   ======================================== */
html, body, .stApp {
    background-color: var(--bg-terminal) !important;
    font-family: var(--font-sans) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden !important;}
.stDeployButton {display: none !important;}
div[data-testid="stDecoration"] {display: none !important;}
div[data-testid="stToolbar"] {display: none !important;}

/* All text defaults */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label {
    color: var(--text-primary) !important;
}

/* ========================================
   TYPOGRAPHY
   ======================================== */
h1, h2, h3 {
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px !important;
}

h1 {
    font-size: 1.75rem !important;
    color: var(--text-amber) !important;
    border-bottom: 2px solid var(--accent-amber) !important;
    padding-bottom: 8px !important;
}

/* ========================================
   SIDEBAR - Terminal Style
   ======================================== */
section[data-testid="stSidebar"] {
    background-color: var(--bg-panel) !important;
    border-right: 1px solid var(--border-default) !important;
}

section[data-testid="stSidebar"] > div {
    background-color: var(--bg-panel) !important;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    color: var(--accent-amber) !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}

/* ========================================
   METRIC CARDS - Terminal Data Display
   ======================================== */
div[data-testid="stMetric"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    border-left: 3px solid var(--accent-amber) !important;
    border-radius: 0 !important;
    padding: 16px 20px !important;
}

div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-amber) !important;
    font-family: var(--font-mono) !important;
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
}

/* Positive delta */
div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg[fill="#09ab3b"] + span,
div[data-testid="stMetric"] [data-testid="stMetricDelta"]:has(svg[fill="#09ab3b"]) {
    color: var(--bullish) !important;
}

/* Negative delta */
div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg[fill="#ff2b2b"] + span,
div[data-testid="stMetric"] [data-testid="stMetricDelta"]:has(svg[fill="#ff2b2b"]) {
    color: var(--bearish) !important;
}

/* ========================================
   BUTTONS - Terminal Style
   ======================================== */
.stButton > button {
    background: linear-gradient(180deg, #ff6600 0%, #cc5500 100%) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(180deg, #ff7700 0%, #ff6600 100%) !important;
    box-shadow: 0 0 10px rgba(255, 102, 0, 0.5) !important;
}

.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--accent-amber) !important;
    color: var(--accent-amber) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(255, 102, 0, 0.1) !important;
}

/* ========================================
   INPUTS - Dark Terminal
   ======================================== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
    color: var(--text-amber) !important;
    font-family: var(--font-mono) !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-amber) !important;
    box-shadow: 0 0 0 1px var(--accent-amber) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
}

/* Slider */
.stSlider > div > div > div {
    background-color: var(--border-default) !important;
}

.stSlider > div > div > div > div {
    background: var(--accent-amber) !important;
}

/* ========================================
   TABS - Terminal Navigation
   ======================================== */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border-default) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 12px 20px !important;
}

.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: var(--accent-amber) !important;
    border-bottom: 2px solid var(--accent-amber) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}

/* ========================================
   ALERTS - Terminal Notifications
   ======================================== */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
}

.stSuccess {
    background-color: rgba(0, 255, 0, 0.1) !important;
    border-left: 3px solid var(--bullish) !important;
}

.stError {
    background-color: rgba(255, 51, 51, 0.1) !important;
    border-left: 3px solid var(--bearish) !important;
}

.stWarning {
    background-color: rgba(255, 170, 0, 0.1) !important;
    border-left: 3px solid var(--accent-gold) !important;
}

.stInfo {
    background-color: rgba(255, 102, 0, 0.1) !important;
    border-left: 3px solid var(--accent-amber) !important;
}

/* ========================================
   DATAFRAMES - Terminal Tables
   ======================================== */
.stDataFrame {
    border: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
}

.stDataFrame [data-testid="stDataFrameContainer"] {
    background-color: var(--bg-card) !important;
}

.stDataFrame th {
    background-color: var(--bg-elevated) !important;
    color: var(--text-amber) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-bottom: 1px solid var(--border-default) !important;
}

.stDataFrame td {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* ========================================
   PROGRESS BARS - Gradient Style
   ======================================== */
.stProgress > div > div > div {
    background-color: var(--bg-elevated) !important;
    border-radius: 0 !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #ff3300 0%, #ff6600 50%, #ffaa00 100%) !important;
    border-radius: 0 !important;
}

/* ========================================
   EXPANDERS
   ======================================== */
.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
}

/* ========================================
   DIVIDERS
   ======================================== */
hr {
    border-color: var(--border-default) !important;
}

/* ========================================
   FORMS
   ======================================== */
[data-testid="stForm"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
    padding: 20px !important;
}

/* ========================================
   CUSTOM COMPONENTS
   ======================================== */
.terminal-header {
    color: var(--accent-amber);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.data-value {
    color: var(--text-amber);
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 600;
}

.badge-bullish {
    background-color: rgba(0, 255, 0, 0.15);
    color: var(--bullish);
    padding: 4px 12px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.badge-bearish {
    background-color: rgba(255, 51, 51, 0.15);
    color: var(--bearish);
    padding: 4px 12px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-panel);
}

::-webkit-scrollbar-thumb {
    background: var(--border-default);
    border-radius: 0;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-amber);
}

/* Custom line chart colors */
.stLineChart {
    background-color: var(--bg-card) !important;
}
</style>
"""


def get_theme_vars() -> Dict[str, str]:
    """Get all theme variables as a dictionary."""
    return {**COLORS, **GRADIENTS}


def get_status_color(status: str) -> str:
    """Get the appropriate color for a status."""
    status_lower = status.lower()
    if status_lower in ("bullish", "buy", "long", "success", "valid", "deploy"):
        return COLORS["bullish"]
    elif status_lower in ("bearish", "sell", "short", "error", "invalid", "reject"):
        return COLORS["bearish"]
    elif status_lower in ("caution", "warning", "neutral"):
        return COLORS["accent_gold"]
    else:
        return COLORS["text_secondary"]
