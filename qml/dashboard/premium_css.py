"""Premium CSS for QML Trading Dashboard."""

PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ========== CSS VARIABLES ========== */
    :root {
        --bg-deep: #0a0a0a;
        --bg-primary: #0d0d0d;
        --bg-card: #111111;
        --bg-elevated: #1a1a1a;
        --bg-hover: #222222;
        --border: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(255, 255, 255, 0.15);
        --text-primary: #ffffff;
        --text-secondary: #888888;
        --text-muted: #555555;
        --cyan: #00d4ff;
        --orange: #ff6b35;
        --pink: #ff3366;
        --purple: #8b5cf6;
        --emerald: #10b981;
        --rose: #ef4444;
    }

    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.15); }
        50% { box-shadow: 0 0 30px rgba(0, 212, 255, 0.3); }
    }

    /* ========== GLOBAL STYLES ========== */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        scrollbar-width: thin;
        scrollbar-color: #333 #0a0a0a;
    }
    *::-webkit-scrollbar { width: 6px; height: 6px; }
    *::-webkit-scrollbar-track { background: #0a0a0a; }
    *::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    *::-webkit-scrollbar-thumb:hover { background: #555; }

    /* ========== APP BACKGROUND ========== */
    .stApp {
        background: #0a0a0a !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Main content */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 100%;
    }

    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d0d 0%, #0a0a0a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 1.75rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.1rem !important; }

    p, span, div, label {
        font-family: 'Inter', sans-serif;
        color: #888888;
    }

    /* Monospace for numbers */
    .mono, [data-testid="stMetricValue"], code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ========== METRICS ========== */
    [data-testid="stMetric"] {
        background: #111111;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
        text-transform: uppercase;
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em;
    }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* ========== GLASSMORPHISM CARDS ========== */
    .glass-card {
        background: rgba(17, 17, 17, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .card-header {
        color: #555555;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 100%) !important;
        color: #000000 !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(0, 212, 255, 0.35) !important;
    }

    /* Secondary buttons */
    button[kind="secondary"], .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #888888 !important;
        box-shadow: none !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
    }

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        background: #111111;
        border-radius: 12px;
        padding: 4px;
        gap: 2px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #555555 !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 10px 16px !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #1a1a1a !important;
        color: #888888 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1a1a !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* ========== INPUTS ========== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #111111 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15) !important;
    }

    /* Dropdown menus */
    [data-baseweb="popover"] {
        background: #111111 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    }
    [data-baseweb="menu"] { background: transparent !important; }
    [data-baseweb="menu"] li {
        color: #888888 !important;
        font-size: 0.85rem !important;
        padding: 10px 14px !important;
    }
    [data-baseweb="menu"] li:hover {
        background: #1a1a1a !important;
        color: #ffffff !important;
    }

    /* ========== SLIDERS ========== */
    .stSlider > div > div > div {
        background: #222222 !important;
        height: 4px !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #8b5cf6) !important;
    }
    .stSlider > div > div > div > div > div {
        background: #ffffff !important;
        border: 2px solid #00d4ff !important;
        width: 14px !important;
        height: 14px !important;
    }

    /* ========== PROGRESS BARS ========== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #8b5cf6) !important;
        border-radius: 4px !important;
    }
    .stProgress > div {
        background: #1a1a1a !important;
        border-radius: 4px !important;
    }

    /* ========== EXPANDERS ========== */
    .streamlit-expanderHeader {
        background: #111111 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderHeader:hover {
        background: #1a1a1a !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    .streamlit-expanderContent {
        background: #0d0d0d !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* ========== DATA TABLES ========== */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: #111111 !important;
    }
    .stDataFrame th {
        background: #1a1a1a !important;
        color: #555555 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    .stDataFrame td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        color: #888888 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
    }

    /* ========== ALERTS ========== */
    .stAlert {
        background: #111111 !important;
        border-radius: 10px !important;
        border-left: 4px solid #00d4ff !important;
    }

    /* ========== CUSTOM COMPONENTS ========== */

    /* Gauge Container */
    .gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 16px;
    }

    /* Metric Card */
    .metric-card {
        background: #111111;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .metric-card.glow-success:hover {
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
    }
    .metric-card.glow-danger:hover {
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
    }
    .metric-card.glow-accent:hover {
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
    }

    /* Trade Row */
    .trade-row {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.2s ease;
    }
    .trade-row:hover {
        background: #1a1a1a;
    }

    /* Stat Row */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .stat-row:last-child { border-bottom: none; }

    /* Badge */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
    }
    .badge-info {
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
    }

    /* Heatmap Cell */
    .heatmap-cell {
        border-radius: 4px;
        transition: transform 0.2s ease;
    }
    .heatmap-cell:hover {
        transform: scale(1.1);
    }

    /* Live Indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #10b981;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }

    /* Aurora Header */
    .aurora-header {
        background: linear-gradient(135deg, #ff6b35 0%, #ff3366 50%, #8b5cf6 100%);
        padding: 2px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .aurora-header-inner {
        background: #0a0a0a;
        border-radius: 10px;
        padding: 16px 24px;
    }

    /* Focus ring */
    *:focus-visible {
        outline: 2px solid #00d4ff !important;
        outline-offset: 2px;
    }
</style>
"""
