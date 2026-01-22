"""
Arctic Pro Theme - Premium Trading Dashboard Design System

A professional trading dashboard theme inspired by TradingView, Binance Pro,
and modern fintech interfaces. Features gradients, depth, and premium styling.
"""

# Arctic Pro Color Palette
ARCTIC_PRO = {
    # Backgrounds - Deep blue gradient tones
    'bg_primary': '#0B1426',
    'bg_secondary': '#0F1A2E',
    'bg_card': '#162032',
    'bg_card_gradient_start': '#1A2742',
    'bg_card_gradient_end': '#141E30',
    'bg_elevated': '#1C2940',
    'bg_hover': '#243352',
    'bg_sidebar': '#0A1220',

    # Accent Colors
    'accent': '#3B82F6',
    'accent_bright': '#60A5FA',
    'accent_hover': '#2563EB',
    'accent_muted': '#1E3A5F',
    'accent_glow': 'rgba(59, 130, 246, 0.4)',

    # Semantic Colors
    'success': '#10B981',
    'success_bright': '#34D399',
    'success_muted': '#065F46',
    'success_glow': 'rgba(16, 185, 129, 0.3)',
    'danger': '#EF4444',
    'danger_bright': '#F87171',
    'danger_muted': '#7F1D1D',
    'danger_glow': 'rgba(239, 68, 68, 0.3)',
    'warning': '#F59E0B',
    'warning_bright': '#FBBF24',
    'warning_muted': '#78350F',

    # Text Colors
    'text_primary': '#F8FAFC',
    'text_secondary': '#CBD5E1',
    'text_muted': '#64748B',
    'text_disabled': '#475569',

    # Borders & Dividers
    'border': '#1E293B',
    'border_light': '#334155',
    'border_accent': '#3B82F6',
    'divider': '#1E293B',

    # Chart Colors
    'chart_green': '#22C55E',
    'chart_red': '#EF4444',
    'chart_blue': '#3B82F6',
    'chart_purple': '#A855F7',
    'chart_orange': '#F97316',
    'chart_grid': '#1E293B',
}

# Typography
TYPOGRAPHY = {
    'font_primary': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    'font_mono': "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
    'font_display': "'Space Grotesk', 'Inter', sans-serif",

    # Font Sizes
    'size_xs': '0.7rem',
    'size_sm': '0.8rem',
    'size_base': '0.9rem',
    'size_lg': '1rem',
    'size_xl': '1.15rem',
    'size_2xl': '1.4rem',
    'size_3xl': '1.75rem',
    'size_4xl': '2.25rem',

    # Font Weights
    'weight_normal': '400',
    'weight_medium': '500',
    'weight_semibold': '600',
    'weight_bold': '700',
}

# Spacing System (8px grid)
SPACING = {
    'xs': '4px',
    'sm': '8px',
    'md': '16px',
    'lg': '24px',
    'xl': '32px',
    '2xl': '48px',
    '3xl': '64px',
}

# Border Radius
RADIUS = {
    'none': '0',
    'sm': '4px',
    'md': '8px',
    'lg': '12px',
    'xl': '16px',
    '2xl': '20px',
    'full': '9999px',
}

# Shadows
SHADOWS = {
    'sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
    'md': '0 4px 12px rgba(0, 0, 0, 0.4)',
    'lg': '0 8px 24px rgba(0, 0, 0, 0.5)',
    'xl': '0 12px 40px rgba(0, 0, 0, 0.6)',
    'inner': 'inset 0 2px 4px rgba(0, 0, 0, 0.3)',
    'glow_blue': '0 0 20px rgba(59, 130, 246, 0.4)',
    'glow_green': '0 0 20px rgba(16, 185, 129, 0.4)',
    'glow_red': '0 0 20px rgba(239, 68, 68, 0.4)',
    'card': '0 4px 20px rgba(0, 0, 0, 0.3), 0 0 1px rgba(255, 255, 255, 0.05)',
}


def get_css() -> str:
    """Generate the complete premium CSS stylesheet for Arctic Pro theme."""
    return f"""
    <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

        /* CSS Variables */
        :root {{
            --bg-primary: {ARCTIC_PRO['bg_primary']};
            --bg-secondary: {ARCTIC_PRO['bg_secondary']};
            --bg-card: {ARCTIC_PRO['bg_card']};
            --accent: {ARCTIC_PRO['accent']};
            --success: {ARCTIC_PRO['success']};
            --danger: {ARCTIC_PRO['danger']};
            --text-primary: {ARCTIC_PRO['text_primary']};
            --text-muted: {ARCTIC_PRO['text_muted']};
        }}

        /* Global Styles */
        .stApp {{
            background: linear-gradient(180deg, {ARCTIC_PRO['bg_primary']} 0%, {ARCTIC_PRO['bg_secondary']} 100%);
            color: {ARCTIC_PRO['text_primary']};
            font-family: {TYPOGRAPHY['font_primary']};
        }}

        /* Hide default Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}
        [data-testid="stSidebarNav"] {{display: none;}}
        .block-container {{padding-top: 1rem;}}

        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {ARCTIC_PRO['bg_secondary']};
        }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, {ARCTIC_PRO['accent']} 0%, {ARCTIC_PRO['accent_muted']} 100%);
            border-radius: {RADIUS['full']};
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {ARCTIC_PRO['accent_bright']};
        }}

        /* ========== SIDEBAR NAVIGATION ========== */
        .sidebar-nav {{
            background: linear-gradient(180deg, {ARCTIC_PRO['bg_sidebar']} 0%, {ARCTIC_PRO['bg_primary']} 100%);
            border-right: 1px solid {ARCTIC_PRO['border']};
            height: 100vh;
            padding: {SPACING['md']};
            position: fixed;
            left: 0;
            top: 0;
            width: 220px;
            z-index: 1000;
        }}

        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: {SPACING['sm']};
            padding: {SPACING['md']} {SPACING['sm']};
            margin-bottom: {SPACING['lg']};
            border-bottom: 1px solid {ARCTIC_PRO['border']};
        }}

        .sidebar-logo-icon {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, {ARCTIC_PRO['accent']} 0%, {ARCTIC_PRO['accent_bright']} 100%);
            border-radius: {RADIUS['md']};
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: {TYPOGRAPHY['weight_bold']};
            font-size: {TYPOGRAPHY['size_lg']};
            box-shadow: {SHADOWS['glow_blue']};
        }}

        .sidebar-logo-text {{
            font-family: {TYPOGRAPHY['font_display']};
            font-size: {TYPOGRAPHY['size_lg']};
            font-weight: {TYPOGRAPHY['weight_bold']};
            color: {ARCTIC_PRO['text_primary']};
        }}

        .nav-item {{
            display: flex;
            align-items: center;
            gap: {SPACING['sm']};
            padding: {SPACING['sm']} {SPACING['md']};
            margin: {SPACING['xs']} 0;
            border-radius: {RADIUS['md']};
            color: {ARCTIC_PRO['text_muted']};
            font-size: {TYPOGRAPHY['size_sm']};
            font-weight: {TYPOGRAPHY['weight_medium']};
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }}

        .nav-item:hover {{
            background: {ARCTIC_PRO['bg_hover']};
            color: {ARCTIC_PRO['text_primary']};
        }}

        .nav-item.active {{
            background: linear-gradient(90deg, {ARCTIC_PRO['accent_muted']} 0%, transparent 100%);
            color: {ARCTIC_PRO['accent_bright']};
            border-left: 3px solid {ARCTIC_PRO['accent']};
        }}

        .nav-icon {{
            width: 20px;
            text-align: center;
            font-size: {TYPOGRAPHY['size_base']};
        }}

        /* ========== PREMIUM COMMAND BAR ========== */
        .command-bar {{
            background: linear-gradient(180deg, {ARCTIC_PRO['bg_card_gradient_start']} 0%, {ARCTIC_PRO['bg_card_gradient_end']} 100%);
            border: 1px solid {ARCTIC_PRO['border']};
            border-radius: {RADIUS['lg']};
            padding: {SPACING['md']};
            margin-bottom: {SPACING['md']};
            box-shadow: {SHADOWS['card']};
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: {SPACING['md']};
        }}

        .kpi-card {{
            background: linear-gradient(145deg, {ARCTIC_PRO['bg_elevated']} 0%, {ARCTIC_PRO['bg_card']} 100%);
            border: 1px solid {ARCTIC_PRO['border']};
            border-radius: {RADIUS['lg']};
            padding: {SPACING['md']};
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: {SHADOWS['sm']};
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 3px;
            background: {ARCTIC_PRO['accent']};
            border-radius: {RADIUS['sm']} 0 0 {RADIUS['sm']};
        }}

        .kpi-card.positive::before {{
            background: linear-gradient(180deg, {ARCTIC_PRO['success']} 0%, {ARCTIC_PRO['success_muted']} 100%);
            box-shadow: {SHADOWS['glow_green']};
        }}

        .kpi-card.negative::before {{
            background: linear-gradient(180deg, {ARCTIC_PRO['danger']} 0%, {ARCTIC_PRO['danger_muted']} 100%);
            box-shadow: {SHADOWS['glow_red']};
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: {ARCTIC_PRO['border_light']};
            box-shadow: {SHADOWS['md']};
        }}

        .kpi-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: {SPACING['sm']};
        }}

        .kpi-label {{
            color: {ARCTIC_PRO['text_muted']};
            font-size: {TYPOGRAPHY['size_xs']};
            font-weight: {TYPOGRAPHY['weight_semibold']};
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .kpi-trend {{
            font-size: {TYPOGRAPHY['size_xs']};
            padding: 2px 6px;
            border-radius: {RADIUS['sm']};
        }}

        .kpi-trend.up {{
            color: {ARCTIC_PRO['success']};
            background: {ARCTIC_PRO['success_muted']};
        }}

        .kpi-trend.down {{
            color: {ARCTIC_PRO['danger']};
            background: {ARCTIC_PRO['danger_muted']};
        }}

        .kpi-value {{
            font-family: {TYPOGRAPHY['font_mono']};
            font-size: {TYPOGRAPHY['size_2xl']};
            font-weight: {TYPOGRAPHY['weight_bold']};
            color: {ARCTIC_PRO['text_primary']};
            line-height: 1.2;
        }}

        .kpi-value.positive {{
            color: {ARCTIC_PRO['success']};
        }}

        .kpi-value.negative {{
            color: {ARCTIC_PRO['danger']};
        }}

        .kpi-sparkline {{
            margin-top: {SPACING['sm']};
            height: 30px;
            opacity: 0.8;
        }}

        .kpi-subtext {{
            color: {ARCTIC_PRO['text_muted']};
            font-size: {TYPOGRAPHY['size_xs']};
            margin-top: {SPACING['xs']};
        }}

        /* ========== PREMIUM PANELS ========== */
        .panel {{
            background: linear-gradient(145deg, {ARCTIC_PRO['bg_card_gradient_start']} 0%, {ARCTIC_PRO['bg_card_gradient_end']} 100%);
            border: 1px solid {ARCTIC_PRO['border']};
            border-radius: {RADIUS['lg']};
            padding: {SPACING['lg']};
            margin-bottom: {SPACING['md']};
            box-shadow: {SHADOWS['card']};
            transition: all 0.3s ease;
        }}

        .panel:hover {{
            border-color: {ARCTIC_PRO['border_light']};
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: {SPACING['md']};
            padding-bottom: {SPACING['sm']};
            border-bottom: 1px solid {ARCTIC_PRO['border']};
        }}

        .panel-title {{
            color: {ARCTIC_PRO['text_secondary']};
            font-size: {TYPOGRAPHY['size_sm']};
            font-weight: {TYPOGRAPHY['weight_semibold']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .panel-actions {{
            display: flex;
            gap: {SPACING['sm']};
        }}

        .panel-action-btn {{
            background: {ARCTIC_PRO['bg_elevated']};
            border: 1px solid {ARCTIC_PRO['border']};
            border-radius: {RADIUS['sm']};
            padding: {SPACING['xs']} {SPACING['sm']};
            color: {ARCTIC_PRO['text_muted']};
            font-size: {TYPOGRAPHY['size_xs']};
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .panel-action-btn:hover {{
            background: {ARCTIC_PRO['bg_hover']};
            color: {ARCTIC_PRO['text_primary']};
            border-color: {ARCTIC_PRO['accent']};
        }}

        /* ========== METRIC CARDS ========== */
        .metric-card {{
            background: linear-gradient(145deg, {ARCTIC_PRO['bg_elevated']} 0%, {ARCTIC_PRO['bg_card']} 100%);
            border: 1px solid {ARCTIC_PRO['border']};
            border-radius: {RADIUS['lg']};
            padding: {SPACING['md']};
            text-align: center;
            transition: all 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: {ARCTIC_PRO['accent_muted']};
            box-shadow: {SHADOWS['md']};
        }}

        .metric-label {{
            color: {ARCTIC_PRO['text_muted']};
            font-size: {TYPOGRAPHY['size_xs']};
            font-weight: {TYPOGRAPHY['weight_semibold']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: {SPACING['xs']};
        }}

        .metric-value {{
            font-family: {TYPOGRAPHY['font_mono']};
            font-size: {TYPOGRAPHY['size_xl']};
            font-weight: {TYPOGRAPHY['weight_bold']};
            color: {ARCTIC_PRO['text_primary']};
        }}

        .metric-value.positive {{ color: {ARCTIC_PRO['success']}; }}
        .metric-value.negative {{ color: {ARCTIC_PRO['danger']}; }}

        /* ========== DATA TABLE ========== */
        .data-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }}

        .data-table th {{
            background: {ARCTIC_PRO['bg_elevated']};
            color: {ARCTIC_PRO['text_muted']};
            font-size: {TYPOGRAPHY['size_xs']};
            font-weight: {TYPOGRAPHY['weight_semibold']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: {SPACING['sm']} {SPACING['md']};
            text-align: left;
            border-bottom: 1px solid {ARCTIC_PRO['border']};
        }}

        .data-table td {{
            padding: {SPACING['sm']} {SPACING['md']};
            border-bottom: 1px solid {ARCTIC_PRO['border']};
            font-size: {TYPOGRAPHY['size_sm']};
            color: {ARCTIC_PRO['text_secondary']};
        }}

        .data-table tr:hover td {{
            background: {ARCTIC_PRO['bg_hover']};
        }}

        .data-table .mono {{
            font-family: {TYPOGRAPHY['font_mono']};
        }}

        .data-table .positive {{ color: {ARCTIC_PRO['success']}; }}
        .data-table .negative {{ color: {ARCTIC_PRO['danger']}; }}

        /* ========== STATUS INDICATORS ========== */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: {RADIUS['full']};
            font-size: {TYPOGRAPHY['size_xs']};
            font-weight: {TYPOGRAPHY['weight_semibold']};
        }}

        .status-badge.success {{
            background: {ARCTIC_PRO['success_muted']};
            color: {ARCTIC_PRO['success']};
        }}

        .status-badge.danger {{
            background: {ARCTIC_PRO['danger_muted']};
            color: {ARCTIC_PRO['danger']};
        }}

        .status-badge.warning {{
            background: {ARCTIC_PRO['warning_muted']};
            color: {ARCTIC_PRO['warning']};
        }}

        .status-badge.neutral {{
            background: {ARCTIC_PRO['bg_elevated']};
            color: {ARCTIC_PRO['text_muted']};
        }}

        .status-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}

        /* ========== ACTIVITY FEED ========== */
        .activity-item {{
            display: flex;
            gap: {SPACING['md']};
            padding: {SPACING['sm']} 0;
            border-bottom: 1px solid {ARCTIC_PRO['border']};
        }}

        .activity-item:last-child {{
            border-bottom: none;
        }}

        .activity-icon {{
            width: 32px;
            height: 32px;
            border-radius: {RADIUS['md']};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: {TYPOGRAPHY['size_sm']};
            flex-shrink: 0;
        }}

        .activity-icon.trade {{
            background: {ARCTIC_PRO['accent_muted']};
            color: {ARCTIC_PRO['accent']};
        }}

        .activity-icon.win {{
            background: {ARCTIC_PRO['success_muted']};
            color: {ARCTIC_PRO['success']};
        }}

        .activity-icon.loss {{
            background: {ARCTIC_PRO['danger_muted']};
            color: {ARCTIC_PRO['danger']};
        }}

        .activity-content {{
            flex: 1;
        }}

        .activity-title {{
            color: {ARCTIC_PRO['text_primary']};
            font-size: {TYPOGRAPHY['size_sm']};
            font-weight: {TYPOGRAPHY['weight_medium']};
        }}

        .activity-meta {{
            color: {ARCTIC_PRO['text_muted']};
            font-size: {TYPOGRAPHY['size_xs']};
            margin-top: 2px;
        }}

        /* ========== PROGRESS BAR ========== */
        .progress-bar {{
            background: {ARCTIC_PRO['bg_elevated']};
            border-radius: {RADIUS['full']};
            height: 6px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            border-radius: {RADIUS['full']};
            transition: width 0.5s ease;
        }}

        .progress-fill.success {{
            background: linear-gradient(90deg, {ARCTIC_PRO['success']} 0%, {ARCTIC_PRO['success_bright']} 100%);
        }}

        .progress-fill.danger {{
            background: linear-gradient(90deg, {ARCTIC_PRO['danger']} 0%, {ARCTIC_PRO['danger_bright']} 100%);
        }}

        .progress-fill.accent {{
            background: linear-gradient(90deg, {ARCTIC_PRO['accent']} 0%, {ARCTIC_PRO['accent_bright']} 100%);
        }}

        /* ========== STREAMLIT OVERRIDES ========== */
        .stButton > button {{
            background: linear-gradient(135deg, {ARCTIC_PRO['accent']} 0%, {ARCTIC_PRO['accent_hover']} 100%);
            color: {ARCTIC_PRO['text_primary']};
            border: none;
            border-radius: {RADIUS['md']};
            font-weight: {TYPOGRAPHY['weight_semibold']};
            padding: {SPACING['sm']} {SPACING['lg']};
            transition: all 0.3s ease;
            box-shadow: {SHADOWS['sm']};
        }}

        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: {SHADOWS['glow_blue']};
        }}

        .stSelectbox > div > div {{
            background: {ARCTIC_PRO['bg_elevated']};
            border: 1px solid {ARCTIC_PRO['border']};
            border-radius: {RADIUS['md']};
        }}

        .stTextInput > div > div > input {{
            background: {ARCTIC_PRO['bg_elevated']};
            border: 1px solid {ARCTIC_PRO['border']};
            border-radius: {RADIUS['md']};
            color: {ARCTIC_PRO['text_primary']};
            font-family: {TYPOGRAPHY['font_mono']};
        }}

        .stSlider > div > div > div {{
            background: {ARCTIC_PRO['accent']};
        }}

        /* Tab overrides for premium look */
        .stTabs [data-baseweb="tab-list"] {{
            background: {ARCTIC_PRO['bg_card']};
            border-radius: {RADIUS['lg']};
            padding: {SPACING['xs']};
            gap: {SPACING['xs']};
            border: 1px solid {ARCTIC_PRO['border']};
        }}

        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            color: {ARCTIC_PRO['text_muted']};
            border-radius: {RADIUS['md']};
            padding: {SPACING['sm']} {SPACING['md']};
            font-weight: {TYPOGRAPHY['weight_medium']};
            font-size: {TYPOGRAPHY['size_sm']};
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            background: {ARCTIC_PRO['bg_hover']};
            color: {ARCTIC_PRO['text_primary']};
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {ARCTIC_PRO['accent']} 0%, {ARCTIC_PRO['accent_hover']} 100%) !important;
            color: {ARCTIC_PRO['text_primary']} !important;
            box-shadow: {SHADOWS['glow_blue']};
        }}

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {{
            display: none;
        }}

        /* Chart container styling */
        .chart-container {{
            background: {ARCTIC_PRO['bg_secondary']};
            border: 1px solid {ARCTIC_PRO['border']};
            border-radius: {RADIUS['lg']};
            padding: {SPACING['sm']};
            overflow: hidden;
        }}
    </style>
    """


def format_value(value: float, format_type: str = 'number', decimals: int = 2) -> str:
    """Format a numeric value for display."""
    if format_type == 'percent':
        return f"{value:.{decimals}f}%"
    elif format_type == 'currency':
        if abs(value) >= 1000:
            return f"${value:,.0f}"
        return f"${value:,.{decimals}f}"
    elif format_type == 'ratio':
        return f"{value:.{decimals}f}x"
    else:
        return f"{value:,.{decimals}f}"


def get_value_class(value: float, neutral_threshold: float = 0) -> str:
    """Get CSS class based on value (positive/negative/neutral)."""
    if value > neutral_threshold:
        return 'positive'
    elif value < -neutral_threshold:
        return 'negative'
    return ''


def generate_sparkline_svg(data: list, width: int = 80, height: int = 24, color: str = None) -> str:
    """Generate an inline SVG sparkline chart."""
    if not data or len(data) < 2:
        return ""

    if color is None:
        color = ARCTIC_PRO['accent']

    # Normalize data to fit in the height
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1

    points = []
    step = width / (len(data) - 1)

    for i, val in enumerate(data):
        x = i * step
        y = height - ((val - min_val) / range_val * height)
        points.append(f"{x:.1f},{y:.1f}")

    path = "M" + " L".join(points)

    # Create gradient fill
    fill_points = points + [f"{width},{height}", f"0,{height}"]
    fill_path = "M" + " L".join(fill_points) + " Z"

    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="sparkGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:{color};stop-opacity:0.3"/>
                <stop offset="100%" style="stop-color:{color};stop-opacity:0"/>
            </linearGradient>
        </defs>
        <path d="{fill_path}" fill="url(#sparkGrad)"/>
        <path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>'''

    return svg
