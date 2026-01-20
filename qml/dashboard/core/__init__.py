"""
QML Dashboard Core Module
=========================
Professional design system and chart components.
"""

from .design_system import (
    COLORS,
    TYPOGRAPHY,
    SPACING,
    LAYOUT,
    CHART_STYLE,
    apply_design_system,
    generate_dashboard_css,
    generate_chart_theme_config,
    generate_candlestick_config,
)

from .pro_chart import (
    render_professional_chart,
    render_simple_chart,
    render_mini_chart,
)

__all__ = [
    # Design system
    "COLORS",
    "TYPOGRAPHY",
    "SPACING",
    "LAYOUT",
    "CHART_STYLE",
    "apply_design_system",
    "generate_dashboard_css",
    "generate_chart_theme_config",
    "generate_candlestick_config",
    # Chart components
    "render_professional_chart",
    "render_simple_chart",
    "render_mini_chart",
]
