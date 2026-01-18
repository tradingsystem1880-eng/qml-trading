"""
Bloomberg Terminal Visualization Components
===========================================

Advanced visualization components for data-dense displays:
- Sparklines (mini trend charts)
- Circular performance gauges
- Horizontal gradient bars
- Data grids
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Union


def sparkline(data: List[float], color: str = "#ff6600", height: int = 40) -> str:
    """
    Generate an SVG sparkline chart.

    Args:
        data: List of values to plot
        color: Line color
        height: Chart height in pixels

    Returns:
        HTML string with SVG sparkline
    """
    if not data or len(data) < 2:
        return ""

    width = len(data) * 3
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1

    # Normalize data to chart height
    points = []
    for i, val in enumerate(data):
        x = i * 3
        y = height - ((val - min_val) / range_val * height)
        points.append(f"{x},{y}")

    polyline = " ".join(points)

    svg = f'<svg width="{width}" height="{height}" style="vertical-align: middle;">'
    svg += f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>'
    svg += '</svg>'
    return svg


def render_sparkline(label: str, value: Union[str, float], data: List[float],
                     trend_up: bool = True) -> None:
    """
    Render a metric with sparkline.

    Args:
        label: Metric label
        value: Current value
        data: Historical data for sparkline
        trend_up: Whether trend is positive
    """
    color = "#00ff00" if trend_up else "#ff3333"
    sparkline_svg = sparkline(data, color=color)

    html = f'<div style="background: #111111; border: 1px solid #333333; border-left: 3px solid {color}; padding: 12px 16px; margin-bottom: 8px;">'
    html += '<div style="display: flex; justify-content: space-between; align-items: center;">'
    html += '<div>'
    html += f'<div style="color: #888888; font-family: JetBrains Mono, monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;">{label}</div>'
    html += f'<div style="color: #ff9900; font-family: JetBrains Mono, monospace; font-size: 1.25rem; font-weight: 600; margin-top: 4px;">{value}</div>'
    html += '</div>'
    html += f'<div>{sparkline_svg}</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def circular_gauge(
    value: float,
    label: str,
    max_value: float = 100,
    color: str = "#ff6600",
    size: int = 120
) -> None:
    """
    Render a circular progress gauge.

    Args:
        value: Current value (0-max_value)
        label: Gauge label
        max_value: Maximum value
        color: Gauge color
        size: Gauge diameter in pixels
    """
    percentage = min(100, (value / max_value) * 100)

    # Calculate arc path
    radius = 45
    circumference = 2 * np.pi * radius
    offset = circumference - (percentage / 100) * circumference
    half_size = size / 2

    html = '<div style="text-align: center; margin: 16px;">'
    html += f'<svg width="{size}" height="{size}" style="transform: rotate(-90deg);">'
    html += f'<circle cx="{half_size}" cy="{half_size}" r="{radius}" fill="none" stroke="#222222" stroke-width="8"/>'
    html += f'<circle cx="{half_size}" cy="{half_size}" r="{radius}" fill="none" stroke="{color}" stroke-width="8" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" stroke-linecap="round"/>'
    html += '</svg>'
    html += '<div style="margin-top: -80px;">'
    html += f'<div style="color: {color}; font-family: JetBrains Mono, monospace; font-size: 1.5rem; font-weight: 700;">{value:.1f}%</div>'
    html += f'<div style="color: #888888; font-family: JetBrains Mono, monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">{label}</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def horizontal_bar(
    label: str,
    value: float,
    max_value: float = 100,
    color_gradient: str = "linear-gradient(90deg, #ff3300 0%, #ff6600 50%, #ffaa00 100%)",
    show_value: bool = True
) -> None:
    """
    Render a horizontal gradient bar.

    Args:
        label: Bar label
        value: Current value
        max_value: Maximum value
        color_gradient: CSS gradient string
        show_value: Whether to show numeric value
    """
    percentage = min(100, (value / max_value) * 100)

    value_display = f'<span style="color: #ff9900; font-weight: 600;">{value:.1f}</span>' if show_value else ""

    html = '<div style="margin-bottom: 12px;">'
    html += '<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">'
    html += f'<span style="color: #888888; font-family: JetBrains Mono, monospace; font-size: 0.75rem; text-transform: uppercase;">{label}</span>'
    html += value_display
    html += '</div>'
    html += '<div style="background: #222222; height: 6px; border-radius: 0; overflow: hidden;">'
    html += f'<div style="background: {color_gradient}; width: {percentage}%; height: 100%;"></div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def data_grid_2x2(
    metric1: dict,
    metric2: dict,
    metric3: dict,
    metric4: dict
) -> None:
    """
    Render a 2x2 data grid - Bloomberg terminal style.

    Args:
        metric1-4: Dicts with keys: label, value, color, trend (optional)
    """
    def render_metric(metric: dict) -> str:
        color = metric.get('color', '#ff9900')
        trend = metric.get('trend', '')
        trend_icon = "▲" if trend == "up" else "▼" if trend == "down" else ""
        trend_color = "#00ff00" if trend == "up" else "#ff3333" if trend == "down" else "#888888"

        m_html = '<div style="background: #111111; border: 1px solid #333333; padding: 20px; text-align: center;">'
        m_html += f'<div style="color: #888888; font-family: JetBrains Mono, monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">{metric["label"]}</div>'
        m_html += f'<div style="color: {color}; font-family: JetBrains Mono, monospace; font-size: 2rem; font-weight: 700; line-height: 1;">{metric["value"]}</div>'
        if trend_icon:
            m_html += f'<div style="color: {trend_color}; font-size: 0.9rem; margin-top: 8px;">{trend_icon}</div>'
        m_html += '</div>'
        return m_html

    html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 16px 0;">'
    html += render_metric(metric1)
    html += render_metric(metric2)
    html += render_metric(metric3)
    html += render_metric(metric4)
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def ticker_tape(symbols: List[dict], scroll_duration: int = 30) -> None:
    """
    Render a scrolling ticker tape.

    Args:
        symbols: List of dicts with keys: symbol, price, change
        scroll_duration: Animation duration in seconds
    """
    ticker_items = []
    for sym in symbols:
        change_color = "#00ff00" if sym['change'] >= 0 else "#ff3333"
        change_symbol = "+" if sym['change'] >= 0 else ""

        item = '<span style="margin: 0 24px;">'
        item += f'<span style="color: #ffffff; font-weight: 600;">{sym["symbol"]}</span>'
        item += f'<span style="color: #ff9900; margin: 0 8px;">${sym["price"]:.2f}</span>'
        item += f'<span style="color: {change_color};">{change_symbol}{sym["change"]:.2f}%</span>'
        item += '</span>'
        ticker_items.append(item)

    ticker_content = "".join(ticker_items) * 3  # Repeat for continuous scroll

    style = '<style>@keyframes scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-33.333%); } }</style>'
    html = style
    html += '<div style="background: #0a0a0a; border-top: 1px solid #333333; border-bottom: 1px solid #333333; overflow: hidden; white-space: nowrap; padding: 8px 0;">'
    html += f'<div style="display: inline-block; animation: scroll {scroll_duration}s linear infinite; font-family: JetBrains Mono, monospace; font-size: 0.85rem;">{ticker_content}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def stats_table(data: pd.DataFrame, highlight_column: Optional[str] = None) -> None:
    """
    Render a Bloomberg-style statistics table.

    Args:
        data: DataFrame to display
        highlight_column: Column to highlight with amber
    """
    # Convert to HTML with custom styling
    html = '<table style="width: 100%; border-collapse: collapse; font-family: JetBrains Mono, monospace;">'

    # Header
    html += '<thead><tr style="background: #1a1a1a; border-bottom: 1px solid #333333;">'
    for col in data.columns:
        html += f'<th style="padding: 12px; text-align: left; color: #ff9900; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;">{col}</th>'
    html += '</tr></thead>'

    # Body
    html += '<tbody>'
    for idx, row in data.iterrows():
        html += '<tr style="border-bottom: 1px solid #222222;">'
        for col in data.columns:
            value = row[col]
            color = "#ff9900" if col == highlight_column else "#ffffff"
            html += f'<td style="padding: 12px; color: {color}; font-size: 0.85rem;">{value}</td>'
        html += '</tr>'
    html += '</tbody></table>'

    st.markdown(f'<div style="background: #111111; border: 1px solid #333333; overflow-x: auto;">{html}</div>', unsafe_allow_html=True)
