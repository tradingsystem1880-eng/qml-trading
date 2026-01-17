"""
QML Dashboard Cards Module
==========================

Reusable card components for displaying metrics and patterns.

Usage:
    from qml.dashboard.components import metric_card, pattern_card
    
    metric_card("Sharpe Ratio", 1.87, delta="+0.12")
    pattern_card(symbol="BTC/USDT", pattern_type="bullish", validity=0.75)
"""

import streamlit as st
from typing import Optional, Union


def metric_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None
) -> None:
    """
    Display a styled metric card.
    
    Args:
        label: Metric label
        value: Metric value 
        delta: Optional change indicator
        delta_color: "normal", "inverse", or "off"
        help_text: Optional tooltip text
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def pattern_card(
    symbol: str,
    pattern_type: str,
    validity: float,
    entry: float = 0,
    stop_loss: float = 0,
    take_profit: float = 0,
    risk_reward: float = 0,
    timeframe: str = "4h"
) -> None:
    """
    Display a pattern card with trading details.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        pattern_type: "bullish" or "bearish"
        validity: Pattern validity score (0-1)
        entry: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        risk_reward: Risk/reward ratio
        timeframe: Chart timeframe
    """
    is_bullish = "bullish" in pattern_type.lower()
    icon = "🟢" if is_bullish else "🔴"
    badge_class = "badge-bullish" if is_bullish else "badge-bearish"
    
    with st.container():
        st.markdown(f"""
        <div class="stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 1.25rem; font-weight: 700;">{icon} {symbol}</span>
                <span class="{badge_class}">{pattern_type.upper()}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; text-align: center;">
                <div>
                    <div style="color: #8b949e; font-size: 0.75rem;">Validity</div>
                    <div style="font-size: 1.25rem; font-weight: 600;">{validity:.0%}</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 0.75rem;">Entry</div>
                    <div style="font-size: 1.25rem; font-weight: 600;">${entry:,.2f}</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 0.75rem;">SL</div>
                    <div style="color: #f85149; font-size: 1.25rem; font-weight: 600;">${stop_loss:,.2f}</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 0.75rem;">TP</div>
                    <div style="color: #3fb950; font-size: 1.25rem; font-weight: 600;">${take_profit:,.2f}</div>
                </div>
            </div>
            <div style="margin-top: 12px; display: flex; gap: 16px; color: #8b949e; font-size: 0.75rem;">
                <span>R:R {risk_reward:.1f}</span>
                <span>TF: {timeframe}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def stat_box(
    label: str,
    value: Union[str, int, float],
    icon: str = "📊",
    color: str = "#58a6ff"
) -> None:
    """
    Display a compact stat box.
    
    Args:
        label: Stat label
        value: Stat value
        icon: Emoji icon
        color: Accent color
    """
    st.markdown(f"""
    <div style="
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
    ">
        <div style="color: #8b949e; font-size: 0.75rem; margin-bottom: 4px;">
            {icon} {label}
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #f0f6fc;">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)


def verdict_banner(
    verdict: str,
    confidence: int = 50,
    message: Optional[str] = None
) -> None:
    """
    Display a verdict banner (DEPLOY/CAUTION/REJECT).
    
    Args:
        verdict: "DEPLOY", "CAUTION", or "REJECT"
        confidence: Confidence score (0-100)
        message: Optional additional message
    """
    verdict_upper = verdict.upper()
    
    if verdict_upper == "DEPLOY":
        icon = "✅"
        color = "#238636"
        bg_color = "rgba(35, 134, 54, 0.15)"
    elif verdict_upper == "CAUTION":
        icon = "⚠️"
        color = "#d29922"
        bg_color = "rgba(210, 153, 34, 0.15)"
    else:
        icon = "❌"
        color = "#f85149"
        bg_color = "rgba(248, 81, 73, 0.15)"
    
    default_messages = {
        "DEPLOY": "Strategy validated for live deployment",
        "CAUTION": "Review recommended before deployment",
        "REJECT": "Not recommended for deployment"
    }
    
    display_message = message or default_messages.get(verdict_upper, "")
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        border: 1px solid {color};
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    ">
        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">
            {icon} VERDICT: {verdict_upper}
        </div>
        <div style="color: #f0f6fc; margin-top: 8px;">
            <strong>Confidence Score: {confidence}/100</strong> — {display_message}
        </div>
    </div>
    """, unsafe_allow_html=True)


def trade_stats_table(stats: dict) -> None:
    """
    Display a trade statistics table.
    
    Args:
        stats: Dictionary with trade statistics
    """
    st.markdown("""
    <style>
    .trade-stats-table {
        width: 100%;
        border-collapse: collapse;
        background: #161b22;
        border-radius: 8px;
        overflow: hidden;
    }
    .trade-stats-table th {
        background: #21262d;
        padding: 12px;
        text-align: left;
        color: #8b949e;
        font-weight: 500;
    }
    .trade-stats-table td {
        padding: 12px;
        border-top: 1px solid #30363d;
        color: #f0f6fc;
    }
    </style>
    """, unsafe_allow_html=True)
    
    rows = "".join([
        f"<tr><td>{key}</td><td><strong>{value}</strong></td></tr>"
        for key, value in stats.items()
    ])
    
    st.markdown(f"""
    <table class="trade-stats-table">
        <thead>
            <tr><th>Metric</th><th>Value</th></tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """, unsafe_allow_html=True)
