"""
Analytics Page - Performance metrics and statistical analysis.

Connects to backtest results from session state or displays placeholder when no data.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from theme import ARCTIC_PRO, TYPOGRAPHY


def get_backtest_results() -> Optional[Dict[str, Any]]:
    """Get backtest results from session state."""
    return st.session_state.get('backtest_results', None)


def calculate_trade_stats(trades: List) -> Dict[str, Any]:
    """Calculate detailed trade statistics."""
    if not trades:
        return {}

    wins = [t for t in trades if hasattr(t, 'pnl_usd') and t.pnl_usd and t.pnl_usd > 0]
    losses = [t for t in trades if hasattr(t, 'pnl_usd') and t.pnl_usd and t.pnl_usd < 0]

    win_pnls = [t.pnl_usd for t in wins]
    loss_pnls = [abs(t.pnl_usd) for t in losses]

    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    largest_win = max(win_pnls) if win_pnls else 0
    largest_loss = max(loss_pnls) if loss_pnls else 0

    # Calculate consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    consec_wins = 0
    consec_losses = 0

    for t in trades:
        pnl = t.pnl_usd if hasattr(t, 'pnl_usd') and t.pnl_usd else 0
        if pnl > 0:
            consec_wins += 1
            consec_losses = 0
            max_consec_wins = max(max_consec_wins, consec_wins)
        elif pnl < 0:
            consec_losses += 1
            consec_wins = 0
            max_consec_losses = max(max_consec_losses, consec_losses)

    return {
        'total_wins': len(wins),
        'total_losses': len(losses),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'max_consec_wins': max_consec_wins,
        'max_consec_losses': max_consec_losses,
    }


def render_equity_chart(equity_curve: List) -> None:
    """Render the equity curve with drawdown overlay."""
    if not equity_curve:
        return

    dates = [e[0] for e in equity_curve]
    values = [e[1] for e in equity_curve]

    # Calculate drawdown
    peak = values[0]
    drawdowns = []
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        drawdowns.append(-dd)

    fig = go.Figure()

    # Add equity line
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Equity',
        line=dict(color=ARCTIC_PRO['accent'], width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>'
    ))

    # Add starting line
    if values:
        fig.add_hline(
            y=values[0],
            line_dash="dot",
            line_color=ARCTIC_PRO['text_muted'],
            annotation_text="Start"
        )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        showlegend=False,
        hovermode='x unified',
        xaxis=dict(
            showgrid=False,
            showline=False,
            tickfont=dict(color=ARCTIC_PRO['text_muted'], size=10),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=ARCTIC_PRO['border'],
            showline=False,
            tickfont=dict(color=ARCTIC_PRO['text_muted'], size=10),
            tickprefix='$',
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_trade_distribution(trades: List) -> None:
    """Render trade P&L distribution histogram."""
    if not trades:
        return

    pnls = [t.pnl_usd for t in trades if hasattr(t, 'pnl_usd') and t.pnl_usd]
    if not pnls:
        return

    # Create histogram with colored bars
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=pnls,
        nbinsx=20,
        marker_color=[ARCTIC_PRO['success'] if x >= 0 else ARCTIC_PRO['danger'] for x in pnls],
        hovertemplate='P&L: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
    ))

    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color=ARCTIC_PRO['text_muted'], line_width=1)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=200,
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showline=False,
            tickfont=dict(color=ARCTIC_PRO['text_muted'], size=10),
            tickprefix='$',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=ARCTIC_PRO['border'],
            showline=False,
            tickfont=dict(color=ARCTIC_PRO['text_muted'], size=10),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_metric_card(label: str, value: str, status: str = "neutral") -> None:
    """Render a single metric card."""
    color = ARCTIC_PRO["text_primary"]
    if status == "positive":
        color = ARCTIC_PRO["success"]
    elif status == "negative":
        color = ARCTIC_PRO["danger"]

    metric_html = '<div class="panel" style="text-align: center;">'
    metric_html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]}; '
    metric_html += f'text-transform: uppercase; margin-bottom: 0.5rem;">{label}</div>'
    metric_html += f'<div style="color: {color}; font-size: {TYPOGRAPHY["size_2xl"]}; '
    metric_html += f'font-weight: {TYPOGRAPHY["weight_bold"]};">{value}</div>'
    metric_html += '</div>'
    st.markdown(metric_html, unsafe_allow_html=True)


def render_analytics_page() -> None:
    """Render the analytics and performance page."""

    # Page header
    html = '<div class="panel">'
    html += '<div class="panel-header">Performance Analytics</div>'
    html += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Detailed performance metrics and statistical analysis.</p>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Get backtest results
    results = get_backtest_results()

    if not results:
        # Empty state - show placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Total Trades", "--", "neutral")
        with col2:
            render_metric_card("Win Rate", "--", "neutral")
        with col3:
            render_metric_card("Profit Factor", "--", "neutral")
        with col4:
            render_metric_card("Sharpe Ratio", "--", "neutral")

        # Placeholder panels
        charts_panel = '<div class="panel" style="margin-top: 1rem;">'
        charts_panel += '<div class="panel-header">Equity Curve</div>'
        charts_panel += f'<div style="height: 300px; display: flex; align-items: center; justify-content: center; '
        charts_panel += f'color: {ARCTIC_PRO["text_muted"]};">Run a backtest to generate equity curve</div>'
        charts_panel += '</div>'
        st.markdown(charts_panel, unsafe_allow_html=True)

        dist_panel = '<div class="panel" style="margin-top: 1rem;">'
        dist_panel += '<div class="panel-header">Trade Distribution</div>'
        dist_panel += f'<div style="height: 200px; display: flex; align-items: center; justify-content: center; '
        dist_panel += f'color: {ARCTIC_PRO["text_muted"]};">No trade data available</div>'
        dist_panel += '</div>'
        st.markdown(dist_panel, unsafe_allow_html=True)

        # Hint to run backtest
        hint_html = f'<div style="text-align: center; padding: 24px; color: {ARCTIC_PRO["text_muted"]};">'
        hint_html += 'Go to the <strong>Backtest</strong> tab and run a backtest to see analytics.'
        hint_html += '</div>'
        st.markdown(hint_html, unsafe_allow_html=True)
        return

    # We have results - show real data
    total_trades = results.get('total_trades', 0)
    win_rate = results.get('win_rate', 0)
    profit_factor = results.get('profit_factor', 0)
    sharpe_ratio = results.get('sharpe_ratio', 0)
    net_profit = results.get('net_profit', 0)
    net_profit_pct = results.get('net_profit_pct', 0)
    max_dd = results.get('max_drawdown', 0)
    final_equity = results.get('final_equity', 0)

    # Primary metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Total Trades", str(total_trades), "neutral")

    with col2:
        wr_status = "positive" if win_rate > 50 else "negative" if win_rate < 50 else "neutral"
        render_metric_card("Win Rate", f"{win_rate:.1f}%", wr_status)

    with col3:
        pf_status = "positive" if profit_factor > 1 else "negative" if profit_factor < 1 else "neutral"
        render_metric_card("Profit Factor", f"{profit_factor:.2f}x", pf_status)

    with col4:
        sh_status = "positive" if sharpe_ratio > 0 else "negative"
        render_metric_card("Sharpe Ratio", f"{sharpe_ratio:.2f}", sh_status)

    # Secondary metrics row
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        np_status = "positive" if net_profit > 0 else "negative"
        sign = "+" if net_profit > 0 else ""
        render_metric_card("Net Profit", f"{sign}${net_profit:,.0f}", np_status)

    with col6:
        render_metric_card("Return", f"{sign}{net_profit_pct:.1f}%", np_status)

    with col7:
        render_metric_card("Max Drawdown", f"-{max_dd:.1f}%", "negative")

    with col8:
        render_metric_card("Final Equity", f"${final_equity:,.0f}", "neutral")

    # Equity Curve
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
    equity_panel = '<div class="panel">'
    equity_panel += '<div class="panel-header">Equity Curve</div>'
    equity_panel += '</div>'
    st.markdown(equity_panel, unsafe_allow_html=True)

    equity_curve = results.get('equity_curve', [])
    if equity_curve:
        render_equity_chart(equity_curve)
    else:
        st.info("No equity curve data available")

    # Trade Distribution & Statistics
    trades = results.get('trades', [])

    col_left, col_right = st.columns(2)

    with col_left:
        dist_panel = '<div class="panel">'
        dist_panel += '<div class="panel-header">Trade P&L Distribution</div>'
        dist_panel += '</div>'
        st.markdown(dist_panel, unsafe_allow_html=True)

        if trades:
            render_trade_distribution(trades)
        else:
            st.info("No trade data available")

    with col_right:
        stats_panel = '<div class="panel">'
        stats_panel += '<div class="panel-header">Trade Statistics</div>'
        stats_panel += '</div>'
        st.markdown(stats_panel, unsafe_allow_html=True)

        if trades:
            stats = calculate_trade_stats(trades)

            stats_html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px;">'

            # Wins/Losses
            stats_html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Total Wins:</div>'
            stats_html += f'<div style="color: {ARCTIC_PRO["success"]}; font-weight: 600;">{stats.get("total_wins", 0)}</div>'

            stats_html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Total Losses:</div>'
            stats_html += f'<div style="color: {ARCTIC_PRO["danger"]}; font-weight: 600;">{stats.get("total_losses", 0)}</div>'

            # Average win/loss
            stats_html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Avg Win:</div>'
            stats_html += f'<div style="color: {ARCTIC_PRO["success"]}; font-weight: 600;">+${stats.get("avg_win", 0):,.0f}</div>'

            stats_html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Avg Loss:</div>'
            stats_html += f'<div style="color: {ARCTIC_PRO["danger"]}; font-weight: 600;">-${stats.get("avg_loss", 0):,.0f}</div>'

            # Largest win/loss
            stats_html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Largest Win:</div>'
            stats_html += f'<div style="color: {ARCTIC_PRO["success"]}; font-weight: 600;">+${stats.get("largest_win", 0):,.0f}</div>'

            stats_html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Largest Loss:</div>'
            stats_html += f'<div style="color: {ARCTIC_PRO["danger"]}; font-weight: 600;">-${stats.get("largest_loss", 0):,.0f}</div>'

            # Consecutive
            stats_html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Max Consec Wins:</div>'
            stats_html += f'<div style="color: {ARCTIC_PRO["text_secondary"]}; font-weight: 600;">{stats.get("max_consec_wins", 0)}</div>'

            stats_html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Max Consec Losses:</div>'
            stats_html += f'<div style="color: {ARCTIC_PRO["text_secondary"]}; font-weight: 600;">{stats.get("max_consec_losses", 0)}</div>'

            stats_html += '</div>'
            st.markdown(stats_html, unsafe_allow_html=True)
        else:
            st.info("No trade data available")
