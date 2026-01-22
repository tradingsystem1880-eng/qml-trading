"""
Dashboard Page - Premium overview with equity curve, trades, and activity.

Features:
- Real-time equity curve chart (Plotly) - uses real data when available
- Recent trades table with P&L
- Activity feed with timestamps
- Quick stats with trends
- Market status indicators

Falls back to mock data when no backtest results exist.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from typing import Dict, Any, List, Optional
from theme import ARCTIC_PRO, TYPOGRAPHY, generate_sparkline_svg


def get_backtest_results() -> Optional[Dict[str, Any]]:
    """Get backtest results from session state."""
    return st.session_state.get('backtest_results', None)


def generate_mock_equity_curve(days: int = 90, starting_balance: float = 10000) -> tuple:
    """Generate mock equity curve data for demonstration."""
    dates = []
    equity = []
    current = starting_balance

    for i in range(days):
        date = datetime.now() - timedelta(days=days - i)
        dates.append(date)

        # Simulate realistic equity changes
        daily_return = random.gauss(0.002, 0.015)  # 0.2% avg return, 1.5% std
        current *= (1 + daily_return)
        equity.append(current)

    return dates, equity


def generate_mock_trades(count: int = 10) -> list:
    """Generate mock trade data for demonstration."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    trades = []

    for i in range(count):
        is_win = random.random() > 0.4  # 60% win rate
        pnl = random.uniform(50, 500) if is_win else -random.uniform(30, 200)
        entry_time = datetime.now() - timedelta(hours=random.randint(1, 72))

        trades.append({
            'id': f'#{1000 + count - i}',
            'symbol': random.choice(symbols),
            'side': random.choice(['LONG', 'SHORT']),
            'entry': entry_time.strftime('%m/%d %H:%M'),
            'pnl': pnl,
            'pnl_pct': pnl / 100,  # Simplified %
            'duration': f'{random.randint(1, 24)}h {random.randint(0, 59)}m',
            'is_win': is_win,
        })

    return sorted(trades, key=lambda x: x['entry'], reverse=True)


def convert_real_trades_to_display(trades: List) -> list:
    """Convert real trade objects to display format."""
    display_trades = []

    for i, trade in enumerate(trades[:10]):  # Last 10 trades
        pnl = trade.pnl_usd if hasattr(trade, 'pnl_usd') and trade.pnl_usd else 0
        is_win = pnl > 0

        entry_time = trade.entry_time if hasattr(trade, 'entry_time') else datetime.now()
        if hasattr(entry_time, 'strftime'):
            entry_str = entry_time.strftime('%m/%d %H:%M')
        else:
            entry_str = str(entry_time)[:16]

        side_str = str(trade.side).split('.')[-1] if hasattr(trade, 'side') else 'LONG'
        if 'LONG' not in side_str.upper() and 'SHORT' not in side_str.upper():
            side_str = 'LONG'

        # Calculate duration if exit time exists
        duration = '--'
        if hasattr(trade, 'exit_time') and trade.exit_time and hasattr(trade, 'entry_time'):
            try:
                delta = trade.exit_time - trade.entry_time
                hours = int(delta.total_seconds() // 3600)
                mins = int((delta.total_seconds() % 3600) // 60)
                duration = f'{hours}h {mins}m'
            except Exception:
                pass

        display_trades.append({
            'id': f'#{i + 1}',
            'symbol': 'BTC/USDT',  # Default since real trades may not have symbol
            'side': side_str.upper(),
            'entry': entry_str,
            'pnl': pnl,
            'pnl_pct': (pnl / 100) if pnl else 0,
            'duration': duration,
            'is_win': is_win,
        })

    return display_trades


def generate_mock_activity() -> list:
    """Generate mock activity feed data."""
    activities = [
        {'type': 'trade', 'title': 'New BTC/USDT Long opened', 'time': '2 min ago', 'icon': '📈'},
        {'type': 'win', 'title': 'ETH/USDT Short closed +$234', 'time': '15 min ago', 'icon': '✅'},
        {'type': 'pattern', 'title': 'QML pattern detected on SOL', 'time': '32 min ago', 'icon': '🔍'},
        {'type': 'loss', 'title': 'BTC/USDT Long stopped out -$89', 'time': '1 hour ago', 'icon': '❌'},
        {'type': 'trade', 'title': 'New ETH/USDT Long opened', 'time': '2 hours ago', 'icon': '📈'},
        {'type': 'backtest', 'title': 'Backtest completed: 67% win rate', 'time': '3 hours ago', 'icon': '📊'},
    ]
    return activities


def generate_activity_from_results(results: Dict[str, Any]) -> list:
    """Generate activity feed from real backtest results."""
    activities = []

    # Add backtest completion activity
    total_trades = results.get('total_trades', 0)
    win_rate = results.get('win_rate', 0)
    net_profit = results.get('net_profit', 0)

    activities.append({
        'type': 'backtest',
        'title': f'Backtest completed: {win_rate:.0f}% win rate',
        'time': 'Just now',
        'icon': '📊'
    })

    if net_profit > 0:
        activities.append({
            'type': 'win',
            'title': f'Net profit: +${net_profit:,.0f}',
            'time': 'Just now',
            'icon': '✅'
        })
    else:
        activities.append({
            'type': 'loss',
            'title': f'Net loss: ${net_profit:,.0f}',
            'time': 'Just now',
            'icon': '❌'
        })

    activities.append({
        'type': 'trade',
        'title': f'{total_trades} trades executed',
        'time': 'Just now',
        'icon': '📈'
    })

    # Add some context activities
    activities.append({
        'type': 'pattern',
        'title': 'Pattern detection completed',
        'time': '1 min ago',
        'icon': '🔍'
    })

    return activities


def render_equity_chart(dates: list, equity: list) -> None:
    """Render the equity curve chart using Plotly."""
    fig = go.Figure()

    # Add equity line with gradient fill
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity,
        mode='lines',
        name='Equity',
        line=dict(color=ARCTIC_PRO['accent'], width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='%{x|%b %d}<br>$%{y:,.0f}<extra></extra>'
    ))

    # Calculate and add moving average
    if len(equity) >= 20:
        ma = []
        for i in range(len(equity)):
            if i < 19:
                ma.append(None)
            else:
                ma.append(sum(equity[i-19:i+1]) / 20)

        fig.add_trace(go.Scatter(
            x=dates,
            y=ma,
            mode='lines',
            name='20-day MA',
            line=dict(color=ARCTIC_PRO['text_muted'], width=1, dash='dot'),
            hoverinfo='skip'
        ))

    # Style the chart
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
            zeroline=False,
            tickfont=dict(color=ARCTIC_PRO['text_muted'], size=10),
            tickformat='%b %d',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=ARCTIC_PRO['border'],
            showline=False,
            zeroline=False,
            tickfont=dict(color=ARCTIC_PRO['text_muted'], size=10),
            tickprefix='$',
            tickformat=',.0f',
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_trades_table(trades: list) -> None:
    """Render the recent trades table."""
    html = '<table class="data-table">'
    html += '<thead><tr>'
    html += '<th>ID</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Duration</th><th>P&L</th>'
    html += '</tr></thead>'
    html += '<tbody>'

    for trade in trades[:8]:  # Show last 8 trades
        pnl_class = 'positive' if trade['is_win'] else 'negative'
        pnl_sign = '+' if trade['pnl'] > 0 else ''
        side_color = ARCTIC_PRO['success'] if trade['side'] == 'LONG' else ARCTIC_PRO['danger']

        html += '<tr>'
        html += f'<td class="mono" style="color: {ARCTIC_PRO["text_muted"]}">{trade["id"]}</td>'
        html += f'<td style="font-weight: 600;">{trade["symbol"]}</td>'
        html += f'<td><span style="color: {side_color}; font-weight: 600;">{trade["side"]}</span></td>'
        html += f'<td class="mono">{trade["entry"]}</td>'
        html += f'<td class="mono" style="color: {ARCTIC_PRO["text_muted"]}">{trade["duration"]}</td>'
        html += f'<td class="mono {pnl_class}" style="font-weight: 600;">{pnl_sign}${abs(trade["pnl"]):,.0f}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


def render_activity_feed(activities: list) -> None:
    """Render the activity feed."""
    for activity in activities[:6]:
        icon_class = 'win' if activity['type'] == 'win' else 'loss' if activity['type'] == 'loss' else 'trade'

        html = '<div class="activity-item">'
        html += f'<div class="activity-icon {icon_class}">{activity["icon"]}</div>'
        html += '<div class="activity-content">'
        html += f'<div class="activity-title">{activity["title"]}</div>'
        html += f'<div class="activity-meta">{activity["time"]}</div>'
        html += '</div></div>'
        st.markdown(html, unsafe_allow_html=True)


def render_dashboard_page() -> None:
    """Render the premium dashboard overview page."""

    # Check for real backtest results
    results = get_backtest_results()
    has_real_data = results is not None

    if has_real_data:
        # Use real data from backtest results
        equity_curve = results.get('equity_curve', [])
        if equity_curve:
            dates = [e[0] for e in equity_curve]
            equity = [e[1] for e in equity_curve]
        else:
            dates, equity = generate_mock_equity_curve(90, 10000)

        real_trades = results.get('trades', [])
        if real_trades:
            trades = convert_real_trades_to_display(real_trades)
        else:
            trades = generate_mock_trades(10)

        activities = generate_activity_from_results(results)

        # Calculate stats from real data
        total_pnl = results.get('net_profit', 0)
        pnl_pct = results.get('net_profit_pct', 0)
        win_rate = results.get('win_rate', 0)
        total_trades_count = results.get('total_trades', 0)
        status_text = "Real Data"
        status_class = "success"
    else:
        # Generate mock data
        dates, equity = generate_mock_equity_curve(90, 10000)
        trades = generate_mock_trades(10)
        activities = generate_mock_activity()

        # Calculate stats from mock data
        total_pnl = equity[-1] - equity[0]
        pnl_pct = (equity[-1] / equity[0] - 1) * 100
        win_count = sum(1 for t in trades if t['is_win'])
        total_trades_count = len(trades)
        win_rate = (win_count / total_trades_count * 100) if total_trades_count > 0 else 0
        status_text = "Demo Data"
        status_class = "neutral"

    # Header with status
    header_html = '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">'
    header_html += f'<div style="font-family: {TYPOGRAPHY["font_display"]}; font-size: {TYPOGRAPHY["size_xl"]}; font-weight: {TYPOGRAPHY["weight_semibold"]}; color: {ARCTIC_PRO["text_primary"]};">Dashboard Overview</div>'
    header_html += f'<div class="status-badge {status_class}"><span class="status-dot"></span>{status_text}</div>'
    header_html += '</div>'
    st.markdown(header_html, unsafe_allow_html=True)

    # Main content - Two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Equity Curve Panel
        panel_html = '<div class="panel">'
        panel_html += '<div class="panel-header">'
        panel_html += '<span class="panel-title">Equity Curve</span>'
        panel_html += '<div class="panel-actions">'
        panel_html += '<span class="panel-action-btn">1W</span>'
        panel_html += '<span class="panel-action-btn">1M</span>'
        panel_html += '<span class="panel-action-btn" style="background: ' + ARCTIC_PRO['accent_muted'] + '; color: ' + ARCTIC_PRO['accent'] + ';">3M</span>'
        panel_html += '<span class="panel-action-btn">1Y</span>'
        panel_html += '</div></div>'
        panel_html += '</div>'
        st.markdown(panel_html, unsafe_allow_html=True)

        render_equity_chart(dates, equity)

        # Quick stats row
        stats_html = '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px;">'

        # Total P&L
        pnl_class = 'positive' if total_pnl > 0 else 'negative'
        pnl_sign = '+' if total_pnl > 0 else ''
        stats_html += '<div class="metric-card">'
        stats_html += '<div class="metric-label">Total P&L</div>'
        stats_html += f'<div class="metric-value {pnl_class}">{pnl_sign}${total_pnl:,.0f}</div>'
        stats_html += '</div>'

        # Return %
        stats_html += '<div class="metric-card">'
        stats_html += '<div class="metric-label">Return</div>'
        stats_html += f'<div class="metric-value {pnl_class}">{pnl_sign}{pnl_pct:.1f}%</div>'
        stats_html += '</div>'

        # Trades
        stats_html += '<div class="metric-card">'
        stats_html += '<div class="metric-label">Trades</div>'
        stats_html += f'<div class="metric-value">{total_trades_count}</div>'
        stats_html += '</div>'

        # Win Rate
        wr_class = 'positive' if win_rate > 50 else 'negative' if win_rate < 50 else ''
        stats_html += '<div class="metric-card">'
        stats_html += '<div class="metric-label">Win Rate</div>'
        stats_html += f'<div class="metric-value {wr_class}">{win_rate:.0f}%</div>'
        stats_html += '</div>'

        stats_html += '</div>'
        st.markdown(stats_html, unsafe_allow_html=True)

        # Recent Trades Panel
        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

        trades_panel = '<div class="panel">'
        trades_panel += '<div class="panel-header">'
        trades_panel += '<span class="panel-title">Recent Trades</span>'
        trades_panel += '<div class="panel-actions">'
        trades_panel += '<span class="panel-action-btn">View All</span>'
        trades_panel += '</div></div>'
        trades_panel += '</div>'
        st.markdown(trades_panel, unsafe_allow_html=True)

        render_trades_table(trades)

    with col2:
        # Activity Feed Panel
        activity_panel = '<div class="panel">'
        activity_panel += '<div class="panel-header">'
        activity_panel += '<span class="panel-title">Activity Feed</span>'
        activity_panel += '</div>'
        activity_panel += '</div>'
        st.markdown(activity_panel, unsafe_allow_html=True)

        render_activity_feed(activities)

        # Market Status Panel
        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

        market_panel = '<div class="panel">'
        market_panel += '<div class="panel-header">'
        market_panel += '<span class="panel-title">Market Status</span>'
        market_panel += '</div>'

        # BTC
        btc_price = 97453.21
        btc_change = 2.34
        btc_sparkline = [94000, 94500, 95200, 94800, 96100, 95800, 96500, 97000, 96800, 97453]
        btc_svg = generate_sparkline_svg(btc_sparkline, width=60, height=20, color=ARCTIC_PRO['success'])

        market_panel += f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid {ARCTIC_PRO["border"]};">'
        market_panel += '<div>'
        market_panel += f'<div style="font-weight: {TYPOGRAPHY["weight_semibold"]}; color: {ARCTIC_PRO["text_primary"]};">BTC/USDT</div>'
        market_panel += f'<div style="font-family: {TYPOGRAPHY["font_mono"]}; font-size: {TYPOGRAPHY["size_lg"]}; color: {ARCTIC_PRO["text_primary"]};">${btc_price:,.2f}</div>'
        market_panel += '</div>'
        market_panel += '<div style="text-align: right;">'
        market_panel += f'{btc_svg}'
        market_panel += f'<div style="color: {ARCTIC_PRO["success"]}; font-size: {TYPOGRAPHY["size_sm"]}; font-weight: {TYPOGRAPHY["weight_semibold"]};">+{btc_change}%</div>'
        market_panel += '</div></div>'

        # ETH
        eth_price = 3287.45
        eth_change = -1.12
        eth_sparkline = [3400, 3380, 3350, 3320, 3340, 3300, 3280, 3260, 3290, 3287]
        eth_svg = generate_sparkline_svg(eth_sparkline, width=60, height=20, color=ARCTIC_PRO['danger'])

        market_panel += f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid {ARCTIC_PRO["border"]};">'
        market_panel += '<div>'
        market_panel += f'<div style="font-weight: {TYPOGRAPHY["weight_semibold"]}; color: {ARCTIC_PRO["text_primary"]};">ETH/USDT</div>'
        market_panel += f'<div style="font-family: {TYPOGRAPHY["font_mono"]}; font-size: {TYPOGRAPHY["size_lg"]}; color: {ARCTIC_PRO["text_primary"]};">${eth_price:,.2f}</div>'
        market_panel += '</div>'
        market_panel += '<div style="text-align: right;">'
        market_panel += f'{eth_svg}'
        market_panel += f'<div style="color: {ARCTIC_PRO["danger"]}; font-size: {TYPOGRAPHY["size_sm"]}; font-weight: {TYPOGRAPHY["weight_semibold"]};">{eth_change}%</div>'
        market_panel += '</div></div>'

        # SOL
        sol_price = 198.73
        sol_change = 5.67
        sol_sparkline = [185, 188, 190, 187, 192, 195, 193, 196, 198, 199]
        sol_svg = generate_sparkline_svg(sol_sparkline, width=60, height=20, color=ARCTIC_PRO['success'])

        market_panel += f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0;">'
        market_panel += '<div>'
        market_panel += f'<div style="font-weight: {TYPOGRAPHY["weight_semibold"]}; color: {ARCTIC_PRO["text_primary"]};">SOL/USDT</div>'
        market_panel += f'<div style="font-family: {TYPOGRAPHY["font_mono"]}; font-size: {TYPOGRAPHY["size_lg"]}; color: {ARCTIC_PRO["text_primary"]};">${sol_price:,.2f}</div>'
        market_panel += '</div>'
        market_panel += '<div style="text-align: right;">'
        market_panel += f'{sol_svg}'
        market_panel += f'<div style="color: {ARCTIC_PRO["success"]}; font-size: {TYPOGRAPHY["size_sm"]}; font-weight: {TYPOGRAPHY["weight_semibold"]};">+{sol_change}%</div>'
        market_panel += '</div></div>'

        market_panel += '</div>'
        st.markdown(market_panel, unsafe_allow_html=True)

        # Hint when no real data
        if not has_real_data:
            hint_html = f'<div style="margin-top: 16px; padding: 12px; background: {ARCTIC_PRO["bg_secondary"]}; '
            hint_html += f'border-radius: 8px; font-size: {TYPOGRAPHY["size_sm"]}; color: {ARCTIC_PRO["text_muted"]}; text-align: center;">'
            hint_html += 'Run a backtest to see real data'
            hint_html += '</div>'
            st.markdown(hint_html, unsafe_allow_html=True)
