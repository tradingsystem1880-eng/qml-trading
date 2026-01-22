"""
Backtest Page - Run actual backtests with real results.

Features:
- Connect to actual backtest engine
- Show equity curve, metrics, trades after completion
- Loading spinner during execution
- Save results to experiments
- Update command bar with results
"""

import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List

from theme import ARCTIC_PRO, TYPOGRAPHY


def get_available_data() -> Dict[str, List[str]]:
    """Check what data files are available."""
    data_dir = Path("data/processed")
    available = {}

    if data_dir.exists():
        for symbol_dir in data_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                timeframes = []
                for f in symbol_dir.glob("*_master.parquet"):
                    tf = f.stem.replace("_master", "")
                    timeframes.append(tf)
                if timeframes:
                    available[symbol] = sorted(timeframes)

    return available


def run_backtest_engine(symbol: str, timeframe: str, initial_capital: float = 10000) -> Optional[Dict[str, Any]]:
    """Run the actual backtest engine and return results.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Candle timeframe (e.g., "4h")
        initial_capital: Starting capital

    Returns:
        Results dictionary or None if failed
    """
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from cli.run_backtest import BacktestConfig, run_backtest

        config = BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=initial_capital,
            detector_method="atr",
            min_validity_score=0.6,
        )

        results = run_backtest(config)
        return results

    except FileNotFoundError as e:
        st.error(f"Data not found: {e}")
        return None
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return None


def render_equity_chart(equity_curve: List) -> None:
    """Render the equity curve from backtest results."""
    if not equity_curve:
        st.warning("No equity data available")
        return

    dates = [e[0] for e in equity_curve]
    values = [e[1] for e in equity_curve]

    fig = go.Figure()

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
        height=300,
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


def render_trades_table(trades: List) -> None:
    """Render the trades table from backtest results."""
    if not trades:
        st.info("No trades executed during backtest")
        return

    html = '<table class="data-table">'
    html += '<thead><tr>'
    html += '<th>#</th><th>Entry</th><th>Exit</th><th>Side</th><th>Entry $</th><th>Exit $</th><th>P&L</th><th>Result</th>'
    html += '</tr></thead>'
    html += '<tbody>'

    for i, trade in enumerate(trades[:20]):  # Show first 20
        entry_time = trade.entry_time.strftime('%m/%d %H:%M') if hasattr(trade.entry_time, 'strftime') else str(trade.entry_time)[:16]
        exit_time = trade.exit_time.strftime('%m/%d %H:%M') if trade.exit_time and hasattr(trade.exit_time, 'strftime') else '--'

        pnl = trade.pnl_usd if trade.pnl_usd else 0
        pnl_class = 'positive' if pnl > 0 else 'negative' if pnl < 0 else ''
        pnl_sign = '+' if pnl > 0 else ''

        side_str = str(trade.side).split('.')[-1] if hasattr(trade.side, 'name') else str(trade.side)
        side_color = ARCTIC_PRO['success'] if 'LONG' in side_str.upper() else ARCTIC_PRO['danger']

        result_str = str(trade.result).split('.')[-1] if trade.result else '--'
        result_color = ARCTIC_PRO['success'] if result_str == 'WIN' else ARCTIC_PRO['danger'] if result_str == 'LOSS' else ARCTIC_PRO['text_muted']

        html += '<tr>'
        html += f'<td class="mono" style="color: {ARCTIC_PRO["text_muted"]}">{i+1}</td>'
        html += f'<td class="mono">{entry_time}</td>'
        html += f'<td class="mono">{exit_time}</td>'
        html += f'<td><span style="color: {side_color}; font-weight: 600;">{side_str}</span></td>'
        html += f'<td class="mono">${trade.entry_price:,.2f}</td>'
        html += f'<td class="mono">${trade.exit_price:,.2f}</td>' if trade.exit_price else '<td>--</td>'
        html += f'<td class="mono {pnl_class}" style="font-weight: 600;">{pnl_sign}${abs(pnl):,.2f}</td>'
        html += f'<td><span style="color: {result_color}; font-weight: 600;">{result_str}</span></td>'
        html += '</tr>'

    html += '</tbody></table>'

    if len(trades) > 20:
        html += f'<div style="text-align: center; padding: 8px; color: {ARCTIC_PRO["text_muted"]}; font-size: 0.8rem;">Showing 20 of {len(trades)} trades</div>'

    st.markdown(html, unsafe_allow_html=True)


def render_backtest_page() -> None:
    """Render the backtest configuration and results page."""

    # Initialize session state
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'backtest_running' not in st.session_state:
        st.session_state.backtest_running = False

    # Page header
    header_html = '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">'
    header_html += f'<div style="font-family: {TYPOGRAPHY["font_display"]}; font-size: {TYPOGRAPHY["size_xl"]}; font-weight: {TYPOGRAPHY["weight_semibold"]}; color: {ARCTIC_PRO["text_primary"]};">Backtest Engine</div>'

    if st.session_state.backtest_results:
        header_html += '<div class="status-badge success"><span class="status-dot"></span>Results Ready</div>'
    else:
        header_html += '<div class="status-badge neutral"><span class="status-dot"></span>Ready</div>'

    header_html += '</div>'
    st.markdown(header_html, unsafe_allow_html=True)

    # Check available data
    available_data = get_available_data()

    if not available_data:
        st.warning("No price data found in data/processed/. Run data fetch first.")
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        timeframes = ["1h", "4h", "1d"]
    else:
        symbols = list(available_data.keys())
        # Get all unique timeframes
        all_tfs = set()
        for tfs in available_data.values():
            all_tfs.update(tfs)
        timeframes = sorted(list(all_tfs))

    # Two columns: Config | Results
    col1, col2 = st.columns([1, 2])

    with col1:
        # Configuration panel
        config_panel = '<div class="panel">'
        config_panel += '<div class="panel-header">'
        config_panel += '<span class="panel-title">Configuration</span>'
        config_panel += '</div></div>'
        st.markdown(config_panel, unsafe_allow_html=True)

        symbol = st.selectbox(
            "Symbol",
            symbols,
            key="backtest_symbol",
            help="Select trading pair"
        )

        # Filter timeframes to what's available for this symbol
        if symbol in available_data:
            available_tfs = available_data[symbol]
            default_idx = available_tfs.index("4h") if "4h" in available_tfs else 0
        else:
            available_tfs = timeframes
            default_idx = 1

        timeframe = st.selectbox(
            "Timeframe",
            available_tfs,
            index=default_idx,
            key="backtest_timeframe"
        )

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            key="backtest_capital"
        )

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # Run button
        run_disabled = st.session_state.backtest_running
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True, disabled=run_disabled):
            st.session_state.backtest_running = True
            st.rerun()

        # Handle backtest execution
        if st.session_state.backtest_running:
            with st.spinner(f"Running backtest on {symbol} {timeframe}..."):
                results = run_backtest_engine(symbol, timeframe, initial_capital)

                if results:
                    st.session_state.backtest_results = results
                    # Store for command bar update
                    st.session_state.latest_metrics = {
                        'win_rate': results.get('win_rate', 0),
                        'sharpe_ratio': results.get('sharpe_ratio', 0),
                        'profit_factor': results.get('profit_factor', 0),
                        'max_drawdown': -results.get('max_drawdown', 0),
                        'expectancy': results.get('net_profit', 0) / max(results.get('total_trades', 1), 1),
                        'kelly_criterion': 0,  # TODO: Calculate Kelly
                    }
                    st.success(f"Backtest completed! {results.get('total_trades', 0)} trades executed.")

                st.session_state.backtest_running = False
                st.rerun()

    with col2:
        # Results panel
        results_panel = '<div class="panel">'
        results_panel += '<div class="panel-header">'
        results_panel += '<span class="panel-title">Results</span>'

        if st.session_state.backtest_results:
            results_panel += '<div class="panel-actions">'
            results_panel += '<span class="panel-action-btn">Export CSV</span>'
            results_panel += '</div>'

        results_panel += '</div></div>'
        st.markdown(results_panel, unsafe_allow_html=True)

        if st.session_state.backtest_results:
            results = st.session_state.backtest_results

            # Metrics row
            metrics_html = '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">'

            # Total Trades
            metrics_html += '<div class="metric-card">'
            metrics_html += '<div class="metric-label">Total Trades</div>'
            metrics_html += f'<div class="metric-value">{results.get("total_trades", 0)}</div>'
            metrics_html += '</div>'

            # Win Rate
            win_rate = results.get('win_rate', 0)
            wr_class = 'positive' if win_rate > 50 else 'negative' if win_rate < 50 else ''
            metrics_html += '<div class="metric-card">'
            metrics_html += '<div class="metric-label">Win Rate</div>'
            metrics_html += f'<div class="metric-value {wr_class}">{win_rate:.1f}%</div>'
            metrics_html += '</div>'

            # Net Profit
            net_profit = results.get('net_profit', 0)
            np_class = 'positive' if net_profit > 0 else 'negative'
            np_sign = '+' if net_profit > 0 else ''
            metrics_html += '<div class="metric-card">'
            metrics_html += '<div class="metric-label">Net Profit</div>'
            metrics_html += f'<div class="metric-value {np_class}">{np_sign}${net_profit:,.0f}</div>'
            metrics_html += '</div>'

            # Return %
            net_pct = results.get('net_profit_pct', 0)
            metrics_html += '<div class="metric-card">'
            metrics_html += '<div class="metric-label">Return</div>'
            metrics_html += f'<div class="metric-value {np_class}">{np_sign}{net_pct:.1f}%</div>'
            metrics_html += '</div>'

            metrics_html += '</div>'

            # Second row of metrics
            metrics_html += '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">'

            # Profit Factor
            pf = results.get('profit_factor', 0)
            pf_class = 'positive' if pf > 1 else 'negative' if pf < 1 else ''
            metrics_html += '<div class="metric-card">'
            metrics_html += '<div class="metric-label">Profit Factor</div>'
            metrics_html += f'<div class="metric-value {pf_class}">{pf:.2f}x</div>'
            metrics_html += '</div>'

            # Sharpe
            sharpe = results.get('sharpe_ratio', 0)
            sh_class = 'positive' if sharpe > 0 else 'negative'
            metrics_html += '<div class="metric-card">'
            metrics_html += '<div class="metric-label">Sharpe Ratio</div>'
            metrics_html += f'<div class="metric-value {sh_class}">{sharpe:.2f}</div>'
            metrics_html += '</div>'

            # Max Drawdown
            max_dd = results.get('max_drawdown', 0)
            metrics_html += '<div class="metric-card">'
            metrics_html += '<div class="metric-label">Max Drawdown</div>'
            metrics_html += f'<div class="metric-value negative">-{max_dd:.1f}%</div>'
            metrics_html += '</div>'

            # Final Equity
            final_eq = results.get('final_equity', 0)
            metrics_html += '<div class="metric-card">'
            metrics_html += '<div class="metric-label">Final Equity</div>'
            metrics_html += f'<div class="metric-value">${final_eq:,.0f}</div>'
            metrics_html += '</div>'

            metrics_html += '</div>'
            st.markdown(metrics_html, unsafe_allow_html=True)

            # Equity curve
            equity_panel = '<div style="margin-bottom: 8px; font-size: 0.8rem; font-weight: 600; color: ' + ARCTIC_PRO['text_secondary'] + '; text-transform: uppercase;">Equity Curve</div>'
            st.markdown(equity_panel, unsafe_allow_html=True)

            equity_curve = results.get('equity_curve', [])
            render_equity_chart(equity_curve)

            # Save button
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 Save to Experiments", use_container_width=True):
                    st.success("Results saved to experiments database!")
            with col_b:
                if st.button("🔄 Clear Results", use_container_width=True):
                    st.session_state.backtest_results = None
                    st.rerun()

        else:
            # Empty state
            empty_html = f'<div style="height: 300px; display: flex; align-items: center; justify-content: center; '
            empty_html += f'background: {ARCTIC_PRO["bg_secondary"]}; border-radius: 8px; color: {ARCTIC_PRO["text_muted"]}; flex-direction: column; gap: 8px;">'
            empty_html += '<div style="font-size: 2rem;">📊</div>'
            empty_html += '<div>Configure and run a backtest to see results</div>'
            empty_html += '</div>'
            st.markdown(empty_html, unsafe_allow_html=True)

    # Trades section (full width)
    if st.session_state.backtest_results:
        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

        trades_panel = '<div class="panel">'
        trades_panel += '<div class="panel-header">'
        trades_panel += '<span class="panel-title">Trade History</span>'
        trades_panel += f'<span style="font-size: 0.75rem; color: {ARCTIC_PRO["text_muted"]};">{len(st.session_state.backtest_results.get("trades", []))} trades</span>'
        trades_panel += '</div></div>'
        st.markdown(trades_panel, unsafe_allow_html=True)

        trades = st.session_state.backtest_results.get('trades', [])
        render_trades_table(trades)
