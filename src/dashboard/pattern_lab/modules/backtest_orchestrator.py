"""
Backtest Orchestrator Module
============================
Interface to run QMLStrategy backtests with parameters.
"""

from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from loguru import logger

# Styling
COLORS = {
    "background": "#0F172A",
    "card_bg": "#1E293B",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
    "border": "#334155",
    "primary": "#0EA5E9",
    "success": "#22C55E",
    "danger": "#EF4444",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "16px",
    "marginBottom": "12px",
}


def create_backtest_tab() -> dbc.Container:
    """Create Backtest Orchestrator tab content."""
    
    return dbc.Container(
        fluid=True,
        className="p-4",
        style={"backgroundColor": COLORS["background"], "minHeight": "80vh"},
        children=[
            dbc.Row([
                # Left: Parameters
                dbc.Col(width=3, children=[
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H5(
                            [html.I(className="fas fa-cog me-2"), "Strategy Parameters"],
                            className="mb-3 text-light",
                        ),
                        
                        # Symbol
                        html.Label("Symbol", className="text-muted small"),
                        dcc.Dropdown(
                            id="bt-symbol",
                            options=[
                                {"label": "BTC/USDT", "value": "BTC/USDT"},
                                {"label": "ETH/USDT", "value": "ETH/USDT"},
                            ],
                            value="BTC/USDT",
                            className="mb-3",
                        ),
                        
                        # Timeframe
                        html.Label("Timeframe", className="text-muted small"),
                        dcc.Dropdown(
                            id="bt-timeframe",
                            options=[
                                {"label": "1h", "value": "1h"},
                                {"label": "4h", "value": "4h"},
                                {"label": "1d", "value": "1d"},
                            ],
                            value="4h",
                            className="mb-3",
                        ),
                        
                        html.Hr(style={"borderColor": COLORS["border"]}),
                        
                        # ATR Period
                        html.Label("ATR Period", className="text-muted small"),
                        dbc.Input(
                            id="bt-atr-period",
                            type="number",
                            value=14,
                            min=5, max=50,
                            className="mb-3",
                        ),
                        
                        # Risk/Reward
                        html.Label("Risk/Reward Ratio", className="text-muted small"),
                        dbc.Input(
                            id="bt-risk-reward",
                            type="number",
                            value=2.0,
                            min=1.0, max=5.0, step=0.5,
                            className="mb-3",
                        ),
                        
                        # Register Patterns
                        dbc.Checkbox(
                            id="bt-register-patterns",
                            label="Register patterns to ML registry",
                            value=True,
                            className="mb-3 text-light",
                        ),
                        
                        html.Hr(style={"borderColor": COLORS["border"]}),
                        
                        # Run Button
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Run Backtest"],
                            id="run-backtest-btn",
                            color="success",
                            className="w-100",
                            size="lg",
                        ),
                        
                        # Status
                        html.Div(id="bt-status", className="mt-3"),
                    ]),
                ]),
                
                # Center: Equity Curve
                dbc.Col(width=6, children=[
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H5(
                            [html.I(className="fas fa-chart-line me-2"), "Equity Curve"],
                            className="mb-3 text-light",
                        ),
                        dcc.Loading(
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="bt-equity-chart",
                                    figure=_create_empty_equity_chart(),
                                    style={"height": "350px"},
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                    ]),
                    
                    # Trade List
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H5(
                            [html.I(className="fas fa-list me-2"), "Trades"],
                            className="mb-3 text-light",
                        ),
                        html.Div(
                            id="bt-trades-list",
                            style={"maxHeight": "200px", "overflowY": "auto"},
                            children=[
                                html.P("Run a backtest to see trades", className="text-muted text-center"),
                            ],
                        ),
                    ]),
                ]),
                
                # Right: Statistics
                dbc.Col(width=3, children=[
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H5(
                            [html.I(className="fas fa-chart-pie me-2"), "Performance"],
                            className="mb-3 text-light",
                        ),
                        html.Div(id="bt-stats", children=[
                            _create_stat_row("Return", "--"),
                            _create_stat_row("Sharpe Ratio", "--"),
                            _create_stat_row("Max Drawdown", "--"),
                            _create_stat_row("Win Rate", "--"),
                            _create_stat_row("# Trades", "--"),
                            _create_stat_row("Exposure", "--"),
                        ]),
                    ]),
                    
                    # Quick Actions
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H6("Quick Actions", className="text-light mb-3"),
                        dbc.Button(
                            [html.I(className="fas fa-flask me-2"), "Run VRD Validation"],
                            id="bt-run-vrd",
                            color="primary",
                            outline=True,
                            className="w-100 mb-2",
                            size="sm",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-brain me-2"), "Train ML Model"],
                            id="bt-train-ml",
                            color="primary",
                            outline=True,
                            className="w-100",
                            size="sm",
                        ),
                    ]),
                ]),
            ]),
        ],
    )


def _create_empty_equity_chart() -> go.Figure:
    """Create empty equity chart placeholder."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font={"color": COLORS["text"]},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": "Run backtest to see equity curve",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 14, "color": COLORS["text_muted"]},
        }],
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _create_stat_row(label: str, value: str) -> html.Div:
    """Create a statistics row."""
    return html.Div(
        className="d-flex justify-content-between mb-2",
        children=[
            html.Span(label, className="text-muted", style={"fontSize": "0.85rem"}),
            html.Span(value, className="text-light", style={"fontSize": "0.85rem", "fontWeight": "600"}),
        ],
    )


def register_backtest_callbacks(app):
    """Register Backtest Orchestrator callbacks."""
    
    @app.callback(
        [
            Output("bt-equity-chart", "figure"),
            Output("bt-trades-list", "children"),
            Output("bt-stats", "children"),
            Output("bt-status", "children"),
        ],
        [Input("run-backtest-btn", "n_clicks")],
        [
            State("bt-symbol", "value"),
            State("bt-timeframe", "value"),
            State("bt-atr-period", "value"),
            State("bt-risk-reward", "value"),
            State("bt-register-patterns", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_backtest_callback(n_clicks, symbol, timeframe, atr_period, risk_reward, register):
        """Run backtest and display results."""
        
        if n_clicks is None:
            return no_update, no_update, no_update, no_update
        
        try:
            from src.dashboard.core.integration import run_backtest
            
            # Run backtest
            results = run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                atr_period=atr_period or 14,
                risk_reward=risk_reward or 2.0,
                register_patterns=register,
            )
            
            if results["status"] == "completed":
                metrics = results["metrics"]
                
                # Create equity chart
                eq_data = results.get("equity_curve", {})
                if eq_data:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=eq_data.get("timestamps", []),
                        y=eq_data.get("values", []),
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color=COLORS["success"], width=2),
                        fillcolor="rgba(34, 197, 94, 0.1)",
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor=COLORS["background"],
                        plot_bgcolor=COLORS["background"],
                        margin=dict(l=40, r=20, t=20, b=40),
                        showlegend=False,
                        xaxis={"showgrid": False},
                        yaxis={"title": "Equity ($)", "showgrid": True, "gridcolor": COLORS["border"]},
                    )
                else:
                    fig = _create_empty_equity_chart()
                
                # Create trades list
                trades = results.get("trades", [])
                if trades:
                    trades_content = html.Div([
                        html.Div(
                            className="d-flex justify-content-between py-1 border-bottom",
                            style={"borderColor": COLORS["border"]},
                            children=[
                                html.Span(t["entry_time"][:10], className="text-muted", style={"fontSize": "0.75rem"}),
                                html.Span(
                                    f"{t['return_pct']:.1f}%",
                                    style={
                                        "fontSize": "0.75rem",
                                        "fontWeight": "600",
                                        "color": COLORS["success"] if t["pnl"] > 0 else COLORS["danger"],
                                    },
                                ),
                            ],
                        ) for t in trades[-10:]  # Last 10 trades
                    ])
                else:
                    trades_content = html.P("No trades", className="text-muted text-center")
                
                # Create stats
                stats = html.Div([
                    _create_stat_row("Return", f"{metrics['return_pct']:.1f}%"),
                    _create_stat_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] else "N/A"),
                    _create_stat_row("Max Drawdown", f"{metrics['max_drawdown']:.1f}%"),
                    _create_stat_row("Win Rate", f"{metrics['win_rate']:.0f}%"),
                    _create_stat_row("# Trades", str(metrics["num_trades"])),
                    _create_stat_row("Exposure", f"{metrics['exposure_time']:.0f}%"),
                ])
                
                status = dbc.Alert(
                    [html.I(className="fas fa-check-circle me-2"), "Backtest completed!"],
                    color="success",
                )
                
                return fig, trades_content, stats, status
                
            else:
                status = dbc.Alert(
                    [html.I(className="fas fa-exclamation-triangle me-2"), results.get("error", "Error")],
                    color="danger",
                )
                return _create_empty_equity_chart(), no_update, no_update, status
                
        except Exception as e:
            logger.error(f"Backtest callback error: {e}")
            status = dbc.Alert(f"Error: {str(e)}", color="danger")
            return _create_empty_equity_chart(), no_update, no_update, status
