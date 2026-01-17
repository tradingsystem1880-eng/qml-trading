"""
Market Scanner Module
=====================
Scan for new patterns in latest market data.
"""

from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
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
    "warning": "#F59E0B",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "16px",
    "marginBottom": "12px",
}


def create_scanner_tab() -> dbc.Container:
    """Create Market Scanner tab content."""
    
    return dbc.Container(
        fluid=True,
        className="p-4",
        style={"backgroundColor": COLORS["background"], "minHeight": "80vh"},
        children=[
            dbc.Row([
                # Left: Scanner Controls
                dbc.Col(width=4, children=[
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H5(
                            [html.I(className="fas fa-radar me-2"), "Scanner Settings"],
                            className="mb-3 text-light",
                        ),
                        
                        # Symbol Selection
                        html.Label("Symbols", className="text-muted small"),
                        dcc.Dropdown(
                            id="scanner-symbols",
                            options=[
                                {"label": "BTC/USDT", "value": "BTC/USDT"},
                                {"label": "ETH/USDT", "value": "ETH/USDT"},
                            ],
                            value=["BTC/USDT"],
                            multi=True,
                            className="mb-3",
                        ),
                        
                        # Timeframes
                        html.Label("Timeframes", className="text-muted small"),
                        dcc.Dropdown(
                            id="scanner-timeframes",
                            options=[
                                {"label": "1h", "value": "1h"},
                                {"label": "4h", "value": "4h"},
                                {"label": "1d", "value": "1d"},
                            ],
                            value=["4h"],
                            multi=True,
                            className="mb-3",
                        ),
                        
                        # Lookback
                        html.Label("Lookback Bars", className="text-muted small"),
                        dbc.Input(
                            id="scanner-lookback",
                            type="number",
                            value=500,
                            min=100, max=2000,
                            className="mb-3",
                        ),
                        
                        # Register patterns
                        dbc.Checkbox(
                            id="scanner-register",
                            label="Register found patterns",
                            value=True,
                            className="mb-3 text-light",
                        ),
                        
                        html.Hr(style={"borderColor": COLORS["border"]}),
                        
                        # Scan Button
                        dbc.Button(
                            [html.I(className="fas fa-search me-2"), "Scan Now"],
                            id="scan-now-btn",
                            color="primary",
                            className="w-100",
                            size="lg",
                        ),
                        
                        # Status
                        html.Div(id="scanner-status", className="mt-3"),
                    ]),
                    
                    # Data Refresh
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H6("Data Management", className="text-light mb-3"),
                        dbc.Button(
                            [html.I(className="fas fa-sync me-2"), "Update Market Data"],
                            id="update-data-btn",
                            color="secondary",
                            outline=True,
                            className="w-100 mb-2",
                            size="sm",
                        ),
                        html.Div(id="data-update-status"),
                    ]),
                ]),
                
                # Right: Results
                dbc.Col(width=8, children=[
                    dbc.Card(style=CARD_STYLE, children=[
                        html.Div(
                            className="d-flex justify-content-between align-items-center mb-3",
                            children=[
                                html.H5(
                                    [html.I(className="fas fa-crosshairs me-2"), "Detected Patterns"],
                                    className="mb-0 text-light",
                                ),
                                html.Span(id="scanner-count", className="badge bg-primary"),
                            ],
                        ),
                        
                        dcc.Loading(
                            type="circle",
                            children=[
                                html.Div(
                                    id="scanner-results",
                                    style={"minHeight": "400px"},
                                    children=[
                                        html.Div(
                                            className="text-center py-5",
                                            children=[
                                                html.I(
                                                    className="fas fa-radar fa-4x text-muted mb-3",
                                                    style={"opacity": "0.3"},
                                                ),
                                                html.P(
                                                    "Configure settings and click 'Scan Now'",
                                                    className="text-muted",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ]),
                ]),
            ]),
        ],
    )


def register_scanner_callbacks(app):
    """Register Market Scanner callbacks."""
    
    @app.callback(
        [
            Output("scanner-results", "children"),
            Output("scanner-count", "children"),
            Output("scanner-status", "children"),
        ],
        [Input("scan-now-btn", "n_clicks")],
        [
            State("scanner-symbols", "value"),
            State("scanner-timeframes", "value"),
            State("scanner-lookback", "value"),
            State("scanner-register", "value"),
        ],
        prevent_initial_call=True,
    )
    def scan_market(n_clicks, symbols, timeframes, lookback, register):
        """Scan market for patterns."""
        
        if n_clicks is None:
            return no_update, no_update, no_update
        
        try:
            from src.dashboard.core.integration import scan_for_patterns
            
            all_patterns = []
            
            for symbol in (symbols or ["BTC/USDT"]):
                for timeframe in (timeframes or ["4h"]):
                    results = scan_for_patterns(
                        symbol=symbol,
                        timeframe=timeframe,
                        lookback_bars=lookback or 500,
                        register=register,
                    )
                    
                    if results["status"] == "completed":
                        for p in results.get("patterns", []):
                            p["symbol"] = symbol
                            p["timeframe"] = timeframe
                            all_patterns.append(p)
            
            if all_patterns:
                # Create pattern cards
                pattern_cards = []
                for p in all_patterns[-20:]:  # Last 20
                    is_bullish = "bullish" in p.get("pattern_type", "").lower()
                    
                    card = html.Div(
                        className="p-3 mb-2",
                        style={
                            "backgroundColor": COLORS["background"],
                            "borderRadius": "8px",
                            "borderLeft": f"4px solid {COLORS['success'] if is_bullish else COLORS['danger']}",
                        },
                        children=[
                            html.Div(
                                className="d-flex justify-content-between",
                                children=[
                                    html.Span(
                                        f"{p.get('symbol', '')} {p.get('timeframe', '')}",
                                        style={"fontWeight": "600", "color": COLORS["text"]},
                                    ),
                                    dbc.Badge(
                                        p.get("pattern_type", "").replace("_", " ").title(),
                                        color="success" if is_bullish else "danger",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="mt-1",
                                style={"fontSize": "0.8rem", "color": COLORS["text_muted"]},
                                children=[
                                    f"Entry: ${p.get('entry_price', 0):,.2f} | ",
                                    f"Features: {p.get('feature_count', 0)}",
                                ],
                            ),
                            html.Small(
                                p.get("detection_time", "")[:16],
                                className="text-muted",
                            ),
                        ],
                    )
                    pattern_cards.append(card)
                
                status = dbc.Alert(
                    [html.I(className="fas fa-check-circle me-2"), f"Found {len(all_patterns)} patterns!"],
                    color="success",
                )
                
                return html.Div(pattern_cards), f"{len(all_patterns)} found", status
                
            else:
                status = dbc.Alert("No patterns found in the scanned range", color="warning")
                return html.Div([html.P("No patterns detected", className="text-muted text-center")]), "0", status
                
        except Exception as e:
            logger.error(f"Scanner error: {e}")
            status = dbc.Alert(f"Error: {str(e)}", color="danger")
            return no_update, "Error", status
    
    
    @app.callback(
        Output("data-update-status", "children"),
        [Input("update-data-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def update_data(n_clicks):
        """Update market data."""
        if n_clicks is None:
            return no_update
        
        try:
            from src.dashboard.core.integration import update_market_data
            
            result = update_market_data()
            
            if result["status"] == "completed":
                return dbc.Alert("Data updated!", color="success", duration=3000)
            else:
                return dbc.Alert(f"Error: {result.get('error', 'Unknown')}", color="danger")
                
        except Exception as e:
            return dbc.Alert(f"Error: {str(e)}", color="danger")
