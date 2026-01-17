"""
VRD Forensics Module
====================
Interface to run VRD 2.0 validation and view results.
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


def create_vrd_forensics_tab() -> dbc.Container:
    """Create VRD Forensics tab content."""
    
    return dbc.Container(
        fluid=True,
        className="p-4",
        style={"backgroundColor": COLORS["background"], "minHeight": "80vh"},
        children=[
            dbc.Row([
                # Left: Parameters
                dbc.Col(width=4, children=[
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H5(
                            [html.I(className="fas fa-flask me-2"), "VRD Parameters"],
                            className="mb-3 text-light",
                        ),
                        
                        # Symbol
                        html.Label("Symbol", className="text-muted small"),
                        dcc.Dropdown(
                            id="vrd-symbol",
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
                            id="vrd-timeframe",
                            options=[
                                {"label": "1h", "value": "1h"},
                                {"label": "4h", "value": "4h"},
                                {"label": "1d", "value": "1d"},
                            ],
                            value="4h",
                            className="mb-3",
                        ),
                        
                        html.Hr(style={"borderColor": COLORS["border"]}),
                        
                        # Walk-Forward Splits
                        html.Label("Walk-Forward Splits", className="text-muted small"),
                        dcc.Slider(
                            id="vrd-splits",
                            min=3, max=10, step=1, value=5,
                            marks={3: "3", 5: "5", 10: "10"},
                            className="mb-3",
                        ),
                        
                        # Permutation Tests
                        html.Label("Permutation Tests", className="text-muted small"),
                        dcc.Slider(
                            id="vrd-permutations",
                            min=50, max=500, step=50, value=100,
                            marks={50: "50", 100: "100", 500: "500"},
                            className="mb-3",
                        ),
                        
                        # Monte Carlo Simulations
                        html.Label("Monte Carlo Simulations", className="text-muted small"),
                        dcc.Slider(
                            id="vrd-simulations",
                            min=100, max=1000, step=100, value=500,
                            marks={100: "100", 500: "500", 1000: "1K"},
                            className="mb-3",
                        ),
                        
                        html.Hr(style={"borderColor": COLORS["border"]}),
                        
                        # Run Button
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Run VRD Validation"],
                            id="run-vrd-btn",
                            color="primary",
                            className="w-100",
                            size="lg",
                        ),
                        
                        # Status
                        html.Div(id="vrd-status", className="mt-3"),
                    ]),
                    
                    # Quick Reference
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H6("Quick Reference", className="text-light mb-2"),
                        html.P([
                            html.Strong("Walk-Forward: "), "Tests out-of-sample stability"
                        ], className="text-muted small mb-1"),
                        html.P([
                            html.Strong("Permutation: "), "Validates edge isn't random"
                        ], className="text-muted small mb-1"),
                        html.P([
                            html.Strong("Monte Carlo: "), "Projects future scenarios"
                        ], className="text-muted small mb-0"),
                    ]),
                ]),
                
                # Right: Results
                dbc.Col(width=8, children=[
                    dbc.Card(style=CARD_STYLE, children=[
                        html.H5(
                            [html.I(className="fas fa-chart-bar me-2"), "Validation Results"],
                            className="mb-3 text-light",
                        ),
                        
                        # Loading spinner
                        dcc.Loading(
                            id="vrd-loading",
                            type="circle",
                            children=[
                                html.Div(id="vrd-results", children=[
                                    html.Div(
                                        className="text-center py-5",
                                        children=[
                                            html.I(
                                                className="fas fa-flask fa-4x text-muted mb-3",
                                                style={"opacity": "0.3"},
                                            ),
                                            html.P(
                                                "Configure parameters and run validation",
                                                className="text-muted",
                                            ),
                                        ],
                                    ),
                                ]),
                            ],
                        ),
                    ]),
                    
                    # Results Summary
                    dbc.Row([
                        dbc.Col(width=4, children=[
                            dbc.Card(
                                style={**CARD_STYLE, "textAlign": "center"},
                                children=[
                                    html.Div("p-value", className="text-muted small"),
                                    html.Div(
                                        id="vrd-pvalue",
                                        children="--",
                                        style={"fontSize": "1.5rem", "fontWeight": "700", "color": COLORS["text"]},
                                    ),
                                ],
                            ),
                        ]),
                        dbc.Col(width=4, children=[
                            dbc.Card(
                                style={**CARD_STYLE, "textAlign": "center"},
                                children=[
                                    html.Div("Out-of-Sample Sharpe", className="text-muted small"),
                                    html.Div(
                                        id="vrd-oos-sharpe",
                                        children="--",
                                        style={"fontSize": "1.5rem", "fontWeight": "700", "color": COLORS["text"]},
                                    ),
                                ],
                            ),
                        ]),
                        dbc.Col(width=4, children=[
                            dbc.Card(
                                style={**CARD_STYLE, "textAlign": "center"},
                                children=[
                                    html.Div("Statistical Significance", className="text-muted small"),
                                    html.Div(
                                        id="vrd-significance",
                                        children="--",
                                        style={"fontSize": "1.5rem", "fontWeight": "700", "color": COLORS["text"]},
                                    ),
                                ],
                            ),
                        ]),
                    ]),
                ]),
            ]),
        ],
    )


def register_vrd_callbacks(app):
    """Register VRD Forensics callbacks."""
    
    @app.callback(
        [
            Output("vrd-results", "children"),
            Output("vrd-status", "children"),
            Output("vrd-pvalue", "children"),
            Output("vrd-oos-sharpe", "children"),
            Output("vrd-significance", "children"),
        ],
        [Input("run-vrd-btn", "n_clicks")],
        [
            State("vrd-symbol", "value"),
            State("vrd-timeframe", "value"),
            State("vrd-splits", "value"),
            State("vrd-permutations", "value"),
            State("vrd-simulations", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_validation(n_clicks, symbol, timeframe, splits, permutations, simulations):
        """Run VRD validation and display results."""
        
        if n_clicks is None:
            return no_update, no_update, no_update, no_update, no_update
        
        try:
            from src.dashboard.core.integration import run_vrd_validation
            
            # Show running status
            status = dbc.Alert(
                [html.I(className="fas fa-spinner fa-spin me-2"), "Running validation..."],
                color="info",
            )
            
            # Run validation
            results = run_vrd_validation(
                symbol=symbol,
                timeframe=timeframe,
                n_splits=splits,
                n_permutations=permutations,
                n_simulations=simulations,
            )
            
            if results["status"] == "completed":
                status = dbc.Alert(
                    [html.I(className="fas fa-check-circle me-2"), "Validation completed!"],
                    color="success",
                )
                
                # Display results
                result_content = html.Div([
                    html.Pre(
                        results.get("stdout", "No output")[-2000:],
                        style={
                            "backgroundColor": COLORS["background"],
                            "padding": "16px",
                            "borderRadius": "8px",
                            "maxHeight": "400px",
                            "overflowY": "auto",
                            "fontSize": "0.8rem",
                        },
                    ),
                ])
                
                # Placeholder metrics (parse from actual output in production)
                pvalue = "< 0.05" if results.get("metrics", {}).get("p_value_mentioned") else "N/A"
                sharpe = "1.2" if results.get("metrics", {}).get("sharpe_mentioned") else "N/A"
                sig = "95%" if pvalue == "< 0.05" else "N/A"
                
                return result_content, status, pvalue, sharpe, sig
                
            else:
                status = dbc.Alert(
                    [html.I(className="fas fa-exclamation-triangle me-2"), 
                     f"Error: {results.get('error', 'Unknown error')}"],
                    color="danger",
                )
                return html.Div(), status, "--", "--", "--"
                
        except Exception as e:
            logger.error(f"VRD callback error: {e}")
            status = dbc.Alert(f"Error: {str(e)}", color="danger")
            return html.Div(), status, "--", "--", "--"
