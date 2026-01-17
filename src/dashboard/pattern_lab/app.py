"""
Unified Trading Command Center
==============================
Single UI for the entire VRD 2.0 trading system.

Modules:
    1. Pattern Discovery - Pattern labeling and ML training
    2. VRD Forensics - Validation and statistical analysis
    3. Backtest Orchestrator - Run and analyze backtests
    4. ML Training - Brain training center
    5. Market Scanner - Real-time pattern detection

Usage:
    python src/dashboard/pattern_lab/app.py
    Then open: http://localhost:8050
"""

from pathlib import Path

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from loguru import logger

# Import modules
from src.dashboard.pattern_lab.layout import create_layout as create_pattern_discovery_layout, COLORS
from src.dashboard.pattern_lab.callbacks import register_callbacks as register_pattern_callbacks
from src.dashboard.pattern_lab.modules.vrd_forensics import create_vrd_forensics_tab, register_vrd_callbacks
from src.dashboard.pattern_lab.modules.backtest_orchestrator import create_backtest_tab, register_backtest_callbacks
from src.dashboard.pattern_lab.modules.market_scanner import create_scanner_tab, register_scanner_callbacks


# =============================================================================
# STYLING
# =============================================================================

EXTERNAL_STYLESHEETS = [
    dbc.themes.DARKLY,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
]

CUSTOM_CSS = """
/* Dark theme overrides for dropdowns */
.dash-dropdown .Select-control {
    background-color: #1E293B !important;
    border-color: #334155 !important;
}
.dash-dropdown .Select-menu-outer {
    background-color: #1E293B !important;
    border-color: #334155 !important;
}
.dash-dropdown .Select-option {
    background-color: #1E293B !important;
    color: #E2E8F0 !important;
}
.dash-dropdown .Select-option.is-focused {
    background-color: #334155 !important;
}
.dash-dropdown .Select-value-label,
.dash-dropdown .Select-placeholder {
    color: #E2E8F0 !important;
}

/* Slider styling */
.rc-slider-track { background-color: #0EA5E9 !important; }
.rc-slider-handle { border-color: #0EA5E9 !important; }
.rc-slider-rail { background-color: #334155 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1E293B; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #64748B; }

/* Pattern list item hover */
.pattern-list-item:hover {
    background-color: #334155 !important;
    border-color: #0EA5E9 !important;
}

/* Pulse animation for live badge */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.pulse-badge { animation: pulse 2s infinite; }

/* Hide Plotly logo */
.modebar-logo { display: none !important; }

/* Tabs styling */
.nav-tabs .nav-link {
    color: #94A3B8 !important;
    border: none !important;
    padding: 12px 20px !important;
}
.nav-tabs .nav-link.active {
    background-color: #1E293B !important;
    color: #0EA5E9 !important;
    border-bottom: 2px solid #0EA5E9 !important;
}
.nav-tabs .nav-link:hover {
    color: #E2E8F0 !important;
}
"""


# =============================================================================
# TAB LAYOUT
# =============================================================================

def create_unified_layout() -> dbc.Container:
    """Create unified tabbed layout."""
    
    return dbc.Container(
        fluid=True,
        style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "0"},
        children=[
            # Header
            dbc.Row(
                className="g-0",
                style={
                    "backgroundColor": "#1E293B",
                    "padding": "12px 24px",
                    "borderBottom": "1px solid #334155",
                },
                children=[
                    dbc.Col(width=4, children=[
                        html.H4(
                            [html.I(className="fas fa-brain me-2"), "Command Center"],
                            className="mb-0 text-light",
                            style={"fontWeight": "700"},
                        ),
                        html.Small("Unified VRD 2.0 Trading System", className="text-muted"),
                    ]),
                    dbc.Col(width=8, className="text-end", children=[
                        html.Div(id="system-status-header", className="d-inline-block"),
                    ]),
                ],
            ),
            
            # Tabs
            dbc.Tabs(
                id="main-tabs",
                active_tab="pattern-discovery",
                className="nav-tabs",
                style={"backgroundColor": "#1E293B", "borderBottom": "1px solid #334155"},
                children=[
                    dbc.Tab(
                        label="🎯 Pattern Discovery",
                        tab_id="pattern-discovery",
                        label_style={"padding": "12px 20px"},
                    ),
                    dbc.Tab(
                        label="🔬 VRD Forensics",
                        tab_id="vrd-forensics",
                        label_style={"padding": "12px 20px"},
                    ),
                    dbc.Tab(
                        label="📊 Backtest",
                        tab_id="backtest",
                        label_style={"padding": "12px 20px"},
                    ),
                    dbc.Tab(
                        label="🧠 ML Training",
                        tab_id="ml-training",
                        label_style={"padding": "12px 20px"},
                    ),
                    dbc.Tab(
                        label="📡 Scanner",
                        tab_id="scanner",
                        label_style={"padding": "12px 20px"},
                    ),
                ],
            ),
            
            # Tab Content
            html.Div(id="tab-content", style={"minHeight": "calc(100vh - 120px)"}),
            
            # Stores
            dcc.Store(id="patterns-store", data=[]),
            dcc.Store(id="selected-pattern-store", data=None),
            dcc.Store(id="similar-patterns-store", data=[]),
            dcc.Store(id="training-state-store", data={}),
            dcc.Interval(id="stats-interval", interval=60000, n_intervals=0),
        ],
    )


# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

def create_app() -> Dash:
    """Create and configure the unified command center."""
    
    app = Dash(
        __name__,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        suppress_callback_exceptions=True,
        title="Command Center | VRD 2.0",
        update_title="Loading...",
    )
    
    # Inject custom CSS
    app.index_string = f'''
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>{CUSTOM_CSS}</style>
        </head>
        <body style="background-color: #0F172A;">
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    '''
    
    # Set layout
    app.layout = create_unified_layout()
    
    # Register tab switching callback
    @app.callback(
        Output("tab-content", "children"),
        [Input("main-tabs", "active_tab")],
    )
    def render_tab_content(active_tab):
        if active_tab == "pattern-discovery":
            return create_pattern_discovery_layout()
        elif active_tab == "vrd-forensics":
            return create_vrd_forensics_tab()
        elif active_tab == "backtest":
            return create_backtest_tab()
        elif active_tab == "ml-training":
            # Reuse pattern discovery with training focus
            return create_pattern_discovery_layout()
        elif active_tab == "scanner":
            return create_scanner_tab()
        return html.Div("Select a tab")
    
    # Register all callbacks
    register_pattern_callbacks(app)
    register_vrd_callbacks(app)
    register_backtest_callbacks(app)
    register_scanner_callbacks(app)
    
    # System status callback
    @app.callback(
        Output("system-status-header", "children"),
        [Input("stats-interval", "n_intervals")],
    )
    def update_system_status(n):
        try:
            from src.dashboard.core.integration import get_system_status
            status = get_system_status()
            
            return html.Div([
                dbc.Badge(
                    f"📊 {status['patterns']} patterns",
                    color="primary",
                    className="me-2",
                ),
                dbc.Badge(
                    f"🏷️ {status['labeled']} labeled",
                    color="info",
                    className="me-2",
                ),
                dbc.Badge(
                    "🧠 Trained" if status['model'] == 'trained' else "⚠️ No model",
                    color="success" if status['model'] == 'trained' else "warning",
                    className="me-2",
                ),
                dbc.Badge(
                    "✅ Data OK" if status['data'] == 'ok' else "❌ No data",
                    color="success" if status['data'] == 'ok' else "danger",
                ),
            ])
        except:
            return html.Span("Status: Loading...", className="text-muted")
    
    logger.info("Unified Command Center initialized")
    
    return app


# Create global app instance
app = create_app()
server = app.server


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("🎯 UNIFIED TRADING COMMAND CENTER")
    print("=" * 60)
    print()
    print("  Complete VRD 2.0 System Interface")
    print()
    print("  Modules:")
    print("    1. Pattern Discovery - Labeling & Training")
    print("    2. VRD Forensics - Validation")
    print("    3. Backtest Orchestrator - Strategy Testing")
    print("    4. ML Training - Brain Center")
    print("    5. Market Scanner - Pattern Detection")
    print()
    print("  🌐 Open: http://localhost:8050")
    print("  🛑 Press Ctrl+C to stop")
    print()
    print("=" * 60)
    print()
    
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050,
    )
