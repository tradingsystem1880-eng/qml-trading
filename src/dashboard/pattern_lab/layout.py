"""
Neuro-Lab Research Engine - Layout Module
==========================================
ML-centric dashboard with pattern labeling, brain training, and similarity search.

This is NOT a visualization tool - it's an adaptive pattern recognition training interface.
Every interaction feeds the ML model.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from typing import List, Dict, Any


# =============================================================================
# STYLING CONSTANTS
# =============================================================================

COLORS = {
    "background": "#0F172A",
    "card_bg": "#1E293B",
    "card_bg_elevated": "#283548",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
    "border": "#334155",
    "primary": "#0EA5E9",
    "success": "#22C55E",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "cyan": "#22D3EE",
    "orange": "#FB923C",
    "purple": "#A855F7",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "16px",
    "marginBottom": "12px",
}


# =============================================================================
# HEADER - BRAIN STATUS BAR
# =============================================================================

def create_header() -> dbc.Row:
    """Create header with brain status and key metrics."""
    
    return dbc.Row(
        className="g-0 mb-3",
        style={
            "backgroundColor": COLORS["card_bg"],
            "padding": "16px 24px",
            "borderBottom": f"1px solid {COLORS['border']}",
        },
        children=[
            # Title
            dbc.Col(width=3, children=[
                html.H4(
                    [html.I(className="fas fa-brain me-2"), "Neuro-Lab"],
                    className="mb-0 text-light",
                    style={"fontWeight": "700"},
                ),
                html.Small("Adaptive Pattern Recognition Engine", className="text-muted"),
            ]),
            
            # Brain Status Indicators
            dbc.Col(width=6, children=[
                dbc.Row(className="g-3", children=[
                    dbc.Col(width=3, children=[
                        create_metric_badge("Total Patterns", "total-patterns-count", "fas fa-chart-line"),
                    ]),
                    dbc.Col(width=3, children=[
                        create_metric_badge("Labeled", "labeled-count", "fas fa-tags"),
                    ]),
                    dbc.Col(width=3, children=[
                        create_metric_badge("Win Rate", "win-rate", "fas fa-percentage"),
                    ]),
                    dbc.Col(width=3, children=[
                        create_metric_badge("Brain Ready", "brain-status", "fas fa-brain"),
                    ]),
                ]),
            ]),
            
            # Brain Training Button
            dbc.Col(width=3, className="text-end", children=[
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-sync me-2"), "Refresh"],
                        id="refresh-stats-btn",
                        color="secondary",
                        outline=True,
                        size="sm",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-graduation-cap me-2"), "Train Brain"],
                        id="train-brain-btn",
                        color="primary",
                        size="sm",
                    ),
                ]),
            ]),
        ],
    )


def create_metric_badge(label: str, element_id: str, icon: str) -> html.Div:
    """Create a small metric badge."""
    return html.Div([
        html.Div([
            html.I(className=f"{icon} text-muted me-1", style={"fontSize": "0.7rem"}),
            html.Span(label, className="text-muted", style={"fontSize": "0.7rem"}),
        ]),
        html.Div(id=element_id, children="--", style={"fontSize": "1.1rem", "fontWeight": "600", "color": COLORS["text"]}),
    ])


# =============================================================================
# LEFT PANEL - PATTERN SCANNER
# =============================================================================

def create_left_panel() -> dbc.Col:
    """Create left panel with pattern scanner and filters."""
    
    return dbc.Col(
        width=3,
        style={
            "height": "calc(100vh - 100px)",
            "overflowY": "auto",
            "padding": "16px",
            "backgroundColor": COLORS["card_bg"],
            "borderRight": f"1px solid {COLORS['border']}",
        },
        children=[
            # Scanner Header
            html.Div(
                className="d-flex justify-content-between align-items-center mb-3",
                children=[
                    html.H6(
                        [html.I(className="fas fa-radar me-2"), "Pattern Scanner"],
                        className="mb-0 text-light",
                    ),
                    dbc.Badge("LIVE", color="success", className="pulse-badge"),
                ],
            ),
            
            # Filters Card
            html.Div(style=CARD_STYLE, children=[
                # Symbol
                html.Label("Symbol", className="text-muted small"),
                dcc.Dropdown(
                    id="symbol-dropdown",
                    options=[
                        {"label": "BTC/USDT", "value": "BTC/USDT"},
                        {"label": "ETH/USDT", "value": "ETH/USDT"},
                    ],
                    value="BTC/USDT",
                    className="mb-2",
                ),
                
                # Timeframe
                html.Label("Timeframe", className="text-muted small"),
                dcc.Dropdown(
                    id="timeframe-dropdown",
                    options=[
                        {"label": "1h", "value": "1h"},
                        {"label": "4h", "value": "4h"},
                        {"label": "1d", "value": "1d"},
                    ],
                    value="4h",
                    className="mb-2",
                ),
                
                # Label Filter
                html.Label("Label Status", className="text-muted small"),
                dcc.Dropdown(
                    id="label-filter-dropdown",
                    options=[
                        {"label": "All Patterns", "value": "all"},
                        {"label": "🏷️ Unlabeled", "value": "unlabeled"},
                        {"label": "✅ Wins", "value": "win"},
                        {"label": "❌ Losses", "value": "loss"},
                    ],
                    value="all",
                    className="mb-3",
                ),
                
                # ML Confidence Filter
                html.Label("Min ML Confidence", className="text-muted small"),
                dcc.Slider(
                    id="confidence-slider",
                    min=0, max=100, step=5, value=0,
                    marks={0: "0%", 50: "50%", 100: "100%"},
                    tooltip={"placement": "bottom", "always_visible": False},
                    className="mb-2",
                ),
                
                # Scan Button
                dbc.Button(
                    [html.I(className="fas fa-search me-2"), "Scan Patterns"],
                    id="scan-patterns-btn",
                    color="primary",
                    className="w-100 mt-2",
                ),
            ]),
            
            # Pattern List
            html.Div(
                id="pattern-list-container",
                style={"maxHeight": "calc(100vh - 450px)", "overflowY": "auto"},
                children=[
                    html.P("Click 'Scan Patterns' to load", className="text-muted text-center py-4"),
                ],
            ),
        ],
    )


# =============================================================================
# CENTER PANEL - PATTERN LAB (FEATURE STUDIO)
# =============================================================================

def create_center_panel() -> dbc.Col:
    """Create center panel with pattern visualization and labeling."""
    
    return dbc.Col(
        width=6,
        style={
            "height": "calc(100vh - 100px)",
            "overflowY": "auto",
            "padding": "16px",
            "backgroundColor": COLORS["background"],
        },
        children=[
            # Chart Container
            dbc.Card(
                style={**CARD_STYLE, "padding": 0, "overflow": "hidden"},
                children=[
                    dcc.Loading(
                        type="circle",
                        children=[
                            dcc.Graph(
                                id="main-chart",
                                figure=create_empty_chart(),
                                style={"height": "400px"},
                                config={"displayModeBar": True, "displaylogo": False},
                            ),
                        ],
                    ),
                ],
            ),
            
            # Labeling Panel
            dbc.Card(
                style=CARD_STYLE,
                children=[
                    html.Div(
                        className="d-flex justify-content-between align-items-center mb-3",
                        children=[
                            html.H6(
                                [html.I(className="fas fa-tag me-2"), "Label This Pattern"],
                                className="mb-0 text-light",
                            ),
                            html.Span(id="pattern-id-display", className="text-muted small"),
                        ],
                    ),
                    
                    # Label Buttons
                    dbc.Row([
                        dbc.Col(width=4, children=[
                            dbc.Button(
                                [html.I(className="fas fa-check-circle me-2"), "WIN"],
                                id="label-win-btn",
                                color="success",
                                className="w-100",
                                disabled=True,
                            ),
                        ]),
                        dbc.Col(width=4, children=[
                            dbc.Button(
                                [html.I(className="fas fa-times-circle me-2"), "LOSS"],
                                id="label-loss-btn",
                                color="danger",
                                className="w-100",
                                disabled=True,
                            ),
                        ]),
                        dbc.Col(width=4, children=[
                            dbc.Button(
                                [html.I(className="fas fa-pause-circle me-2"), "SKIP"],
                                id="label-skip-btn",
                                color="secondary",
                                outline=True,
                                className="w-100",
                                disabled=True,
                            ),
                        ]),
                    ]),
                    
                    # Outcome Input
                    html.Div(
                        className="mt-3",
                        children=[
                            html.Label("Trade Outcome (optional)", className="text-muted small"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="outcome-input",
                                    type="number",
                                    placeholder="PnL %",
                                    style={"backgroundColor": COLORS["background"]},
                                ),
                                dbc.InputGroupText("%"),
                            ]),
                        ],
                    ),
                    
                    # Label Feedback
                    html.Div(id="label-feedback", className="mt-2"),
                ],
            ),
            
            # Feature Importance Card
            dbc.Card(
                style=CARD_STYLE,
                children=[
                    html.Div(
                        className="d-flex justify-content-between align-items-center mb-2",
                        children=[
                            html.H6(
                                [html.I(className="fas fa-chart-bar me-2"), "ML Confidence Drivers"],
                                className="mb-0 text-light",
                            ),
                            html.Span(id="ml-confidence-display", className="badge bg-primary"),
                        ],
                    ),
                    html.Div(id="feature-importance-display", children=[
                        html.P("Select a pattern to view feature importance", className="text-muted text-center py-3"),
                    ]),
                ],
            ),
        ],
    )


def create_empty_chart() -> dict:
    """Create empty placeholder chart."""
    return {
        "data": [],
        "layout": {
            "template": "plotly_dark",
            "paper_bgcolor": COLORS["background"],
            "plot_bgcolor": COLORS["background"],
            "font": {"color": COLORS["text"]},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": "Select a pattern from the scanner",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "showarrow": False,
                "font": {"size": 16, "color": COLORS["text_muted"]},
            }],
        },
    }


# =============================================================================
# RIGHT PANEL - SIMILARITY ENGINE & BRAIN STATUS
# =============================================================================

def create_right_panel() -> dbc.Col:
    """Create right panel with similar patterns and brain training."""
    
    return dbc.Col(
        width=3,
        style={
            "height": "calc(100vh - 100px)",
            "overflowY": "auto",
            "padding": "16px",
            "backgroundColor": COLORS["card_bg"],
            "borderLeft": f"1px solid {COLORS['border']}",
        },
        children=[
            # Pattern Details
            dbc.Card(
                style=CARD_STYLE,
                children=[
                    html.H6(
                        [html.I(className="fas fa-info-circle me-2"), "Pattern Details"],
                        className="mb-3 text-light",
                    ),
                    html.Div(id="pattern-details", children=[
                        html.P("Select a pattern", className="text-muted text-center"),
                    ]),
                ],
            ),
            
            # Similar Patterns (Similarity Engine)
            dbc.Card(
                style=CARD_STYLE,
                children=[
                    html.Div(
                        className="d-flex justify-content-between align-items-center mb-3",
                        children=[
                            html.H6(
                                [html.I(className="fas fa-project-diagram me-2"), "Similar Patterns"],
                                className="mb-0 text-light",
                            ),
                            dbc.Button(
                                html.I(className="fas fa-search"),
                                id="find-similar-btn",
                                color="secondary",
                                size="sm",
                                outline=True,
                            ),
                        ],
                    ),
                    html.Div(id="similar-patterns-list", children=[
                        html.P("Click search to find similar historical patterns", className="text-muted text-center small"),
                    ]),
                ],
            ),
            
            # Brain Training Status
            dbc.Card(
                style={**CARD_STYLE, "borderColor": COLORS["primary"]},
                children=[
                    html.H6(
                        [html.I(className="fas fa-brain me-2"), "Brain Training Center"],
                        className="mb-3 text-light",
                    ),
                    
                    # Training Status
                    html.Div(id="training-status", children=[
                        create_training_status_default(),
                    ]),
                    
                    # Training Button
                    dbc.Button(
                        [html.I(className="fas fa-play me-2"), "Start Training"],
                        id="start-training-btn",
                        color="primary",
                        className="w-100 mt-3",
                    ),
                    
                    # Training Log
                    html.Div(
                        id="training-log",
                        className="mt-2",
                        style={
                            "maxHeight": "150px",
                            "overflowY": "auto",
                            "fontSize": "0.75rem",
                        },
                    ),
                ],
            ),
        ],
    )


def create_training_status_default() -> html.Div:
    """Create default training status display."""
    return html.Div([
        create_status_row("Model Version", "v0.0", COLORS["text_muted"]),
        create_status_row("Training Samples", "0", COLORS["text_muted"]),
        create_status_row("Last Trained", "Never", COLORS["text_muted"]),
        create_status_row("Accuracy", "--", COLORS["text_muted"]),
    ])


def create_status_row(label: str, value: str, color: str) -> html.Div:
    """Create a status row."""
    return html.Div(
        className="d-flex justify-content-between mb-1",
        children=[
            html.Span(label, className="text-muted", style={"fontSize": "0.85rem"}),
            html.Span(value, style={"color": color, "fontSize": "0.85rem", "fontWeight": "500"}),
        ],
    )


# =============================================================================
# PATTERN LIST ITEMS
# =============================================================================

def create_pattern_list_item(pattern: Dict[str, Any]) -> html.Div:
    """Create a pattern list item for the scanner."""
    
    pattern_id = pattern.get("pattern_id", "")
    pattern_type = pattern.get("pattern_type", "")
    is_bullish = "bullish" in pattern_type.lower()
    
    # ML Confidence color
    ml_conf = pattern.get("ml_confidence")
    if ml_conf is not None:
        conf_pct = ml_conf * 100 if ml_conf <= 1 else ml_conf
        if conf_pct >= 70:
            conf_color = COLORS["success"]
        elif conf_pct >= 50:
            conf_color = COLORS["warning"]
        else:
            conf_color = COLORS["danger"]
        conf_text = f"{conf_pct:.0f}%"
    else:
        conf_color = COLORS["text_muted"]
        conf_text = "N/A"
    
    # Label status
    label = pattern.get("user_label")
    if label == "win":
        label_badge = dbc.Badge("WIN", color="success", className="ms-1")
    elif label == "loss":
        label_badge = dbc.Badge("LOSS", color="danger", className="ms-1")
    elif label == "ignore":
        label_badge = dbc.Badge("SKIP", color="secondary", className="ms-1")
    else:
        label_badge = dbc.Badge("UNLABELED", color="warning", className="ms-1")
    
    return html.Div(
        id={"type": "pattern-item", "index": pattern_id},
        className="pattern-list-item mb-2 p-2",
        style={
            "backgroundColor": COLORS["card_bg_elevated"],
            "borderRadius": "8px",
            "cursor": "pointer",
            "border": f"1px solid {COLORS['border']}",
            "transition": "all 0.2s",
        },
        children=[
            html.Div(
                className="d-flex justify-content-between align-items-center",
                children=[
                    html.Div([
                        html.Span(
                            "🟢" if is_bullish else "🔴",
                            className="me-2",
                        ),
                        html.Span(
                            pattern_id[:8] + "...",
                            style={"fontSize": "0.85rem", "color": COLORS["text"]},
                        ),
                        label_badge,
                    ]),
                    html.Span(
                        conf_text,
                        style={"color": conf_color, "fontWeight": "600", "fontSize": "0.85rem"},
                    ),
                ],
            ),
            html.Div(
                className="mt-1",
                style={"fontSize": "0.75rem", "color": COLORS["text_muted"]},
                children=[
                    f"{pattern.get('symbol', '')} · {pattern.get('timeframe', '')} · ",
                    format_timestamp_short(pattern.get("detection_time")),
                ],
            ),
        ],
    )


def format_timestamp_short(ts) -> str:
    """Format timestamp for short display."""
    if ts is None:
        return "N/A"
    try:
        from datetime import datetime
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = ts
        return dt.strftime("%m/%d %H:%M")
    except:
        return str(ts)[:10]


# =============================================================================
# PATTERN DETAILS COMPONENT
# =============================================================================

def create_pattern_details_panel(pattern: Dict[str, Any]) -> html.Div:
    """Create detailed pattern info panel."""
    
    features = pattern.get("features", {})
    
    return html.Div([
        # Type and Symbol
        create_detail_row("Type", pattern.get("pattern_type", "").replace("_", " ").title()),
        create_detail_row("Symbol", pattern.get("symbol", "N/A")),
        create_detail_row("Timeframe", pattern.get("timeframe", "N/A")),
        
        html.Hr(style={"borderColor": COLORS["border"], "margin": "8px 0"}),
        
        # Trade Levels
        html.Div(className="mb-2", style={"fontSize": "0.8rem"}, children=[
            html.Span("Trade Levels", className="text-muted d-block mb-1"),
            create_price_line("Entry", features.get("entry_price"), COLORS["primary"]),
            create_price_line("Stop", features.get("stop_loss_price"), COLORS["danger"]),
            create_price_line("Target", features.get("take_profit_price"), COLORS["success"]),
        ]),
        
        html.Hr(style={"borderColor": COLORS["border"], "margin": "8px 0"}),
        
        # Key Metrics
        create_detail_row("Validity", f"{pattern.get('validity_score', 0):.0%}" if pattern.get('validity_score') else "N/A"),
        create_detail_row("Regime", pattern.get("regime_at_detection") or "N/A"),
        create_detail_row("Features", f"{len(features)} extracted"),
    ])


def create_detail_row(label: str, value: str) -> html.Div:
    """Create a detail row."""
    return html.Div(
        className="d-flex justify-content-between mb-1",
        children=[
            html.Span(label, className="text-muted", style={"fontSize": "0.8rem"}),
            html.Span(str(value), className="text-light", style={"fontSize": "0.8rem", "fontWeight": "500"}),
        ],
    )


def create_price_line(label: str, price: float, color: str) -> html.Div:
    """Create a price display line."""
    return html.Div(
        className="d-flex justify-content-between",
        children=[
            html.Div([
                html.Span("●", style={"color": color, "marginRight": "4px", "fontSize": "0.6rem"}),
                html.Span(label, style={"fontSize": "0.75rem"}),
            ]),
            html.Span(
                f"${price:,.2f}" if price else "--",
                style={"fontSize": "0.75rem", "fontWeight": "500"},
            ),
        ],
    )


# =============================================================================
# SIMILAR PATTERNS COMPONENT
# =============================================================================

def create_similar_pattern_item(pattern: Dict[str, Any], similarity: float) -> html.Div:
    """Create a similar pattern display item."""
    
    label = pattern.get("user_label")
    outcome = pattern.get("trade_outcome")
    
    if label == "win":
        outcome_color = COLORS["success"]
        outcome_icon = "✅"
    elif label == "loss":
        outcome_color = COLORS["danger"]
        outcome_icon = "❌"
    else:
        outcome_color = COLORS["text_muted"]
        outcome_icon = "⏸️"
    
    return html.Div(
        className="mb-2 p-2",
        style={
            "backgroundColor": COLORS["background"],
            "borderRadius": "6px",
            "borderLeft": f"3px solid {outcome_color}",
        },
        children=[
            html.Div(
                className="d-flex justify-content-between",
                children=[
                    html.Span(
                        f"{outcome_icon} {pattern.get('pattern_id', '')[:8]}...",
                        style={"fontSize": "0.8rem"},
                    ),
                    html.Span(
                        f"{similarity:.0%} match",
                        className="text-muted",
                        style={"fontSize": "0.75rem"},
                    ),
                ],
            ),
            html.Div(
                style={"fontSize": "0.7rem", "color": COLORS["text_muted"]},
                children=[
                    f"Outcome: {outcome:+.1f}%" if outcome else "No outcome",
                ],
            ),
        ],
    )


# =============================================================================
# MAIN LAYOUT
# =============================================================================

def create_layout() -> dbc.Container:
    """Create the main Neuro-Lab layout."""
    
    return dbc.Container(
        fluid=True,
        style={
            "backgroundColor": COLORS["background"],
            "minHeight": "100vh",
            "padding": "0",
        },
        children=[
            # Data Stores
            dcc.Store(id="patterns-store", data=[]),
            dcc.Store(id="selected-pattern-store", data=None),
            dcc.Store(id="similar-patterns-store", data=[]),
            dcc.Store(id="training-state-store", data={}),
            
            # Interval for auto-refresh
            dcc.Interval(id="stats-interval", interval=60000, n_intervals=0),
            
            # Header
            create_header(),
            
            # Main Content
            dbc.Row(
                className="g-0",
                style={"minHeight": "calc(100vh - 100px)"},
                children=[
                    create_left_panel(),
                    create_center_panel(),
                    create_right_panel(),
                ],
            ),
        ],
    )
