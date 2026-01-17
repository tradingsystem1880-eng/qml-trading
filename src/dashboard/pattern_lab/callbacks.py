"""
Neuro-Lab Research Engine - Callbacks Module
=============================================
ML-centric callbacks for pattern labeling, brain training, and similarity search.

Every interaction feeds the ML model. This is the adaptive brain.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, no_update, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from loguru import logger

# ML Infrastructure
from src.ml.pattern_registry import PatternRegistry
from src.ml.feature_extractor import PatternFeatureExtractor

# Pattern Visualization
from src.dashboard.components.pattern_viz import add_pattern_to_figure

# Layout Components
from src.dashboard.pattern_lab.layout import (
    COLORS,
    create_pattern_list_item,
    create_pattern_details_panel,
    create_similar_pattern_item,
    create_empty_chart,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = Path("results/experiments.db")
DATA_PATH = Path("data/processed")

# Initialize ML infrastructure
registry = PatternRegistry(str(DB_PATH))
feature_extractor = PatternFeatureExtractor()


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_ohlcv_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data from parquet files."""
    symbol_clean = symbol.replace("/", "")
    parquet_path = DATA_PATH / symbol_clean / f"{timeframe}_master.parquet"
    
    if not parquet_path.exists():
        logger.warning(f"Parquet file not found: {parquet_path}")
        return None
    
    try:
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.lower()
        
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            if df["time"].dt.tz is not None:
                df["time"] = df["time"].dt.tz_localize(None)
        
        return df
    except Exception as e:
        logger.error(f"Failed to load OHLCV data: {e}")
        return None


def create_pattern_chart(pattern: Dict[str, Any]) -> Optional[go.Figure]:
    """Create pattern visualization chart."""
    
    ohlcv_df = load_ohlcv_data(pattern["symbol"], pattern["timeframe"])
    if ohlcv_df is None:
        return None
    
    # Get detection time
    detection_time = pd.to_datetime(pattern["detection_time"])
    if detection_time.tz is not None:
        detection_time = detection_time.tz_localize(None)
    
    # Filter around detection
    mask = (ohlcv_df["time"] >= detection_time - pd.Timedelta(days=10)) & \
           (ohlcv_df["time"] <= detection_time + pd.Timedelta(days=5))
    display_df = ohlcv_df[mask].copy()
    
    if len(display_df) == 0:
        return None
    
    # Create candlestick chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=display_df["time"],
        open=display_df["open"],
        high=display_df["high"],
        low=display_df["low"],
        close=display_df["close"],
        name="Price",
        increasing_line_color="#22C55E",
        decreasing_line_color="#EF4444",
    ))
    
    # Add pattern overlay using pattern_viz
    fig = add_pattern_to_figure(fig, pattern, ohlcv_df)
    
    # Update layout
    pattern_type = pattern["pattern_type"].replace("_", " ").title()
    ml_conf = pattern.get("ml_confidence")
    conf_text = f" | ML: {ml_conf*100:.0f}%" if ml_conf else ""
    
    fig.update_layout(
        title=dict(
            text=f"<b>{pattern['symbol']}</b> {pattern_type}{conf_text}",
            x=0.5, xanchor="center",
            font=dict(size=14, color="#F1F5F9"),
        ),
        height=400,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        margin=dict(l=50, r=50, t=50, b=30),
    )
    
    return fig


# =============================================================================
# CALLBACKS
# =============================================================================

def register_callbacks(app):
    """Register all Neuro-Lab callbacks."""
    
    # =========================================================================
    # STATS & HEADER UPDATES
    # =========================================================================
    
    @app.callback(
        [
            Output("total-patterns-count", "children"),
            Output("labeled-count", "children"),
            Output("win-rate", "children"),
            Output("brain-status", "children"),
        ],
        [
            Input("stats-interval", "n_intervals"),
            Input("refresh-stats-btn", "n_clicks"),
        ],
    )
    def update_stats(n_intervals, n_clicks):
        """Update header statistics."""
        try:
            stats = registry.get_statistics()
            
            total = stats.get("total_patterns", 0)
            labeled = stats.get("labeled_patterns", 0)
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            
            # Calculate win rate
            if wins + losses > 0:
                win_rate = wins / (wins + losses) * 100
                win_rate_text = f"{win_rate:.0f}%"
            else:
                win_rate_text = "N/A"
            
            # Brain status
            if labeled >= 30:
                brain_status = "✅ Ready"
            elif labeled >= 10:
                brain_status = f"🟡 {30 - labeled} more"
            else:
                brain_status = "❌ Need data"
            
            return str(total), str(labeled), win_rate_text, brain_status
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            return "--", "--", "--", "Error"
    
    
    # =========================================================================
    # PATTERN SCANNER
    # =========================================================================
    
    @app.callback(
        [
            Output("patterns-store", "data"),
            Output("pattern-list-container", "children"),
        ],
        [Input("scan-patterns-btn", "n_clicks")],
        [
            State("symbol-dropdown", "value"),
            State("timeframe-dropdown", "value"),
            State("label-filter-dropdown", "value"),
            State("confidence-slider", "value"),
        ],
        prevent_initial_call=True,
    )
    def scan_patterns(n_clicks, symbol, timeframe, label_filter, min_confidence):
        """Scan and load patterns from registry."""
        
        if n_clicks is None:
            raise PreventUpdate
        
        try:
            # Determine label filter
            label = None if label_filter == "all" else (None if label_filter == "unlabeled" else label_filter)
            
            # Get patterns
            if label_filter == "unlabeled":
                patterns = registry.get_unlabeled_patterns(limit=50, symbol=symbol)
            else:
                patterns = registry.get_patterns(
                    symbol=symbol,
                    timeframe=timeframe,
                    label=label,
                    limit=50,
                )
            
            # Filter by confidence if specified
            if min_confidence > 0:
                patterns = [
                    p for p in patterns 
                    if p.get("ml_confidence") and p["ml_confidence"] * 100 >= min_confidence
                ]
            
            if not patterns:
                return [], html.P("No patterns found", className="text-muted text-center py-4")
            
            # Create list items
            list_items = [create_pattern_list_item(p) for p in patterns]
            
            # Store patterns data
            patterns_data = [
                {
                    "pattern_id": p["pattern_id"],
                    "symbol": p["symbol"],
                    "timeframe": p["timeframe"],
                    "pattern_type": p["pattern_type"],
                    "detection_time": p.get("detection_time"),
                    "ml_confidence": p.get("ml_confidence"),
                    "validity_score": p.get("validity_score"),
                    "user_label": p.get("user_label"),
                    "trade_outcome": p.get("trade_outcome"),
                    "features": p.get("features", {}),
                    "regime_at_detection": p.get("regime_at_detection"),
                }
                for p in patterns
            ]
            
            logger.info(f"Scanned {len(patterns)} patterns")
            
            return patterns_data, list_items
            
        except Exception as e:
            logger.error(f"Failed to scan patterns: {e}")
            return [], html.P(f"Error: {str(e)}", className="text-danger text-center py-4")
    
    
    # =========================================================================
    # PATTERN SELECTION & DISPLAY
    # =========================================================================
    
    @app.callback(
        [
            Output("selected-pattern-store", "data"),
            Output("main-chart", "figure"),
            Output("pattern-id-display", "children"),
            Output("pattern-details", "children"),
            Output("ml-confidence-display", "children"),
            Output("feature-importance-display", "children"),
            Output("label-win-btn", "disabled"),
            Output("label-loss-btn", "disabled"),
            Output("label-skip-btn", "disabled"),
        ],
        [Input({"type": "pattern-item", "index": ALL}, "n_clicks")],
        [State("patterns-store", "data")],
    )
    def select_pattern(n_clicks_list, patterns_data):
        """Handle pattern selection from list."""
        
        if not ctx.triggered or all(nc is None for nc in n_clicks_list):
            raise PreventUpdate
        
        # Get clicked pattern ID
        triggered_id = ctx.triggered_id
        if triggered_id is None:
            raise PreventUpdate
        
        pattern_id = triggered_id["index"]
        
        # Find pattern in store
        pattern = next((p for p in patterns_data if p["pattern_id"] == pattern_id), None)
        if pattern is None:
            raise PreventUpdate
        
        try:
            # Create chart
            fig = create_pattern_chart(pattern)
            if fig is None:
                fig = create_empty_chart()
            
            # Pattern details
            details = create_pattern_details_panel(pattern)
            
            # ML Confidence display
            ml_conf = pattern.get("ml_confidence")
            if ml_conf:
                conf_text = f"{ml_conf*100:.0f}% Confidence"
            else:
                conf_text = "No ML Score"
            
            # Feature importance (top 5 features)
            features = pattern.get("features", {})
            if features:
                # Sort by absolute value and take top 5
                sorted_features = sorted(
                    [(k, v) for k, v in features.items() if isinstance(v, (int, float)) and not k.startswith("p")],
                    key=lambda x: abs(x[1]) if x[1] else 0,
                    reverse=True
                )[:5]
                
                feature_display = html.Div([
                    html.Div(
                        className="d-flex justify-content-between mb-1",
                        children=[
                            html.Span(k.replace("_", " ")[:20], style={"fontSize": "0.75rem"}),
                            html.Span(f"{v:.2f}" if isinstance(v, float) else str(v), 
                                     style={"fontSize": "0.75rem", "fontWeight": "500"}),
                        ],
                    ) for k, v in sorted_features
                ])
            else:
                feature_display = html.P("No features extracted", className="text-muted text-center small")
            
            return (
                pattern,
                fig,
                pattern_id[:12] + "...",
                details,
                conf_text,
                feature_display,
                False, False, False,  # Enable labeling buttons
            )
            
        except Exception as e:
            logger.error(f"Failed to display pattern: {e}\n{traceback.format_exc()}")
            return (
                pattern,
                create_empty_chart(),
                "Error",
                html.P(str(e), className="text-danger"),
                "Error",
                html.P("Error loading features", className="text-danger"),
                True, True, True,
            )
    
    
    # =========================================================================
    # PATTERN LABELING
    # =========================================================================
    
    @app.callback(
        [
            Output("label-feedback", "children"),
            Output("patterns-store", "data", allow_duplicate=True),
        ],
        [
            Input("label-win-btn", "n_clicks"),
            Input("label-loss-btn", "n_clicks"),
            Input("label-skip-btn", "n_clicks"),
        ],
        [
            State("selected-pattern-store", "data"),
            State("outcome-input", "value"),
            State("patterns-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def label_pattern(win_clicks, loss_clicks, skip_clicks, pattern, outcome, patterns_data):
        """Label the selected pattern."""
        
        if pattern is None:
            raise PreventUpdate
        
        # Determine which button was clicked
        triggered = ctx.triggered_id
        if triggered == "label-win-btn":
            label = "win"
        elif triggered == "label-loss-btn":
            label = "loss"
        elif triggered == "label-skip-btn":
            label = "ignore"
        else:
            raise PreventUpdate
        
        pattern_id = pattern["pattern_id"]
        
        try:
            # Update in registry
            success = registry.label_pattern(
                pattern_id=pattern_id,
                label=label,
                outcome=float(outcome) if outcome else None,
            )
            
            if success:
                # Update local store
                for p in patterns_data:
                    if p["pattern_id"] == pattern_id:
                        p["user_label"] = label
                        if outcome:
                            p["trade_outcome"] = float(outcome)
                        break
                
                feedback = dbc.Alert(
                    [
                        html.I(className="fas fa-check-circle me-2"),
                        f"Labeled as {label.upper()}",
                    ],
                    color="success" if label == "win" else "danger" if label == "loss" else "secondary",
                    dismissable=True,
                    is_open=True,
                )
                
                logger.info(f"Labeled pattern {pattern_id[:8]} as {label}")
                
                return feedback, patterns_data
            else:
                return dbc.Alert("Failed to save label", color="danger"), no_update
                
        except Exception as e:
            logger.error(f"Failed to label pattern: {e}")
            return dbc.Alert(f"Error: {str(e)}", color="danger"), no_update
    
    
    # =========================================================================
    # SIMILARITY SEARCH
    # =========================================================================
    
    @app.callback(
        Output("similar-patterns-list", "children"),
        [Input("find-similar-btn", "n_clicks")],
        [State("selected-pattern-store", "data")],
        prevent_initial_call=True,
    )
    def find_similar_patterns(n_clicks, pattern):
        """Find similar historical patterns."""
        
        if n_clicks is None or pattern is None:
            raise PreventUpdate
        
        try:
            features = pattern.get("features", {})
            if not features:
                return html.P("No features to compare", className="text-muted text-center small")
            
            # Find similar patterns
            similar = registry.find_similar_patterns(
                current_features=features,
                n=5,
                label_filter=["win", "loss"],  # Only labeled patterns
            )
            
            if not similar:
                return html.P("No similar patterns found", className="text-muted text-center small")
            
            # Create display items
            items = [
                create_similar_pattern_item(s["pattern"], s["similarity"])
                for s in similar
            ]
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}")
            return html.P(f"Error: {str(e)}", className="text-danger text-center small")
    
    
    # =========================================================================
    # BRAIN TRAINING
    # =========================================================================
    
    @app.callback(
        [
            Output("training-status", "children"),
            Output("training-log", "children"),
        ],
        [Input("start-training-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def train_brain(n_clicks):
        """Trigger ML model training."""
        
        if n_clicks is None:
            raise PreventUpdate
        
        logs = []
        
        try:
            logs.append(html.Div("📊 Fetching training data...", className="mb-1"))
            
            # Get training data
            try:
                X, y = registry.get_training_data(min_labels=10)
                feature_names = list(X.columns)
                logs.append(html.Div(f"✅ Got {len(y)} samples, {len(feature_names)} features", className="mb-1 text-success"))
            except ValueError as e:
                logs.append(html.Div(f"❌ {str(e)}", className="mb-1 text-danger"))
                status = html.Div([
                    html.Div(className="d-flex justify-content-between mb-1", children=[
                        html.Span("Status", className="text-muted", style={"fontSize": "0.85rem"}),
                        html.Span("Need more labels", style={"color": COLORS["warning"], "fontSize": "0.85rem"}),
                    ]),
                ])
                return status, logs
            
            logs.append(html.Div("🧠 Training XGBoost model...", className="mb-1"))
            
            # Try to import and train XGBoost
            try:
                from xgboost import XGBClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                
                logs.append(html.Div(f"✅ Model trained!", className="mb-1 text-success"))
                logs.append(html.Div(f"   Accuracy: {accuracy:.1%}", className="mb-1"))
                logs.append(html.Div(f"   Precision: {precision:.1%}", className="mb-1"))
                logs.append(html.Div(f"   Recall: {recall:.1%}", className="mb-1"))
                
                # Save model (simplified - in production use joblib)
                model_path = Path("results/ml_models/pattern_classifier.json")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(str(model_path))
                
                logs.append(html.Div(f"💾 Model saved to {model_path.name}", className="mb-1"))
                
                # Update status
                status = html.Div([
                    html.Div(className="d-flex justify-content-between mb-1", children=[
                        html.Span("Model Version", className="text-muted", style={"fontSize": "0.85rem"}),
                        html.Span("v1.0", style={"color": COLORS["success"], "fontSize": "0.85rem", "fontWeight": "500"}),
                    ]),
                    html.Div(className="d-flex justify-content-between mb-1", children=[
                        html.Span("Training Samples", className="text-muted", style={"fontSize": "0.85rem"}),
                        html.Span(str(len(y)), style={"fontSize": "0.85rem", "fontWeight": "500"}),
                    ]),
                    html.Div(className="d-flex justify-content-between mb-1", children=[
                        html.Span("Last Trained", className="text-muted", style={"fontSize": "0.85rem"}),
                        html.Span(datetime.now().strftime("%H:%M"), style={"fontSize": "0.85rem", "fontWeight": "500"}),
                    ]),
                    html.Div(className="d-flex justify-content-between mb-1", children=[
                        html.Span("Accuracy", className="text-muted", style={"fontSize": "0.85rem"}),
                        html.Span(f"{accuracy:.1%}", style={"color": COLORS["success"], "fontSize": "0.85rem", "fontWeight": "500"}),
                    ]),
                ])
                
                return status, logs
                
            except ImportError:
                logs.append(html.Div("❌ XGBoost not installed", className="mb-1 text-danger"))
                logs.append(html.Div("   Run: pip install xgboost scikit-learn", className="mb-1 text-muted"))
                return no_update, logs
                
        except Exception as e:
            logs.append(html.Div(f"❌ Error: {str(e)}", className="mb-1 text-danger"))
            logger.error(f"Training failed: {e}\n{traceback.format_exc()}")
            return no_update, logs
