"""
Sensitivity Analysis & Visualization
=====================================
VRD 2.0 Module 4: Parameter Sensitivity Analysis

Visualizes the "Stability Landscape" of parameter combinations:
- Heatmaps showing performance across parameter grid
- 3D surface plots for multi-dimensional analysis
- Identifies robust parameter regions vs brittle outliers
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class SensitivityVisualizer:
    """
    Visualize parameter sensitivity from grid search results.
    
    Queries experiments database and creates:
    - 2D heatmaps (SL vs TP, colored by metric)
    - 3D surface plots
    - Parameter distribution charts
    
    Usage:
        viz = SensitivityVisualizer()
        viz.plot_heatmap(metric='sharpe_ratio')
        viz.plot_3d_surface(metric='pnl_percent')
    """
    
    def __init__(self, db_path: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            db_path: Path to experiments database
            output_dir: Directory for output files
        """
        from src.reporting.storage import ExperimentLogger
        
        self.logger = ExperimentLogger(db_path)
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "results"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_grid_search_results(self, limit: int = 500) -> pd.DataFrame:
        """
        Get all grid search results from database.
        
        Returns:
            DataFrame with parameters and metrics
        """
        import sqlite3
        
        with sqlite3.connect(self.logger.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    run_id, timestamp, pnl_percent, sharpe_ratio, 
                    max_drawdown, win_rate, profit_factor, total_trades,
                    config_json
                FROM experiments
                WHERE strategy_name = 'grid_search'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = [dict(row) for row in cursor.fetchall()]
        
        if not rows:
            return pd.DataFrame()
        
        # Parse config JSON to get parameters
        data = []
        for row in rows:
            config = json.loads(row['config_json'])
            data.append({
                'run_id': row['run_id'],
                'atr_period': config.get('atr_period', 14),
                'stop_loss_atr': config.get('stop_loss_atr', 1.5),
                'take_profit_atr': config.get('take_profit_atr', 3.0),
                'min_validity_score': config.get('min_validity_score', 0.5),
                'pnl_percent': row['pnl_percent'] or 0,
                'sharpe_ratio': row['sharpe_ratio'] or 0,
                'max_drawdown': row['max_drawdown'] or 0,
                'win_rate': row['win_rate'] or 0,
                'profit_factor': row['profit_factor'] or 0,
                'total_trades': row['total_trades'] or 0,
            })
        
        return pd.DataFrame(data)
    
    def plot_heatmap(
        self,
        metric: str = 'sharpe_ratio',
        x_param: str = 'stop_loss_atr',
        y_param: str = 'take_profit_atr',
        atr_period: Optional[int] = None
    ) -> str:
        """
        Create heatmap of metric across parameter grid.
        
        Args:
            metric: Metric to visualize (sharpe_ratio, pnl_percent, etc.)
            x_param: Parameter for X axis
            y_param: Parameter for Y axis
            atr_period: Filter by specific ATR period (None = aggregate)
        
        Returns:
            Path to saved HTML file
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for visualization")
        
        df = self.get_grid_search_results()
        
        if df.empty:
            raise ValueError("No grid search results found in database")
        
        # Filter by ATR period if specified
        if atr_period is not None:
            df = df[df['atr_period'] == atr_period]
        
        # Get unique parameter values
        x_values = sorted(df[x_param].unique())
        y_values = sorted(df[y_param].unique())
        
        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index=y_param,
            columns=x_param,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure()
        
        # Color scale based on metric
        if metric in ('sharpe_ratio', 'pnl_percent', 'profit_factor'):
            colorscale = 'RdYlGn'  # Red-Yellow-Green (low-to-high is bad-to-good)
            zmid = 0 if metric in ('sharpe_ratio', 'pnl_percent') else 1
        else:
            colorscale = 'RdYlGn_r'  # Reversed for max_drawdown
            zmid = None
        
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=colorscale,
            zmid=zmid,
            text=np.round(pivot.values, 2),
            texttemplate='%{text}',
            textfont={'size': 12},
            hovertemplate=(
                f'{x_param}: %{{x}}<br>'
                f'{y_param}: %{{y}}<br>'
                f'{metric}: %{{z:.3f}}<extra></extra>'
            )
        ))
        
        # Layout
        title = f"Sensitivity Analysis: {metric.replace('_', ' ').title()}"
        if atr_period:
            title += f" (ATR Period: {atr_period})"
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title=x_param.replace('_', ' ').title(),
            yaxis_title=y_param.replace('_', ' ').title(),
            height=600,
            width=800,
            font=dict(family='system-ui, -apple-system, sans-serif'),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Save
        output_path = self.output_dir / f"sensitivity_{metric}.html"
        fig.write_html(str(output_path), include_plotlyjs='cdn')
        
        return str(output_path)
    
    def plot_3d_surface(
        self,
        metric: str = 'sharpe_ratio',
        atr_period: Optional[int] = None
    ) -> str:
        """
        Create 3D surface plot of metric across SL and TP.
        
        Args:
            metric: Metric for Z axis
            atr_period: Filter by ATR period
        
        Returns:
            Path to saved HTML file
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for visualization")
        
        df = self.get_grid_search_results()
        
        if df.empty:
            raise ValueError("No grid search results found")
        
        if atr_period:
            df = df[df['atr_period'] == atr_period]
        
        # Create pivot
        pivot = df.pivot_table(
            values=metric,
            index='take_profit_atr',
            columns='stop_loss_atr',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=[go.Surface(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale='RdYlGn',
            hovertemplate=(
                'Stop Loss ATR: %{x}<br>'
                'Take Profit ATR: %{y}<br>'
                f'{metric}: %{{z:.3f}}<extra></extra>'
            )
        )])
        
        title = f"3D Sensitivity: {metric.replace('_', ' ').title()}"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Stop Loss (ATR mult)',
                yaxis_title='Take Profit (ATR mult)',
                zaxis_title=metric.replace('_', ' ').title()
            ),
            height=700,
            width=900
        )
        
        output_path = self.output_dir / f"sensitivity_3d_{metric}.html"
        fig.write_html(str(output_path), include_plotlyjs='cdn')
        
        return str(output_path)
    
    def plot_parameter_distributions(self) -> str:
        """
        Create box plots showing metric distributions by parameter value.
        
        Returns:
            Path to saved HTML file
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required")
        
        df = self.get_grid_search_results()
        
        if df.empty:
            raise ValueError("No grid search results found")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sharpe by ATR Period',
                'Sharpe by Stop Loss ATR',
                'Sharpe by Take Profit ATR',
                'Sharpe by Validity Score'
            )
        )
        
        # ATR Period
        for atr in sorted(df['atr_period'].unique()):
            subset = df[df['atr_period'] == atr]
            fig.add_trace(
                go.Box(y=subset['sharpe_ratio'], name=str(atr), showlegend=False),
                row=1, col=1
            )
        
        # Stop Loss ATR
        for sl in sorted(df['stop_loss_atr'].unique()):
            subset = df[df['stop_loss_atr'] == sl]
            fig.add_trace(
                go.Box(y=subset['sharpe_ratio'], name=str(sl), showlegend=False),
                row=1, col=2
            )
        
        # Take Profit ATR
        for tp in sorted(df['take_profit_atr'].unique()):
            subset = df[df['take_profit_atr'] == tp]
            fig.add_trace(
                go.Box(y=subset['sharpe_ratio'], name=str(tp), showlegend=False),
                row=2, col=1
            )
        
        # Validity Score
        for val in sorted(df['min_validity_score'].unique()):
            subset = df[df['min_validity_score'] == val]
            fig.add_trace(
                go.Box(y=subset['sharpe_ratio'], name=str(val), showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Parameter Sensitivity Distributions',
            height=800,
            width=1000
        )
        
        output_path = self.output_dir / "sensitivity_distributions.html"
        fig.write_html(str(output_path), include_plotlyjs='cdn')
        
        return str(output_path)
    
    def get_optimal_parameters(
        self,
        metric: str = 'sharpe_ratio',
        min_trades: int = 50
    ) -> Dict[str, Any]:
        """
        Find optimal parameter combination.
        
        Args:
            metric: Metric to optimize
            min_trades: Minimum trades required
        
        Returns:
            Dictionary with optimal parameters
        """
        df = self.get_grid_search_results()
        
        if df.empty:
            return {}
        
        # Filter by minimum trades
        df = df[df['total_trades'] >= min_trades]
        
        if df.empty:
            return {}
        
        # Find best row
        best_idx = df[metric].idxmax() if metric != 'max_drawdown' else df[metric].idxmin()
        best_row = df.loc[best_idx]
        
        return {
            'atr_period': int(best_row['atr_period']),
            'stop_loss_atr': float(best_row['stop_loss_atr']),
            'take_profit_atr': float(best_row['take_profit_atr']),
            'min_validity_score': float(best_row['min_validity_score']),
            'metric_value': float(best_row[metric]),
            'total_trades': int(best_row['total_trades'])
        }
