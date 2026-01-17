"""
Dossier Generator - Strategy Autopsy Reports
=============================================
Generates standalone HTML reports for each backtest run.

Each dossier contains:
- Header with strategy name, run ID, date
- Config card with all parameters
- Large metrics display (P&L, Win Rate, Profit Factor)
- Interactive Plotly charts (Equity Curve, Drawdown)
- Searchable trade log table

The HTML is self-contained (Plotly embedded as JSON) for portability.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict, is_dataclass

import pandas as pd

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class DossierGenerator:
    """
    Generate professional HTML reports for backtest runs.
    
    Each report is a standalone HTML file with embedded charts
    and full configuration details for reproducibility.
    
    Usage:
        generator = DossierGenerator()
        report_path = generator.generate_html(
            run_id='abc123',
            config=config,
            results=results,
            trades_df=trades_df
        )
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize dossier generator.
        
        Args:
            output_dir: Base directory for reports. Defaults to results/
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "results"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _serialize_config(self, config: Any) -> Dict[str, Any]:
        """Convert config to dictionary."""
        if is_dataclass(config):
            return asdict(config)
        elif hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        elif isinstance(config, dict):
            return config
        else:
            return {'raw': str(config)}
    
    def _create_equity_chart(self, equity_curve: List) -> str:
        """
        Create interactive equity curve chart.
        
        Returns:
            Plotly chart as HTML div string
        """
        if not HAS_PLOTLY or not equity_curve:
            return "<p>Plotly not available or no data</p>"
        
        times = [e[0] for e in equity_curve]
        values = [e[1] for e in equity_curve]
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Equity Curve', 'Drawdown')
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=times, y=values,
                mode='lines',
                name='Equity',
                line=dict(color='#2E86AB', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.1)'
            ),
            row=1, col=1
        )
        
        # Calculate drawdown
        import numpy as np
        values_arr = np.array(values)
        running_max = np.maximum.accumulate(values_arr)
        drawdown = (running_max - values_arr) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=times, y=drawdown,
                mode='lines',
                name='Drawdown %',
                line=dict(color='#E74C3C', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ),
            row=2, col=1
        )
        
        # Layout
        fig.update_layout(
            height=500,
            showlegend=False,
            margin=dict(l=50, r=30, t=50, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,249,250,1)',
            font=dict(family='system-ui, -apple-system, sans-serif')
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="DD %", row=2, col=1)
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def _create_trades_table(self, trades_df: pd.DataFrame, limit: int = 50) -> str:
        """
        Create HTML table of trades.
        
        Args:
            trades_df: DataFrame of trades
            limit: Max trades to show
        
        Returns:
            HTML table string
        """
        if trades_df is None or trades_df.empty:
            return "<p>No trades to display</p>"
        
        # Select and format columns
        display_cols = [
            'entry_time', 'exit_time', 'side', 'entry_price', 'exit_price',
            'pnl_pct', 'result', 'pattern_type'
        ]
        
        available_cols = [c for c in display_cols if c in trades_df.columns]
        df = trades_df[available_cols].head(limit).copy()
        
        # Format columns
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        if 'pnl_pct' in df.columns:
            df['pnl_pct'] = df['pnl_pct'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
        if 'entry_price' in df.columns:
            df['entry_price'] = df['entry_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        if 'exit_price' in df.columns:
            df['exit_price'] = df['exit_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        
        # Generate HTML
        html = '<table class="trades-table">\n<thead>\n<tr>'
        
        col_names = {
            'entry_time': 'Entry Time',
            'exit_time': 'Exit Time',
            'side': 'Side',
            'entry_price': 'Entry',
            'exit_price': 'Exit',
            'pnl_pct': 'P&L',
            'result': 'Result',
            'pattern_type': 'Pattern'
        }
        
        for col in available_cols:
            html += f'<th>{col_names.get(col, col)}</th>'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        for _, row in df.iterrows():
            result = row.get('result', '')
            row_class = 'win' if result == 'WIN' else 'loss' if result == 'LOSS' else ''
            html += f'<tr class="{row_class}">'
            for col in available_cols:
                val = row[col] if pd.notna(row[col]) else ''
                html += f'<td>{val}</td>'
            html += '</tr>\n'
        
        html += '</tbody>\n</table>'
        
        if len(trades_df) > limit:
            html += f'<p class="note">Showing {limit} of {len(trades_df)} trades</p>'
        
        return html
    
    def _format_config_table(self, config_dict: Dict[str, Any]) -> str:
        """Format config as HTML table."""
        html = '<table class="config-table">\n'
        
        for key, value in config_dict.items():
            if key.startswith('_'):
                continue
            
            # Format value
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            elif isinstance(value, datetime):
                formatted = value.strftime('%Y-%m-%d %H:%M')
            elif value is None:
                formatted = '<em>None</em>'
            else:
                formatted = str(value)
            
            # Format key
            key_display = key.replace('_', ' ').title()
            
            html += f'<tr><td class="key">{key_display}</td><td class="value">{formatted}</td></tr>\n'
        
        html += '</table>'
        return html
    
    def _get_metric_color(self, value: float, metric: str) -> str:
        """Get color based on metric value."""
        if metric in ('pnl', 'pnl_pct', 'sharpe', 'sortino', 'profit_factor'):
            return '#27ae60' if value > 0 else '#e74c3c'
        elif metric == 'win_rate':
            return '#27ae60' if value >= 50 else '#e74c3c'
        elif metric == 'max_dd':
            return '#e74c3c' if value > 20 else '#f39c12' if value > 10 else '#27ae60'
        return '#333'
    
    def _create_validation_section(self, validation_suite: Optional[Any]) -> str:
        """
        Create HTML section for validation results.
        
        Args:
            validation_suite: ValidationSuite object with results
        
        Returns:
            HTML string for validation section
        """
        if validation_suite is None:
            return ""
        
        # Import here to avoid circular imports
        try:
            from src.validation.base import ValidationStatus
        except ImportError:
            return ""
        
        status_icons = {
            'pass': '✅',
            'warn': '⚠️',
            'fail': '❌',
            'error': '💥'
        }
        
        # Overall status
        overall_status = validation_suite.overall_status.value if hasattr(validation_suite.overall_status, 'value') else str(validation_suite.overall_status)
        overall_icon = status_icons.get(overall_status, '❓')
        
        cards_html = ""
        for result in validation_suite.results:
            status = result.status.value if hasattr(result.status, 'value') else str(result.status)
            icon = status_icons.get(status, '❓')
            
            # Format metrics
            metrics_html = ""
            for key, val in result.metrics.items():
                if isinstance(val, float):
                    metrics_html += f"<span><strong>{key}:</strong> {val:.4f}</span>"
                elif isinstance(val, list):
                    metrics_html += f"<span><strong>{key}:</strong> {val}</span>"
                else:
                    metrics_html += f"<span><strong>{key}:</strong> {val}</span>"
            
            p_value_str = f"<span><strong>p-value:</strong> {result.p_value:.4f}</span>" if result.p_value is not None else ""
            
            cards_html += f'''
            <div class="validation-card {status}">
                <h3>{icon} {result.validator_name.upper().replace('_', ' ')}</h3>
                <div class="interpretation">{result.interpretation}</div>
                <div class="metrics">
                    {p_value_str}
                    {metrics_html}
                </div>
            </div>
            '''
        
        return f'''
        <!-- Validation Results -->
        <div class="card">
            <h2>🔬 Statistical Validation ({overall_icon} {overall_status.upper()})</h2>
            <div class="validation-grid">
                {cards_html}
            </div>
        </div>
        '''
    
    def generate_html(
        self,
        run_id: str,
        config: Any,
        results: Dict[str, Any],
        trades_df: Optional[pd.DataFrame] = None,
        strategy_name: Optional[str] = None,
        validation_suite: Optional[Any] = None
    ) -> str:
        """
        Generate standalone HTML dossier.
        
        Args:
            run_id: Unique run identifier
            config: Configuration (dataclass or dict)
            results: Results dictionary from BacktestEngine
            trades_df: DataFrame of trades
            strategy_name: Strategy name override
            validation_suite: Optional ValidationSuite with test results
        
        Returns:
            Path to generated HTML file
        """
        config_dict = self._serialize_config(config)
        strategy = strategy_name or config_dict.get('detector_method', 'QML_ATR')
        timestamp = datetime.now()
        
        # Create strategy-specific output directory
        strategy_dir = self.output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metrics
        pnl_pct = results.get('net_profit_pct', 0)
        pnl_usd = results.get('net_profit', 0)
        win_rate = results.get('win_rate', 0)
        profit_factor = results.get('profit_factor', 0)
        sharpe = results.get('sharpe_ratio', 0)
        max_dd = results.get('max_drawdown', 0)
        total_trades = results.get('total_trades', 0)
        winning_trades = results.get('winning_trades', 0)
        losing_trades = results.get('losing_trades', 0)
        avg_win = results.get('avg_win', 0)
        avg_loss = results.get('avg_loss', 0)
        
        # Generate charts
        equity_chart = self._create_equity_chart(results.get('equity_curve', []))
        
        # Generate trades table
        trades_table = self._create_trades_table(trades_df)
        
        # Generate config table
        config_table = self._format_config_table(config_dict)
        
        # Generate validation section if available
        validation_html = self._create_validation_section(validation_suite)
        
        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategy Dossier: {run_id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8f9fa;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        
        .header .meta {{
            opacity: 0.8;
            font-size: 14px;
        }}
        
        .header .run-id {{
            font-family: monospace;
            background: rgba(255,255,255,0.1);
            padding: 2px 8px;
            border-radius: 4px;
        }}
        
        /* Cards */
        .card {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .card h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #f0f0f0;
        }}
        
        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
        }}
        
        .metric {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        
        .metric .value {{
            font-size: 32px;
            font-weight: 700;
            line-height: 1.2;
        }}
        
        .metric .label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
            margin-top: 4px;
        }}
        
        .metric.positive .value {{ color: #27ae60; }}
        .metric.negative .value {{ color: #e74c3c; }}
        .metric.neutral .value {{ color: #333; }}
        
        /* Config Table */
        .config-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .config-table tr {{
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .config-table td {{
            padding: 10px 0;
        }}
        
        .config-table .key {{
            font-weight: 500;
            color: #666;
            width: 40%;
        }}
        
        .config-table .value {{
            font-family: monospace;
            color: #333;
        }}
        
        /* Trades Table */
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        
        .trades-table th {{
            background: #f8f9fa;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .trades-table td {{
            padding: 10px 8px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .trades-table tr.win td {{ background: rgba(39, 174, 96, 0.05); }}
        .trades-table tr.loss td {{ background: rgba(231, 76, 60, 0.05); }}
        
        .trades-table tr:hover td {{
            background: rgba(0,0,0,0.02);
        }}
        
        /* Summary Grid */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }}
        
        .summary-item {{
            text-align: center;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .summary-item .num {{
            font-size: 24px;
            font-weight: 700;
        }}
        
        .summary-item .lbl {{
            font-size: 11px;
            text-transform: uppercase;
            color: #666;
        }}
        
        /* Notes */
        .note {{
            font-size: 12px;
            color: #888;
            margin-top: 12px;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 12px;
        }}
        
        /* Chart container */
        .chart-container {{
            margin: 0 -24px;
        }}
        
        /* Validation section */
        .validation-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }}
        
        .validation-card {{
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid #ddd;
        }}
        
        .validation-card.pass {{
            background: rgba(39, 174, 96, 0.05);
            border-left-color: #27ae60;
        }}
        
        .validation-card.warn {{
            background: rgba(243, 156, 18, 0.05);
            border-left-color: #f39c12;
        }}
        
        .validation-card.fail {{
            background: rgba(231, 76, 60, 0.05);
            border-left-color: #e74c3c;
        }}
        
        .validation-card h3 {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .validation-card .interpretation {{
            font-size: 13px;
            color: #555;
            margin-bottom: 8px;
        }}
        
        .validation-card .metrics {{
            font-size: 12px;
            color: #888;
        }}
        
        .validation-card .metrics span {{
            display: inline-block;
            margin-right: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 Strategy Dossier</h1>
            <div class="meta">
                <strong>{strategy.upper()}</strong> &nbsp;|&nbsp;
                Run ID: <span class="run-id">{run_id}</span> &nbsp;|&nbsp;
                {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <!-- Key Metrics -->
        <div class="card">
            <h2>📈 Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric {'positive' if pnl_pct > 0 else 'negative' if pnl_pct < 0 else 'neutral'}">
                    <div class="value">{pnl_pct:+.2f}%</div>
                    <div class="label">Net P&L</div>
                </div>
                <div class="metric {'positive' if pnl_usd > 0 else 'negative' if pnl_usd < 0 else 'neutral'}">
                    <div class="value">${pnl_usd:,.0f}</div>
                    <div class="label">Profit/Loss</div>
                </div>
                <div class="metric {'positive' if win_rate >= 50 else 'negative'}">
                    <div class="value">{win_rate:.1f}%</div>
                    <div class="label">Win Rate</div>
                </div>
                <div class="metric {'positive' if profit_factor > 1 else 'negative' if profit_factor < 1 else 'neutral'}">
                    <div class="value">{profit_factor:.2f}</div>
                    <div class="label">Profit Factor</div>
                </div>
                <div class="metric {'positive' if sharpe > 0 else 'negative'}">
                    <div class="value">{sharpe:.2f}</div>
                    <div class="label">Sharpe Ratio</div>
                </div>
                <div class="metric {'negative' if max_dd > 20 else 'neutral'}">
                    <div class="value">{max_dd:.1f}%</div>
                    <div class="label">Max Drawdown</div>
                </div>
            </div>
        </div>
        
        <!-- Trade Summary -->
        <div class="card">
            <h2>🎯 Trade Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="num">{total_trades}</div>
                    <div class="lbl">Total Trades</div>
                </div>
                <div class="summary-item">
                    <div class="num" style="color: #27ae60">{winning_trades}</div>
                    <div class="lbl">Winners</div>
                </div>
                <div class="summary-item">
                    <div class="num" style="color: #e74c3c">{losing_trades}</div>
                    <div class="lbl">Losers</div>
                </div>
            </div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="num" style="color: #27ae60">{avg_win:+.2f}%</div>
                    <div class="lbl">Avg Win</div>
                </div>
                <div class="summary-item">
                    <div class="num" style="color: #e74c3c">{avg_loss:+.2f}%</div>
                    <div class="lbl">Avg Loss</div>
                </div>
                <div class="summary-item">
                    <div class="num">{abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}</div>
                    <div class="lbl">Win/Loss Ratio</div>
                </div>
            </div>
        </div>
        
        <!-- Equity Curve -->
        <div class="card">
            <h2>📉 Equity Curve & Drawdown</h2>
            <div class="chart-container">
                {equity_chart}
            </div>
        </div>
        
        <!-- Configuration -->
        <div class="card">
            <h2>⚙️ Configuration</h2>
            {config_table}
        </div>
        
        <!-- Trade Log -->
        <div class="card">
            <h2>📋 Trade Log</h2>
            {trades_table}
        </div>
        
        {validation_html}
        
        <!-- Footer -->
        <div class="footer">
            Generated by QML Trading System &nbsp;|&nbsp;
            VRD 2.0 Compliant &nbsp;|&nbsp;
            {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        # Write file
        output_path = strategy_dir / f"{run_id}_dossier.html"
        output_path.write_text(html)
        
        return str(output_path)
