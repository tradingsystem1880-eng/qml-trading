#!/usr/bin/env python3
"""
Phase 9.6: Comprehensive Strategy Report Generator
==================================================
Generates hedge fund-grade HTML strategy validation report.

Sections:
1. Cover Page
2. Executive Summary
3. Methodology
4. Backtest Performance
5. Statistical Validation
6. Feature Analysis (SHAP + correlations)
7. Risk Assessment
8. Market Regime Analysis
9. Forward Testing Plan
10. Appendices

Uses Arctic Pro theme and Plotly.js for interactive charts.
"""

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Arctic Pro Theme Colors
THEME = {
    'bg_primary': '#0B1426',
    'bg_secondary': '#0F1A2E',
    'bg_card': '#162032',
    'accent': '#3B82F6',
    'success': '#10B981',
    'danger': '#EF4444',
    'warning': '#F59E0B',
    'text_primary': '#F8FAFC',
    'text_secondary': '#94A3B8',
    'border': '#334155',
    'chart_green': '#22C55E',
    'chart_red': '#EF4444',
}


class StrategyReportGenerator:
    """
    Generates comprehensive strategy validation reports.

    Usage:
        generator = StrategyReportGenerator()
        html = generator.generate_report(
            trades_df=trades_df,
            validation_results=validation_dict,
            metrics=metrics_dict,
        )
        generator.save_report(html, 'reports/strategy_report.html')
    """

    def __init__(self):
        """Initialize report generator."""
        self.generated_at = datetime.now()

    def _get_css(self) -> str:
        """Get CSS styles for the report."""
        return f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: {THEME['bg_primary']};
                color: {THEME['text_primary']};
                line-height: 1.6;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }}

            /* Cover Page */
            .cover-page {{
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                background: linear-gradient(135deg, {THEME['bg_primary']} 0%, {THEME['bg_secondary']} 100%);
                border-bottom: 1px solid {THEME['border']};
                margin-bottom: 40px;
            }}

            .cover-title {{
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                background: linear-gradient(90deg, {THEME['accent']}, {THEME['success']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            .cover-subtitle {{
                font-size: 1.5rem;
                color: {THEME['text_secondary']};
                margin-bottom: 2rem;
            }}

            .cover-meta {{
                color: {THEME['text_secondary']};
                font-size: 0.9rem;
            }}

            /* Section Headers */
            .section {{
                margin-bottom: 60px;
                page-break-inside: avoid;
            }}

            .section-header {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 24px;
                padding-bottom: 12px;
                border-bottom: 2px solid {THEME['accent']};
            }}

            .section-number {{
                width: 36px;
                height: 36px;
                background: {THEME['accent']};
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 0.9rem;
            }}

            .section-title {{
                font-size: 1.75rem;
                font-weight: 600;
            }}

            /* Cards */
            .card {{
                background: {THEME['bg_card']};
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 20px;
                border: 1px solid {THEME['border']};
            }}

            .card-title {{
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 16px;
                color: {THEME['text_primary']};
            }}

            /* Metrics Grid */
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }}

            .metric-card {{
                background: {THEME['bg_secondary']};
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                border: 1px solid {THEME['border']};
            }}

            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 4px;
            }}

            .metric-label {{
                font-size: 0.85rem;
                color: {THEME['text_secondary']};
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}

            .metric-positive {{ color: {THEME['success']}; }}
            .metric-negative {{ color: {THEME['danger']}; }}
            .metric-neutral {{ color: {THEME['accent']}; }}

            /* Tables */
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
            }}

            th, td {{
                padding: 12px 16px;
                text-align: left;
                border-bottom: 1px solid {THEME['border']};
            }}

            th {{
                background: {THEME['bg_secondary']};
                font-weight: 600;
                color: {THEME['text_secondary']};
                text-transform: uppercase;
                font-size: 0.8rem;
                letter-spacing: 0.05em;
            }}

            tr:hover {{
                background: rgba(59, 130, 246, 0.05);
            }}

            /* Status badges */
            .badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 500;
            }}

            .badge-pass {{
                background: rgba(16, 185, 129, 0.2);
                color: {THEME['success']};
            }}

            .badge-fail {{
                background: rgba(239, 68, 68, 0.2);
                color: {THEME['danger']};
            }}

            .badge-warn {{
                background: rgba(245, 158, 11, 0.2);
                color: {THEME['warning']};
            }}

            /* Charts */
            .chart-container {{
                background: {THEME['bg_secondary']};
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 20px;
            }}

            /* Collapsible sections */
            .collapsible {{
                cursor: pointer;
                user-select: none;
            }}

            .collapsible:after {{
                content: ' ▼';
                font-size: 0.7em;
                color: {THEME['text_secondary']};
            }}

            .collapsible.active:after {{
                content: ' ▲';
            }}

            .collapsible-content {{
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
            }}

            .collapsible-content.show {{
                max-height: none;
            }}

            /* Print styles */
            @media print {{
                body {{
                    background: white;
                    color: #1a1a1a;
                }}

                .cover-page {{
                    height: auto;
                    padding: 60px 20px;
                }}

                .card, .metric-card, .chart-container {{
                    background: #f5f5f5;
                    border-color: #ddd;
                }}

                .no-print {{
                    display: none;
                }}

                .section {{
                    page-break-before: always;
                }}

                .section:first-of-type {{
                    page-break-before: avoid;
                }}
            }}

            /* Code blocks */
            code {{
                font-family: 'JetBrains Mono', monospace;
                background: {THEME['bg_secondary']};
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.9em;
            }}

            pre {{
                background: {THEME['bg_secondary']};
                padding: 16px;
                border-radius: 8px;
                overflow-x: auto;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                line-height: 1.5;
            }}

            /* Verdict box */
            .verdict-box {{
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                margin: 24px 0;
            }}

            .verdict-pass {{
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
                border: 2px solid {THEME['success']};
            }}

            .verdict-fail {{
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
                border: 2px solid {THEME['danger']};
            }}

            .verdict-title {{
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 8px;
            }}

            .verdict-subtitle {{
                color: {THEME['text_secondary']};
            }}
        </style>
        """

    def _get_js(self) -> str:
        """Get JavaScript for interactivity."""
        return """
        <script>
            // Collapsible sections
            document.querySelectorAll('.collapsible').forEach(function(element) {
                element.addEventListener('click', function() {
                    this.classList.toggle('active');
                    var content = this.nextElementSibling;
                    content.classList.toggle('show');
                });
            });
        </script>
        """

    def _render_cover_page(
        self,
        strategy_name: str = "QML Pattern Trading System",
        version: str = "Phase 9.6",
    ) -> str:
        """Render the cover page."""
        return f"""
        <div class="cover-page">
            <h1 class="cover-title">{strategy_name}</h1>
            <p class="cover-subtitle">Comprehensive Strategy Validation Report</p>
            <div class="cover-meta">
                <p>Version: {version}</p>
                <p>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Framework: QML Trading System</p>
            </div>
        </div>
        """

    def _render_section_header(self, number: int, title: str) -> str:
        """Render a section header."""
        return f"""
        <div class="section-header">
            <div class="section-number">{number}</div>
            <h2 class="section-title">{title}</h2>
        </div>
        """

    def _render_metric_card(
        self,
        value: str,
        label: str,
        status: str = "neutral",
    ) -> str:
        """Render a metric card."""
        status_class = f"metric-{status}"
        return f"""
        <div class="metric-card">
            <div class="metric-value {status_class}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """

    def _render_executive_summary(
        self,
        metrics: Dict[str, Any],
        validation_passed: bool,
    ) -> str:
        """Render executive summary section."""
        verdict_class = "verdict-pass" if validation_passed else "verdict-fail"
        verdict_text = "VALIDATED" if validation_passed else "NOT VALIDATED"
        verdict_desc = (
            "Strategy demonstrates statistically significant edge"
            if validation_passed
            else "Strategy requires further optimization"
        )

        win_rate = metrics.get('win_rate', 0)
        pf = metrics.get('profit_factor', 0)
        expectancy = metrics.get('expectancy', 0)
        total_trades = metrics.get('total_trades', 0)

        win_status = "positive" if win_rate > 0.5 else "negative"
        pf_status = "positive" if pf > 1.5 else ("neutral" if pf > 1 else "negative")
        exp_status = "positive" if expectancy > 0.5 else ("neutral" if expectancy > 0 else "negative")

        html = f"""
        <div class="section">
            {self._render_section_header(2, "Executive Summary")}

            <div class="verdict-box {verdict_class}">
                <div class="verdict-title">{verdict_text}</div>
                <div class="verdict-subtitle">{verdict_desc}</div>
            </div>

            <div class="metrics-grid">
                {self._render_metric_card(f"{win_rate:.1%}", "Win Rate", win_status)}
                {self._render_metric_card(f"{pf:.2f}", "Profit Factor", pf_status)}
                {self._render_metric_card(f"{expectancy:.2f}R", "Expectancy", exp_status)}
                {self._render_metric_card(str(total_trades), "Total Trades", "neutral")}
            </div>

            <div class="card">
                <h3 class="card-title">Key Findings</h3>
                <ul style="padding-left: 20px; color: {THEME['text_secondary']};">
                    <li>Pattern detection system identifies QML formations with {win_rate:.1%} accuracy</li>
                    <li>Risk-adjusted returns show {'positive' if pf > 1 else 'negative'} expectancy of {expectancy:.2f}R per trade</li>
                    <li>Statistical validation {'confirms' if validation_passed else 'does not confirm'} edge significance</li>
                    <li>Sample size of {total_trades} trades {'meets' if total_trades >= 100 else 'below'} minimum threshold for reliability</li>
                </ul>
            </div>
        </div>
        """
        return html

    def _render_methodology(self) -> str:
        """Render methodology section."""
        return f"""
        <div class="section">
            {self._render_section_header(3, "Methodology")}

            <div class="card">
                <h3 class="card-title">Pattern Detection Algorithm</h3>
                <p style="color: {THEME['text_secondary']}; margin-bottom: 16px;">
                    The QML (Quasimodo) pattern detection system uses a hierarchical swing detection
                    algorithm to identify five-point reversal patterns in price action.
                </p>

                <h4 style="margin: 16px 0 8px; color: {THEME['text_primary']};">Detection Layers</h4>
                <ol style="padding-left: 20px; color: {THEME['text_secondary']};">
                    <li><strong>Geometric Layer:</strong> Identifies swing highs/lows using ATR-based thresholds</li>
                    <li><strong>Significance Layer:</strong> Filters swings by magnitude and market context</li>
                    <li><strong>Pattern Layer:</strong> Validates 5-point QML structure (P1→P2→P3→P4→P5)</li>
                </ol>

                <h4 style="margin: 16px 0 8px; color: {THEME['text_primary']};">Scoring Components</h4>
                <ul style="padding-left: 20px; color: {THEME['text_secondary']};">
                    <li>Head Extension Score (22%)</li>
                    <li>Break of Structure Efficiency (18%)</li>
                    <li>Shoulder Symmetry (12%)</li>
                    <li>Swing Significance (8%)</li>
                    <li>Volume Spike (10%)</li>
                    <li>Path Efficiency (10%)</li>
                    <li>Trend Strength (10%)</li>
                    <li>Regime Suitability (10%)</li>
                </ul>
            </div>

            <div class="card">
                <h3 class="card-title">Trade Management</h3>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Rationale</th></tr>
                    <tr><td>Stop Loss</td><td>1.0 ATR</td><td>Below/above pattern invalidation point</td></tr>
                    <tr><td>Take Profit</td><td>4.6 ATR</td><td>Optimized via Bayesian search</td></tr>
                    <tr><td>Risk:Reward</td><td>1:4.6</td><td>Compensates for ~50% win rate</td></tr>
                    <tr><td>Max Hold</td><td>100 bars</td><td>Prevent capital lockup</td></tr>
                </table>
            </div>
        </div>
        """

    def _render_backtest_performance(
        self,
        trades_df: pd.DataFrame,
        metrics: Dict[str, Any],
    ) -> str:
        """Render backtest performance section with charts."""
        # Calculate equity curve
        if 'pnl_r' in trades_df.columns:
            cumulative_pnl = trades_df['pnl_r'].cumsum()
        else:
            cumulative_pnl = pd.Series([0])

        # Create equity chart
        equity_chart = ""
        if HAS_PLOTLY and len(trades_df) > 0:
            fig = go.Figure()

            # Equity curve
            fig.add_trace(go.Scatter(
                x=list(range(len(cumulative_pnl))),
                y=cumulative_pnl.values,
                mode='lines',
                name='Equity (R)',
                line=dict(color=THEME['accent'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(59, 130, 246, 0.1)",
            ))

            fig.update_layout(
                title="Cumulative P&L (R-Multiple)",
                height=400,
                paper_bgcolor=THEME['bg_secondary'],
                plot_bgcolor=THEME['bg_secondary'],
                font=dict(color=THEME['text_primary']),
                xaxis=dict(title="Trade #", gridcolor=THEME['border']),
                yaxis=dict(title="Cumulative R", gridcolor=THEME['border']),
            )

            equity_chart = f'<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>'

        # Monthly returns
        monthly_chart = ""
        if HAS_PLOTLY and 'entry_time' in trades_df.columns and len(trades_df) > 0:
            trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
            monthly_pnl = trades_df.groupby('month')['pnl_r'].sum()

            colors = [THEME['chart_green'] if x > 0 else THEME['chart_red'] for x in monthly_pnl.values]

            fig2 = go.Figure(go.Bar(
                x=[str(m) for m in monthly_pnl.index],
                y=monthly_pnl.values,
                marker_color=colors,
            ))

            fig2.update_layout(
                title="Monthly Returns (R-Multiple)",
                height=300,
                paper_bgcolor=THEME['bg_secondary'],
                plot_bgcolor=THEME['bg_secondary'],
                font=dict(color=THEME['text_primary']),
                xaxis=dict(gridcolor=THEME['border']),
                yaxis=dict(title="Monthly R", gridcolor=THEME['border']),
            )

            monthly_chart = f'<div class="chart-container">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>'

        # Performance metrics
        avg_win = metrics.get('avg_win', 0)
        avg_loss = metrics.get('avg_loss', 0)
        max_dd = metrics.get('max_drawdown', 0)
        sharpe = metrics.get('sharpe_ratio', 0)

        return f"""
        <div class="section">
            {self._render_section_header(4, "Backtest Performance")}

            {equity_chart}
            {monthly_chart}

            <div class="metrics-grid">
                {self._render_metric_card(f"{avg_win:.2f}R", "Avg Win", "positive")}
                {self._render_metric_card(f"{avg_loss:.2f}R", "Avg Loss", "negative")}
                {self._render_metric_card(f"{max_dd:.1%}", "Max Drawdown", "negative" if max_dd > 0.2 else "neutral")}
                {self._render_metric_card(f"{sharpe:.2f}", "Sharpe Ratio", "positive" if sharpe > 1 else "neutral")}
            </div>
        </div>
        """

    def _render_statistical_validation(
        self,
        validation_results: Dict[str, Any],
    ) -> str:
        """Render statistical validation section."""
        rows = ""
        for test_name, result in validation_results.items():
            if isinstance(result, dict):
                passed = result.get('passed', result.get('verdict') == 'PASS')
                p_value = result.get('p_value', result.get('results', {}).get('p_value', 'N/A'))
                badge_class = "badge-pass" if passed else "badge-fail"
                status_text = "PASS" if passed else "FAIL"

                if isinstance(p_value, float):
                    p_value_str = f"{p_value:.4f}"
                else:
                    p_value_str = str(p_value)

                rows += f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{p_value_str}</td>
                    <td><span class="badge {badge_class}">{status_text}</span></td>
                </tr>
                """

        return f"""
        <div class="section">
            {self._render_section_header(5, "Statistical Validation")}

            <div class="card">
                <h3 class="card-title">Validation Tests</h3>
                <table>
                    <tr><th>Test</th><th>P-Value</th><th>Result</th></tr>
                    {rows}
                </table>
            </div>

            <div class="card">
                <h3 class="card-title">Interpretation</h3>
                <ul style="padding-left: 20px; color: {THEME['text_secondary']};">
                    <li><strong>Permutation Test:</strong> Compares real profit factor against shuffled returns</li>
                    <li><strong>Walk-Forward:</strong> Tests out-of-sample performance across time periods</li>
                    <li><strong>Monte Carlo:</strong> Simulates equity paths to assess risk of ruin</li>
                    <li><strong>Bootstrap:</strong> Provides confidence intervals on key metrics</li>
                </ul>
            </div>
        </div>
        """

    def _render_feature_analysis(
        self,
        shap_image_path: Optional[str] = None,
        correlation_image_path: Optional[str] = None,
        feature_importance: Optional[pd.DataFrame] = None,
    ) -> str:
        """Render feature analysis section."""
        shap_img = ""
        if shap_image_path and Path(shap_image_path).exists():
            with open(shap_image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            shap_img = f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; border-radius: 8px;">'

        corr_img = ""
        if correlation_image_path and Path(correlation_image_path).exists():
            with open(correlation_image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            corr_img = f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; border-radius: 8px;">'

        importance_table = ""
        if feature_importance is not None and len(feature_importance) > 0:
            rows = ""
            for _, row in feature_importance.head(10).iterrows():
                importance = row.get('importance', 0)
                bar_width = min(100, importance * 100 / feature_importance['importance'].max())
                rows += f"""
                <tr>
                    <td>{row['feature']}</td>
                    <td>{importance:.4f}</td>
                    <td>
                        <div style="background: {THEME['bg_secondary']}; border-radius: 4px; height: 20px; width: 100%;">
                            <div style="background: {THEME['accent']}; border-radius: 4px; height: 100%; width: {bar_width}%;"></div>
                        </div>
                    </td>
                </tr>
                """
            importance_table = f"""
            <div class="card">
                <h3 class="card-title">Feature Importance (SHAP)</h3>
                <table>
                    <tr><th>Feature</th><th>Importance</th><th>Relative</th></tr>
                    {rows}
                </table>
            </div>
            """

        return f"""
        <div class="section">
            {self._render_section_header(6, "Feature Analysis")}

            {importance_table}

            <div class="card">
                <h3 class="card-title">SHAP Summary</h3>
                {shap_img if shap_img else '<p style="color: ' + THEME['text_secondary'] + ';">SHAP analysis not available. Run scripts/run_shap_analysis.py to generate.</p>'}
            </div>

            <div class="card">
                <h3 class="card-title">Feature Correlation Matrix</h3>
                {corr_img if corr_img else '<p style="color: ' + THEME['text_secondary'] + ';">Correlation analysis not available. Run scripts/generate_feature_scatter.py to generate.</p>'}
            </div>
        </div>
        """

    def _render_risk_assessment(self, metrics: Dict[str, Any]) -> str:
        """Render risk assessment section."""
        max_dd = metrics.get('max_drawdown', 0)
        var_95 = metrics.get('var_95', 0)
        cvar_95 = metrics.get('cvar_95', 0)
        risk_of_ruin = metrics.get('risk_of_ruin', 0)

        return f"""
        <div class="section">
            {self._render_section_header(7, "Risk Assessment")}

            <div class="metrics-grid">
                {self._render_metric_card(f"{max_dd:.1%}", "Max Drawdown", "negative" if max_dd > 0.2 else "neutral")}
                {self._render_metric_card(f"{var_95:.2f}R", "VaR (95%)", "neutral")}
                {self._render_metric_card(f"{cvar_95:.2f}R", "CVaR (95%)", "neutral")}
                {self._render_metric_card(f"{risk_of_ruin:.1%}", "Risk of Ruin", "negative" if risk_of_ruin > 0.05 else "positive")}
            </div>

            <div class="card">
                <h3 class="card-title">Position Sizing Recommendations</h3>
                <table>
                    <tr><th>Risk Profile</th><th>Position Size</th><th>Max Drawdown Est.</th></tr>
                    <tr><td>Conservative</td><td>0.5% per trade</td><td>~10%</td></tr>
                    <tr><td>Moderate</td><td>1.0% per trade</td><td>~20%</td></tr>
                    <tr><td>Aggressive</td><td>2.0% per trade</td><td>~40%</td></tr>
                </table>
            </div>
        </div>
        """

    def _render_regime_analysis(self, trades_df: pd.DataFrame) -> str:
        """Render market regime analysis section."""
        regime_stats = ""
        if 'regime' in trades_df.columns:
            regime_perf = trades_df.groupby('regime').agg({
                'pnl_r': ['count', 'mean', 'sum'],
            })
            regime_perf.columns = ['trades', 'avg_r', 'total_r']

            rows = ""
            for regime, row in regime_perf.iterrows():
                status = "positive" if row['avg_r'] > 0 else "negative"
                rows += f"""
                <tr>
                    <td>{regime}</td>
                    <td>{int(row['trades'])}</td>
                    <td class="metric-{status}">{row['avg_r']:.2f}R</td>
                    <td class="metric-{status}">{row['total_r']:.2f}R</td>
                </tr>
                """

            regime_stats = f"""
            <table>
                <tr><th>Regime</th><th>Trades</th><th>Avg R</th><th>Total R</th></tr>
                {rows}
            </table>
            """
        else:
            regime_stats = f'<p style="color: {THEME["text_secondary"]};">Regime data not available.</p>'

        return f"""
        <div class="section">
            {self._render_section_header(8, "Market Regime Analysis")}

            <div class="card">
                <h3 class="card-title">Performance by Market Regime</h3>
                {regime_stats}
            </div>

            <div class="card">
                <h3 class="card-title">Regime Definitions</h3>
                <ul style="padding-left: 20px; color: {THEME['text_secondary']};">
                    <li><strong>RANGING:</strong> Low ADX (&lt;20), price oscillating in range - Ideal for QML patterns</li>
                    <li><strong>TRENDING:</strong> High ADX (&gt;30), sustained directional move - Lower pattern success</li>
                    <li><strong>VOLATILE:</strong> High ATR relative to recent history - Wider stops needed</li>
                    <li><strong>EXTREME:</strong> ADX &gt;35 with high volatility - Avoid trading</li>
                </ul>
            </div>
        </div>
        """

    def _render_forward_testing_plan(self) -> str:
        """Render forward testing plan section."""
        return f"""
        <div class="section">
            {self._render_section_header(9, "Forward Testing Plan")}

            <div class="card">
                <h3 class="card-title">Phase 1: Paper Trading (4-8 weeks)</h3>
                <ul style="padding-left: 20px; color: {THEME['text_secondary']};">
                    <li>Execute all signals in paper trading mode</li>
                    <li>Track every trade with entry, exit, and R-multiple</li>
                    <li>Minimum 100 trades before proceeding</li>
                    <li>Compare results to backtest expectations (within 20%)</li>
                </ul>
            </div>

            <div class="card">
                <h3 class="card-title">Phase 2: Small Size Live (4-8 weeks)</h3>
                <ul style="padding-left: 20px; color: {THEME['text_secondary']};">
                    <li>0.25% risk per trade (quarter size)</li>
                    <li>Same symbols and timeframes as backtest</li>
                    <li>Monitor for execution slippage</li>
                    <li>Track psychological factors</li>
                </ul>
            </div>

            <div class="card">
                <h3 class="card-title">Phase 3: Full Deployment</h3>
                <ul style="padding-left: 20px; color: {THEME['text_secondary']};">
                    <li>Scale to 1% risk per trade if Phase 2 successful</li>
                    <li>Implement ForwardTestMonitor for edge degradation detection</li>
                    <li>Review performance weekly</li>
                    <li>Halt trading if 3-month rolling metrics deviate &gt;30% from backtest</li>
                </ul>
            </div>

            <div class="card">
                <h3 class="card-title">Edge Degradation Triggers</h3>
                <table>
                    <tr><th>Metric</th><th>Warning</th><th>Halt</th></tr>
                    <tr><td>Win Rate</td><td>&lt;40%</td><td>&lt;35%</td></tr>
                    <tr><td>Profit Factor</td><td>&lt;1.2</td><td>&lt;1.0</td></tr>
                    <tr><td>Max Drawdown</td><td>&gt;25%</td><td>&gt;35%</td></tr>
                    <tr><td>Consecutive Losses</td><td>8</td><td>12</td></tr>
                </table>
            </div>
        </div>
        """

    def _render_appendices(self, trades_df: pd.DataFrame) -> str:
        """Render appendices section."""
        # Recent trades table
        trades_table = ""
        if len(trades_df) > 0:
            display_cols = ['symbol', 'direction', 'entry_time', 'pnl_r', 'result']
            display_cols = [c for c in display_cols if c in trades_df.columns]

            if display_cols:
                recent = trades_df[display_cols].tail(20)
                rows = ""
                for _, row in recent.iterrows():
                    pnl_class = "metric-positive" if row.get('pnl_r', 0) > 0 else "metric-negative"
                    entry_time = row.get('entry_time', '')
                    if hasattr(entry_time, 'strftime'):
                        entry_time = entry_time.strftime('%Y-%m-%d %H:%M')
                    rows += f"""
                    <tr>
                        <td>{row.get('symbol', 'N/A')}</td>
                        <td>{row.get('direction', 'N/A')}</td>
                        <td>{entry_time}</td>
                        <td class="{pnl_class}">{row.get('pnl_r', 0):.2f}R</td>
                        <td>{row.get('result', 'N/A')}</td>
                    </tr>
                    """
                trades_table = f"""
                <div class="card">
                    <h3 class="card-title collapsible">Recent Trades (Last 20)</h3>
                    <div class="collapsible-content show">
                        <table>
                            <tr><th>Symbol</th><th>Direction</th><th>Entry</th><th>P&L</th><th>Result</th></tr>
                            {rows}
                        </table>
                    </div>
                </div>
                """

        return f"""
        <div class="section">
            {self._render_section_header(10, "Appendices")}

            {trades_table}

            <div class="card">
                <h3 class="card-title">System Configuration</h3>
                <pre>
Swing Detection:
  min_bar_separation: 3
  min_move_atr: 0.85
  forward_confirm_pct: 0.2
  lookback: 6
  lookforward: 8

Pattern Validation:
  p3_min_extension_atr: 0.3
  p3_max_extension_atr: 5.0
  p5_max_symmetry_atr: 4.6
  min_pattern_bars: 16
  max_pattern_bars: 200

Trade Management:
  tp_atr_mult: 4.6
  sl_atr_mult: 1.0
  max_bars_held: 100
  min_risk_reward: 3.0
                </pre>
            </div>

            <div class="card">
                <h3 class="card-title">Data Sources</h3>
                <table>
                    <tr><th>Source</th><th>Description</th></tr>
                    <tr><td>Binance</td><td>Historical OHLCV data (spot/futures)</td></tr>
                    <tr><td>Bybit</td><td>Forward testing / live execution</td></tr>
                    <tr><td>Timeframes</td><td>1H, 4H, 1D</td></tr>
                    <tr><td>Symbols</td><td>30 major crypto pairs</td></tr>
                </table>
            </div>
        </div>
        """

    def generate_report(
        self,
        trades_df: pd.DataFrame,
        metrics: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None,
        shap_image_path: Optional[str] = None,
        correlation_image_path: Optional[str] = None,
        feature_importance: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate the full HTML report.

        Args:
            trades_df: DataFrame with trade data
            metrics: Dict with performance metrics
            validation_results: Dict with validation test results
            shap_image_path: Path to SHAP summary image
            correlation_image_path: Path to correlation heatmap image
            feature_importance: DataFrame with feature importance

        Returns:
            Complete HTML string
        """
        validation_results = validation_results or {}
        validation_passed = all(
            r.get('passed', r.get('verdict') == 'PASS')
            for r in validation_results.values()
            if isinstance(r, dict)
        ) if validation_results else False

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QML Strategy Validation Report</title>
    {self._get_css()}
</head>
<body>
    {self._render_cover_page()}

    <div class="container">
        {self._render_executive_summary(metrics, validation_passed)}
        {self._render_methodology()}
        {self._render_backtest_performance(trades_df, metrics)}
        {self._render_statistical_validation(validation_results)}
        {self._render_feature_analysis(shap_image_path, correlation_image_path, feature_importance)}
        {self._render_risk_assessment(metrics)}
        {self._render_regime_analysis(trades_df)}
        {self._render_forward_testing_plan()}
        {self._render_appendices(trades_df)}
    </div>

    {self._get_js()}
</body>
</html>
        """
        return html

    def save_report(self, html: str, output_path: str) -> Path:
        """
        Save HTML report to file.

        Args:
            html: HTML content
            output_path: Path to save file

        Returns:
            Path to saved file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return path
