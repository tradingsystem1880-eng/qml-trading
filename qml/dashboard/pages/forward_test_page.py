"""
Forward Test Monitoring Page - Phase 9.4

Real-time monitoring of forward test performance vs baseline expectations.
Displays alerts when edge degradation is detected.
"""

import json
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

# Try to import theme, fallback to defaults
try:
    from theme import ARCTIC_PRO, TYPOGRAPHY
except ImportError:
    ARCTIC_PRO = {
        'background': '#0a0f1a',
        'surface': '#131b2e',
        'border': '#1e293b',
        'accent': '#3b82f6',
        'success': '#22c55e',
        'danger': '#ef4444',
        'warning': '#f59e0b',
        'text_primary': '#f1f5f9',
        'text_secondary': '#94a3b8',
        'text_muted': '#64748b',
    }
    TYPOGRAPHY = {
        'size_sm': '0.75rem',
        'size_base': '0.875rem',
        'size_lg': '1rem',
        'size_xl': '1.25rem',
        'size_2xl': '1.5rem',
        'weight_normal': 400,
        'weight_medium': 500,
        'weight_bold': 700,
    }

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_forward_config() -> Dict[str, Any]:
    """Load forward test configuration."""
    config_path = PROJECT_ROOT / "config" / "forward_test_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def load_validation_results() -> Optional[Dict[str, Any]]:
    """Load latest validation results from Phase 9.4."""
    results_dir = PROJECT_ROOT / "results" / "phase94_validation"
    if not results_dir.exists():
        return None

    # Find latest pf_validation file
    pf_files = sorted(results_dir.glob("pf_validation_*.json"), reverse=True)
    if pf_files:
        with open(pf_files[0]) as f:
            return json.load(f)

    return None


def load_walk_forward_results() -> Optional[Dict[str, Any]]:
    """Load latest walk-forward validation results."""
    results_dir = PROJECT_ROOT / "results" / "phase94_validation"
    if not results_dir.exists():
        return None

    wf_files = sorted(results_dir.glob("walk_forward_*.json"), reverse=True)
    if wf_files:
        with open(wf_files[0]) as f:
            return json.load(f)

    return None


def get_forward_test_trades() -> List[Dict]:
    """Get forward test trades from session state."""
    return st.session_state.get('forward_test_trades', [])


def calculate_current_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """Calculate current forward test metrics."""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'consecutive_losses': 0,
        }

    winners = [t for t in trades if t.get('pnl_r', 0) > 0]
    losers = [t for t in trades if t.get('pnl_r', 0) <= 0]

    total = len(trades)
    wr = len(winners) / total if total > 0 else 0

    gross_profit = sum(t.get('pnl_r', 0) for t in winners)
    gross_loss = abs(sum(t.get('pnl_r', 0) for t in losers))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([t.get('pnl_r', 0) for t in winners]) if winners else 0
    avg_loss = abs(np.mean([t.get('pnl_r', 0) for t in losers])) if losers else 0
    expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)

    # Count current consecutive losses
    consec_losses = 0
    for t in reversed(trades):
        if t.get('pnl_r', 0) <= 0:
            consec_losses += 1
        else:
            break

    return {
        'total_trades': total,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': wr,
        'profit_factor': pf,
        'expectancy': expectancy,
        'avg_win_r': avg_win,
        'avg_loss_r': avg_loss,
        'consecutive_losses': consec_losses,
    }


def check_alerts(metrics: Dict, config: Dict) -> List[Dict]:
    """Check for alert conditions."""
    alerts = []

    if not config:
        return alerts

    monitoring = config.get('monitoring', {})
    thresholds = monitoring.get('alert_thresholds', {})
    live_exp = monitoring.get('live_expectations', {})

    # Minimum trades check
    min_trades = thresholds.get('min_trades_for_comparison', 30)
    if metrics['total_trades'] < min_trades:
        alerts.append({
            'level': 'info',
            'message': f"Need {min_trades - metrics['total_trades']} more trades for valid comparison",
        })
        return alerts  # Don't check other alerts until we have enough trades

    # PF degradation
    baseline_pf = live_exp.get('profit_factor', 2.5)
    pf_threshold = thresholds.get('degradation_pf_pct', 0.30)
    if metrics['profit_factor'] < baseline_pf * (1 - pf_threshold):
        alerts.append({
            'level': 'warning',
            'message': f"PF {metrics['profit_factor']:.2f} is {pf_threshold*100:.0f}%+ below expected {baseline_pf:.2f}",
        })

    # WR degradation
    baseline_wr = live_exp.get('win_rate', 0.54)
    wr_threshold = thresholds.get('degradation_wr_pct', 0.15)
    if metrics['win_rate'] < baseline_wr * (1 - wr_threshold):
        alerts.append({
            'level': 'warning',
            'message': f"Win Rate {metrics['win_rate']:.1%} is {wr_threshold*100:.0f}%+ below expected {baseline_wr:.1%}",
        })

    # Consecutive losses
    max_consec = thresholds.get('max_consecutive_losses', 7)
    if metrics['consecutive_losses'] >= max_consec:
        alerts.append({
            'level': 'danger',
            'message': f"{metrics['consecutive_losses']} consecutive losses (max: {max_consec})",
        })

    # PF below 1.0 with sufficient trades
    if metrics['profit_factor'] < 1.0 and metrics['total_trades'] >= 50:
        alerts.append({
            'level': 'danger',
            'message': f"System is UNPROFITABLE (PF {metrics['profit_factor']:.2f} < 1.0)",
        })

    return alerts


def render_metric_card(label: str, value: str, status: str = "neutral",
                       baseline: Optional[str] = None) -> None:
    """Render a metric card with optional baseline comparison."""
    color = ARCTIC_PRO["text_primary"]
    if status == "positive":
        color = ARCTIC_PRO["success"]
    elif status == "negative":
        color = ARCTIC_PRO["danger"]
    elif status == "warning":
        color = ARCTIC_PRO["warning"]

    html = '<div class="panel" style="text-align: center;">'
    html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]}; '
    html += f'text-transform: uppercase; margin-bottom: 0.5rem;">{label}</div>'
    html += f'<div style="color: {color}; font-size: {TYPOGRAPHY["size_2xl"]}; '
    html += f'font-weight: {TYPOGRAPHY["weight_bold"]};">{value}</div>'

    if baseline:
        html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]}; '
        html += f'margin-top: 0.25rem;">Expected: {baseline}</div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_alert_banner(alerts: List[Dict]) -> None:
    """Render alert banners."""
    for alert in alerts:
        level = alert.get('level', 'info')
        message = alert.get('message', '')

        if level == 'danger':
            bg_color = 'rgba(239, 68, 68, 0.2)'
            border_color = ARCTIC_PRO['danger']
            icon = '🚨'
        elif level == 'warning':
            bg_color = 'rgba(245, 158, 11, 0.2)'
            border_color = ARCTIC_PRO['warning']
            icon = '⚠️'
        else:
            bg_color = 'rgba(59, 130, 246, 0.2)'
            border_color = ARCTIC_PRO['accent']
            icon = 'ℹ️'

        html = f'<div style="background: {bg_color}; border-left: 4px solid {border_color}; '
        html += f'padding: 12px 16px; margin-bottom: 8px; border-radius: 4px;">'
        html += f'<span style="margin-right: 8px;">{icon}</span>'
        html += f'<span style="color: {ARCTIC_PRO["text_primary"]};">{message}</span>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)


def render_phase_progress(config: Dict, metrics: Dict) -> None:
    """Render forward test phase progress."""
    phases = config.get('forward_test_phases', {})

    if not phases:
        return

    current_trades = metrics.get('total_trades', 0)

    for phase_id, phase_config in phases.items():
        min_trades = phase_config.get('min_trades', 0)
        success_criteria = phase_config.get('success_criteria', {})
        min_pf = success_criteria.get('min_pf', 0)
        min_wr = success_criteria.get('min_wr', 0)

        progress = min(current_trades / min_trades * 100, 100) if min_trades > 0 else 0

        # Determine status
        if progress >= 100:
            pf_pass = metrics.get('profit_factor', 0) >= min_pf
            wr_pass = metrics.get('win_rate', 0) >= min_wr
            if pf_pass and wr_pass:
                status = "pass"
                status_color = ARCTIC_PRO['success']
                status_text = "PASSED"
            else:
                status = "fail"
                status_color = ARCTIC_PRO['danger']
                status_text = "FAILED"
        else:
            status = "in_progress"
            status_color = ARCTIC_PRO['accent']
            status_text = f"{progress:.0f}%"

        # Format phase name
        phase_name = phase_id.replace('_', ' ').title()

        html = '<div style="margin-bottom: 12px;">'
        html += f'<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">'
        html += f'<span style="color: {ARCTIC_PRO["text_secondary"]};">{phase_name}</span>'
        html += f'<span style="color: {status_color}; font-weight: 600;">{status_text}</span>'
        html += '</div>'

        # Progress bar
        html += f'<div style="background: {ARCTIC_PRO["border"]}; border-radius: 4px; height: 8px; overflow: hidden;">'
        html += f'<div style="background: {status_color}; width: {progress}%; height: 100%;"></div>'
        html += '</div>'

        # Criteria
        html += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]}; margin-top: 4px;">'
        html += f'{min_trades} trades | PF ≥ {min_pf} | WR ≥ {min_wr*100:.0f}%'
        html += '</div>'

        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)


def render_validation_summary(pf_results: Dict, wf_results: Dict) -> None:
    """Render Phase 9.4 validation summary."""
    html = '<div class="panel">'
    html += '<div class="panel-header">Phase 9.4 Validation Summary</div>'

    if pf_results:
        verdict = pf_results.get('verdict', 'N/A')
        stats = pf_results.get('aggregate_stats', {})

        verdict_color = ARCTIC_PRO['success'] if verdict == 'PASS' else ARCTIC_PRO['danger']

        html += f'<div style="margin-bottom: 12px;">'
        html += f'<strong style="color: {ARCTIC_PRO["text_secondary"]};">Distribution Validation:</strong> '
        html += f'<span style="color: {verdict_color}; font-weight: bold;">{verdict}</span>'
        html += '</div>'

        html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">'
        html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Avg Win (R):</div>'
        html += f'<div style="color: {ARCTIC_PRO["text_primary"]};">{stats.get("avg_win_r", 0):.2f}</div>'
        html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Avg Loss (R):</div>'
        html += f'<div style="color: {ARCTIC_PRO["text_primary"]};">{stats.get("avg_loss_r", 0):.2f}</div>'
        html += '</div>'
    else:
        html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Run validate_pf_distribution.py first</div>'

    html += '<div style="height: 16px;"></div>'

    if wf_results:
        verdict = wf_results.get('verdict', 'N/A')
        fold_results = wf_results.get('fold_results', [])

        verdict_color = ARCTIC_PRO['success'] if verdict == 'PASS' else ARCTIC_PRO['danger']

        html += f'<div style="margin-bottom: 12px;">'
        html += f'<strong style="color: {ARCTIC_PRO["text_secondary"]};">Walk-Forward Validation:</strong> '
        html += f'<span style="color: {verdict_color}; font-weight: bold;">{verdict}</span>'
        html += '</div>'

        # Show fold summary
        passing = sum(1 for f in fold_results if f.get('status') == 'PASS')
        html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">'
        html += f'{passing}/{len(fold_results)} folds passed (PF > 1.5)'
        html += '</div>'
    else:
        html += f'<div style="color: {ARCTIC_PRO["text_muted"]};">Run walk_forward_validation.py first</div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_forward_test_page() -> None:
    """Render the forward test monitoring page."""

    # Page header
    html = '<div class="panel">'
    html += '<div class="panel-header">Forward Test Monitor</div>'
    html += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Real-time monitoring of forward test performance vs baseline expectations.</p>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Load config and data
    config = load_forward_config()
    pf_results = load_validation_results()
    wf_results = load_walk_forward_results()
    trades = get_forward_test_trades()

    # Calculate current metrics
    metrics = calculate_current_metrics(trades)

    # Check for alerts
    alerts = check_alerts(metrics, config)

    # Render alerts at top
    if alerts:
        for alert in alerts:
            render_alert_banner([alert])

    # Get expected values from config
    live_exp = config.get('monitoring', {}).get('live_expectations', {})
    baseline_pf = live_exp.get('profit_factor', 2.5)
    baseline_wr = live_exp.get('win_rate', 0.54)
    baseline_exp = live_exp.get('expectancy_r', 1.0)

    # Primary metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Total Trades", str(metrics['total_trades']), "neutral")

    with col2:
        wr = metrics['win_rate']
        wr_status = "positive" if wr >= baseline_wr else "negative" if wr < baseline_wr * 0.85 else "warning"
        render_metric_card("Win Rate", f"{wr:.1%}", wr_status, f"{baseline_wr:.1%}")

    with col3:
        pf = metrics['profit_factor']
        pf_status = "positive" if pf >= baseline_pf else "negative" if pf < 1.0 else "warning"
        pf_display = f"{pf:.2f}" if pf != float('inf') else "∞"
        render_metric_card("Profit Factor", pf_display, pf_status, f"{baseline_pf:.2f}")

    with col4:
        exp = metrics['expectancy']
        exp_status = "positive" if exp >= baseline_exp else "negative" if exp < 0 else "warning"
        render_metric_card("Expectancy (R)", f"{exp:.2f}", exp_status, f"{baseline_exp:.2f}")

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Layout: Left = Phase Progress, Right = Validation Summary
    col_left, col_right = st.columns(2)

    with col_left:
        progress_panel = '<div class="panel">'
        progress_panel += '<div class="panel-header">Forward Test Progress</div>'
        progress_panel += '</div>'
        st.markdown(progress_panel, unsafe_allow_html=True)

        if config:
            render_phase_progress(config, metrics)
        else:
            html = f'<div style="color: {ARCTIC_PRO["text_muted"]}; padding: 16px;">No config loaded. '
            html += 'Create config/forward_test_config.json</div>'
            st.markdown(html, unsafe_allow_html=True)

    with col_right:
        render_validation_summary(pf_results, wf_results)

    # Trade log section
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    log_panel = '<div class="panel">'
    log_panel += '<div class="panel-header">Recent Trades</div>'
    log_panel += '</div>'
    st.markdown(log_panel, unsafe_allow_html=True)

    if trades:
        # Show last 10 trades
        recent = trades[-10:][::-1]

        table_html = '<table style="width: 100%; border-collapse: collapse;">'
        table_html += '<thead><tr style="border-bottom: 1px solid ' + ARCTIC_PRO['border'] + ';">'
        for header in ['Date', 'Symbol', 'Side', 'Entry', 'Exit', 'P&L (R)', 'Outcome']:
            table_html += f'<th style="padding: 8px; text-align: left; color: {ARCTIC_PRO["text_muted"]};">{header}</th>'
        table_html += '</tr></thead><tbody>'

        for trade in recent:
            pnl_r = trade.get('pnl_r', 0)
            outcome_color = ARCTIC_PRO['success'] if pnl_r > 0 else ARCTIC_PRO['danger']
            outcome_text = 'WIN' if pnl_r > 0 else 'LOSS'

            table_html += f'<tr style="border-bottom: 1px solid {ARCTIC_PRO["border"]};">'
            table_html += f'<td style="padding: 8px; color: {ARCTIC_PRO["text_secondary"]};">{trade.get("date", "N/A")}</td>'
            table_html += f'<td style="padding: 8px; color: {ARCTIC_PRO["text_primary"]};">{trade.get("symbol", "N/A")}</td>'
            table_html += f'<td style="padding: 8px; color: {ARCTIC_PRO["text_secondary"]};">{trade.get("side", "N/A")}</td>'
            table_html += f'<td style="padding: 8px; color: {ARCTIC_PRO["text_secondary"]};">{trade.get("entry_price", 0):,.2f}</td>'
            table_html += f'<td style="padding: 8px; color: {ARCTIC_PRO["text_secondary"]};">{trade.get("exit_price", 0):,.2f}</td>'
            table_html += f'<td style="padding: 8px; color: {outcome_color}; font-weight: 600;">{pnl_r:+.2f}</td>'
            table_html += f'<td style="padding: 8px; color: {outcome_color}; font-weight: 600;">{outcome_text}</td>'
            table_html += '</tr>'

        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        html = f'<div style="padding: 32px; text-align: center; color: {ARCTIC_PRO["text_muted"]};">'
        html += 'No forward test trades recorded yet.<br><br>'
        html += 'To start forward testing:<br>'
        html += '1. Run validation scripts first<br>'
        html += '2. Use paper trading to record trades<br>'
        html += '3. Trades will appear here automatically'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # Commands reference
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    cmd_panel = '<div class="panel">'
    cmd_panel += '<div class="panel-header">Quick Commands</div>'
    cmd_panel += f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 8px;">'

    cmd_panel += f'<div><code style="background: {ARCTIC_PRO["surface"]}; padding: 4px 8px; border-radius: 4px; font-size: {TYPOGRAPHY["size_sm"]};">'
    cmd_panel += 'python scripts/validate_pf_distribution.py</code>'
    cmd_panel += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]}; margin-top: 4px;">Validate R:R distribution</div></div>'

    cmd_panel += f'<div><code style="background: {ARCTIC_PRO["surface"]}; padding: 4px 8px; border-radius: 4px; font-size: {TYPOGRAPHY["size_sm"]};">'
    cmd_panel += 'python scripts/walk_forward_validation.py</code>'
    cmd_panel += f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]}; margin-top: 4px;">Walk-forward validation (5 folds)</div></div>'

    cmd_panel += '</div></div>'
    st.markdown(cmd_panel, unsafe_allow_html=True)
