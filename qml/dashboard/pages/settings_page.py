"""
Settings Page - Configure dashboard, detection, and prop firm parameters.
"""

import streamlit as st
import yaml
import json
from pathlib import Path
from theme import ARCTIC_PRO, TYPOGRAPHY

# Config paths - use project root to handle running from any directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
YAML_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
JSON_CONFIG_PATH = PROJECT_ROOT / "qml" / "dashboard" / "config" / "user_config.json"

# Prop firm presets
FIRM_PRESETS = {
    "Breakout": {
        "account_size": 100000,
        "profit_target_pct": 8.0,
        "max_daily_dd_pct": 4.0,
        "max_total_dd_pct": 8.0,
        "min_trading_days": 5,
        "consistency_rule": True,
    },
    "FTMO": {
        "account_size": 100000,
        "profit_target_pct": 10.0,
        "max_daily_dd_pct": 5.0,
        "max_total_dd_pct": 10.0,
        "min_trading_days": 4,
        "consistency_rule": False,
    },
    "MFF": {
        "account_size": 100000,
        "profit_target_pct": 8.0,
        "max_daily_dd_pct": 5.0,
        "max_total_dd_pct": 12.0,
        "min_trading_days": 5,
        "consistency_rule": True,
    },
    "Custom": None,  # User-defined
}


def load_yaml_config() -> dict:
    """Load configuration from YAML file."""
    if YAML_CONFIG_PATH.exists():
        with open(YAML_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_yaml_config(config: dict) -> bool:
    """Save configuration to YAML file."""
    try:
        YAML_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(YAML_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"Failed to save YAML config: {e}")
        return False


def load_prop_firm_config() -> dict:
    """Load prop firm configuration from JSON file."""
    if JSON_CONFIG_PATH.exists():
        with open(JSON_CONFIG_PATH) as f:
            return json.load(f)
    # Return Breakout defaults
    return {
        "firm_name": "Breakout",
        "account_size": 100000,
        "profit_target_pct": 8.0,
        "max_daily_dd_pct": 4.0,
        "max_total_dd_pct": 8.0,
        "min_trading_days": 5,
        "max_position_size_pct": 2.0,
        "consistency_rule": True,
        "kelly_fraction": 0.5,
        "max_concurrent": 3,
        "primary_symbols": ["BTC/USDT", "ETH/USDT"],
        "primary_timeframe": "4H"
    }


def save_prop_firm_config(config: dict) -> bool:
    """Save prop firm configuration to JSON file."""
    try:
        JSON_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JSON_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save prop firm config: {e}")
        return False


def render_settings_page() -> None:
    """Render the settings configuration page."""

    # Load configs
    yaml_config = load_yaml_config()
    prop_config = load_prop_firm_config()

    # Page header
    html = '<div class="panel">'
    html += '<div class="panel-header">Settings</div>'
    html += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Configure detection parameters, trade settings, and prop firm rules.</p>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # === PROP FIRM CONFIGURATION ===
    st.markdown("---")
    prop_panel = '<div class="panel">'
    prop_panel += '<div class="panel-header">Prop Firm Configuration</div>'
    prop_panel += '</div>'
    st.markdown(prop_panel, unsafe_allow_html=True)

    # Firm preset selection
    firm_names = list(FIRM_PRESETS.keys())
    current_firm = prop_config.get('firm_name', 'Breakout')
    if current_firm not in firm_names:
        current_firm = 'Custom'

    firm_name = st.selectbox(
        "Prop Firm Preset",
        firm_names,
        index=firm_names.index(current_firm),
        key="settings_firm_preset"
    )

    # Load preset values or use custom
    if firm_name != "Custom" and FIRM_PRESETS.get(firm_name):
        preset = FIRM_PRESETS[firm_name]
        default_account = preset["account_size"]
        default_target = preset["profit_target_pct"]
        default_daily = preset["max_daily_dd_pct"]
        default_total = preset["max_total_dd_pct"]
        default_min_days = preset["min_trading_days"]
        default_consistency = preset["consistency_rule"]
    else:
        default_account = prop_config.get('account_size', 100000)
        default_target = prop_config.get('profit_target_pct', 8.0)
        default_daily = prop_config.get('max_daily_dd_pct', 4.0)
        default_total = prop_config.get('max_total_dd_pct', 8.0)
        default_min_days = prop_config.get('min_trading_days', 5)
        default_consistency = prop_config.get('consistency_rule', True)

    col_pf1, col_pf2 = st.columns(2)

    with col_pf1:
        account_size = st.number_input(
            "Account Size ($)",
            min_value=1000,
            max_value=1000000,
            value=int(default_account),
            step=1000,
            key="settings_account_size"
        )

        max_daily_dd = st.number_input(
            "Max Daily Drawdown (%)",
            min_value=1.0,
            max_value=20.0,
            value=float(default_daily),
            step=0.5,
            key="settings_daily_dd"
        )

        max_total_dd = st.number_input(
            "Max Total Drawdown (%)",
            min_value=1.0,
            max_value=30.0,
            value=float(default_total),
            step=0.5,
            key="settings_total_dd"
        )

        consistency_rule = st.checkbox(
            "Consistency Rule (no day > 30% of profits)",
            value=default_consistency,
            key="settings_consistency"
        )

    with col_pf2:
        profit_target = st.number_input(
            "Profit Target (%)",
            min_value=1.0,
            max_value=30.0,
            value=float(default_target),
            step=0.5,
            key="settings_profit_target"
        )

        min_trading_days = st.number_input(
            "Min Trading Days",
            min_value=1,
            max_value=30,
            value=int(default_min_days),
            key="settings_min_days"
        )

        max_position_size = st.number_input(
            "Max Position Size (%)",
            min_value=0.5,
            max_value=10.0,
            value=float(prop_config.get('max_position_size_pct', 2.0)),
            step=0.5,
            key="settings_max_position"
        )

        kelly_fraction = st.slider(
            "Kelly Fraction",
            min_value=0.1,
            max_value=1.0,
            value=float(prop_config.get('kelly_fraction', 0.5)),
            step=0.1,
            help="0.5 = Half-Kelly (recommended)",
            key="settings_kelly"
        )

    # Trading parameters row
    col_tp1, col_tp2 = st.columns(2)

    with col_tp1:
        max_concurrent = st.number_input(
            "Max Concurrent Positions",
            min_value=1,
            max_value=10,
            value=int(prop_config.get('max_concurrent', 3)),
            key="settings_max_concurrent"
        )

    with col_tp2:
        primary_timeframe = st.selectbox(
            "Primary Timeframe",
            ["1H", "4H", "1D"],
            index=["1H", "4H", "1D"].index(prop_config.get('primary_timeframe', '4H')),
            key="settings_timeframe"
        )

    # Save Prop Firm Config button
    if st.button("Save Prop Firm Config", type="primary", key="save_prop_firm"):
        new_prop_config = {
            'firm_name': firm_name,
            'account_size': account_size,
            'profit_target_pct': profit_target,
            'max_daily_dd_pct': max_daily_dd,
            'max_total_dd_pct': max_total_dd,
            'min_trading_days': min_trading_days,
            'max_position_size_pct': max_position_size,
            'consistency_rule': consistency_rule,
            'kelly_fraction': kelly_fraction,
            'max_concurrent': max_concurrent,
            'primary_timeframe': primary_timeframe,
            'primary_symbols': prop_config.get('primary_symbols', ["BTC/USDT", "ETH/USDT"])
        }

        if save_prop_firm_config(new_prop_config):
            st.success("Prop firm configuration saved!")

    # Show current PropFirmRules for reference
    with st.expander("View PropFirmRules Object"):
        rules_dict = {
            "account_size": account_size,
            "profit_target_pct": profit_target,
            "daily_loss_limit_pct": max_daily_dd,
            "total_loss_limit_pct": max_total_dd,
            "min_trading_days": min_trading_days,
            "max_position_size_pct": max_position_size,
            "consistency_rule": consistency_rule
        }
        st.json(rules_dict)

    # === DETECTION PARAMETERS ===
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        det_panel = '<div class="panel">'
        det_panel += '<div class="panel-header">Detection Parameters</div>'
        det_panel += '</div>'
        st.markdown(det_panel, unsafe_allow_html=True)

        detection_config = yaml_config.get('detection', {})

        atr_multiplier = st.number_input(
            "ATR Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=float(detection_config.get('atr_multiplier', 1.5)),
            step=0.1,
            key="settings_atr_mult"
        )

        min_validity = st.slider(
            "Min Validity Score",
            min_value=0.0,
            max_value=1.0,
            value=float(detection_config.get('min_validity', 0.6)),
            step=0.05,
            key="settings_min_validity"
        )

        lookback_bars = st.number_input(
            "Lookback Bars",
            min_value=50,
            max_value=500,
            value=int(detection_config.get('lookback_bars', 100)),
            step=10,
            key="settings_lookback"
        )

    with col2:
        trade_panel = '<div class="panel">'
        trade_panel += '<div class="panel-header">Trade Parameters</div>'
        trade_panel += '</div>'
        st.markdown(trade_panel, unsafe_allow_html=True)

        trade_config = yaml_config.get('trade', {})

        risk_reward = st.number_input(
            "Target R:R Ratio",
            min_value=1.0,
            max_value=5.0,
            value=float(trade_config.get('risk_reward', 2.0)),
            step=0.1,
            key="settings_rr"
        )

        max_risk_pct = st.slider(
            "Max Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=float(trade_config.get('max_risk_pct', 2.0)),
            step=0.5,
            key="settings_max_risk"
        )

        sl_buffer_pct = st.number_input(
            "SL Buffer (%)",
            min_value=0.0,
            max_value=2.0,
            value=float(trade_config.get('sl_buffer_pct', 0.1)),
            step=0.05,
            key="settings_sl_buffer"
        )

    # Save Detection/Trade Config button
    if st.button("Save Detection/Trade Settings", type="secondary", key="save_detection"):
        new_yaml_config = {
            'detection': {
                'atr_multiplier': atr_multiplier,
                'min_validity': min_validity,
                'lookback_bars': lookback_bars,
            },
            'trade': {
                'risk_reward': risk_reward,
                'max_risk_pct': max_risk_pct,
                'sl_buffer_pct': sl_buffer_pct,
            }
        }

        yaml_config.update(new_yaml_config)

        if save_yaml_config(yaml_config):
            st.success("Detection/Trade settings saved!")

    # Config file locations
    st.markdown("---")
    info_html = f'<div style="color: {ARCTIC_PRO["text_muted"]}; font-size: {TYPOGRAPHY["size_sm"]};">'
    info_html += '<p><strong>Config Files:</strong></p>'
    info_html += '<ul>'
    info_html += '<li>Prop Firm: <code>qml/dashboard/config/user_config.json</code></li>'
    info_html += '<li>Detection/Trade: <code>config/default.yaml</code></li>'
    info_html += '</ul>'
    info_html += '</div>'
    st.markdown(info_html, unsafe_allow_html=True)
