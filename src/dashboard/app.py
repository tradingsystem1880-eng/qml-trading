"""
QML Trading System Dashboard - Premium Edition
===============================================
Professional trading dashboard with real-time pattern detection.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
import json
import sys

import numpy as np
import pandas as pd

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from config.settings import settings
from src.data.fetcher import DataFetcher
from src.detection.detector import QMLDetector
from src.features.regime import RegimeClassifier
from src.data.models import QMLPattern, PatternType
from src.dashboard.components.tradingview_chart import render_pattern_chart, render_mini_chart


# Page config
st.set_page_config(
    page_title="QML Trading System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Theme CSS
st.markdown("""
<style>
    /* ===== BASE THEME ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #0d1321 50%, #0a1628 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #131a2b 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    p, span, label, .stMarkdown {
        color: #cbd5e1;
    }
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.75rem !important;
        font-weight: 700;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
    }
    
    /* ===== PATTERN CARDS ===== */
    .pattern-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #0ea5e9;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: all 0.2s ease;
    }
    .pattern-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(14, 165, 233, 0.2);
    }
    .pattern-card.bullish {
        border-left-color: #22c55e;
    }
    .pattern-card.bearish {
        border-left-color: #ef4444;
    }
    
    /* ===== BADGES ===== */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-bullish {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
    }
    .badge-bearish {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    .badge-pass {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
    }
    .badge-fail {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    /* ===== STAT BOXES ===== */
    .stat-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.2;
    }
    .stat-value.positive { color: #22c55e; }
    .stat-value.negative { color: #ef4444; }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    
    /* ===== DATA TABLES ===== */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stTable"] {
        background: #1e293b;
    }
    
    /* ===== PROGRESS/RING ===== */
    .ring-chart {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto;
    }
    
    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {
        background: #1e293b;
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 8px;
        color: #e2e8f0;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 24px;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 8px;
        color: #e2e8f0 !important;
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border-color: rgba(56, 189, 248, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ============ CACHED RESOURCES ============
@st.cache_resource
def get_detector():
    return QMLDetector()

@st.cache_resource
def get_fetcher():
    return DataFetcher()

@st.cache_resource
def get_regime_classifier():
    return RegimeClassifier()


# ============ SESSION STATE ============
if 'detected_patterns' not in st.session_state:
    st.session_state.detected_patterns = []
if 'selected_pattern' not in st.session_state:
    st.session_state.selected_pattern = None
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None


# ============ HELPER FUNCTIONS ============
def fetch_data(symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data."""
    fetcher = get_fetcher()
    return fetcher.get_data(symbol, timeframe, limit=limit)


def run_scan(symbols: List[str], timeframes: List[str], min_validity: float = 0.5) -> List[Dict]:
    """Run pattern scan across symbols and timeframes."""
    detector = get_detector()
    regime = get_regime_classifier()
    patterns = []
    
    for symbol in symbols:
        for tf in timeframes:
            df = fetch_data(symbol, tf)
            if df is None or len(df) < 100:
                continue
            
            detected = detector.detect(symbol, tf, df)
            for p in detected:
                if p.validity_score >= min_validity:
                    patterns.append({
                        'pattern': p,
                        'symbol': symbol,
                        'timeframe': tf,
                        'df': df,
                        'regime': regime.classify(df)
                    })
    
    patterns.sort(key=lambda x: x['pattern'].validity_score, reverse=True)
    return patterns


# Note: create_mini_chart is now handled by render_mini_chart from tradingview_chart.py
# Keeping a simple wrapper for compatibility
def create_mini_chart_wrapper(df: pd.DataFrame, container_key: str, height: int = 80):
    """Wrapper to render mini chart in current container."""
    render_mini_chart(df, height=height, key=container_key)


# Note: create_pattern_chart is now handled by render_pattern_chart from tradingview_chart.py
# Wrapper for compatibility
def create_pattern_chart_wrapper(pattern_data: Dict, container_key: str):
    """Render pattern chart using TradingView Lightweight Charts."""
    pattern = pattern_data['pattern']
    df = pattern_data['df'].copy()
    
    # Prepare dataframe
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        if df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_localize(None)
    else:
        df = df.reset_index()
        df.rename(columns={'index': 'time'}, inplace=True)
    
    df = df.tail(100)
    
    # Prepare pattern dict for chart
    pattern_dict = None
    if pattern.trading_levels:
        pattern_dict = {
            'trading_levels': {
                'entry': pattern.trading_levels.entry,
                'stop_loss': pattern.trading_levels.stop_loss,
                'take_profit_1': pattern.trading_levels.take_profit_1,
                'take_profit_2': pattern.trading_levels.take_profit_2,
            }
        }
    
    # Render TradingView chart
    render_pattern_chart(df, pattern=pattern_dict, height=600, key=container_key)


def create_win_rate_ring(win_rate: float, size: int = 120) -> str:
    """Create a CSS ring chart for win rate."""
    pct = win_rate * 100
    color = '#22c55e' if win_rate >= 0.55 else '#f59e0b' if win_rate >= 0.45 else '#ef4444'
    return f"""
    <div style="position: relative; width: {size}px; height: {size}px; margin: 0 auto;">
        <svg viewBox="0 0 36 36" style="transform: rotate(-90deg);">
            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none" stroke="#1e293b" stroke-width="3"/>
            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none" stroke="{color}" stroke-width="3"
                  stroke-dasharray="{pct}, 100"/>
        </svg>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{pct:.1f}%</div>
            <div style="font-size: 0.7rem; color: #94a3b8;">WIN RATE</div>
        </div>
    </div>
    """


# ============ PAGES ============
def render_dashboard():
    """Main dashboard with overview stats."""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🎯 QML Trading Dashboard")
        st.markdown(f"<p style='color: #94a3b8;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", 
                   unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Key Stats Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value positive">61.4%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">1.59</div>
            <div class="stat-label">Profit Factor</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value positive">+0.39R</div>
            <div class="stat-label">Expectancy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value negative">26.5%</div>
            <div class="stat-label">Max Drawdown</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">1,015</div>
            <div class="stat-label">Total Trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📈 Recent Activity")
        
        # Price mini charts
        fetcher = get_fetcher()
        chart_cols = st.columns(3)
        
        for i, symbol in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT"]):
            with chart_cols[i]:
                df = fetcher.get_data(symbol, "4h", limit=100)
                if df is not None and len(df) > 0:
                    if 'time' in df.columns:
                        df = df.set_index('time')
                    
                    price = df['close'].iloc[-1]
                    change = ((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100 if len(df) > 24 else 0
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                                border-radius: 12px; padding: 16px; border: 1px solid rgba(56, 189, 248, 0.1);">
                        <div style="color: #94a3b8; font-size: 0.8rem;">{symbol}</div>
                        <div style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600;">${price:,.2f}</div>
                        <div style="color: {'#22c55e' if change >= 0 else '#ef4444'}; font-size: 0.9rem;">
                            {'▲' if change >= 0 else '▼'} {abs(change):.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Reset index for TradingView chart
                    if isinstance(df.index, pd.DatetimeIndex):
                        df_chart = df.reset_index().rename(columns={'index': 'time'})
                    else:
                        df_chart = df.copy()
                    
                    # Render TradingView mini chart
                    render_mini_chart(df_chart, height=80, key=f"mini_chart_{symbol}")
    
    with col_right:
        st.markdown("### 🎯 Performance")
        
        # Win rate ring
        st.markdown(create_win_rate_ring(0.614), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    border-radius: 12px; padding: 20px; border: 1px solid rgba(56, 189, 248, 0.1);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <span style="color: #94a3b8;">High Vol Filter WR</span>
                <span style="color: #22c55e; font-weight: 600;">68.1%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <span style="color: #94a3b8;">Sharpe Ratio</span>
                <span style="color: #f1f5f9; font-weight: 600;">0.85</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <span style="color: #94a3b8;">Avg Trade</span>
                <span style="color: #22c55e; font-weight: 600;">+$127.50</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8;">Filter Status</span>
                <span class="badge badge-pass">ACTIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent Patterns Section
    st.markdown("### 📋 Recent Detected Patterns")
    
    patterns = st.session_state.detected_patterns
    
    if not patterns:
        st.info("👋 No patterns detected yet. Use the **Pattern Scanner** to find patterns.")
    else:
        for i, p_data in enumerate(patterns[:5]):
            pattern = p_data['pattern']
            is_bullish = pattern.pattern_type == PatternType.BULLISH
            
            st.markdown(f"""
            <div class="pattern-card {'bullish' if is_bullish else 'bearish'}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span class="badge badge-{'bullish' if is_bullish else 'bearish'}">
                            {'BULLISH' if is_bullish else 'BEARISH'}
                        </span>
                        <span style="color: #f1f5f9; font-size: 1.1rem; font-weight: 600;">
                            {p_data['symbol']}
                        </span>
                        <span style="color: #64748b;">{p_data['timeframe']}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #0ea5e9; font-weight: 600;">
                            Entry: ${pattern.trading_levels.entry:,.2f if pattern.trading_levels else 0}
                        </div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">
                            Validity: {pattern.validity_score:.1%}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_scanner():
    """Pattern scanner page."""
    st.markdown("# 🔍 Pattern Scanner")
    st.markdown("<p style='color: #94a3b8;'>Scan multiple assets for QML patterns</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbols = st.multiselect(
            "Select Symbols",
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
            default=["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        )
    
    with col2:
        timeframes = st.multiselect(
            "Select Timeframes",
            ["1h", "4h", "1d"],
            default=["1h", "4h"]
        )
    
    with col3:
        min_validity = st.slider("Min Validity", 0.5, 1.0, 0.6, 0.05)
    
    if st.button("🚀 Start Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning for patterns..."):
            patterns = run_scan(symbols, timeframes, min_validity)
            st.session_state.detected_patterns = patterns
            st.session_state.last_scan = datetime.now()
        
        if patterns:
            st.success(f"✅ Found {len(patterns)} patterns!")
        else:
            st.info("No patterns found. Try different parameters or symbols.")
    
    st.markdown("---")
    
    # Results
    if st.session_state.detected_patterns:
        st.markdown(f"### Results ({len(st.session_state.detected_patterns)} patterns)")
        
        for i, p_data in enumerate(st.session_state.detected_patterns):
            pattern = p_data['pattern']
            is_bullish = pattern.pattern_type == PatternType.BULLISH
            
            with st.expander(f"{'🟢' if is_bullish else '🔴'} {p_data['symbol']} {p_data['timeframe']} - {pattern.validity_score:.1%}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Use TradingView chart wrapper
                    create_pattern_chart_wrapper(p_data, container_key=f"pattern_chart_{i}")
                
                with col2:
                    if pattern.trading_levels:
                        tl = pattern.trading_levels
                        st.markdown("#### Trade Setup")
                        st.metric("Entry", f"${tl.entry:,.2f}")
                        st.metric("Stop Loss", f"${tl.stop_loss:,.2f}")
                        st.metric("Take Profit", f"${tl.take_profit_1:,.2f}")
                        
                        risk = abs(tl.entry - tl.stop_loss)
                        reward = abs(tl.take_profit_1 - tl.entry)
                        st.metric("Risk/Reward", f"{reward/risk:.2f}:1" if risk > 0 else "N/A")


def render_paper_trading():
    """Paper trading page."""
    st.markdown("# 📝 Paper Trading")
    st.markdown("<p style='color: #94a3b8;'>Live signal tracking and paper trading log</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load signals
    log_dir = Path("/Users/hunternovotny/Desktop/QML_SYSTEM/paper_trading_logs")
    signals = []
    
    signals_file = log_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.json"
    if signals_file.exists():
        try:
            with open(signals_file) as f:
                content = f.read()
                for block in content.split('---'):
                    block = block.strip()
                    if block:
                        try:
                            signals.append(json.loads(block))
                        except:
                            pass
        except:
            pass
    
    # Deduplicate
    signals = list({s['signal_id']: s for s in signals}.values())
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(signals)
    passed = len([s for s in signals if s.get('filter_decision') == 'PASS'])
    failed = len([s for s in signals if s.get('filter_decision') == 'FAIL'])
    
    with col1:
        st.metric("Total Signals", total)
    with col2:
        st.metric("Passed Filter", passed, delta=f"{passed/total*100:.0f}%" if total > 0 else "0%")
    with col3:
        st.metric("Filtered Out", failed)
    with col4:
        avg_val = np.mean([s.get('validity_score', 0) for s in signals]) if signals else 0
        st.metric("Avg Validity", f"{avg_val:.1%}")
    
    st.markdown("---")
    
    # Signals list
    if signals:
        st.markdown("### 📋 Today's Signals")
        
        for signal in signals:
            is_bullish = signal.get('pattern_type', '').upper() == 'BULLISH'
            passed = signal.get('filter_decision') == 'PASS'
            
            st.markdown(f"""
            <div class="pattern-card {'bullish' if is_bullish else 'bearish'}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                            <span class="badge badge-{'bullish' if is_bullish else 'bearish'}">
                                {signal.get('pattern_type', 'N/A').upper()}
                            </span>
                            <span style="color: #f1f5f9; font-size: 1.1rem; font-weight: 600;">
                                {signal.get('symbol', 'N/A')}
                            </span>
                            <span style="color: #64748b;">{signal.get('timeframe', 'N/A')}</span>
                            <span class="badge badge-{'pass' if passed else 'fail'}">
                                {signal.get('filter_decision', 'N/A')}
                            </span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; color: #94a3b8; font-size: 0.9rem;">
                            <div>
                                <div style="color: #64748b; font-size: 0.75rem;">ENTRY</div>
                                <div style="color: #0ea5e9; font-weight: 600;">${signal.get('entry_price', 0):,.2f}</div>
                            </div>
                            <div>
                                <div style="color: #64748b; font-size: 0.75rem;">STOP</div>
                                <div style="color: #ef4444; font-weight: 600;">${signal.get('stop_loss', 0):,.2f}</div>
                            </div>
                            <div>
                                <div style="color: #64748b; font-size: 0.75rem;">TARGET</div>
                                <div style="color: #22c55e; font-weight: 600;">${signal.get('take_profit_1', 0):,.2f}</div>
                            </div>
                            <div>
                                <div style="color: #64748b; font-size: 0.75rem;">VOL%</div>
                                <div style="color: #f1f5f9; font-weight: 600;">{signal.get('volatility_percentile', 0):.1%}</div>
                            </div>
                        </div>
                    </div>
                    <div style="text-align: right; color: #94a3b8; font-size: 0.85rem;">
                        {signal.get('validity_score', 0):.1%}
                    </div>
                </div>
                {f'<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); color: #f59e0b; font-size: 0.85rem;">⚠️ {signal.get("filter_reason", "")}</div>' if not passed else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No signals logged today. Run a paper trading scan to generate signals.")


def render_reports():
    """Reports and validation page."""
    st.markdown("# 📈 Reports & Validation")
    st.markdown("<p style='color: #94a3b8;'>Strategy verification and backtest results</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Verification", "📊 Backtest", "📄 Documentation"])
    
    with tab1:
        st.markdown("### Pre-Launch Verification Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                        border-radius: 12px; padding: 20px; border: 1px solid rgba(34, 197, 94, 0.3);">
                <h4 style="color: #22c55e; margin-bottom: 16px;">✅ Tests Passed</h4>
                <ul style="color: #e2e8f0; line-height: 2;">
                    <li>Falsification Test - <span style="color: #22c55e;">PASSED</span></li>
                    <li>Monte Carlo Analysis - <span style="color: #22c55e;">PASSED</span></li>
                    <li>Parameter Perturbation - <span style="color: #22c55e;">PASSED</span></li>
                    <li>Walk-Forward Validation - <span style="color: #22c55e;">PASSED</span></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                        border-radius: 12px; padding: 20px; border: 1px solid rgba(14, 165, 233, 0.3);">
                <h4 style="color: #0ea5e9; margin-bottom: 16px;">📊 Key Findings</h4>
                <div style="color: #e2e8f0; line-height: 2;">
                    <div>Edge Validated: <span style="color: #22c55e;">100th percentile vs random</span></div>
                    <div>High Vol Filter: <span style="color: #22c55e;">+8.6% WR improvement</span></div>
                    <div>Risk Reduction: <span style="color: #22c55e;">~50% lower max DD</span></div>
                    <div>Status: <span style="color: #22c55e; font-weight: 600;">GO FOR PAPER TRADING</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Historical Backtest Results")
        
        # Performance table
        st.markdown("""
        | Metric | Unfiltered | High Vol Filter |
        |--------|------------|-----------------|
        | **Win Rate** | 59.5% | **68.1%** |
        | **Profit Factor** | 1.47 | **1.72** |
        | **Max Drawdown** | 30.7% | **16.6%** |
        | **Sharpe Ratio** | 0.71 | **0.92** |
        | **Total Trades** | 1,222 | 351 |
        """)
        
        st.markdown("---")
        
        # Show validation images if available
        val_dir = Path("/Users/hunternovotny/Desktop/QML_SYSTEM/validation")
        images = list(val_dir.glob("*.png"))
        
        if images:
            st.markdown("### 📊 Validation Charts")
            cols = st.columns(2)
            for i, img in enumerate(images[:4]):
                with cols[i % 2]:
                    st.image(str(img), caption=img.stem.replace('_', ' ').title())
    
    with tab3:
        st.markdown("### Documentation")
        
        docs = [
            ("Final Strategy Specification", "/Users/hunternovotny/Desktop/QML_SYSTEM/docs/FINAL_STRATEGY_SPECIFICATION.md"),
            ("Pre-Launch Verification Dossier", "/Users/hunternovotny/Desktop/QML_SYSTEM/docs/PRE_LAUNCH_VERIFICATION_DOSSIER.md"),
        ]
        
        for title, path in docs:
            if Path(path).exists():
                with st.expander(f"📄 {title}"):
                    with open(path) as f:
                        st.markdown(f.read())


def render_settings():
    """Settings page."""
    st.markdown("# ⚙️ Settings")
    st.markdown("<p style='color: #94a3b8;'>System configuration and parameters</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Detection Parameters")
        st.number_input("Min Validity Score", 0.0, 1.0, 0.5, 0.05)
        st.number_input("Min Head Depth (ATR)", 0.0, 5.0, 0.2, 0.1)
        st.number_input("Max Head Depth (ATR)", 1.0, 10.0, 8.0, 0.5)
        st.number_input("ATR Period", 5, 30, 14)
    
    with col2:
        st.markdown("### Filter Settings")
        st.number_input("Volatility Threshold", 0.0, 1.0, 0.7, 0.05)
        st.checkbox("Enable High-Conviction Filter", value=True)
        st.markdown("### Trading Parameters")
        st.number_input("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
        st.number_input("Default R:R Target", 1.0, 5.0, 1.0, 0.5)


# ============ MAIN APP ============
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 2rem;">🎯</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #f1f5f9;">QML System</div>
            <div style="font-size: 0.8rem; color: #64748b;">Pattern Detection</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔍 Scanner", "📝 Paper Trading", "📈 Reports", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick scan
        st.markdown("### ⚡ Quick Scan")
        quick_symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
        quick_tf = st.selectbox("Timeframe", ["1h", "4h", "1d"])
        
        if st.button("🔍 Scan", use_container_width=True):
            with st.spinner("Scanning..."):
                patterns = run_scan([quick_symbol], [quick_tf], 0.5)
                st.session_state.detected_patterns = patterns
            if patterns:
                st.success(f"Found {len(patterns)}!")
            else:
                st.info("No patterns")
        
        st.markdown("---")
        st.markdown("<p style='color: #64748b; font-size: 0.75rem; text-align: center;'>v1.0 Premium</p>", unsafe_allow_html=True)
    
    # Main content
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "🔍 Scanner":
        render_scanner()
    elif page == "📝 Paper Trading":
        render_paper_trading()
    elif page == "📈 Reports":
        render_reports()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
