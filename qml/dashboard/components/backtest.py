"""
QML Dashboard Backtest Runner
=============================

Integration component for running backtests from the dashboard.

Usage:
    from qml.dashboard.components import BacktestRunner
    
    runner = BacktestRunner()
    results = runner.run(symbol="BTC/USDT", timeframe="4h", days=365)
    runner.display_results(results)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from loguru import logger

# Project root for imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BacktestResult:
    """Container for backtest results."""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    trades: Optional[pd.DataFrame] = None
    equity_curve: Optional[pd.DataFrame] = None


class BacktestRunner:
    """
    Dashboard integration for running backtests.
    
    Wraps the existing backtesting infrastructure for use in Streamlit.
    """
    
    def __init__(self):
        """Initialize the backtest runner."""
        self.results_dir = PROJECT_ROOT / "results" / "backtests"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._engine = None
        self._strategy = None
    
    def _load_backtest_engine(self):
        """Lazy load the backtest engine."""
        if self._engine is None:
            try:
                from qml.backtest.engine import BacktestEngine
                self._engine = BacktestEngine()
                logger.info("Backtest engine loaded")
            except ImportError as e:
                logger.error(f"Could not load backtest engine: {e}")
                return None
        return self._engine
    
    def run(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "4h",
        days: int = 365,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.02,
        show_progress: bool = True
    ) -> Optional[BacktestResult]:
        """
        Run a backtest with the given parameters.
        
        Args:
            symbol: Trading pair
            timeframe: Chart timeframe
            days: Historical data period
            initial_capital: Starting capital
            risk_per_trade: Risk per trade (fraction)
            show_progress: Show progress bar
            
        Returns:
            BacktestResult or None if failed
        """
        engine = self._load_backtest_engine()
        if not engine:
            st.error("Backtest engine not available")
            return None
        
        if show_progress:
            progress = st.progress(0, text="Loading data...")
        
        try:
            # Run the backtest
            if show_progress:
                progress.progress(20, text="Fetching historical data...")
            
            result = engine.run(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade
            )
            
            if show_progress:
                progress.progress(80, text="Processing results...")
            
            # Convert to BacktestResult
            backtest_result = BacktestResult(
                symbol=symbol,
                timeframe=timeframe,
                start_date=result.start_date.strftime("%Y-%m-%d") if hasattr(result, 'start_date') else "N/A",
                end_date=result.end_date.strftime("%Y-%m-%d") if hasattr(result, 'end_date') else "N/A",
                initial_capital=initial_capital,
                final_equity=result.final_equity if hasattr(result, 'final_equity') else initial_capital,
                total_return=result.total_return if hasattr(result, 'total_return') else 0,
                sharpe_ratio=result.sharpe_ratio if hasattr(result, 'sharpe_ratio') else 0,
                max_drawdown=result.max_drawdown if hasattr(result, 'max_drawdown') else 0,
                win_rate=result.win_rate if hasattr(result, 'win_rate') else 0,
                total_trades=result.total_trades if hasattr(result, 'total_trades') else 0,
                profit_factor=result.profit_factor if hasattr(result, 'profit_factor') else 0,
                trades=result.trades if hasattr(result, 'trades') else None,
                equity_curve=result.equity_curve if hasattr(result, 'equity_curve') else None
            )
            
            if show_progress:
                progress.progress(100, text="Complete!")
                progress.empty()
            
            logger.info(f"Backtest complete: {backtest_result.total_trades} trades")
            return backtest_result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            if show_progress:
                progress.empty()
            st.error(f"Backtest failed: {e}")
            return None
    
    def display_results(self, result: BacktestResult) -> None:
        """
        Display backtest results in the dashboard.
        
        Args:
            result: BacktestResult object
        """
        if result is None:
            st.warning("No backtest results to display")
            return
        
        # Summary metrics
        st.subheader(f"📊 {result.symbol} Backtest Results")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{result.total_return:.1%}")
        col2.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        col3.metric("Win Rate", f"{result.win_rate:.1%}")
        col4.metric("Max Drawdown", f"{result.max_drawdown:.1%}")
        col5.metric("Total Trades", result.total_trades)
        
        st.divider()
        
        # Equity curve
        if result.equity_curve is not None:
            st.subheader("📈 Equity Curve")
            st.line_chart(result.equity_curve, use_container_width=True)
        
        # Trade details
        if result.trades is not None and len(result.trades) > 0:
            st.subheader("📋 Trade History")
            st.dataframe(result.trades, use_container_width=True, hide_index=True)
    
    def get_saved_results(self) -> List[Path]:
        """Get list of saved backtest results."""
        return sorted(
            self.results_dir.glob("*.csv"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
    
    def load_saved_result(self, path: Path) -> Optional[pd.DataFrame]:
        """Load a saved backtest result."""
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Could not load {path}: {e}")
            return None


def render_backtest_page():
    """Render the backtest page in the dashboard."""
    st.title("📈 Backtest Runner")
    st.caption("Run and analyze strategy backtests")
    
    runner = BacktestRunner()
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
        with c2:
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
        with c3:
            days = st.slider("History (days)", 90, 730, 365)
        
        c4, c5 = st.columns(2)
        with c4:
            initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)
        with c5:
            risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
    
    with col2:
        st.subheader("🎯 Actions")
        
        if st.button("🚀 Run Backtest", use_container_width=True, type="primary"):
            result = runner.run(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade
            )
            
            if result:
                st.session_state.backtest_result = result
                st.success(f"✅ Backtest complete: {result.total_trades} trades")
                st.rerun()
        
        # Saved results
        saved = runner.get_saved_results()
        if saved:
            st.selectbox("📂 Load Previous", [p.stem for p in saved[:10]])
    
    st.divider()
    
    # Display results
    if "backtest_result" in st.session_state:
        runner.display_results(st.session_state.backtest_result)
    else:
        st.info("Configure parameters above and click 'Run Backtest'")
