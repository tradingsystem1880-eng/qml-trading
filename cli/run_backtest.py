#!/usr/bin/env python3
"""
CLI Backtest Runner
===================
Command-line entry point for running backtests.

This module wires the Detection "Brain" to the Backtest "Body":
1. Loads price data from parquet files
2. Runs pattern detection using the selected detector
3. Simulates trades based on signals
4. Reports performance metrics

Usage:
    python -m cli.run_backtest
    python -m cli.run_backtest --symbol BTCUSDT --timeframe 4h
    python -m cli.run_backtest --detector atr --config config/strategies/qml_bullish.yaml
"""

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import Signal, Trade, TradeResult, Side, SignalType
from src.detection import get_detector, list_available_detectors
from src.reporting.storage import ExperimentLogger
from src.reporting.dossier import DossierGenerator
from src.data.integrity import DataValidator


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    
    # Data settings
    symbol: str = "BTCUSDT"
    timeframe: str = "4h"
    data_path: Optional[str] = None  # If None, auto-detect
    
    # Date range (None = use all data)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Capital and position sizing
    initial_capital: float = 10000.0
    position_size_pct: float = 0.1  # 10% of capital per trade
    
    # Risk management
    use_stop_loss: bool = True
    use_take_profit: bool = True
    max_concurrent_trades: int = 1
    
    # Detection settings
    detector_method: str = "atr"  # "atr" or "rolling_window"
    min_validity_score: float = 0.7
    
    # Commission and slippage
    commission_pct: float = 0.1  # 0.1%
    slippage_pct: float = 0.05  # 0.05%


# =============================================================================
# SIMPLE BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Simple backtesting engine.
    
    This is a clean, minimal implementation that:
    1. Takes signals from the detector
    2. Simulates trade execution with SL/TP
    3. Tracks equity curve and performance
    
    The engine is intentionally simple and stateless per-run
    to ensure reproducibility.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        
        # State
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_capital = config.initial_capital
        self.open_trade: Optional[Trade] = None
    
    def run(self, df: pd.DataFrame, signals: List[Signal]) -> Dict[str, Any]:
        """
        Run backtest on data with given signals.
        
        Args:
            df: OHLCV DataFrame with 'time', 'open', 'high', 'low', 'close', 'volume'
            signals: List of Signal objects from detector
        
        Returns:
            Dictionary with backtest results
        """
        self._reset()
        
        # Sort signals by timestamp
        signals = sorted(signals, key=lambda s: s.timestamp)
        
        # Build signal lookup by timestamp
        signal_map: Dict[datetime, Signal] = {}
        for sig in signals:
            ts = sig.timestamp
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            signal_map[ts] = sig
        
        # Iterate through each bar
        for idx, row in df.iterrows():
            bar_time = row['time']
            if hasattr(bar_time, 'to_pydatetime'):
                bar_time = bar_time.to_pydatetime()
            
            # Check for exit on open trade
            if self.open_trade is not None:
                exit_price, exit_reason = self._check_exit(
                    self.open_trade, row['high'], row['low'], row['close']
                )
                
                if exit_price is not None:
                    self._close_trade(bar_time, exit_price, exit_reason)
            
            # Check for new signal
            if bar_time in signal_map and self.open_trade is None:
                signal = signal_map[bar_time]
                
                # Only take signals with sufficient validity
                if signal.validity_score >= self.config.min_validity_score:
                    self._open_trade(signal, row['close'])
            
            # Record equity
            equity = self._calculate_equity(row['close'])
            self.equity_curve.append((bar_time, equity))
        
        # Close any remaining open trade at last bar
        if self.open_trade is not None:
            last_row = df.iloc[-1]
            last_time = last_row['time']
            if hasattr(last_time, 'to_pydatetime'):
                last_time = last_time.to_pydatetime()
            self._close_trade(last_time, last_row['close'], 'end_of_data')
        
        return self._calculate_metrics()
    
    def _reset(self) -> None:
        """Reset engine state for new run."""
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        self.open_trade = None
    
    def _open_trade(self, signal: Signal, current_price: float) -> None:
        """
        Open a new trade based on signal.
        
        Args:
            signal: Signal that triggered the trade
            current_price: Current bar close price
        """
        # Use signal price or current price
        entry_price = signal.price if signal.price else current_price
        
        # Apply slippage
        slippage = entry_price * (self.config.slippage_pct / 100)
        if signal.signal_type == SignalType.BUY:
            entry_price += slippage  # Pay more for longs
            side = Side.LONG
        else:
            entry_price -= slippage  # Get less for shorts
            side = Side.SHORT
        
        # Calculate position size
        position_value = self.current_capital * self.config.position_size_pct
        quantity = position_value / entry_price
        
        # Create trade
        self.open_trade = Trade(
            entry_time=signal.timestamp,
            entry_price=entry_price,
            side=side,
            quantity=quantity,
            position_value=position_value,
            stop_loss=signal.stop_loss if self.config.use_stop_loss else None,
            take_profit=signal.take_profit if self.config.use_take_profit else None,
            symbol=signal.symbol or self.config.symbol,
            timeframe=signal.timeframe or self.config.timeframe,
            strategy_name=signal.strategy_name,
            signal_id=signal.pattern_id,
            pattern_type=signal.pattern_type,
            commission=self.config.commission_pct,
            metadata=signal.metadata,
        )
    
    def _check_exit(
        self, 
        trade: Trade, 
        high: float, 
        low: float, 
        close: float
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Check if trade should be exited this bar.
        
        Args:
            trade: Open trade
            high: Bar high
            low: Bar low
            close: Bar close
        
        Returns:
            (exit_price, exit_reason) or (None, None) if no exit
        """
        if trade.side == Side.LONG:
            # Check stop loss (hit if price goes below SL)
            if trade.stop_loss and low <= trade.stop_loss:
                return trade.stop_loss, 'stop_loss'
            
            # Check take profit (hit if price goes above TP)
            if trade.take_profit and high >= trade.take_profit:
                return trade.take_profit, 'take_profit'
        
        else:  # SHORT
            # Check stop loss (hit if price goes above SL)
            if trade.stop_loss and high >= trade.stop_loss:
                return trade.stop_loss, 'stop_loss'
            
            # Check take profit (hit if price goes below TP)
            if trade.take_profit and low <= trade.take_profit:
                return trade.take_profit, 'take_profit'
        
        return None, None
    
    def _close_trade(self, exit_time: datetime, exit_price: float, reason: str) -> None:
        """
        Close the open trade.
        
        Args:
            exit_time: Time of exit
            exit_price: Exit price
            reason: Exit reason (stop_loss, take_profit, end_of_data)
        """
        if self.open_trade is None:
            return
        
        # Apply slippage on exit
        slippage = exit_price * (self.config.slippage_pct / 100)
        if self.open_trade.side == Side.LONG:
            exit_price -= slippage  # Get less when closing long
        else:
            exit_price += slippage  # Pay more when closing short
        
        # Close the trade (calculates P&L)
        self.open_trade.close(exit_time, exit_price)
        self.open_trade.metadata['exit_reason'] = reason
        
        # Update capital
        if self.open_trade.pnl_usd:
            self.current_capital += self.open_trade.pnl_usd
        
        # Store trade
        self.trades.append(self.open_trade)
        self.open_trade = None
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open position."""
        equity = self.current_capital
        
        if self.open_trade is not None:
            # Mark to market
            entry = self.open_trade.entry_price
            qty = self.open_trade.quantity
            
            if self.open_trade.side == Side.LONG:
                unrealized = (current_price - entry) * qty
            else:
                unrealized = (entry - current_price) * qty
            
            equity += unrealized
        
        return equity
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with all performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'net_profit': 0,
                'net_profit_pct': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'initial_capital': self.config.initial_capital,
                'final_equity': self.config.initial_capital,
                'trades': [],
                'equity_curve': self.equity_curve,
            }
        
        # Basic stats
        total_trades = len(self.trades)
        winners = [t for t in self.trades if t.result == TradeResult.WIN]
        losers = [t for t in self.trades if t.result == TradeResult.LOSS]
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        
        # P&L
        total_pnl = sum(t.pnl_usd for t in self.trades if t.pnl_usd)
        gross_profit = sum(t.pnl_usd for t in winners if t.pnl_usd)
        gross_loss = abs(sum(t.pnl_usd for t in losers if t.pnl_usd))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns for Sharpe
        returns = [t.pnl_pct for t in self.trades if t.pnl_pct is not None]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0
            sharpe = (avg_return / std_return) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Drawdown
        equity_values = [e[1] for e in self.equity_curve]
        if equity_values:
            running_max = np.maximum.accumulate(equity_values)
            drawdowns = (running_max - equity_values) / running_max
            max_drawdown = np.max(drawdowns) * 100  # As percentage
        else:
            max_drawdown = 0
        
        # Final equity
        final_equity = self.equity_curve[-1][1] if self.equity_curve else self.config.initial_capital
        net_profit = final_equity - self.config.initial_capital
        net_profit_pct = (net_profit / self.config.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': round(win_rate * 100, 2),
            'net_profit': round(net_profit, 2),
            'net_profit_pct': round(net_profit_pct, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2),
            'avg_win': round(np.mean([t.pnl_pct for t in winners if t.pnl_pct]) if winners else 0, 2),
            'avg_loss': round(np.mean([t.pnl_pct for t in losers if t.pnl_pct]) if losers else 0, 2),
            'initial_capital': self.config.initial_capital,
            'final_equity': round(final_equity, 2),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: BacktestConfig) -> pd.DataFrame:
    """
    Load price data from parquet file.
    
    Args:
        config: Backtest configuration
    
    Returns:
        DataFrame with OHLCV data
    """
    # Auto-detect data path if not specified
    if config.data_path:
        data_path = Path(config.data_path)
    else:
        # Normalize symbol: "BTCUSDT" -> "BTCUSDT", "BTC/USDT" -> "BTCUSDT"
        symbol_normalized = config.symbol.replace('/', '').replace('-', '').upper()
        
        # Look in standard locations
        possible_paths = [
            PROJECT_ROOT / f"data/processed/{symbol_normalized}/{config.timeframe}_master.parquet",
            # Backward compatibility: try BTC folder for BTCUSDT
            PROJECT_ROOT / f"data/processed/BTC/{config.timeframe}_master.parquet" if 'BTC' in symbol_normalized else None,
            PROJECT_ROOT / f"data/{symbol_normalized}_{config.timeframe}.parquet",
        ]
        
        # Filter out None paths
        possible_paths = [p for p in possible_paths if p is not None]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"No data found for {config.symbol} {config.timeframe}. "
                f"Searched: {[str(p) for p in possible_paths]}\n"
                f"Run: python -m src.data_engine --symbol {config.symbol} to fetch data."
            )
    
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Standardize column names (handle capitalized and various formats)
    column_map = {
        'timestamp': 'time',
        'datetime': 'time',
        'date': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'OPEN': 'open',
        'HIGH': 'high',
        'LOW': 'low',
        'CLOSE': 'close',
        'VOLUME': 'volume',
    }
    df.rename(columns=column_map, inplace=True)
    
    # Ensure time column is datetime
    if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # Filter by date range if specified
    if config.start_date:
        df = df[df['time'] >= config.start_date]
    if config.end_date:
        df = df[df['time'] <= config.end_date]
    
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"📊 Loaded {len(df)} bars: {df['time'].min()} to {df['time'].max()}")
    
    return df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_backtest(config: BacktestConfig) -> Dict[str, Any]:
    """
    Run a complete backtest.
    
    This is the main function that wires the "Brain" (Detector) to the "Body" (Engine).
    
    Args:
        config: Backtest configuration
    
    Returns:
        Backtest results dictionary
    """
    print("\n" + "=" * 70)
    print("  🧪 QML BACKTEST RUNNER")
    print("=" * 70)
    print(f"  Symbol:    {config.symbol}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Detector:  {config.detector_method}")
    print(f"  Capital:   ${config.initial_capital:,.2f}")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n📥 Step 1: Loading Data...")
    df = load_data(config)
    
    # Step 1b: Data integrity check (THE DATA DOCTOR)
    print("\n🩺 Step 1b: Data Health Check...")
    validator = DataValidator()
    health_report = validator.check_health(df, timeframe=config.timeframe)
    print(f"   {health_report}")
    
    if health_report.status == 'fail':
        print("   ⚠️  WARNING: Data integrity issues detected!")
        print("   Proceeding anyway... (use --strict to abort on failure)")
    
    # Step 2: Initialize detector (THE BRAIN)
    print(f"\n🧠 Step 2: Initializing Detector ({config.detector_method})...")
    detector = get_detector(config.detector_method, {
        'min_validity_score': config.min_validity_score,
    })
    print(f"   Using: {detector}")
    
    # Step 3: Run detection
    print("\n🔍 Step 3: Running Pattern Detection...")
    signals = detector.detect(df, symbol=config.symbol, timeframe=config.timeframe)
    print(f"   Found {len(signals)} signals")
    
    if signals:
        buy_signals = len([s for s in signals if s.signal_type == SignalType.BUY])
        sell_signals = len([s for s in signals if s.signal_type == SignalType.SELL])
        print(f"   📈 BUY signals:  {buy_signals}")
        print(f"   📉 SELL signals: {sell_signals}")
    
    # Step 4: Run backtest engine (THE BODY)
    print("\n⚙️  Step 4: Running Backtest Engine...")
    engine = BacktestEngine(config)
    results = engine.run(df, signals)
    
    # Step 5: Display results
    print("\n" + "=" * 70)
    print("  📊 BACKTEST RESULTS")
    print("=" * 70)
    print(f"""
    Total Trades:     {results['total_trades']}
    Win Rate:         {results['win_rate']}%
    Net Profit:       ${results['net_profit']:,.2f} ({results['net_profit_pct']:+.2f}%)
    
    Winning Trades:   {results['winning_trades']}
    Losing Trades:    {results['losing_trades']}
    
    Gross Profit:     ${results['gross_profit']:,.2f}
    Gross Loss:       ${results['gross_loss']:,.2f}
    Profit Factor:    {results['profit_factor']}
    
    Average Win:      {results['avg_win']}%
    Average Loss:     {results['avg_loss']}%
    
    Sharpe Ratio:     {results['sharpe_ratio']}
    Max Drawdown:     {results['max_drawdown']}%
    
    Initial Capital:  ${results['initial_capital']:,.2f}
    Final Equity:     ${results['final_equity']:,.2f}
    """)
    print("=" * 70)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run QML pattern backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data options
    parser.add_argument(
        "--symbol", "-s",
        default="BTCUSDT",
        help="Trading symbol"
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="4h",
        help="Candle timeframe"
    )
    parser.add_argument(
        "--data-path",
        help="Path to parquet data file (auto-detect if not specified)"
    )
    
    # Date range
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD)"
    )
    
    # Capital
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=10000.0,
        help="Initial capital"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.1,
        help="Position size as fraction of capital (0.1 = 10%%)"
    )
    
    # Detection
    parser.add_argument(
        "--detector", "-d",
        default="atr",
        choices=["atr", "rolling_window", "v1", "v2"],
        help="Detection method"
    )
    parser.add_argument(
        "--min-validity",
        type=float,
        default=0.7,
        help="Minimum pattern validity score"
    )
    
    # ML Training
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train XGBoost model on trade outcomes"
    )
    
    args = parser.parse_args()
    
    # Build config
    config = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_path=args.data_path,
        start_date=datetime.fromisoformat(args.start_date) if args.start_date else None,
        end_date=datetime.fromisoformat(args.end_date) if args.end_date else None,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        detector_method=args.detector,
        min_validity_score=args.min_validity,
    )
    
    # Run backtest
    results = run_backtest(config)
    
    # Create trades DataFrame
    trades_df = None
    if results['trades']:
        trades_df = pd.DataFrame([t.to_dict() for t in results['trades']])
    
    # =========================================================================
    # FLIGHT RECORDER: Log to DB + Generate Dossier
    # =========================================================================
    print("\n📼 Step 5: Recording to Flight Recorder...")
    
    # Log to experiment database
    logger = ExperimentLogger()
    run_id = logger.log_run(config, results)
    run_count = logger.count_runs()
    print(f"   Logged run: {run_id} (Total experiments: {run_count})")
    
    # Generate HTML Dossier
    print("\n📋 Step 6: Generating Strategy Dossier...")
    generator = DossierGenerator()
    report_path = generator.generate_html(
        run_id=run_id,
        config=config,
        results=results,
        trades_df=trades_df,
        strategy_name=config.detector_method
    )
    print(f"   Report generated: {report_path}")
    
    # Update DB with report path
    with sqlite3.connect(logger.db_path) as conn:
        conn.execute(
            "UPDATE experiments SET report_path = ? WHERE run_id = ?",
            (report_path, run_id)
        )
        conn.commit()
    
    # Export trades CSV (keep for data analysis)
    if trades_df is not None:
        csv_path = PROJECT_ROOT / "results" / config.detector_method / f"{run_id}_trades.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(csv_path, index=False)
        print(f"   Trades exported: {csv_path}")
    
    # =========================================================================
    # ML TRAINING (optional)
    # =========================================================================
    if args.train_ml and trades_df is not None and len(trades_df) >= 20:
        print("\n🤖 Step 7: Training ML Model...")
        try:
            from src.ml.predictor import XGBoostPredictor
            
            predictor = XGBoostPredictor()
            X, y = predictor.prepare_data(trades_df)
            metrics = predictor.train(X, y)
            
            # Save model
            model_path = PROJECT_ROOT / "results" / "models" / "xgb_latest.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            predictor.save(str(model_path))
            
            print(f"   Model trained on {len(trades_df)} trades")
            print(f"   Accuracy: {metrics['accuracy']:.2%}, AUC: {metrics['auc']:.3f}")
            print(f"   Model saved: {model_path}")
            
            # Print top features
            print("   Top 5 Features:")
            for feat, imp in predictor.get_top_features(5):
                print(f"     - {feat}: {imp:.4f}")
        except ImportError:
            print("   ⚠️ XGBoost not installed: pip install xgboost scikit-learn")
        except Exception as e:
            print(f"   ⚠️ ML training failed: {e}")
    elif args.train_ml:
        print("\n⚠️ Skipping ML training: Need at least 20 trades")
    
    print("\n" + "=" * 70)
    print(f"  ✅ BACKTEST COMPLETE")
    print(f"  📊 Report: file://{report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
