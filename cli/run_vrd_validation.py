#!/usr/bin/env python3
"""
VRD Validation Runner for Backtesting.py
=========================================
Bridge script connecting the backtesting.py QMLStrategy to VRD 2.0 analytics.

This script:
1. Loads OHLCV data for a specified symbol
2. Runs parameter optimization via backtesting.py
3. Runs final backtest with best parameters
4. Converts trades to VRD format
5. Runs full VRD validation suite (Permutation, Monte Carlo, Bootstrap)
6. Generates HTML dossier report

Usage:
    python -m cli.run_vrd_validation
    python -m cli.run_vrd_validation --symbol BTC/USDT --timeframe 4h
    python -m cli.run_vrd_validation --quick  # Skip optimization for speed
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Suppress backtesting.py warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting import Backtest
from src.strategies.qml_backtestingpy import QMLStrategy, prepare_backtesting_data
from src.data_engine import load_master_data
from src.validation import run_validation_suite
from src.reporting.dossier import DossierGenerator


# =============================================================================
# TRADE CONVERSION
# =============================================================================

def extract_trade_returns(bt_stats) -> np.ndarray:
    """
    Extract trade returns from backtesting.py stats.
    
    Args:
        bt_stats: Stats object from bt.run()
        
    Returns:
        Array of percentage returns per trade
    """
    trades_df = bt_stats._trades
    
    if trades_df is None or len(trades_df) == 0:
        return np.array([])
    
    # backtesting.py stores return as 'ReturnPct'
    if 'ReturnPct' in trades_df.columns:
        returns = trades_df['ReturnPct'].values
    elif 'return_pct' in trades_df.columns:
        returns = trades_df['return_pct'].values
    else:
        # Calculate from PnL if available
        returns = trades_df['PnL'].values / trades_df['EntryPrice'].values * 100
    
    return returns


def convert_trades_for_dossier(bt_stats) -> pd.DataFrame:
    """
    Convert backtesting.py trades to format expected by DossierGenerator.
    
    Args:
        bt_stats: Stats object from bt.run()
        
    Returns:
        DataFrame with columns expected by dossier
    """
    trades_df = bt_stats._trades
    
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame()
    
    # Copy and keep reference to original column names
    df = trades_df.copy()
    original_columns = list(trades_df.columns)
    
    # Determine side before renaming
    if 'Size' in original_columns:
        df['side'] = trades_df['Size'].apply(lambda x: 'LONG' if x > 0 else 'SHORT')
    else:
        df['side'] = 'LONG'
    
    # Rename columns to match expected format
    column_map = {
        'EntryTime': 'entry_time',
        'ExitTime': 'exit_time',
        'EntryPrice': 'entry_price',
        'ExitPrice': 'exit_price',
        'PnL': 'pnl_usd',
        'ReturnPct': 'pnl_pct',
        'Size': 'quantity',
        'Tag': 'pattern_type',
    }
    
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    # Add missing columns
    if 'symbol' not in df.columns:
        df['symbol'] = 'BTCUSDT'
    
    return df


def create_backtest_result_dict(bt_stats, config: Dict) -> Dict[str, Any]:
    """
    Create result dictionary matching BacktestEngine format for VRD.
    
    Args:
        bt_stats: Stats object from bt.run()
        config: Configuration dictionary
        
    Returns:
        Dictionary compatible with VRD validation
    """
    trades_df = convert_trades_for_dossier(bt_stats)
    
    # Build equity curve from series
    equity_series = bt_stats._equity_curve['Equity']
    equity_curve = list(zip(equity_series.index, equity_series.values))
    
    return {
        'total_trades': int(bt_stats['# Trades']),
        'winning_trades': int(bt_stats['# Trades'] * bt_stats['Win Rate [%]'] / 100) if bt_stats['# Trades'] > 0 else 0,
        'losing_trades': int(bt_stats['# Trades'] * (100 - bt_stats['Win Rate [%]']) / 100) if bt_stats['# Trades'] > 0 else 0,
        'win_rate': float(bt_stats['Win Rate [%]']),
        'net_profit': float(bt_stats['Equity Final [$]'] - config.get('initial_capital', 100000)),
        'net_profit_pct': float(bt_stats['Return [%]']),
        'gross_profit': 0,  # Not directly available
        'gross_loss': 0,
        'profit_factor': float(bt_stats['Profit Factor']) if bt_stats['Profit Factor'] != float('inf') else 0,
        'sharpe_ratio': float(bt_stats['Sharpe Ratio']),
        'max_drawdown': float(bt_stats['Max. Drawdown [%]']),
        'avg_win': float(bt_stats['Best Trade [%]']) if pd.notna(bt_stats['Best Trade [%]']) else 0,
        'avg_loss': float(bt_stats['Worst Trade [%]']) if pd.notna(bt_stats['Worst Trade [%]']) else 0,
        'initial_capital': config.get('initial_capital', 100000),
        'final_equity': float(bt_stats['Equity Final [$]']),
        'trades': trades_df.to_dict('records') if len(trades_df) > 0 else [],
        'equity_curve': equity_curve,
    }


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def run_vrd_validation(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    initial_capital: float = 100000,
    quick_mode: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete VRD validation pipeline.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        initial_capital: Starting capital
        quick_mode: If True, skip optimization
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "=" * 70)
    print("  🔬 VRD 2.0 VALIDATION RUNNER")
    print("=" * 70)
    print(f"  Symbol:    {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Capital:   ${initial_capital:,.0f}")
    print(f"  Mode:      {'Quick (no optimization)' if quick_mode else 'Full (with optimization)'}")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n📥 Step 1: Loading Data...")
    df = load_master_data(timeframe, symbol=symbol)
    df = prepare_backtesting_data(df)
    print(f"   Loaded {len(df)} bars")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Step 2: Initialize backtest
    print("\n⚙️  Step 2: Initializing Backtest Engine...")
    bt = Backtest(df, QMLStrategy, cash=initial_capital, commission=0.001)
    
    # Step 3: Optimization (optional)
    best_params = {}
    if not quick_mode:
        print("\n🔍 Step 3: Running Parameter Optimization...")
        print("   Optimizing: atr_period, min_depth_ratio")
        print("   Maximizing: Sharpe Ratio")
        
        opt_stats = bt.optimize(
            atr_period=[10, 14, 20],
            min_depth_ratio=[0.3, 0.5, 0.7],
            maximize='Sharpe Ratio',
            return_heatmap=False,
        )
        
        best_params = {
            'atr_period': opt_stats._strategy.atr_period,
            'min_depth_ratio': opt_stats._strategy.min_depth_ratio,
        }
        
        print(f"\n   Best Parameters Found:")
        print(f"   - atr_period: {best_params['atr_period']}")
        print(f"   - min_depth_ratio: {best_params['min_depth_ratio']}")
        print(f"   - Sharpe Ratio: {opt_stats['Sharpe Ratio']:.2f}")
    else:
        print("\n⏩ Step 3: Skipping optimization (quick mode)")
        best_params = {'atr_period': 14, 'min_depth_ratio': 0.5}
    
    # Step 4: Final Backtest with OPTIMIZED parameters
    print("\n📊 Step 4: Running Final Backtest with OPTIMIZED Parameters...")
    
    # Initialize pattern registry for ML training
    from src.ml.pattern_registry import PatternRegistry
    from src.ml.feature_extractor import PatternFeatureExtractor
    
    registry = PatternRegistry()
    extractor = PatternFeatureExtractor()
    print(f"   Pattern registry initialized for ML training")
    
    # Show the parameters being used
    is_optimized = not quick_mode
    param_source = "OPTIMIZED" if is_optimized else "DEFAULT"
    
    print(f"\n   ╔{'═' * 50}╗")
    print(f"   ║  {param_source} PARAMETERS".ljust(51) + "║")
    print(f"   ╠{'═' * 50}╣")
    for param, value in best_params.items():
        print(f"   ║  {param}: {value}".ljust(51) + "║")
    print(f"   ╚{'═' * 50}╝\n")
    
    class OptimizedQML(QMLStrategy):
        atr_period = best_params.get('atr_period', 14)
        min_depth_ratio = best_params.get('min_depth_ratio', 0.5)
        # Enable pattern registration
        _register_patterns = True
        _pattern_registry = registry
        _feature_extractor = extractor
        _symbol = symbol
        _timeframe = timeframe
    
    bt_final = Backtest(df, OptimizedQML, cash=initial_capital, commission=0.001)
    final_stats = bt_final.run()
    
    print(f"   Backtest Results ({param_source} params):")
    print(f"   - # Trades:      {final_stats['# Trades']}")
    print(f"   - Win Rate:      {final_stats['Win Rate [%]']:.2f}%")
    print(f"   - Return:        {final_stats['Return [%]']:.2f}%")
    print(f"   - Sharpe Ratio:  {final_stats['Sharpe Ratio']:.2f}")
    print(f"   - Max Drawdown:  {final_stats['Max. Drawdown [%]']:.2f}%")
    
    # Check if we have enough trades
    trade_returns = extract_trade_returns(final_stats)
    if len(trade_returns) < 20:
        print(f"\n⚠️  WARNING: Only {len(trade_returns)} trades. VRD requires 20+ for statistical validity.")
    
    # Step 5: VRD Statistical Validation
    print("\n🧪 Step 5: Running VRD Statistical Validation...")
    
    if len(trade_returns) >= 5:
        # Use run_validation_suite for quick validation
        config = {
            'symbol': symbol,
            'timeframe': timeframe,
            'initial_capital': initial_capital,
            **best_params
        }
        
        backtest_result = create_backtest_result_dict(final_stats, config)
        
        validation_suite = run_validation_suite(
            backtest_result=backtest_result,
            trades=[{'pnl_pct': r} for r in trade_returns],
            validators=['permutation_test', 'monte_carlo', 'bootstrap'],
        )
        
        print(f"   Validation suite complete!")
        print(f"   {validation_suite}")
    else:
        print("   ⚠️  Not enough trades for validation")
        validation_suite = None
    
    # Step 6: Generate HTML Dossier
    print("\n📋 Step 6: Generating HTML Dossier...")
    
    run_id = f"vrd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    generator = DossierGenerator(output_dir=output_dir)
    trades_df = convert_trades_for_dossier(final_stats)
    
    # Config includes all parameters used
    config_dict = {
        'symbol': symbol,
        'timeframe': timeframe,
        'initial_capital': initial_capital,
        'optimization_mode': param_source,
        'atr_period': best_params.get('atr_period', 14),
        'min_depth_ratio': best_params.get('min_depth_ratio', 0.5),
        'max_depth_ratio': QMLStrategy.max_depth_ratio,
        'stop_loss_atr': QMLStrategy.stop_loss_atr,
        'take_profit_rr': QMLStrategy.take_profit_rr,
    }
    
    # Strategy name reflects optimization status
    strategy_name = f"QMLStrategy [{param_source}] ({symbol} {timeframe})"
    
    report_path = generator.generate_html(
        run_id=run_id,
        config=config_dict,
        results=backtest_result if len(trade_returns) >= 5 else create_backtest_result_dict(final_stats, config_dict),
        trades_df=trades_df if len(trades_df) > 0 else None,
        strategy_name=strategy_name,
        validation_suite=validation_suite,
    )
    
    print(f"   Report saved: {report_path}")
    
    # Final Summary with Optimal Parameters
    print("\n" + "=" * 70)
    print("  ✅ VRD VALIDATION COMPLETE")
    print("=" * 70)
    
    # Prominent optimal parameters section
    print(f"\n  ╔{'═' * 66}╗")
    print(f"  ║  OPTIMAL PARAMETERS FOUND ({param_source})".ljust(67) + "║")
    print(f"  ╠{'═' * 66}╣")
    for param, value in best_params.items():
        print(f"  ║    {param}: {value}".ljust(67) + "║")
    print(f"  ╚{'═' * 66}╝")
    
    print(f"\n  Performance Summary:")
    print(f"    Trades:       {len(trade_returns)}")
    print(f"    Win Rate:     {final_stats['Win Rate [%]']:.1f}%")
    print(f"    Return:       {final_stats['Return [%]']:.2f}%")
    print(f"    Sharpe:       {final_stats['Sharpe Ratio']:.2f}")
    
    print(f"\n  Report: file://{report_path}")
    print("=" * 70 + "\n")
    
    return {
        'run_id': run_id,
        'stats': final_stats,
        'best_params': best_params,
        'is_optimized': is_optimized,
        'trade_returns': trade_returns,
        'validation_suite': validation_suite,
        'report_path': report_path,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run VRD 2.0 validation on QML Strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--symbol', '-s',
        default='BTC/USDT',
        help='Trading pair symbol'
    )
    parser.add_argument(
        '--timeframe', '-t',
        default='4h',
        help='Candle timeframe'
    )
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=100000,
        help='Initial capital'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Skip optimization for faster execution'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_vrd_validation(
            symbol=args.symbol,
            timeframe=args.timeframe,
            initial_capital=args.capital,
            quick_mode=args.quick,
            output_dir=args.output_dir,
        )
        
        # Return success if we got results
        return 0 if result['stats']['# Trades'] > 0 else 1
        
    except FileNotFoundError as e:
        print(f"\n❌ Data not found: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
