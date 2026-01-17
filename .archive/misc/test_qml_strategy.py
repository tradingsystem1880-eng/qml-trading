#!/usr/bin/env python3
"""
QML Strategy Verification Script
=================================
Standalone test to verify the QMLStrategy works correctly with backtesting.py.

Success criteria:
- # Trades > 0
- Equity curve generated
- Plot saved to test_plot.html

Usage:
    python test_qml_strategy.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

from backtesting import Backtest
from src.strategies.qml_backtestingpy import QMLStrategy, prepare_backtesting_data
from src.data_engine import load_master_data


def main():
    print("=" * 60)
    print("QML STRATEGY VERIFICATION TEST")
    print("=" * 60)
    print()
    
    # 1. Load data (last 500 bars)
    print("1. Loading data...")
    df = load_master_data(timeframe='4h')
    df = df.tail(500).reset_index(drop=True)
    print(f"   Loaded {len(df)} bars")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    
    # 2. Prepare for backtesting.py
    print()
    print("2. Preparing data...")
    df = prepare_backtesting_data(df)
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index type: {type(df.index).__name__}")
    
    # 3. Run backtest
    print()
    print("3. Running backtest...")
    bt = Backtest(df, QMLStrategy, cash=100000, commission=0.001)
    stats = bt.run()
    
    # 4. Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"# Trades:        {stats['# Trades']}")
    print(f"Equity Final:    ${stats['Equity Final [$]']:,.2f}")
    print(f"Return:          {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio:    {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:    {stats['Max. Drawdown [%]']:.2f}%")
    
    if stats['# Trades'] > 0:
        print(f"Win Rate:        {stats['Win Rate [%]']:.2f}%")
        print(f"Profit Factor:   {stats['Profit Factor']:.2f}")
    
    print()
    
    # 5. Plot to HTML
    output_path = Path(__file__).parent / "test_plot.html"
    print(f"4. Saving plot to {output_path}...")
    bt.plot(filename=str(output_path), open_browser=False)
    print("   Done!")
    
    # 6. Verification status
    print()
    print("=" * 60)
    if stats['# Trades'] > 0:
        print("✅ VERIFICATION PASSED: Trades are being executed and closed!")
    else:
        print("❌ VERIFICATION FAILED: No trades recorded")
    print("=" * 60)
    
    return stats['# Trades'] > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
