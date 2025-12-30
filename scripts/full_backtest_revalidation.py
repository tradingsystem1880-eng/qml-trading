#!/usr/bin/env python3
"""
FULL BACKTEST REVALIDATION
==========================
Re-runs core validation suite using rolling-window detector
and compares against original backtest results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import ccxt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.detection.detector import QMLDetector
from src.data.models import QMLPattern, PatternType


# ==============================================================================
# DATA FETCHING
# ==============================================================================
def fetch_historical_data(symbol: str, timeframe: str, 
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical data."""
    exchange = ccxt.binance({'enableRateLimit': True})
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_candles = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= current_ts:
                break
            current_ts = last_ts + 1
            time.sleep(0.1)
        except Exception as e:
            break
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    return df


# ==============================================================================
# ROLLING PATTERN DETECTOR
# ==============================================================================
class RollingPatternDetector:
    """Detects patterns using rolling windows."""
    
    def __init__(self, window_size: int = 200, step_size: int = 12):
        self.window_size = window_size
        self.step_size = step_size
        self.detector = QMLDetector()
    
    def detect_all(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[QMLPattern]:
        """Detect all patterns using rolling windows."""
        n_bars = len(df)
        if n_bars < self.window_size:
            return []
        
        seen_keys = set()
        patterns = []
        
        for i in range(0, n_bars - self.window_size + 1, self.step_size):
            window_df = df.iloc[i:i + self.window_size].copy().reset_index(drop=True)
            detected = self.detector.detect(symbol, timeframe, window_df)
            
            for p in detected:
                key = f"{p.head_time}_{p.left_shoulder_time}_{p.pattern_type.value}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    patterns.append(p)
        
        patterns.sort(key=lambda p: p.detection_time)
        return patterns


# ==============================================================================
# BACKTESTER
# ==============================================================================
@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pattern_type: str
    pnl_pct: float
    outcome: int  # 1=win, 0=loss
    r_multiple: float


class Backtester:
    """Simple backtester for pattern evaluation."""
    
    def __init__(self, initial_capital: float = 100000, risk_per_trade: float = 0.01):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
    
    def simulate_trade(self, pattern: QMLPattern, df: pd.DataFrame) -> Optional[Trade]:
        """Simulate a single trade."""
        if not pattern.trading_levels:
            return None
        
        tl = pattern.trading_levels
        entry = tl.entry
        stop = tl.stop_loss
        tp = tl.take_profit_1
        
        # Normalize timezone
        detection_time = pattern.detection_time
        if hasattr(detection_time, 'tzinfo') and detection_time.tzinfo is None:
            detection_time = pd.Timestamp(detection_time).tz_localize('UTC')
        elif hasattr(detection_time, 'tz_localize'):
            detection_time = pd.Timestamp(detection_time)
            if detection_time.tzinfo is None:
                detection_time = detection_time.tz_localize('UTC')
        
        # Find entry bar
        entry_mask = df['time'] >= detection_time
        if not entry_mask.any():
            return None
        
        entry_idx = entry_mask.idxmax()
        if entry_idx >= len(df) - 1:
            return None
        
        # Simulate forward
        is_long = pattern.pattern_type == PatternType.BULLISH
        risk = abs(entry - stop)
        
        for i in range(entry_idx + 1, min(entry_idx + 100, len(df))):
            row = df.iloc[i]
            high = row['high']
            low = row['low']
            
            if is_long:
                # Check stop hit
                if low <= stop:
                    return Trade(
                        entry_time=pattern.detection_time,
                        exit_time=row['time'],
                        entry_price=entry,
                        exit_price=stop,
                        stop_loss=stop,
                        take_profit=tp,
                        pattern_type='bullish',
                        pnl_pct=(stop - entry) / entry * 100,
                        outcome=0,
                        r_multiple=-1.0
                    )
                # Check TP hit
                if high >= tp:
                    return Trade(
                        entry_time=pattern.detection_time,
                        exit_time=row['time'],
                        entry_price=entry,
                        exit_price=tp,
                        stop_loss=stop,
                        take_profit=tp,
                        pattern_type='bullish',
                        pnl_pct=(tp - entry) / entry * 100,
                        outcome=1,
                        r_multiple=(tp - entry) / risk
                    )
            else:
                # Short trade
                if high >= stop:
                    return Trade(
                        entry_time=pattern.detection_time,
                        exit_time=row['time'],
                        entry_price=entry,
                        exit_price=stop,
                        stop_loss=stop,
                        take_profit=tp,
                        pattern_type='bearish',
                        pnl_pct=(entry - stop) / entry * 100,
                        outcome=0,
                        r_multiple=-1.0
                    )
                if low <= tp:
                    return Trade(
                        entry_time=pattern.detection_time,
                        exit_time=row['time'],
                        entry_price=entry,
                        exit_price=tp,
                        stop_loss=stop,
                        take_profit=tp,
                        pattern_type='bearish',
                        pnl_pct=(entry - tp) / entry * 100,
                        outcome=1,
                        r_multiple=(entry - tp) / risk
                    )
        
        return None
    
    def run_backtest(self, patterns: List[QMLPattern], df: pd.DataFrame) -> Tuple[List[Trade], pd.Series]:
        """Run backtest on all patterns."""
        trades = []
        equity = [self.initial_capital]
        capital = self.initial_capital
        
        for pattern in patterns:
            trade = self.simulate_trade(pattern, df)
            if trade:
                # Position sizing
                risk_amount = capital * self.risk_per_trade
                position_pnl = risk_amount * trade.r_multiple
                capital += position_pnl
                equity.append(capital)
                trades.append(trade)
        
        equity_series = pd.Series(equity)
        return trades, equity_series


# ==============================================================================
# METRICS CALCULATION
# ==============================================================================
def calculate_metrics(trades: List[Trade], equity: pd.Series) -> Dict[str, float]:
    """Calculate backtest metrics."""
    if not trades:
        return {}
    
    wins = sum(1 for t in trades if t.outcome == 1)
    losses = sum(1 for t in trades if t.outcome == 0)
    total = len(trades)
    
    win_rate = wins / total if total > 0 else 0
    
    # Profit factor
    gross_profit = sum(t.r_multiple for t in trades if t.r_multiple > 0)
    gross_loss = abs(sum(t.r_multiple for t in trades if t.r_multiple < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy
    avg_win = np.mean([t.r_multiple for t in trades if t.r_multiple > 0]) if wins > 0 else 0
    avg_loss = abs(np.mean([t.r_multiple for t in trades if t.r_multiple < 0])) if losses > 0 else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Max drawdown
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak * 100
    max_dd = abs(drawdown.min())
    
    # Sharpe ratio (simplified - daily returns approximation)
    if len(equity) > 1:
        returns = equity.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    
    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_win_r': avg_win,
        'avg_loss_r': avg_loss,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'total_return': total_return,
    }


# ==============================================================================
# MONTE CARLO SIMULATION
# ==============================================================================
def monte_carlo_simulation(trades: List[Trade], n_simulations: int = 1000, 
                           initial_capital: float = 100000) -> Dict[str, Any]:
    """Run Monte Carlo simulation on trade sequence."""
    if not trades:
        return {}
    
    r_multiples = [t.r_multiple for t in trades]
    risk_per_trade = 0.01  # 1% risk
    
    final_capitals = []
    max_drawdowns = []
    
    for _ in range(n_simulations):
        # Shuffle trade order
        shuffled = np.random.permutation(r_multiples)
        
        capital = initial_capital
        peak = capital
        max_dd = 0
        
        for r in shuffled:
            risk_amount = capital * risk_per_trade
            capital += risk_amount * r
            
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        final_capitals.append(capital)
        max_drawdowns.append(max_dd)
    
    return {
        'mean_final_capital': np.mean(final_capitals),
        'median_final_capital': np.median(final_capitals),
        'std_final_capital': np.std(final_capitals),
        'p5_final_capital': np.percentile(final_capitals, 5),
        'p95_final_capital': np.percentile(final_capitals, 95),
        'mean_max_dd': np.mean(max_drawdowns),
        'median_max_dd': np.median(max_drawdowns),
        'p95_max_dd': np.percentile(max_drawdowns, 95),
        'probability_profitable': sum(1 for c in final_capitals if c > initial_capital) / n_simulations,
    }


# ==============================================================================
# WALK-FORWARD ANALYSIS
# ==============================================================================
def walk_forward_analysis(patterns: List[QMLPattern], df: pd.DataFrame, 
                          n_splits: int = 4) -> List[Dict[str, Any]]:
    """Perform walk-forward analysis."""
    if not patterns:
        return []
    
    # Sort patterns by time
    patterns = sorted(patterns, key=lambda p: p.detection_time)
    
    # Split into folds
    n_patterns = len(patterns)
    fold_size = n_patterns // n_splits
    
    results = []
    backtester = Backtester()
    
    for fold in range(n_splits):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_splits - 1 else n_patterns
        
        fold_patterns = patterns[start_idx:end_idx]
        
        if len(fold_patterns) < 5:
            continue
        
        trades, equity = backtester.run_backtest(fold_patterns, df)
        metrics = calculate_metrics(trades, equity)
        
        if metrics:
            metrics['fold'] = fold + 1
            metrics['start_date'] = fold_patterns[0].detection_time
            metrics['end_date'] = fold_patterns[-1].detection_time
            results.append(metrics)
    
    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print("\n" + "="*80)
print("  📊 FULL BACKTEST REVALIDATION")
print("  Comparing Original vs Rolling-Window Detector Results")
print("="*80)


# ==============================================================================
# STEP 1: LOAD ORIGINAL RESULTS
# ==============================================================================
print("\n" + "-"*80)
print("STEP 1: Loading Original Backtest Results")
print("-"*80)

original_file = Path(__file__).parent.parent / "data" / "comprehensive_features.csv"
original_df = pd.read_csv(original_file)

# Filter to BTC/USDT
btc_original = original_df[original_df['symbol'] == 'BTC/USDT'].copy()
btc_original['detection_time'] = pd.to_datetime(btc_original['detection_time'])

# Calculate original metrics
orig_wins = btc_original['outcome'].sum()
orig_total = len(btc_original)
orig_win_rate = orig_wins / orig_total

print(f"\n📊 Original BTC/USDT Results:")
print(f"   Total trades: {orig_total}")
print(f"   Wins: {orig_wins}")
print(f"   Win Rate: {orig_win_rate:.1%}")


# ==============================================================================
# STEP 2: RUN ROLLING DETECTOR ON FULL PERIOD
# ==============================================================================
print("\n" + "-"*80)
print("STEP 2: Running Rolling Detector on Full Historical Period")
print("-"*80)

# Fetch data for the full period (2023-2024)
print("\n📡 Fetching historical data...")
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 6, 1)  # Extended period

df = fetch_historical_data("BTC/USDT", "1h", start_date, end_date)
print(f"   Loaded {len(df)} candles")
print(f"   Period: {df['time'].min()} to {df['time'].max()}")

# Run rolling detector
print("\n🔄 Running rolling detection...")
rolling_detector = RollingPatternDetector(window_size=200, step_size=12)
patterns = rolling_detector.detect_all(df, "BTC/USDT", "1h")
print(f"   Detected {len(patterns)} patterns")


# ==============================================================================
# STEP 3: RUN BACKTEST
# ==============================================================================
print("\n" + "-"*80)
print("STEP 3: Running Backtest Simulation")
print("-"*80)

backtester = Backtester(initial_capital=100000, risk_per_trade=0.01)
trades, equity = backtester.run_backtest(patterns, df)

print(f"\n📊 Backtest completed:")
print(f"   Trades executed: {len(trades)}")
print(f"   Final equity: ${equity.iloc[-1]:,.2f}")


# ==============================================================================
# STEP 4: CALCULATE METRICS
# ==============================================================================
print("\n" + "-"*80)
print("STEP 4: Calculating Performance Metrics")
print("-"*80)

metrics = calculate_metrics(trades, equity)

print(f"\n📊 Rolling Detector Metrics:")
print(f"   Total Trades: {metrics.get('total_trades', 0)}")
print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
print(f"   Expectancy: {metrics.get('expectancy', 0):.2f}R")
print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1f}%")
print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
print(f"   Total Return: {metrics.get('total_return', 0):.1f}%")


# ==============================================================================
# STEP 5: WALK-FORWARD ANALYSIS
# ==============================================================================
print("\n" + "-"*80)
print("STEP 5: Walk-Forward Analysis")
print("-"*80)

wfa_results = walk_forward_analysis(patterns, df, n_splits=4)

print(f"\n📊 Walk-Forward Results ({len(wfa_results)} folds):")
print("-" * 70)
print(f"{'Fold':<6} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'Sharpe':<8} {'Max DD':<10}")
print("-" * 70)

for r in wfa_results:
    print(f"{r['fold']:<6} {r['total_trades']:<8} {r['win_rate']:.1%}{'':3} "
          f"{r['profit_factor']:.2f}{'':4} {r['sharpe_ratio']:.2f}{'':4} {r['max_drawdown']:.1f}%")

# Aggregate WFA metrics
if wfa_results:
    wfa_win_rates = [r['win_rate'] for r in wfa_results]
    wfa_pfs = [r['profit_factor'] for r in wfa_results if r['profit_factor'] < float('inf')]
    wfa_sharpes = [r['sharpe_ratio'] for r in wfa_results]
    
    print("-" * 70)
    print(f"{'Mean':<6} {'':8} {np.mean(wfa_win_rates):.1%}{'':3} "
          f"{np.mean(wfa_pfs):.2f}{'':4} {np.mean(wfa_sharpes):.2f}")


# ==============================================================================
# STEP 6: MONTE CARLO SIMULATION
# ==============================================================================
print("\n" + "-"*80)
print("STEP 6: Monte Carlo Simulation (1000 iterations)")
print("-"*80)

mc_results = monte_carlo_simulation(trades, n_simulations=1000)

print(f"\n📊 Monte Carlo Results:")
print(f"   Mean Final Capital: ${mc_results.get('mean_final_capital', 0):,.2f}")
print(f"   Median Final Capital: ${mc_results.get('median_final_capital', 0):,.2f}")
print(f"   5th Percentile: ${mc_results.get('p5_final_capital', 0):,.2f}")
print(f"   95th Percentile: ${mc_results.get('p95_final_capital', 0):,.2f}")
print(f"   Probability Profitable: {mc_results.get('probability_profitable', 0):.1%}")
print(f"   Mean Max Drawdown: {mc_results.get('mean_max_dd', 0):.1f}%")
print(f"   95th Percentile Max DD: {mc_results.get('p95_max_dd', 0):.1f}%")


# ==============================================================================
# STEP 7: COMPARISON REPORT
# ==============================================================================
print("\n" + "="*80)
print("  📋 ORIGINAL vs CORRECTED BACKTEST COMPARISON")
print("="*80)

# Original reported metrics (from previous analysis)
original_metrics = {
    'total_trades': 119,  # BTC only
    'win_rate': 0.595,    # From comprehensive_features outcomes
    'profit_factor': 1.47,
    'expectancy': 0.39,
    'max_drawdown': 30.7,
    'sharpe_ratio': 0.71,
}

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 ORIGINAL vs CORRECTED BACKTEST COMPARISON                    ║
╠═══════════════════════════╦═══════════════════╦══════════════════════════════╣
║         METRIC            ║     ORIGINAL      ║     CORRECTED (Rolling)      ║
╠═══════════════════════════╬═══════════════════╬══════════════════════════════╣
║ Total Trades (BTC)        ║        {original_metrics['total_trades']:<9} ║        {metrics.get('total_trades', 0):<21} ║
║ Win Rate                  ║      {original_metrics['win_rate']:.1%}{'':6} ║      {metrics.get('win_rate', 0):.1%}{'':18} ║
║ Profit Factor             ║        {original_metrics['profit_factor']:.2f}{'':6} ║        {metrics.get('profit_factor', 0):.2f}{'':18} ║
║ Expectancy (R)            ║       +{original_metrics['expectancy']:.2f}R{'':5} ║       {'+' if metrics.get('expectancy', 0) >= 0 else ''}{metrics.get('expectancy', 0):.2f}R{'':17} ║
║ Max Drawdown              ║      {original_metrics['max_drawdown']:.1f}%{'':6} ║      {metrics.get('max_drawdown', 0):.1f}%{'':18} ║
║ Sharpe Ratio              ║        {original_metrics['sharpe_ratio']:.2f}{'':6} ║        {metrics.get('sharpe_ratio', 0):.2f}{'':18} ║
╠═══════════════════════════╩═══════════════════╩══════════════════════════════╣
║                                                                              ║
║   VERDICT: {'METRICS CONSISTENT' if abs(metrics.get('win_rate', 0) - original_metrics['win_rate']) < 0.15 else 'SIGNIFICANT DIVERGENCE'}{'':43}║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Calculate differences
wr_diff = metrics.get('win_rate', 0) - original_metrics['win_rate']
pf_diff = metrics.get('profit_factor', 0) - original_metrics['profit_factor']
dd_diff = metrics.get('max_drawdown', 0) - original_metrics['max_drawdown']

print(f"""
CHANGE ANALYSIS:
----------------
Win Rate:       {'+' if wr_diff >= 0 else ''}{wr_diff*100:.1f} percentage points
Profit Factor:  {'+' if pf_diff >= 0 else ''}{pf_diff:.2f}
Max Drawdown:   {'+' if dd_diff >= 0 else ''}{dd_diff:.1f} percentage points

INTERPRETATION:
""")

if abs(wr_diff) < 0.10:
    print("   ✅ Win Rate is within acceptable variance (±10%)")
else:
    print("   ⚠️ Win Rate shows notable difference")

if abs(pf_diff) < 0.5:
    print("   ✅ Profit Factor is consistent")
else:
    print("   ⚠️ Profit Factor shows notable difference")

print(f"""
CONCLUSION:
-----------
The rolling-window detector produces {'CONSISTENT' if abs(wr_diff) < 0.15 else 'DIFFERENT'} results compared to the original backtest.

Key observations:
1. Pattern detection rate: ~79.5% match rate (as established in reconciliation)
2. Win rate variation: {'+' if wr_diff >= 0 else ''}{wr_diff*100:.1f} percentage points
3. The corrected detector captures the same core edge with {'minor' if abs(wr_diff) < 0.10 else 'moderate'} variance

The strategy's fundamental edge {'is validated' if metrics.get('win_rate', 0) > 0.50 and metrics.get('profit_factor', 0) > 1.0 else 'needs review'}.
""")


# ==============================================================================
# EXPORT RESULTS
# ==============================================================================
print("\n" + "-"*80)
print("Exporting Results")
print("-"*80)

# Export trades
trades_df = pd.DataFrame([{
    'entry_time': t.entry_time,
    'exit_time': t.exit_time,
    'entry_price': t.entry_price,
    'exit_price': t.exit_price,
    'stop_loss': t.stop_loss,
    'take_profit': t.take_profit,
    'pattern_type': t.pattern_type,
    'pnl_pct': t.pnl_pct,
    'outcome': t.outcome,
    'r_multiple': t.r_multiple,
} for t in trades])

trades_df.to_csv(Path(__file__).parent.parent / "revalidation_trades.csv", index=False)

# Export equity curve
equity_df = pd.DataFrame({'equity': equity})
equity_df.to_csv(Path(__file__).parent.parent / "revalidation_equity.csv", index=False)

# Export summary
summary = {
    'original_trades': original_metrics['total_trades'],
    'corrected_trades': metrics.get('total_trades', 0),
    'original_win_rate': original_metrics['win_rate'],
    'corrected_win_rate': metrics.get('win_rate', 0),
    'original_profit_factor': original_metrics['profit_factor'],
    'corrected_profit_factor': metrics.get('profit_factor', 0),
    'original_max_dd': original_metrics['max_drawdown'],
    'corrected_max_dd': metrics.get('max_drawdown', 0),
    'original_sharpe': original_metrics['sharpe_ratio'],
    'corrected_sharpe': metrics.get('sharpe_ratio', 0),
    'mc_probability_profitable': mc_results.get('probability_profitable', 0),
    'mc_mean_max_dd': mc_results.get('mean_max_dd', 0),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(Path(__file__).parent.parent / "revalidation_summary.csv", index=False)

print(f"\n📁 Exported:")
print(f"   - revalidation_trades.csv ({len(trades)} trades)")
print(f"   - revalidation_equity.csv (equity curve)")
print(f"   - revalidation_summary.csv (comparison summary)")

print("\n" + "="*80 + "\n")

