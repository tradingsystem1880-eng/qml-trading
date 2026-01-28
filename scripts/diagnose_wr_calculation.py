"""
Phase 9.3: Win Rate Calculation Diagnostic
==========================================
Investigates the discrepancy between expected (22.4%) and observed (57-95%) win rates.

Key Questions:
1. How is win rate calculated? (strict TP/SL vs any pnl > 0)
2. What's the exit reason distribution?
3. What's the ACTUAL achieved R:R (not target)?
4. Are "dust profits" inflating win rate?

Mathematical Analysis:
- Phase 7.9 claims: 22.4% WR, 1.23 PF, 4.6:1 target R:R
- Math check: 22.4% WR with 4.6:1 R:R → PF = (0.224 * 4.6) / (0.776 * 1.0) = 1.33
- This is close to 1.23, so Phase 7.9 metrics ARE internally consistent

The problem: We're getting 57-95% WR which is impossible without a bug.

Usage:
    python scripts/diagnose_wr_calculation.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import get_symbol_data_dir, normalize_symbol
from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
from src.detection.pattern_validator import PatternValidator
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.backtest_adapter import BacktestAdapter
from src.detection.regime import MarketRegimeDetector
from src.detection.config import PatternValidationConfig, PatternScoringConfig
from src.optimization.trade_simulator import (
    TradeSimulator,
    TradeManagementConfig,
    ExitReason,
)

# Phase 7.9 configs (MUST match exactly)
SWING_CONFIG = HierarchicalSwingConfig(
    min_bar_separation=3,
    min_move_atr=0.85,
    forward_confirm_pct=0.2,
    lookback=6,
    lookforward=8,
)

PATTERN_CONFIG = PatternValidationConfig(
    p3_min_extension_atr=0.3,
    p3_max_extension_atr=5.0,
    p5_max_symmetry_atr=4.6,
    min_pattern_bars=16,
    max_pattern_bars=200,
)

SCORING_CONFIG = PatternScoringConfig()

# Test symbols
TEST_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
]


def load_data(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    """Load price data for a symbol."""
    data_dir = get_symbol_data_dir(symbol)
    data_path = data_dir / f"{timeframe}_master.parquet"

    if not data_path.exists():
        return None

    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'time'})

    return df


def run_detection(df: pd.DataFrame, symbol: str, timeframe: str = "4h") -> List[Dict]:
    """Run detection pipeline and return signals."""
    normalized = normalize_symbol(symbol)

    detector = HierarchicalSwingDetector(
        config=SWING_CONFIG,
        symbol=normalized,
        timeframe=timeframe,
    )
    swings = detector.detect(df)

    validator = PatternValidator(PATTERN_CONFIG)
    patterns = validator.find_patterns(swings, df['close'].values)
    valid_patterns = [p for p in patterns if p.is_valid]

    scorer = PatternScorer(SCORING_CONFIG)
    regime_detector = MarketRegimeDetector()

    scored = []
    for p in valid_patterns:
        p5_idx = p.p5.bar_index
        window_start = max(0, p5_idx - 150)
        window_df = df.iloc[window_start:p5_idx + 1].copy()
        regime_result = regime_detector.get_regime(window_df)

        score_result = scorer.score(p, df=df, regime_result=regime_result)
        if score_result.tier != PatternTier.REJECT:
            scored.append((p, score_result))

    adapter = BacktestAdapter()
    validation_results = [vr for vr, sr in scored]
    scoring_results = [sr for vr, sr in scored]

    signals_raw = adapter.batch_convert_to_signals(
        validation_results=validation_results,
        scoring_results=scoring_results,
        symbol=normalized,
        min_tier=PatternTier.C,
    )

    signals = []
    for sig in signals_raw:
        sig_time = sig.timestamp
        if hasattr(sig_time, 'tzinfo') and sig_time.tzinfo is not None:
            sig_time = sig_time.replace(tzinfo=None)

        df_time = df['time']
        if df_time.dt.tz is not None:
            df_time = df_time.dt.tz_localize(None)

        bar_indices = df[df_time >= sig_time].index
        if len(bar_indices) == 0:
            continue
        bar_idx = bar_indices[0]

        signal_atr = sig.atr_at_signal if hasattr(sig, 'atr_at_signal') and sig.atr_at_signal else None
        if signal_atr is None and 'atr' in df.columns:
            signal_atr = df.iloc[bar_idx]['atr']
        if signal_atr is None:
            signal_atr = 0

        signals.append({
            'bar_idx': bar_idx,
            'direction': sig.signal_type.value.upper().replace('BUY', 'LONG').replace('SELL', 'SHORT'),
            'entry_price': sig.price,
            'atr': signal_atr,
            'score': sig.validity_score if hasattr(sig, 'validity_score') else 0.5,
        })

    return signals


def analyze_trades(trades: List) -> Dict:
    """Deep analysis of trade outcomes."""
    if not trades:
        return {"error": "No trades"}

    # Basic counts
    total = len(trades)

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason.value if t.exit_reason else 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Win rate calculations - MULTIPLE METHODS

    # Method 1: Inclusive (any pnl > 0)
    winners_inclusive = [t for t in trades if t.pnl_r > 0]
    wr_inclusive = len(winners_inclusive) / total

    # Method 2: Strict TP only (exit_reason == take_profit)
    tp_exits = [t for t in trades if t.exit_reason == ExitReason.TAKE_PROFIT]
    wr_strict_tp = len(tp_exits) / total

    # Method 3: TP or profitable time exits
    tp_or_profitable_time = [t for t in trades if
        t.exit_reason == ExitReason.TAKE_PROFIT or
        (t.exit_reason == ExitReason.TIME_EXIT and t.pnl_r > 0)]
    wr_tp_or_time = len(tp_or_profitable_time) / total

    # R-multiple analysis
    all_pnl_r = [t.pnl_r for t in trades]
    winner_pnl_r = [t.pnl_r for t in winners_inclusive]
    loser_pnl_r = [t.pnl_r for t in trades if t.pnl_r <= 0]

    avg_win_r = np.mean(winner_pnl_r) if winner_pnl_r else 0
    avg_loss_r = abs(np.mean(loser_pnl_r)) if loser_pnl_r else 0

    # Actual achieved R:R
    actual_rr = avg_win_r / avg_loss_r if avg_loss_r > 0 else 0

    # Profit factor calculation
    gross_profit = sum(p for p in all_pnl_r if p > 0)
    gross_loss = abs(sum(p for p in all_pnl_r if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Mathematical check: Does PF match WR and R:R?
    # PF = (WR * avg_win) / ((1-WR) * avg_loss)
    expected_pf_from_wr = (wr_inclusive * avg_win_r) / ((1 - wr_inclusive) * avg_loss_r) if avg_loss_r > 0 and wr_inclusive < 1 else 0

    # Dust profit analysis (wins with tiny profit)
    dust_threshold = 0.1  # Less than 0.1R profit
    dust_wins = [t for t in winners_inclusive if 0 < t.pnl_r < dust_threshold]
    dust_pct = len(dust_wins) / len(winners_inclusive) if winners_inclusive else 0

    # Medium wins (0.1R to 1.0R)
    medium_wins = [t for t in winners_inclusive if dust_threshold <= t.pnl_r < 1.0]
    medium_pct = len(medium_wins) / len(winners_inclusive) if winners_inclusive else 0

    # Full wins (>= 1R, approaching target TP)
    full_wins = [t for t in winners_inclusive if t.pnl_r >= 1.0]
    full_pct = len(full_wins) / len(winners_inclusive) if winners_inclusive else 0

    # Bars held analysis
    bars_held = [t.bars_held for t in trades]
    winner_bars = [t.bars_held for t in winners_inclusive]
    loser_bars = [t.bars_held for t in trades if t.pnl_r <= 0]

    # MFE/MAE analysis (if available)
    mfe_values = [t.mfe_r for t in trades if hasattr(t, 'mfe_r') and t.mfe_r is not None]
    mae_values = [t.mae_r for t in trades if hasattr(t, 'mae_r') and t.mae_r is not None]

    return {
        "total_trades": total,

        # Exit distribution
        "exit_distribution": exit_reasons,
        "tp_exits": exit_reasons.get('take_profit', 0),
        "sl_exits": exit_reasons.get('stop_loss', 0),
        "time_exits": exit_reasons.get('time_exit', 0),

        # Win rate methods
        "wr_inclusive": wr_inclusive,  # Any pnl > 0
        "wr_strict_tp": wr_strict_tp,  # Only TP exits
        "wr_tp_or_profitable_time": wr_tp_or_time,

        # R-multiples
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "actual_rr": actual_rr,
        "target_rr": 4.6,  # TP/SL = 4.6/1.0

        # Profit factor
        "profit_factor": pf,
        "expected_pf_from_wr": expected_pf_from_wr,
        "pf_discrepancy": abs(pf - expected_pf_from_wr),

        # Win quality breakdown
        "dust_wins": len(dust_wins),
        "dust_pct_of_wins": dust_pct,
        "medium_wins": len(medium_wins),
        "medium_pct_of_wins": medium_pct,
        "full_wins": len(full_wins),
        "full_pct_of_wins": full_pct,

        # Bars held
        "avg_bars_held": np.mean(bars_held),
        "median_bars_held": np.median(bars_held),
        "avg_winner_bars": np.mean(winner_bars) if winner_bars else 0,
        "avg_loser_bars": np.mean(loser_bars) if loser_bars else 0,

        # MFE/MAE
        "avg_mfe_r": np.mean(mfe_values) if mfe_values else None,
        "avg_mae_r": np.mean(mae_values) if mae_values else None,
        "mfe_capture": avg_win_r / np.mean(mfe_values) if mfe_values and np.mean(mfe_values) > 0 else None,
    }


def run_diagnostic(symbols: List[str] = None, verbose: bool = True) -> Dict:
    """Run full diagnostic on win rate calculation."""
    if symbols is None:
        symbols = TEST_SYMBOLS

    config = TradeManagementConfig(
        tp_decay_enabled=False,
        tp_atr_mult=4.6,
        sl_atr_mult=1.0,
        trailing_mode="none",
        trailing_activation_atr=0.0,
        trailing_step_atr=0.0,
        max_bars_held=100,
        min_risk_reward=3.0,
    )

    simulator = TradeSimulator(config)
    all_trades = []

    for symbol in symbols:
        if verbose:
            print(f"Processing {symbol}...", end=" ")

        try:
            df = load_data(symbol)
            if df is None:
                if verbose:
                    print("NO DATA")
                continue

            signals = run_detection(df, symbol)
            if not signals:
                if verbose:
                    print("NO SIGNALS")
                continue

            result = simulator.simulate_trades(
                df=df,
                signals=signals,
                symbol=normalize_symbol(symbol),
                timeframe="4h",
            )
            all_trades.extend(result.trades)

            if verbose:
                print(f"{result.total_trades} trades")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

    if not all_trades:
        return {"error": "No trades generated"}

    analysis = analyze_trades(all_trades)
    analysis["symbols_tested"] = len(symbols)

    return analysis


def print_report(analysis: Dict):
    """Print formatted diagnostic report."""
    print("\n" + "=" * 70)
    print("PHASE 9.3: WIN RATE CALCULATION DIAGNOSTIC")
    print("=" * 70)

    print(f"\nTotal Trades: {analysis['total_trades']}")
    print(f"Symbols Tested: {analysis['symbols_tested']}")

    print("\n--- EXIT DISTRIBUTION ---")
    print(f"Take Profit:  {analysis['tp_exits']:4d} ({analysis['tp_exits']/analysis['total_trades']*100:.1f}%)")
    print(f"Stop Loss:    {analysis['sl_exits']:4d} ({analysis['sl_exits']/analysis['total_trades']*100:.1f}%)")
    print(f"Time Exit:    {analysis['time_exits']:4d} ({analysis['time_exits']/analysis['total_trades']*100:.1f}%)")

    print("\n--- WIN RATE METHODS ---")
    print(f"Inclusive (pnl > 0):        {analysis['wr_inclusive']:.1%}")
    print(f"Strict TP only:             {analysis['wr_strict_tp']:.1%}")
    print(f"TP or profitable time:      {analysis['wr_tp_or_profitable_time']:.1%}")

    print("\n--- R-MULTIPLE ANALYSIS ---")
    print(f"Avg Win (R):    {analysis['avg_win_r']:.2f}")
    print(f"Avg Loss (R):   {analysis['avg_loss_r']:.2f}")
    print(f"Actual R:R:     {analysis['actual_rr']:.2f}")
    print(f"Target R:R:     {analysis['target_rr']:.1f}")
    print(f"R:R Capture:    {analysis['actual_rr']/analysis['target_rr']*100:.1f}% of target")

    print("\n--- PROFIT FACTOR ---")
    print(f"Actual PF:      {analysis['profit_factor']:.2f}")
    print(f"Expected PF:    {analysis['expected_pf_from_wr']:.2f} (from WR & R:R)")
    print(f"Discrepancy:    {analysis['pf_discrepancy']:.2f}")

    print("\n--- WIN QUALITY BREAKDOWN ---")
    total_wins = analysis['dust_wins'] + analysis['medium_wins'] + analysis['full_wins']
    print(f"Dust (<0.1R):   {analysis['dust_wins']:4d} ({analysis['dust_pct_of_wins']*100:.1f}% of wins)")
    print(f"Medium (0.1-1R):{analysis['medium_wins']:4d} ({analysis['medium_pct_of_wins']*100:.1f}% of wins)")
    print(f"Full (>=1R):    {analysis['full_wins']:4d} ({analysis['full_pct_of_wins']*100:.1f}% of wins)")

    print("\n--- BARS HELD ---")
    print(f"Avg (all):      {analysis['avg_bars_held']:.1f}")
    print(f"Median:         {analysis['median_bars_held']:.0f}")
    print(f"Avg (winners):  {analysis['avg_winner_bars']:.1f}")
    print(f"Avg (losers):   {analysis['avg_loser_bars']:.1f}")

    if analysis.get('avg_mfe_r'):
        print("\n--- MFE/MAE ---")
        print(f"Avg MFE (R):    {analysis['avg_mfe_r']:.2f}")
        print(f"Avg MAE (R):    {analysis['avg_mae_r']:.2f}")
        print(f"MFE Capture:    {analysis['mfe_capture']*100:.1f}%")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Check for issues
    issues = []

    if analysis['wr_inclusive'] > 0.60:
        issues.append(f"WIN RATE TOO HIGH: {analysis['wr_inclusive']:.1%} > 60% is unrealistic for 4.6:1 R:R")

    if analysis['dust_pct_of_wins'] > 0.30:
        issues.append(f"TOO MANY DUST WINS: {analysis['dust_pct_of_wins']*100:.1f}% of wins are <0.1R (suggests early exits)")

    if analysis['actual_rr'] < 1.0:
        issues.append(f"ACTUAL R:R BELOW 1: {analysis['actual_rr']:.2f} means avg win < avg loss")

    if analysis['avg_bars_held'] < 5:
        issues.append(f"TRADES TOO SHORT: {analysis['avg_bars_held']:.1f} bars avg (suggests premature exits)")

    if analysis['profit_factor'] > 3.0:
        issues.append(f"PF UNREALISTIC: {analysis['profit_factor']:.2f} > 3.0 (world-class is 2-3)")

    # Phase 7.9 comparison
    print("\n--- PHASE 7.9 COMPARISON ---")
    print("Expected:  22.4% WR, 1.23 PF, 2,155 trades")
    print(f"Actual:    {analysis['wr_inclusive']:.1%} WR, {analysis['profit_factor']:.2f} PF, {analysis['total_trades']} trades")

    wr_diff = abs(analysis['wr_inclusive'] - 0.224) / 0.224 * 100
    pf_diff = abs(analysis['profit_factor'] - 1.23) / 1.23 * 100
    print(f"WR Diff:   {wr_diff:.0f}%")
    print(f"PF Diff:   {pf_diff:.0f}%")

    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No major issues detected")

    # Mathematical consistency check
    print("\n--- MATHEMATICAL CONSISTENCY ---")
    # Using actual numbers: PF = (WR * AvgWin) / ((1-WR) * AvgLoss)
    calculated_pf = (analysis['wr_inclusive'] * analysis['avg_win_r']) / ((1 - analysis['wr_inclusive']) * analysis['avg_loss_r']) if analysis['avg_loss_r'] > 0 and analysis['wr_inclusive'] < 1 else 0
    print(f"WR={analysis['wr_inclusive']:.3f}, AvgWin={analysis['avg_win_r']:.2f}R, AvgLoss={analysis['avg_loss_r']:.2f}R")
    print(f"Calculated PF: ({analysis['wr_inclusive']:.3f} * {analysis['avg_win_r']:.2f}) / ({1-analysis['wr_inclusive']:.3f} * {analysis['avg_loss_r']:.2f}) = {calculated_pf:.2f}")
    print(f"Actual PF: {analysis['profit_factor']:.2f}")

    if abs(calculated_pf - analysis['profit_factor']) < 0.05:
        print("✅ PF is mathematically consistent with WR and R-multiples")
    else:
        print(f"⚠️  PF discrepancy of {abs(calculated_pf - analysis['profit_factor']):.2f} - check calculation")


def main():
    parser = argparse.ArgumentParser(description='Phase 9.3 Win Rate Diagnostic')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (default: test set)')
    parser.add_argument('--all', action='store_true', help='Use all 15 symbols')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-symbol output')
    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    elif args.all:
        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
            "MATIC/USDT", "ATOM/USDT", "UNI/USDT", "LTC/USDT", "FIL/USDT",
        ]
    else:
        symbols = TEST_SYMBOLS

    print(f"Running diagnostic on {len(symbols)} symbols...")

    analysis = run_diagnostic(symbols, verbose=not args.quiet)

    if "error" in analysis:
        print(f"\nERROR: {analysis['error']}")
        return

    print_report(analysis)

    # Save results
    results_dir = PROJECT_ROOT / "results" / "phase93_diagnostic"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"wr_diagnostic_{timestamp}.json"

    # Convert numpy types for JSON
    serializable = {}
    for k, v in analysis.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        elif isinstance(v, dict):
            serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
        else:
            serializable[k] = v

    serializable['timestamp'] = timestamp

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
