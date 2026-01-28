"""
Phase 9.3: Trade Outcome Diagnostic
====================================
Analyzes individual trades to understand why TP hit rate is 52% instead of expected 22%.

Checks:
1. Entry price calculation
2. SL/TP distances in ATR
3. Actual price movement to TP vs SL
4. Time to exit for each trade

Usage:
    python scripts/diagnose_trade_outcomes.py --symbol BTCUSDT
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

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

# Phase 7.9 configs
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


def load_data(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
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
    normalized = normalize_symbol(symbol)

    detector = HierarchicalSwingDetector(
        config=SWING_CONFIG, symbol=normalized, timeframe=timeframe
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
            # Include BacktestAdapter's calculated SL/TP for comparison
            'signal_sl': sig.stop_loss,
            'signal_tp': sig.take_profit,
        })

    return signals


def analyze_trade_outcomes(df: pd.DataFrame, signals: List[Dict], symbol: str):
    """Analyze trade outcomes in detail."""
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
    result = simulator.simulate_trades(
        df=df, signals=signals, symbol=symbol, timeframe="4h"
    )

    print(f"\n{'='*70}")
    print(f"TRADE OUTCOME ANALYSIS: {symbol}")
    print(f"{'='*70}")
    print(f"Total signals: {len(signals)}")
    print(f"Total trades: {result.total_trades}")
    print(f"Win rate: {result.win_rate:.1%}")
    print(f"Profit factor: {result.profit_factor:.2f}")

    # Analyze TP/SL distances
    print(f"\n--- ENTRY/EXIT ANALYSIS ---")

    tp_hits = [t for t in result.trades if t.exit_reason == ExitReason.TAKE_PROFIT]
    sl_hits = [t for t in result.trades if t.exit_reason == ExitReason.STOP_LOSS]
    time_exits = [t for t in result.trades if t.exit_reason == ExitReason.TIME_EXIT]

    print(f"TP hits: {len(tp_hits)} ({len(tp_hits)/result.total_trades*100:.1f}%)")
    print(f"SL hits: {len(sl_hits)} ({len(sl_hits)/result.total_trades*100:.1f}%)")
    print(f"Time exits: {len(time_exits)} ({len(time_exits)/result.total_trades*100:.1f}%)")

    # Analyze SL/TP distances
    print(f"\n--- SL/TP DISTANCES ---")

    sl_distances_atr = []
    tp_distances_atr = []

    for trade in result.trades:
        entry = trade.entry_price
        sl = trade.stop_loss
        tp = trade.take_profit
        atr = trade.atr_at_entry

        sl_dist = abs(entry - sl) / atr if atr > 0 else 0
        tp_dist = abs(tp - entry) / atr if atr > 0 else 0

        sl_distances_atr.append(sl_dist)
        tp_distances_atr.append(tp_dist)

    print(f"Avg SL distance: {np.mean(sl_distances_atr):.2f} ATR (expected: 1.0)")
    print(f"Avg TP distance: {np.mean(tp_distances_atr):.2f} ATR (expected: 4.6)")
    print(f"Min SL distance: {np.min(sl_distances_atr):.2f} ATR")
    print(f"Max SL distance: {np.max(sl_distances_atr):.2f} ATR")
    print(f"Min TP distance: {np.min(tp_distances_atr):.2f} ATR")
    print(f"Max TP distance: {np.max(tp_distances_atr):.2f} ATR")

    # Analyze actual R:R achieved
    print(f"\n--- ACHIEVED R:R ---")

    rr_ratios = [tp/sl if sl > 0 else 0 for tp, sl in zip(tp_distances_atr, sl_distances_atr)]
    print(f"Avg R:R ratio: {np.mean(rr_ratios):.2f} (expected: 4.6)")
    print(f"Min R:R ratio: {np.min(rr_ratios):.2f}")
    print(f"Max R:R ratio: {np.max(rr_ratios):.2f}")

    # Analyze bars held
    print(f"\n--- BARS HELD ---")
    bars_held = [t.bars_held for t in result.trades]
    tp_bars = [t.bars_held for t in tp_hits]
    sl_bars = [t.bars_held for t in sl_hits]

    print(f"Avg bars held (all): {np.mean(bars_held):.1f}")
    print(f"Avg bars held (TP hits): {np.mean(tp_bars):.1f}" if tp_bars else "N/A")
    print(f"Avg bars held (SL hits): {np.mean(sl_bars):.1f}" if sl_bars else "N/A")

    # Show sample trades
    print(f"\n--- SAMPLE TRADES ---")
    print(f"{'Dir':<6} {'Entry':>10} {'SL':>10} {'TP':>10} {'Exit':>10} {'R:R':>6} {'Bars':>5} {'Reason':<12} {'PnL R':>8}")
    print("-" * 85)

    for trade in result.trades[:10]:
        dir_str = trade.side.value[:5]
        rr = abs(trade.take_profit - trade.entry_price) / abs(trade.entry_price - trade.stop_loss) if abs(trade.entry_price - trade.stop_loss) > 0 else 0
        reason = trade.exit_reason.value if trade.exit_reason else "unknown"
        print(f"{dir_str:<6} {trade.entry_price:>10.2f} {trade.stop_loss:>10.2f} {trade.take_profit:>10.2f} {trade.exit_price:>10.2f} {rr:>6.2f} {trade.bars_held:>5} {reason:<12} {trade.pnl_r:>+8.2f}")

    # Check MFE/MAE
    print(f"\n--- MFE/MAE ANALYSIS ---")
    mfe_r = [t.mfe_r for t in result.trades]
    mae_r = [t.mae_r for t in result.trades]

    print(f"Avg MFE (R): {np.mean(mfe_r):.2f}")
    print(f"Avg MAE (R): {np.mean(mae_r):.2f}")
    print(f"Avg MFE for TP hits: {np.mean([t.mfe_r for t in tp_hits]):.2f}" if tp_hits else "N/A")
    print(f"Avg MAE for SL hits: {np.mean([t.mae_r for t in sl_hits]):.2f}" if sl_hits else "N/A")

    # Mathematical check: For 22% WR with 4.6:1 R:R, 22% should hit TP
    # For 57% WR, that means trades are hitting TP more often
    # This could be because:
    # 1. TP is closer than expected (ATR miscalculation)
    # 2. SL is further than expected
    # 3. Price movement is more favorable than expected

    print(f"\n--- MATHEMATICAL VERIFICATION ---")
    # What WR would we expect with actual R:R?
    actual_rr = np.mean(rr_ratios)
    actual_avg_win_r = np.mean([t.pnl_r for t in result.trades if t.pnl_r > 0])
    actual_avg_loss_r = abs(np.mean([t.pnl_r for t in result.trades if t.pnl_r <= 0]))

    # PF = (WR * avg_win) / ((1-WR) * avg_loss)
    # Rearranging: WR = PF * avg_loss / (avg_win + PF * avg_loss)
    if result.profit_factor > 0 and actual_avg_loss_r > 0:
        implied_wr_from_pf = (result.profit_factor * actual_avg_loss_r) / (actual_avg_win_r + result.profit_factor * actual_avg_loss_r)
        print(f"WR implied by PF and R-multiples: {implied_wr_from_pf:.1%}")
        print(f"Actual WR: {result.win_rate:.1%}")
        print(f"Match: {'YES' if abs(implied_wr_from_pf - result.win_rate) < 0.02 else 'NO (calculation error?)'}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    args = parser.parse_args()

    symbol = args.symbol if '/' in args.symbol else f"{args.symbol[:-4]}/{args.symbol[-4:]}"

    print(f"Loading data for {symbol}...")
    df = load_data(symbol)
    if df is None:
        print("No data found")
        return

    print(f"Running detection...")
    signals = run_detection(df, symbol)

    if not signals:
        print("No signals generated")
        return

    analyze_trade_outcomes(df, signals, normalize_symbol(symbol))


if __name__ == "__main__":
    main()
