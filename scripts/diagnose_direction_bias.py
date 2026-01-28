"""
Phase 9.3: Direction Bias Diagnostic
=====================================
Checks if there's a directional bias causing high win rate.

Hypothesis: If data is from a bullish period, LONG signals would have higher WR than SHORT.
This could explain 55% WR vs expected 22%.

Usage:
    python scripts/diagnose_direction_bias.py
"""

import argparse
import sys
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
from src.core.models import Side

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

TEST_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
]


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
        })

    return signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = TEST_SYMBOLS

    config = TradeManagementConfig(
        tp_decay_enabled=False,
        tp_atr_mult=4.6,
        sl_atr_mult=1.0,
        trailing_mode="none",
        max_bars_held=100,
        min_risk_reward=3.0,
    )

    simulator = TradeSimulator(config)

    # Aggregate stats
    all_trades = []
    long_trades = []
    short_trades = []

    for symbol in symbols:
        print(f"Processing {symbol}...", end=" ")
        df = load_data(symbol)
        if df is None:
            print("NO DATA")
            continue

        signals = run_detection(df, symbol)
        if not signals:
            print("NO SIGNALS")
            continue

        result = simulator.simulate_trades(
            df=df, signals=signals, symbol=normalize_symbol(symbol), timeframe="4h"
        )
        all_trades.extend(result.trades)

        for t in result.trades:
            if t.side == Side.LONG:
                long_trades.append(t)
            else:
                short_trades.append(t)

        long_count = sum(1 for s in signals if s['direction'] == 'LONG')
        short_count = sum(1 for s in signals if s['direction'] == 'SHORT')
        print(f"{len(signals)} trades (L:{long_count}, S:{short_count})")

    print(f"\n{'='*70}")
    print("DIRECTION BIAS ANALYSIS")
    print(f"{'='*70}")

    def calc_stats(trades):
        if not trades:
            return {}
        winners = [t for t in trades if t.pnl_r > 0]
        losers = [t for t in trades if t.pnl_r <= 0]
        total = len(trades)
        wr = len(winners) / total
        gross_profit = sum(t.pnl_r for t in winners)
        gross_loss = abs(sum(t.pnl_r for t in losers))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        avg_win = np.mean([t.pnl_r for t in winners]) if winners else 0
        avg_loss = abs(np.mean([t.pnl_r for t in losers])) if losers else 0
        tp_hits = sum(1 for t in trades if t.exit_reason == ExitReason.TAKE_PROFIT)
        sl_hits = sum(1 for t in trades if t.exit_reason == ExitReason.STOP_LOSS)
        return {
            'total': total,
            'win_rate': wr,
            'profit_factor': pf,
            'avg_win_r': avg_win,
            'avg_loss_r': avg_loss,
            'tp_hits': tp_hits,
            'tp_pct': tp_hits / total,
            'sl_hits': sl_hits,
            'sl_pct': sl_hits / total,
        }

    all_stats = calc_stats(all_trades)
    long_stats = calc_stats(long_trades)
    short_stats = calc_stats(short_trades)

    print(f"\n{'Metric':<20} {'ALL':<15} {'LONG':<15} {'SHORT':<15}")
    print("-" * 65)
    print(f"{'Total Trades':<20} {all_stats['total']:<15} {long_stats.get('total', 0):<15} {short_stats.get('total', 0):<15}")
    print(f"{'Win Rate':<20} {all_stats['win_rate']*100:>5.1f}%{'':8} {long_stats.get('win_rate', 0)*100:>5.1f}%{'':8} {short_stats.get('win_rate', 0)*100:>5.1f}%")
    print(f"{'Profit Factor':<20} {all_stats['profit_factor']:>6.2f}{'':8} {long_stats.get('profit_factor', 0):>6.2f}{'':8} {short_stats.get('profit_factor', 0):>6.2f}")
    print(f"{'TP Hit Rate':<20} {all_stats['tp_pct']*100:>5.1f}%{'':8} {long_stats.get('tp_pct', 0)*100:>5.1f}%{'':8} {short_stats.get('tp_pct', 0)*100:>5.1f}%")
    print(f"{'SL Hit Rate':<20} {all_stats['sl_pct']*100:>5.1f}%{'':8} {long_stats.get('sl_pct', 0)*100:>5.1f}%{'':8} {short_stats.get('sl_pct', 0)*100:>5.1f}%")
    print(f"{'Avg Win (R)':<20} {all_stats['avg_win_r']:>6.2f}{'':8} {long_stats.get('avg_win_r', 0):>6.2f}{'':8} {short_stats.get('avg_win_r', 0):>6.2f}")
    print(f"{'Avg Loss (R)':<20} {all_stats['avg_loss_r']:>6.2f}{'':8} {long_stats.get('avg_loss_r', 0):>6.2f}{'':8} {short_stats.get('avg_loss_r', 0):>6.2f}")

    # Calculate expected WR based on Phase 7.9 formula
    print(f"\n--- ANALYSIS ---")

    if long_stats and short_stats:
        long_pct = long_stats['total'] / all_stats['total'] * 100
        short_pct = short_stats['total'] / all_stats['total'] * 100
        print(f"Direction distribution: {long_pct:.1f}% LONG, {short_pct:.1f}% SHORT")

        wr_diff = abs(long_stats['win_rate'] - short_stats['win_rate']) * 100
        print(f"Win rate difference: {wr_diff:.1f} percentage points")

        if wr_diff > 10:
            print(f"⚠️  SIGNIFICANT DIRECTION BIAS DETECTED")
            if long_stats['win_rate'] > short_stats['win_rate']:
                print(f"   LONG trades outperform SHORT by {wr_diff:.1f}pp - possible bullish data bias")
            else:
                print(f"   SHORT trades outperform LONG by {wr_diff:.1f}pp - possible bearish data bias")
        else:
            print(f"✅ No significant direction bias (difference < 10pp)")

    # Check data period
    print(f"\n--- DATA PERIOD CHECK ---")
    for symbol in symbols[:3]:
        df = load_data(symbol)
        if df is not None and 'time' in df.columns:
            start = df['time'].iloc[0]
            end = df['time'].iloc[-1]
            print(f"{symbol}: {start} to {end}")

    # Phase 7.9 comparison
    print(f"\n--- PHASE 7.9 COMPARISON ---")
    print(f"Expected: 22.4% WR, 1.23 PF")
    print(f"Actual:   {all_stats['win_rate']*100:.1f}% WR, {all_stats['profit_factor']:.2f} PF")

    if all_stats['win_rate'] > 0.40:
        print(f"\n⚠️  WIN RATE IS ANOMALOUSLY HIGH")
        print(f"   Possible causes:")
        print(f"   1. Data from favorable market conditions (bull/bear bias)")
        print(f"   2. Survivorship bias in pattern detection")
        print(f"   3. Different parameters than Phase 7.9")
        print(f"   4. Bug in trade simulation")


if __name__ == "__main__":
    main()
