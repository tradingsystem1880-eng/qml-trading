#!/usr/bin/env python3
"""
Generate trades from 2025+ data for funding filter validation.
Only includes trades after 2025-01-29 to match funding data availability.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import get_symbol_data_dir, normalize_symbol
from src.detection.hierarchical_swing import HierarchicalSwingDetector, HierarchicalSwingConfig
from src.detection.pattern_validator import PatternValidator
from src.detection.pattern_scorer import PatternScorer, PatternTier
from src.detection.backtest_adapter import BacktestAdapter
from src.detection.regime import MarketRegimeDetector
from src.detection.config import (
    PatternValidationConfig,
    PatternScoringConfig,
)
from src.optimization.trade_simulator import (
    TradeSimulator,
    TradeManagementConfig,
    SimulationResult,
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

# All symbols with funding data
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
    "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    "FIL/USDT", "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT",
    "INJ/USDT", "SUI/USDT", "TIA/USDT", "SEI/USDT", "RUNE/USDT",
    "SAND/USDT", "MANA/USDT", "AAVE/USDT", "CRV/USDT",
    "ALGO/USDT", "XLM/USDT",
]

# Date filter for funding data availability
MIN_DATE = pd.Timestamp("2025-01-29", tz="UTC")


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

    # Ensure timezone-aware
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')

    return df


def run_detection(df: pd.DataFrame, symbol: str, timeframe: str = "4h") -> List[Dict]:
    """Run detection pipeline and return signals."""
    normalized = normalize_symbol(symbol)

    detector = HierarchicalSwingDetector(config=SWING_CONFIG)
    swings = detector.detect(df)

    if len(swings) < 5:
        return []

    validator = PatternValidator(PATTERN_CONFIG)
    scorer = PatternScorer(SCORING_CONFIG)
    adapter = BacktestAdapter()

    regime_detector = MarketRegimeDetector()

    # Pass close prices as numpy array (not DataFrame)
    patterns = validator.find_patterns(swings, df['close'].values)
    valid_patterns = [p for p in patterns if p.is_valid]

    signals = []
    for pattern in valid_patterns:
        # Get regime at pattern time
        p5_idx = pattern.p5.bar_index
        window_start = max(0, p5_idx - 150)
        window_df = df.iloc[window_start:p5_idx + 1].copy()

        regime_result = regime_detector.get_regime(window_df)
        regime = regime_result.regime.name if regime_result else "UNKNOWN"

        # Score the pattern
        score_result = scorer.score(pattern, df=df, regime_result=regime_result)
        if score_result.tier == PatternTier.REJECT:
            continue

        # Convert to signal
        signal = adapter.validation_to_signal(
            validation_result=pattern,
            scoring_result=score_result,
            symbol=normalized,
            timeframe=timeframe,
        )
        if signal is None:
            continue

        # Map signal_type (BUY/SELL) to direction (LONG/SHORT)
        direction = "LONG" if signal.signal_type.value == "BUY" else "SHORT"

        # Convert Signal object to dict for simulation
        signal_dict = {
            'direction': direction,
            'entry_price': signal.price,
            'atr': signal.atr_at_signal if signal.atr_at_signal else df.iloc[p5_idx].get('atr', 1.0),
            'bar_idx': p5_idx,  # TradeSimulator expects 'bar_idx'
            'pattern_score': score_result.total_score,
            'tier': score_result.tier.value,
            'regime': regime,
        }

        signals.append(signal_dict)

    return signals


def simulate_trades(signals: List[Dict], df: pd.DataFrame, symbol: str) -> List[Dict]:
    """Simulate trades and return detailed results."""
    config = TradeManagementConfig(
        sl_atr_mult=1.0,
        tp_atr_mult=4.0,
        trailing_mode="none",
        max_bars_held=100,
    )
    simulator = TradeSimulator(config)

    # Filter signals for 2025+ entries
    signals_2025 = []
    for sig in signals:
        entry_idx = sig.get('bar_idx', 0)
        if entry_idx >= len(df) - 1:
            continue
        entry_time = df.iloc[entry_idx]['time']
        if entry_time >= MIN_DATE:
            signals_2025.append(sig)

    if not signals_2025:
        return []

    # Simulate all trades at once
    sim_result = simulator.simulate_trades(
        df=df,
        signals=signals_2025,
        symbol=symbol,
        timeframe="4h",
    )

    # Convert to trade dicts
    trades = []
    for sim_trade in sim_result.trades:
        # Find matching signal by bar_idx
        matching_sigs = [s for s in signals_2025 if s['bar_idx'] == sim_trade.entry_bar_idx]
        if not matching_sigs:
            continue
        sig = matching_sigs[0]

        # Calculate bars held
        bars_held = (sim_trade.exit_bar_idx - sim_trade.entry_bar_idx) if sim_trade.exit_bar_idx else 0

        trade = {
            'symbol': symbol,
            'direction': sig['direction'],
            'entry_time': sim_trade.entry_time,
            'exit_time': sim_trade.exit_time,
            'entry_price': sim_trade.entry_price,
            'exit_price': sim_trade.exit_price,
            'pnl_r': sim_trade.pnl_r,
            'result': 'WIN' if sim_trade.pnl_r > 0 else 'LOSS',
            'exit_reason': sim_trade.exit_reason.value if sim_trade.exit_reason else 'UNKNOWN',
            'bars_held': bars_held,
            'mae_r': sim_trade.mae_r,
            'mfe_r': sim_trade.mfe_r,
            'pattern_score': sig.get('pattern_score', 0),
            'tier': sig.get('tier', 'C'),
            'regime': sig.get('regime', 'UNKNOWN'),
        }
        trades.append(trade)

    return trades


def main():
    print("=" * 70)
    print("GENERATING 2025+ TRADES FOR FUNDING FILTER VALIDATION")
    print("=" * 70)
    print(f"Minimum date: {MIN_DATE}")
    print(f"Symbols: {len(SYMBOLS)}")
    print("=" * 70)
    print()

    all_trades = []

    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {symbol}...", end=" ", flush=True)

        df = load_data(symbol)
        if df is None:
            print("no data")
            continue

        # Filter to data that includes 2025+
        df_recent = df[df['time'] >= pd.Timestamp("2024-01-01", tz="UTC")]
        if len(df_recent) < 100:
            print(f"insufficient data ({len(df_recent)} bars)")
            continue

        try:
            signals = run_detection(df, symbol)
            if not signals:
                print("no patterns")
                continue

            trades = simulate_trades(signals, df, symbol)
            if not trades:
                print("no 2025+ trades")
                continue

            all_trades.extend(trades)
            print(f"{len(trades)} trades")

        except Exception as e:
            print(f"error: {e}")
            continue

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total 2025+ trades: {len(all_trades)}")

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        wins = (trades_df['pnl_r'] > 0).sum()
        losses = len(trades_df) - wins
        win_rate = wins / len(trades_df) * 100

        print(f"Win rate: {win_rate:.1f}% ({wins}W / {losses}L)")
        print(f"Date range: {trades_df['entry_time'].min()} to {trades_df['entry_time'].max()}")
        print(f"Symbols: {trades_df['symbol'].nunique()}")

        # Save
        output_path = PROJECT_ROOT / 'data' / 'backtest' / 'validated_trades.parquet'
        trades_df.to_parquet(output_path, index=False)
        print(f"\nSaved to: {output_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
