#!/usr/bin/env python3
"""
Phase 9.6: Collect Trade Data with Features for Report Generation
==================================================================
Runs backtest and collects all trade-level features needed for:
- SHAP analysis
- Feature scatter plots
- Correlation analysis

Outputs a parquet file with trades and all available features.

Usage:
    python scripts/collect_report_data.py
    python scripts/collect_report_data.py --symbols BTCUSDT,ETHUSDT
    python scripts/collect_report_data.py --output results/report_trades.parquet
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
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
    SimulatedTrade,
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

TRADE_CONFIG = TradeManagementConfig(
    tp_decay_enabled=False,
    tp_atr_mult=4.6,
    sl_atr_mult=1.0,
    trailing_mode="none",
    max_bars_held=100,
    min_risk_reward=3.0,
)

DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT",
]


def load_data(symbol: str, timeframe: str = "4h") -> Optional[pd.DataFrame]:
    """Load OHLCV data for a symbol."""
    data_dir = get_symbol_data_dir(symbol)
    data_path = data_dir / f"{timeframe}_master.parquet"
    if not data_path.exists():
        return None
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'time'})
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    return adx


def collect_trades_with_features(
    symbols: List[str],
    timeframe: str = "4h",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Collect all trades with enriched feature data.

    Returns DataFrame with columns:
    - Trade info: symbol, direction, entry_time, exit_time, pnl_r, result, exit_reason
    - Pattern features: pattern_score, total_score, head_extension, bos_efficiency, etc.
    - Market features: atr_at_signal, atr_percentile, adx, volume_ratio, regime
    - Time features: entry_hour, entry_day_of_week
    """
    all_trades = []

    simulator = TradeSimulator(TRADE_CONFIG)

    for symbol in symbols:
        if verbose:
            print(f"Processing {symbol}...", end=" ")

        df = load_data(symbol, timeframe)
        if df is None:
            if verbose:
                print("NO DATA")
            continue

        # Add technical indicators
        df['atr'] = calculate_atr(df)
        df['adx'] = calculate_adx(df)
        df['atr_percentile'] = df['atr'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        normalized = normalize_symbol(symbol)

        # Run detection
        detector = HierarchicalSwingDetector(
            config=SWING_CONFIG, symbol=normalized, timeframe=timeframe
        )
        swings = detector.detect(df)

        validator = PatternValidator(PATTERN_CONFIG)
        patterns = validator.find_patterns(swings, df['close'].values)
        valid_patterns = [p for p in patterns if p.is_valid]

        scorer = PatternScorer(SCORING_CONFIG)
        regime_detector = MarketRegimeDetector()

        scored_patterns = []
        for p in valid_patterns:
            p5_idx = p.p5.bar_index
            window_start = max(0, p5_idx - 150)
            window_df = df.iloc[window_start:p5_idx + 1].copy()
            regime_result = regime_detector.get_regime(window_df)
            score_result = scorer.score(p, df=df, regime_result=regime_result)
            if score_result.tier != PatternTier.REJECT:
                scored_patterns.append((p, score_result, regime_result))

        adapter = BacktestAdapter()

        for pattern, score_result, regime_result in scored_patterns:
            validation_results = [pattern]
            scoring_results = [score_result]

            signals_raw = adapter.batch_convert_to_signals(
                validation_results=validation_results,
                scoring_results=scoring_results,
                symbol=normalized,
                min_tier=PatternTier.C,
            )

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

                signal_atr = df.iloc[bar_idx]['atr'] if 'atr' in df.columns else 0

                signal = {
                    'bar_idx': bar_idx,
                    'direction': sig.signal_type.value.upper().replace('BUY', 'LONG').replace('SELL', 'SHORT'),
                    'entry_price': sig.price,
                    'atr': signal_atr,
                    'score': sig.validity_score if hasattr(sig, 'validity_score') else 0.5,
                    'timestamp': sig_time,
                }

                # Simulate trade
                result = simulator.simulate_trades(
                    df=df,
                    signals=[signal],
                    symbol=normalized,
                    timeframe=timeframe
                )

                if result.trades:
                    trade = result.trades[0]

                    # Collect all features
                    trade_data = {
                        # Trade info
                        'symbol': symbol,
                        'direction': signal['direction'],
                        'entry_time': trade.entry_time,
                        'exit_time': trade.exit_time,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'pnl_r': trade.pnl_r,
                        'result': trade.result.value if trade.result else 'PENDING',
                        'exit_reason': trade.exit_reason.value if trade.exit_reason else None,
                        'bars_held': trade.bars_held,
                        'mae_r': trade.mae_r,
                        'mfe_r': trade.mfe_r,

                        # Pattern features
                        'pattern_score': signal['score'],
                        'total_score': score_result.total_score,
                        'tier': score_result.tier.name,
                        'head_extension_score': score_result.head_extension_score,
                        'bos_efficiency_score': score_result.bos_efficiency_score,
                        'shoulder_symmetry_score': score_result.shoulder_symmetry_score,
                        'swing_significance_score': score_result.swing_significance_score,

                        # Geometric features
                        'head_extension_atr': pattern.p3.price - pattern.p1.price if pattern.direction == 'BULLISH' else pattern.p1.price - pattern.p3.price,
                        'pattern_bars': pattern.p5.bar_index - pattern.p1.bar_index,
                        'bos_efficiency': score_result.bos_efficiency_score,

                        # Market features
                        'atr_at_signal': signal_atr,
                        'atr_percentile': df.iloc[bar_idx]['atr_percentile'] if 'atr_percentile' in df.columns else 0.5,
                        'adx': df.iloc[bar_idx]['adx'] if 'adx' in df.columns else 0,
                        'volume_ratio': df.iloc[bar_idx]['volume_ratio'] if 'volume_ratio' in df.columns else 1.0,

                        # Regime features
                        'regime': regime_result.regime.name if regime_result else 'UNKNOWN',
                        'regime_score': regime_result.confidence if regime_result else 0.5,

                        # Time features
                        'entry_hour': trade.entry_time.hour if trade.entry_time else 0,
                        'entry_day_of_week': trade.entry_time.weekday() if trade.entry_time else 0,
                    }

                    all_trades.append(trade_data)

        if verbose:
            print(f"{len([p for p in scored_patterns])} patterns")

    return pd.DataFrame(all_trades)


def main():
    parser = argparse.ArgumentParser(description="Collect trade data with features")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--output', type=str, default='results/report_trades.parquet')
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s[:-4]}/{s[-4:]}" for s in symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("COLLECTING TRADE DATA WITH FEATURES")
    print("=" * 70)
    print(f"\nSymbols: {len(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Output: {args.output}")

    print("\n" + "=" * 70)
    print("RUNNING DETECTION AND SIMULATION")
    print("=" * 70 + "\n")

    trades_df = collect_trades_with_features(symbols, args.timeframe, verbose=True)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nTotal trades collected: {len(trades_df)}")

    if len(trades_df) > 0:
        winners = (trades_df['pnl_r'] > 0).sum()
        losers = (trades_df['pnl_r'] <= 0).sum()
        win_rate = winners / len(trades_df)
        avg_win = trades_df[trades_df['pnl_r'] > 0]['pnl_r'].mean() if winners > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl_r'] <= 0]['pnl_r'].mean()) if losers > 0 else 0
        pf = (winners * avg_win) / (losers * avg_loss) if losers > 0 and avg_loss > 0 else float('inf')

        print(f"Win Rate:       {win_rate:.1%}")
        print(f"Profit Factor:  {pf:.2f}")
        print(f"Avg Win:        {avg_win:.2f}R")
        print(f"Avg Loss:       {avg_loss:.2f}R")

        print(f"\nFeatures collected: {len(trades_df.columns)}")
        print(f"Columns: {list(trades_df.columns)}")

    # Save to parquet
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
