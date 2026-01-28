"""
Phase 9.5: Shared Utilities for Validation Scripts
==================================================
Common functions used across all Phase 9.5 validation tests.

Provides:
- Pattern detection wrapper (HierarchicalSwingDetector)
- Trade simulation wrapper (TradeSimulator)
- Data loading utilities
- Metric calculation helpers
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
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


# Phase 7.9 optimized configs
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

# Phase 9.1 optimized trade management
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


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    metric_value: float
    threshold: float
    details: Dict


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


def run_detection(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "4h"
) -> List[Dict]:
    """
    Run pattern detection on dataframe and return signals.

    Returns list of signal dicts with: bar_idx, direction, entry_price, atr, score, timestamp
    """
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
            'timestamp': sig_time,
        })

    return signals


def simulate_trades(
    df: pd.DataFrame,
    signals: List[Dict],
    symbol: str,
    timeframe: str = "4h",
    config: Optional[TradeManagementConfig] = None,
) -> List[SimulatedTrade]:
    """
    Simulate trades from signals and return list of SimulatedTrade objects.
    """
    if config is None:
        config = TRADE_CONFIG

    simulator = TradeSimulator(config)
    result = simulator.simulate_trades(
        df=df,
        signals=signals,
        symbol=normalize_symbol(symbol),
        timeframe=timeframe
    )
    return result.trades


def calculate_metrics(trades: List[SimulatedTrade]) -> Dict:
    """
    Calculate key metrics from list of trades.

    Returns dict with: total_trades, win_rate, profit_factor, expectancy,
                       avg_win_r, avg_loss_r, max_drawdown, sharpe
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    winners = [t for t in trades if t.pnl_r > 0]
    losers = [t for t in trades if t.pnl_r <= 0]

    total = len(trades)
    wr = len(winners) / total if total > 0 else 0

    gross_profit = sum(t.pnl_r for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_r for t in losers)) if losers else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([t.pnl_r for t in winners]) if winners else 0
    avg_loss = abs(np.mean([t.pnl_r for t in losers])) if losers else 0

    expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)

    # Calculate max drawdown from R returns
    r_returns = [t.pnl_r for t in trades]
    equity_curve = np.cumsum(r_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Calculate Sharpe (annualized, assuming 4h bars = 6 per day = 2190 per year)
    if len(r_returns) > 1:
        mean_r = np.mean(r_returns)
        std_r = np.std(r_returns, ddof=1)
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0
    else:
        sharpe = 0

    return {
        "total_trades": total,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": wr,
        "profit_factor": pf,
        "expectancy": expectancy,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


def run_full_backtest(
    symbols: List[str],
    timeframe: str = "4h",
    config: Optional[TradeManagementConfig] = None,
    date_filter: Optional[Tuple[str, str]] = None,
    verbose: bool = True,
) -> Tuple[List[SimulatedTrade], Dict]:
    """
    Run full backtest across multiple symbols.

    Args:
        symbols: List of symbols to test
        timeframe: Timeframe to use
        config: Trade management config (uses default if None)
        date_filter: Optional (start_date, end_date) strings to filter data
        verbose: Print progress

    Returns:
        Tuple of (all_trades, aggregate_metrics)
    """
    all_trades = []

    for symbol in symbols:
        if verbose:
            print(f"Processing {symbol}...", end=" ")

        df = load_data(symbol, timeframe)
        if df is None:
            if verbose:
                print("NO DATA")
            continue

        # Apply date filter if provided
        if date_filter:
            start_date, end_date = date_filter
            df_time = df['time']
            if df_time.dt.tz is not None:
                df_time = df_time.dt.tz_localize(None)
            mask = (df_time >= pd.to_datetime(start_date)) & (df_time <= pd.to_datetime(end_date))
            df = df[mask].reset_index(drop=True)
            if len(df) < 100:
                if verbose:
                    print(f"INSUFFICIENT DATA ({len(df)} bars)")
                continue

        signals = run_detection(df, symbol, timeframe)
        if not signals:
            if verbose:
                print("NO SIGNALS")
            continue

        trades = simulate_trades(df, signals, symbol, timeframe, config)
        all_trades.extend(trades)

        if verbose:
            print(f"{len(signals)} signals -> {len(trades)} trades")

    metrics = calculate_metrics(all_trades)
    return all_trades, metrics


def get_r_returns(trades: List[SimulatedTrade]) -> np.ndarray:
    """Extract R-multiple returns from trades as numpy array."""
    return np.array([t.pnl_r for t in trades])
