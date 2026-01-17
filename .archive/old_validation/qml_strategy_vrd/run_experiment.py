#!/usr/bin/env python3
"""
RUN EXPERIMENT - Execute detection and log to VRD
==================================================
Run a QML detection experiment and automatically log results to the VRD.

Usage:
    python run_experiment.py --symbol ETH/USDT --timeframe 1h --logic v1.1.0_rolling_window
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import ccxt

from src.detection.detector import QMLDetector
from src.data.models import QMLPattern, PatternType


# VRD directories
VRD_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = VRD_ROOT / "experiments"
DETECTION_LOGIC_DIR = VRD_ROOT / "detection_logic"


def fetch_historical_data(
    symbol: str, 
    timeframe: str, 
    start_date: datetime, 
    end_date: datetime = None
) -> pd.DataFrame:
    """Fetch historical OHLCV data from Binance."""
    
    print(f"📡 Fetching {symbol} {timeframe}...")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int((end_date or datetime.now()).timestamp() * 1000)
    
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
            print(f"   ⚠️ Error: {e}")
            break
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    
    if end_date:
        end_ts_pd = pd.Timestamp(end_date).tz_localize('UTC')
        df = df[df['time'] <= end_ts_pd]
    
    df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    print(f"   ✅ {len(df)} candles: {df['time'].min()} to {df['time'].max()}")
    return df


class RollingPatternDetector:
    """Rolling window pattern detector for point-in-time detection."""
    
    def __init__(self, window_size: int = 200, step_size: int = 12):
        self.window_size = window_size
        self.step_size = step_size
        self.detector = QMLDetector()
    
    def _pattern_key(self, pattern: QMLPattern) -> str:
        return f"{pattern.head_time}_{pattern.left_shoulder_time}_{pattern.pattern_type.value}"
    
    def detect_all(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[QMLPattern]:
        n_bars = len(df)
        
        if n_bars < self.window_size:
            print(f"⚠️ Insufficient data: {n_bars} bars < {self.window_size} window")
            return []
        
        print(f"\n🔄 Running rolling detection on {symbol} {timeframe}")
        print(f"   Data: {n_bars} bars")
        print(f"   Window: {self.window_size} bars, Step: {self.step_size} bars")
        
        detected_patterns = []
        seen_patterns = set()
        
        n_windows = (n_bars - self.window_size) // self.step_size + 1
        
        for i in range(0, n_bars - self.window_size + 1, self.step_size):
            window_df = df.iloc[i:i + self.window_size].copy().reset_index(drop=True)
            patterns = self.detector.detect(symbol, timeframe, window_df)
            
            for p in patterns:
                key = self._pattern_key(p)
                if key not in seen_patterns:
                    seen_patterns.add(key)
                    detected_patterns.append(p)
            
            if (i // self.step_size) % 50 == 0:
                window_time = window_df['time'].iloc[-1]
                print(f"   Progress: {i // self.step_size + 1}/{n_windows} - {len(detected_patterns)} patterns")
        
        detected_patterns.sort(key=lambda p: p.detection_time)
        print(f"\n✅ Total unique patterns: {len(detected_patterns)}")
        
        return detected_patterns


def simulate_trades(patterns: List[QMLPattern], df: pd.DataFrame) -> List[Dict]:
    """Simulate trades from patterns and calculate outcomes."""
    trades = []
    
    for pattern in patterns:
        if not pattern.trading_levels:
            continue
        
        entry = pattern.trading_levels.entry
        stop = pattern.trading_levels.stop_loss
        target = pattern.trading_levels.take_profit_1
        
        # Find entry bar
        entry_idx = None
        for idx, row in df.iterrows():
            if row['time'] >= pattern.detection_time:
                entry_idx = idx
                break
        
        if entry_idx is None:
            continue
        
        # Simulate forward
        outcome = None
        exit_price = None
        exit_time = None
        
        for idx in range(entry_idx, min(entry_idx + 100, len(df))):
            row = df.iloc[idx]
            
            if pattern.pattern_type == PatternType.BULLISH:
                if row['low'] <= stop:
                    outcome = -1
                    exit_price = stop
                    exit_time = row['time']
                    break
                elif row['high'] >= target:
                    outcome = 1
                    exit_price = target
                    exit_time = row['time']
                    break
            else:
                if row['high'] >= stop:
                    outcome = -1
                    exit_price = stop
                    exit_time = row['time']
                    break
                elif row['low'] <= target:
                    outcome = 1
                    exit_price = target
                    exit_time = row['time']
                    break
        
        if outcome is None:
            continue
        
        risk = abs(entry - stop)
        r_multiple = (exit_price - entry) / risk if risk > 0 else 0
        if pattern.pattern_type == PatternType.BEARISH:
            r_multiple = (entry - exit_price) / risk if risk > 0 else 0
        
        trades.append({
            'entry_time': pattern.detection_time,
            'exit_time': exit_time,
            'entry_price': entry,
            'exit_price': exit_price,
            'stop_loss': stop,
            'take_profit': target,
            'pattern_type': pattern.pattern_type.value,
            'outcome': outcome,
            'r_multiple': r_multiple,
            'pnl_pct': r_multiple * 0.01  # Assuming 1% risk
        })
    
    return trades


def calculate_metrics(trades: List[Dict]) -> Dict:
    """Calculate performance metrics."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0
        }
    
    wins = sum(1 for t in trades if t['outcome'] == 1)
    losses = sum(1 for t in trades if t['outcome'] == -1)
    
    win_rate = wins / len(trades) if trades else 0
    
    gross_profit = sum(t['r_multiple'] for t in trades if t['outcome'] == 1)
    gross_loss = abs(sum(t['r_multiple'] for t in trades if t['outcome'] == -1))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    r_multiples = [t['r_multiple'] for t in trades]
    expectancy = np.mean(r_multiples) if r_multiples else 0
    
    # Sharpe ratio (annualized)
    if len(r_multiples) > 1:
        std = np.std(r_multiples)
        sharpe = (np.mean(r_multiples) / std) * np.sqrt(252) if std > 0 else 0
    else:
        sharpe = 0
    
    # Max drawdown
    equity = [100000]
    for t in trades:
        equity.append(equity[-1] * (1 + t['pnl_pct']))
    
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    
    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 3),
        "profit_factor": round(profit_factor, 2),
        "expectancy_r": round(expectancy, 3),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe_ratio": round(sharpe, 2),
        "total_return_pct": round((equity[-1] / equity[0] - 1) * 100, 1)
    }


def create_experiment(
    symbol: str,
    timeframe: str,
    logic_version: str,
    patterns: List[QMLPattern],
    trades: List[Dict],
    metrics: Dict,
    data_start: datetime,
    data_end: datetime
) -> str:
    """Create experiment directory and save all artifacts."""
    
    # Generate experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_clean = symbol.replace("/", "")
    logic_short = logic_version.split("_")[0] if "_" in logic_version else logic_version
    
    exp_id = f"{timestamp}_{symbol_clean}_{timeframe}_{logic_short}"
    exp_path = EXPERIMENTS_DIR / exp_id
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # Save meta.json
    meta = {
        "experiment_id": exp_id,
        "detection_logic_version": logic_version,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_range": {
            "start": data_start.strftime("%Y-%m-%d"),
            "end": data_end.strftime("%Y-%m-%d")
        },
        "run_timestamp": datetime.now().isoformat(),
        "patterns_detected": len(patterns),
        "trades_executed": len(trades)
    }
    
    with open(exp_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Save metrics.json
    with open(exp_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save trades.csv
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(exp_path / "trades.csv", index=False)
    
    # Save patterns.csv
    if patterns:
        patterns_data = []
        for p in patterns:
            patterns_data.append({
                "detection_time": p.detection_time,
                "pattern_type": p.pattern_type.value,
                "validity_score": p.validity_score,
                "head_price": p.head_price,
                "left_shoulder_price": p.left_shoulder_price,
                "entry": p.trading_levels.entry if p.trading_levels else None,
                "stop_loss": p.trading_levels.stop_loss if p.trading_levels else None,
                "take_profit": p.trading_levels.take_profit_1 if p.trading_levels else None
            })
        patterns_df = pd.DataFrame(patterns_data)
        patterns_df.to_csv(exp_path / "patterns.csv", index=False)
    
    return exp_id


def run_experiment(
    symbol: str,
    timeframe: str,
    logic_version: str,
    start_date: datetime = None,
    end_date: datetime = None
) -> str:
    """Run a complete experiment and log to VRD."""
    
    print("\n" + "="*80)
    print(f"  🧪 VRD EXPERIMENT: {symbol} {timeframe}")
    print(f"  📋 Logic: {logic_version}")
    print("="*80)
    
    # Default dates
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    if end_date is None:
        end_date = datetime.now()
    
    # Validate logic version exists
    logic_path = DETECTION_LOGIC_DIR / logic_version
    if not logic_path.exists():
        print(f"❌ Logic version not found: {logic_version}")
        print(f"   Available: {[d.name for d in DETECTION_LOGIC_DIR.iterdir() if d.is_dir()]}")
        return None
    
    # Load logic parameters
    params_path = logic_path / "params.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        print(f"\n📋 Logic parameters:")
        print(f"   Window size: {params.get('parameters', {}).get('window_size', 200)}")
        print(f"   Step size: {params.get('parameters', {}).get('step_size', 12)}")
    
    # Fetch data
    print(f"\n📊 Fetching data: {start_date.date()} to {end_date.date()}")
    df = fetch_historical_data(symbol, timeframe, start_date, end_date)
    
    if len(df) < 200:
        print("❌ Insufficient data")
        return None
    
    # Run detection
    if "rolling" in logic_version.lower():
        window_size = params.get('parameters', {}).get('window_size', 200)
        step_size = params.get('parameters', {}).get('step_size', 12)
        detector = RollingPatternDetector(window_size=window_size, step_size=step_size)
        patterns = detector.detect_all(df, symbol, timeframe)
    else:
        detector = QMLDetector()
        patterns = detector.detect(symbol, timeframe, df)
    
    print(f"\n📊 Patterns detected: {len(patterns)}")
    
    # Simulate trades
    trades = simulate_trades(patterns, df)
    print(f"📊 Trades simulated: {len(trades)}")
    
    # Calculate metrics
    metrics = calculate_metrics(trades)
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
    
    # Save to VRD
    exp_id = create_experiment(
        symbol=symbol,
        timeframe=timeframe,
        logic_version=logic_version,
        patterns=patterns,
        trades=trades,
        metrics=metrics,
        data_start=start_date,
        data_end=end_date
    )
    
    print(f"\n✅ Experiment logged to VRD: {exp_id}")
    print(f"   Path: {EXPERIMENTS_DIR / exp_id}")
    print("\n" + "="*80 + "\n")
    
    return exp_id


def main():
    parser = argparse.ArgumentParser(description="Run VRD experiment")
    parser.add_argument("--symbol", type=str, default="ETH/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe")
    parser.add_argument("--logic", type=str, default="v1.1.0_rolling_window", 
                       help="Detection logic version")
    parser.add_argument("--start", type=str, default="2023-01-01", 
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, 
                       help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    
    exp_id = run_experiment(
        symbol=args.symbol,
        timeframe=args.timeframe,
        logic_version=args.logic,
        start_date=start_date,
        end_date=end_date
    )
    
    if exp_id:
        print(f"🎉 Success! View experiment with:")
        print(f"   python vr_dashboard.py show {exp_id}")


if __name__ == "__main__":
    main()
