"""
Core Integration Layer
======================
Connects dashboard to all existing subsystems:
- VRD Validation Pipeline
- Backtesting Engine
- Pattern Detection
- Data Pipeline
"""

import subprocess
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


# =============================================================================
# VRD VALIDATION INTEGRATION
# =============================================================================

def run_vrd_validation(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    n_splits: int = 5,
    n_permutations: int = 100,
    n_simulations: int = 500,
) -> Dict[str, Any]:
    """
    Run VRD 2.0 validation pipeline.
    
    Calls cli/run_vrd_validation.py with specified parameters.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "params": {
            "symbol": symbol,
            "timeframe": timeframe,
            "n_splits": n_splits,
            "n_permutations": n_permutations,
            "n_simulations": n_simulations,
        },
    }
    
    try:
        # Build command
        cmd = [
            sys.executable,
            "cli/run_vrd_validation.py",
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--splits", str(n_splits),
            "--permutations", str(n_permutations),
            "--simulations", str(n_simulations),
        ]
        
        logger.info(f"Running VRD validation: {' '.join(cmd)}")
        
        # Run subprocess
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        results["status"] = "completed" if process.returncode == 0 else "failed"
        results["stdout"] = process.stdout
        results["stderr"] = process.stderr
        results["return_code"] = process.returncode
        results["completed_at"] = datetime.now().isoformat()
        
        # Try to parse output for key metrics
        if process.returncode == 0:
            results["metrics"] = _parse_vrd_output(process.stdout)
        
    except subprocess.TimeoutExpired:
        results["status"] = "timeout"
        results["error"] = "Validation timed out after 10 minutes"
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results


def _parse_vrd_output(stdout: str) -> Dict[str, Any]:
    """Parse VRD output for key metrics."""
    metrics = {}
    
    # Look for common metrics in output
    lines = stdout.split("\n")
    for line in lines:
        if "p-value" in line.lower():
            metrics["p_value_mentioned"] = True
        if "sharpe" in line.lower():
            metrics["sharpe_mentioned"] = True
        if "monte carlo" in line.lower():
            metrics["monte_carlo_mentioned"] = True
    
    return metrics


# =============================================================================
# BACKTEST INTEGRATION
# =============================================================================

def run_backtest(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    atr_period: int = 14,
    risk_reward: float = 2.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    register_patterns: bool = True,
) -> Dict[str, Any]:
    """
    Run QMLStrategy backtest.
    
    Returns:
        Dictionary with backtest results
    """
    results = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "params": {
            "symbol": symbol,
            "timeframe": timeframe,
            "atr_period": atr_period,
            "risk_reward": risk_reward,
        },
    }
    
    try:
        # Import backtesting components
        from backtesting import Backtest
        from src.strategies.qml_backtestingpy import QMLStrategy
        
        # Load data
        symbol_clean = symbol.replace("/", "")
        parquet_path = Path(f"data/processed/{symbol_clean}/{timeframe}_master.parquet")
        
        if not parquet_path.exists():
            results["status"] = "error"
            results["error"] = f"Data file not found: {parquet_path}"
            return results
        
        df = pd.read_parquet(parquet_path)
        
        # Normalize columns
        df.columns = df.columns.str.lower()
        
        # Rename for backtesting.py
        column_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=column_map)
        
        # Set index
        if "time" in df.columns:
            df = df.set_index("time")
        
        # Filter by date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Run backtest
        bt = Backtest(
            df,
            QMLStrategy,
            cash=100000,
            commission=0.001,
        )
        
        stats = bt.run(
            atr_period=atr_period,
            risk_reward_ratio=risk_reward,
            register_patterns=register_patterns,
        )
        
        # Extract key metrics
        results["status"] = "completed"
        results["completed_at"] = datetime.now().isoformat()
        results["metrics"] = {
            "return_pct": float(stats["Return [%]"]),
            "sharpe_ratio": float(stats["Sharpe Ratio"]) if pd.notna(stats["Sharpe Ratio"]) else None,
            "max_drawdown": float(stats["Max. Drawdown [%]"]),
            "win_rate": float(stats["Win Rate [%]"]),
            "num_trades": int(stats["# Trades"]),
            "exposure_time": float(stats["Exposure Time [%]"]),
        }
        
        # Get trades
        trades = stats["_trades"]
        results["trades"] = [
            {
                "entry_time": str(t.EntryTime),
                "exit_time": str(t.ExitTime),
                "entry_price": float(t.EntryPrice),
                "exit_price": float(t.ExitPrice),
                "pnl": float(t.PnL),
                "return_pct": float(t.ReturnPct),
            }
            for t in trades.itertuples()
        ]
        
        # Get equity curve data
        equity = stats["_equity_curve"]
        results["equity_curve"] = {
            "timestamps": [str(t) for t in equity.index.tolist()[-100:]],  # Last 100 points
            "values": equity["Equity"].tolist()[-100:],
        }
        
    except ImportError as e:
        results["status"] = "error"
        results["error"] = f"Import error: {e}"
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        logger.error(f"Backtest failed: {e}")
    
    return results


# =============================================================================
# PATTERN SCANNING INTEGRATION
# =============================================================================

def scan_for_patterns(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    lookback_bars: int = 500,
    register: bool = True,
) -> Dict[str, Any]:
    """
    Scan latest data for QML patterns.
    
    Uses the same detection logic as populate_pattern_registry.py
    """
    results = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "patterns_found": 0,
        "patterns": [],
    }
    
    try:
        # Import population script functions
        from src.scripts.populate_pattern_registry import (
            populate_from_historical,
            calculate_atr,
            detect_bullish_qml,
            detect_bearish_qml,
        )
        from src.ml.pattern_registry import PatternRegistry
        from src.ml.feature_extractor import PatternFeatureExtractor
        
        # Load latest data
        symbol_clean = symbol.replace("/", "")
        parquet_path = Path(f"data/processed/{symbol_clean}/{timeframe}_master.parquet")
        
        if not parquet_path.exists():
            results["status"] = "error"
            results["error"] = f"Data not found: {parquet_path}"
            return results
        
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.lower()
        
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        
        # Use only recent data
        df = df.tail(lookback_bars).reset_index(drop=True)
        
        # Calculate ATR
        df["atr"] = calculate_atr(df)
        
        # Initialize extractor
        extractor = PatternFeatureExtractor()
        registry = PatternRegistry() if register else None
        
        # Scan for patterns
        patterns_found = []
        
        for i in range(50, len(df) - 5, 5):
            atr = df.iloc[i]["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            
            pattern = detect_bullish_qml(df, i, atr)
            if pattern is None:
                pattern = detect_bearish_qml(df, i, atr)
            
            if pattern is None:
                continue
            
            # Build pattern data
            pattern_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "pattern_type": pattern["pattern_type"],
                "detection_time": df.iloc[i]["time"],
                "detection_idx": i,
                "entry_price": pattern["entry_price"],
                "stop_loss": pattern["stop_loss"],
                "take_profit": pattern["take_profit"],
                "atr": pattern["atr"],
            }
            
            # Extract features
            features = extractor.extract_pattern_features(pattern_data, df, i)
            
            pattern_info = {
                "detection_time": str(pattern_data["detection_time"]),
                "pattern_type": pattern["pattern_type"],
                "entry_price": pattern["entry_price"],
                "stop_loss": pattern["stop_loss"],
                "take_profit": pattern["take_profit"],
                "feature_count": len(features),
            }
            patterns_found.append(pattern_info)
            
            # Register if requested
            if register and registry:
                registry.register_pattern(pattern_data, features)
        
        results["status"] = "completed"
        results["patterns_found"] = len(patterns_found)
        results["patterns"] = patterns_found[-10:]  # Last 10
        results["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        logger.error(f"Pattern scan failed: {e}")
    
    return results


# =============================================================================
# DATA PIPELINE INTEGRATION
# =============================================================================

def update_market_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
) -> Dict[str, Any]:
    """
    Update market data from CCXT.
    
    Calls data_engine.py to refresh parquet files.
    """
    results = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
    }
    
    try:
        cmd = [
            sys.executable,
            "src/data_engine.py",
            "--symbol", symbol,
            "--timeframe", timeframe,
        ]
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        results["status"] = "completed" if process.returncode == 0 else "failed"
        results["stdout"] = process.stdout[-500:]  # Last 500 chars
        results["return_code"] = process.returncode
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results


# =============================================================================
# SYSTEM STATUS
# =============================================================================

def get_system_status() -> Dict[str, Any]:
    """Get status of all subsystems."""
    
    status = {
        "database": "unknown",
        "data": "unknown",
        "model": "unknown",
        "patterns": 0,
        "labeled": 0,
        "last_update": None,
    }
    
    # Check database
    db_path = Path("results/experiments.db")
    if db_path.exists():
        status["database"] = "ok"
        
        # Get pattern counts
        try:
            from src.ml.pattern_registry import PatternRegistry
            registry = PatternRegistry()
            stats = registry.get_statistics()
            status["patterns"] = stats.get("total_patterns", 0)
            status["labeled"] = stats.get("labeled", 0)
        except:
            pass
    else:
        status["database"] = "missing"
    
    # Check data
    data_path = Path("data/processed/BTCUSDT/4h_master.parquet")
    if data_path.exists():
        status["data"] = "ok"
        status["last_update"] = datetime.fromtimestamp(
            data_path.stat().st_mtime
        ).isoformat()
    else:
        status["data"] = "missing"
    
    # Check model
    model_path = Path("results/ml_models/pattern_classifier.json")
    if model_path.exists():
        status["model"] = "trained"
    else:
        status["model"] = "not_trained"
    
    return status
