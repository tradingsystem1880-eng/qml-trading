"""
QML Paper Trading Simulation System
====================================
Live paper trading simulator for QML strategy validation.

This module:
1. Monitors real-time market data
2. Detects QML patterns in real-time
3. Logs all signals with full context
4. Tracks paper positions and P&L
5. Applies high-conviction filter

Usage:
    python -m src.trading.paper_trader --symbols BTC/USDT,ETH/USDT --filter high_vol
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from loguru import logger

from src.data.database import DatabaseManager
from src.data.fetcher import DataFetcher
from src.detection.detector import QMLDetector
from src.utils.indicators import calculate_atr, calculate_volatility_percentile
from config.settings import settings


@dataclass
class PaperSignal:
    """Represents a trading signal for paper trading."""
    
    timestamp: str
    signal_id: str
    symbol: str
    timeframe: str
    pattern_type: str  # BULLISH or BEARISH
    
    # Detection context
    choch_time: str
    bos_time: str
    head_price: float
    left_shoulder_price: float
    right_shoulder_price: float
    
    # Trading levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    risk_reward: float
    
    # Market context
    volatility_percentile: float
    adx: float
    rsi: float
    trend_state: str
    
    # Filter decision
    filter_enabled: bool
    filter_decision: str  # PASS or FAIL
    filter_reason: str
    
    # Validity
    validity_score: float
    
    # Position tracking (updated as trade resolves)
    position_status: str = "PENDING"  # PENDING, OPEN, CLOSED
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl_r: Optional[float] = None
    outcome: Optional[str] = None  # WIN, LOSS, TIMEOUT


class PaperTradingEngine:
    """
    Paper trading engine for QML strategy.
    
    Monitors markets, detects patterns, and logs all signals
    for paper trading validation.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        filter_enabled: bool = True,
        vol_threshold: float = 0.7,
        log_dir: str = "paper_trading_logs"
    ):
        """
        Initialize paper trading engine.
        
        Args:
            symbols: List of trading pairs to monitor
            timeframe: Primary timeframe for detection
            filter_enabled: Whether to apply high-conviction filter
            vol_threshold: Volatility threshold for filter
            log_dir: Directory for signal logs
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.filter_enabled = filter_enabled
        self.vol_threshold = vol_threshold
        self.log_dir = Path(log_dir)
        
        # Initialize components
        self.db = DatabaseManager()
        self.fetcher = DataFetcher()
        self.detector = QMLDetector()
        
        # State tracking
        self.active_signals: Dict[str, PaperSignal] = {}
        self.completed_signals: List[PaperSignal] = []
        self.last_detection_time: Dict[str, datetime] = {}
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        self.log_file = self.log_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.json"
        
        logger.info(f"Paper trading engine initialized")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Filter enabled: {filter_enabled}")
        logger.info(f"Volatility threshold: {vol_threshold}")
    
    def calculate_market_context(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate market context features at current bar."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # ATR
        atr = calculate_atr(high, low, close, 14)
        
        # Volatility percentile
        returns = np.diff(close) / close[:-1]
        returns = np.insert(returns, 0, 0)
        vol_20 = pd.Series(returns).rolling(20).std().values
        
        vol_lookback = vol_20[-100:]
        current_vol = vol_20[-1] if not np.isnan(vol_20[-1]) else 0
        vol_percentile = (current_vol - np.nanmin(vol_lookback)) / (np.nanmax(vol_lookback) - np.nanmin(vol_lookback) + 1e-8)
        
        # ADX (simplified)
        from src.utils.indicators import calculate_adx
        adx = calculate_adx(high, low, close, 14)
        
        # RSI (simplified)
        from src.utils.indicators import calculate_rsi
        rsi = calculate_rsi(close, 14)
        
        # Trend state
        adx_val = adx[-1] if not np.isnan(adx[-1]) else 0
        if adx_val > 25:
            # Strong trend - check direction
            if close[-1] > close[-20]:
                trend = "UPTREND"
            else:
                trend = "DOWNTREND"
        else:
            trend = "CONSOLIDATION"
        
        return {
            'volatility_percentile': float(vol_percentile),
            'adx': float(adx[-1]) if not np.isnan(adx[-1]) else 0.0,
            'rsi': float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0,
            'trend_state': trend
        }
    
    def apply_filter(
        self,
        context: Dict[str, float]
    ) -> tuple[str, str]:
        """
        Apply high-conviction filter.
        
        Returns:
            Tuple of (decision, reason)
        """
        if not self.filter_enabled:
            return "PASS", "Filter disabled"
        
        vol_pctl = context['volatility_percentile']
        
        if vol_pctl > self.vol_threshold:
            return "PASS", f"vol_percentile ({vol_pctl:.2f}) > {self.vol_threshold}"
        else:
            return "FAIL", f"vol_percentile ({vol_pctl:.2f}) <= {self.vol_threshold}"
    
    def detect_patterns(
        self,
        symbol: str
    ) -> List[PaperSignal]:
        """
        Detect QML patterns for a symbol.
        
        Returns:
            List of new signals
        """
        signals = []
        
        # Get recent data
        df = self.db.get_ohlcv(symbol, self.timeframe, limit=1000)
        
        if df.empty or len(df) < 500:
            return signals
        
        # Run detection
        patterns = self.detector.detect(symbol, self.timeframe, df=df)
        
        for pattern in patterns:
            if not pattern.trading_levels:
                continue
            
            # Check if this is a new pattern (not already logged)
            pattern_time = pd.Timestamp(pattern.detection_time)
            last_time = self.last_detection_time.get(symbol, datetime.min.replace(tzinfo=pattern_time.tzinfo))
            
            if pattern_time <= last_time:
                continue
            
            # Calculate market context
            context = self.calculate_market_context(df)
            
            # Apply filter
            decision, reason = self.apply_filter(context)
            
            # Create signal
            signal = PaperSignal(
                timestamp=datetime.now().isoformat(),
                signal_id=f"qml_{symbol.replace('/', '')}_{pattern_time.strftime('%Y%m%d_%H%M')}",
                symbol=symbol,
                timeframe=self.timeframe,
                pattern_type=pattern.pattern_type.value.upper(),
                choch_time=pattern.head_time.isoformat() if pattern.head_time else "",  # Use head_time as proxy
                bos_time=pattern.detection_time.isoformat() if pattern.detection_time else "",  # Use detection_time
                head_price=pattern.head_price,
                left_shoulder_price=pattern.left_shoulder_price,
                right_shoulder_price=pattern.right_shoulder_price if pattern.right_shoulder_price else pattern.left_shoulder_price,
                entry_price=pattern.trading_levels.entry,
                stop_loss=pattern.trading_levels.stop_loss,
                take_profit_1=pattern.trading_levels.take_profit_1,
                risk_reward=1.0,
                volatility_percentile=context['volatility_percentile'],
                adx=context['adx'],
                rsi=context['rsi'],
                trend_state=context['trend_state'],
                filter_enabled=self.filter_enabled,
                filter_decision=decision,
                filter_reason=reason,
                validity_score=pattern.validity_score if hasattr(pattern, 'validity_score') else 0.5
            )
            
            signals.append(signal)
            self.last_detection_time[symbol] = pattern_time
        
        return signals
    
    def log_signal(self, signal: PaperSignal):
        """Log a signal to file."""
        # Append to daily log
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(asdict(signal), indent=2) + "\n---\n")
        
        logger.info(f"Signal logged: {signal.signal_id}")
    
    def get_stats(self) -> Dict:
        """Get current paper trading statistics."""
        all_signals = self.completed_signals + list(self.active_signals.values())
        
        if not all_signals:
            return {
                'total_signals': 0,
                'passed_filter': 0,
                'failed_filter': 0,
                'wins': 0,
                'losses': 0,
                'pending': 0
            }
        
        passed = [s for s in all_signals if s.filter_decision == "PASS"]
        failed = [s for s in all_signals if s.filter_decision == "FAIL"]
        wins = [s for s in self.completed_signals if s.outcome == "WIN"]
        losses = [s for s in self.completed_signals if s.outcome == "LOSS"]
        pending = [s for s in all_signals if s.position_status == "PENDING"]
        
        return {
            'total_signals': len(all_signals),
            'passed_filter': len(passed),
            'failed_filter': len(failed),
            'wins': len(wins),
            'losses': len(losses),
            'pending': len(pending),
            'win_rate': len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0
        }
    
    async def run_scan(self):
        """Run a single scan across all symbols."""
        all_signals = []
        
        for symbol in self.symbols:
            try:
                signals = self.detect_patterns(symbol)
                
                for signal in signals:
                    self.log_signal(signal)
                    all_signals.append(signal)
                    
                    # Log to console
                    emoji = "🟢" if signal.filter_decision == "PASS" else "🔴"
                    logger.info(f"{emoji} {signal.symbol} {signal.pattern_type}")
                    logger.info(f"   Entry: {signal.entry_price:.2f}")
                    logger.info(f"   Stop: {signal.stop_loss:.2f}")
                    logger.info(f"   Target: {signal.take_profit_1:.2f}")
                    logger.info(f"   Vol%: {signal.volatility_percentile:.2f}")
                    logger.info(f"   Filter: {signal.filter_decision}")
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return all_signals
    
    def generate_log_entry(self, signal: PaperSignal) -> str:
        """Generate formatted log entry for a signal."""
        return f"""
================================================================================
SIGNAL ID: {signal.signal_id}
================================================================================
Timestamp: {signal.timestamp}
Symbol: {signal.symbol}
Timeframe: {signal.timeframe}
Pattern Type: {signal.pattern_type}

DETECTION CONTEXT:
  CHoCH Time: {signal.choch_time}
  BoS Time: {signal.bos_time}
  Head Price: {signal.head_price}
  Left Shoulder: {signal.left_shoulder_price}
  Right Shoulder: {signal.right_shoulder_price}

TRADING LEVELS:
  Entry: {signal.entry_price}
  Stop Loss: {signal.stop_loss}
  Take Profit 1: {signal.take_profit_1}
  Risk/Reward: {signal.risk_reward}

MARKET CONTEXT:
  Volatility Percentile: {signal.volatility_percentile:.2f}
  ADX: {signal.adx:.1f}
  RSI: {signal.rsi:.1f}
  Trend State: {signal.trend_state}

FILTER DECISION:
  Filter Enabled: {signal.filter_enabled}
  Decision: {signal.filter_decision}
  Reason: {signal.filter_reason}

VALIDITY SCORE: {signal.validity_score:.2f}
================================================================================
"""


def create_paper_trader(
    symbols: List[str],
    filter_enabled: bool = True
) -> PaperTradingEngine:
    """Factory function for paper trading engine."""
    return PaperTradingEngine(
        symbols=symbols,
        filter_enabled=filter_enabled
    )


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QML Paper Trading Simulator")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT",
                        help="Comma-separated list of symbols")
    parser.add_argument("--filter", type=str, choices=["on", "off"], default="on",
                        help="Enable/disable high-conviction filter")
    parser.add_argument("--scan", action="store_true",
                        help="Run single scan")
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    filter_enabled = args.filter == "on"
    
    engine = PaperTradingEngine(
        symbols=symbols,
        filter_enabled=filter_enabled
    )
    
    if args.scan:
        import asyncio
        asyncio.run(engine.run_scan())
        
        stats = engine.get_stats()
        print(f"\nScan complete. Signals: {stats['total_signals']}")

