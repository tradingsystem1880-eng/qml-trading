"""
QML Strategy Adapter for Validation Framework
==============================================
Wraps existing QML detection and backtest logic into the format
required by the ValidationOrchestrator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.detection.detector import QMLDetector, DetectorConfig, create_detector
from src.detection.swing import SwingDetector
from src.detection.structure import StructureAnalyzer
from src.detection.choch import CHoCHDetector
from src.detection.bos import BoSDetector
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.utils.indicators import calculate_atr


@dataclass
class QMLAdapterConfig:
    """Configuration for QML strategy adapter."""
    
    # Detection parameters
    min_validity_score: float = 0.7
    min_head_depth_atr: float = 0.5
    max_head_depth_atr: float = 3.0
    default_risk_reward: float = 3.0
    stop_loss_atr_multiplier: float = 1.5
    
    # Backtest parameters
    initial_capital: float = 100000.0
    risk_per_trade_pct: float = 1.0
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    
    # Symbol/timeframe
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"


class QMLStrategyAdapter:
    """
    QML Strategy Adapter.
    
    Wraps the complete QML detection and backtest pipeline into a single
    callable function compatible with the ValidationOrchestrator.
    
    Usage:
        adapter = QMLStrategyAdapter()
        metrics = adapter.run(df, params)
    """
    
    def __init__(self, config: Optional[QMLAdapterConfig] = None):
        """
        Initialize strategy adapter.
        
        Args:
            config: Adapter configuration
        """
        self.config = config or QMLAdapterConfig()
        self._detector = None
        self._engine = None
        
        logger.info(f"QMLStrategyAdapter initialized for {self.config.symbol} / {self.config.timeframe}")
    
    def _get_detector(self, params: Dict[str, Any]) -> QMLDetector:
        """Get or create detector with current params."""
        detector_config = DetectorConfig(
            min_validity_score=params.get("min_validity_score", self.config.min_validity_score),
            min_head_depth_atr=params.get("min_head_depth_atr", self.config.min_head_depth_atr),
            max_head_depth_atr=params.get("max_head_depth_atr", self.config.max_head_depth_atr),
            default_risk_reward=params.get("risk_reward", self.config.default_risk_reward),
            stop_loss_atr_multiplier=params.get("stop_loss_atr", self.config.stop_loss_atr_multiplier),
        )
        return create_detector(config=detector_config)
    
    def _get_backtest_engine(self, params: Dict[str, Any]) -> BacktestEngine:
        """Get or create backtest engine with current params."""
        bt_config = BacktestConfig(
            initial_capital=params.get("initial_capital", self.config.initial_capital),
            risk_per_trade_pct=params.get("risk_per_trade_pct", self.config.risk_per_trade_pct),
            commission_pct=params.get("commission_pct", self.config.commission_pct),
            slippage_pct=params.get("slippage_pct", self.config.slippage_pct),
            min_validity_score=params.get("min_validity_score", self.config.min_validity_score),
        )
        return BacktestEngine(config=bt_config)
    
    def run(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Run QML strategy and return performance metrics.
        
        This is the main entry point called by ValidationOrchestrator.
        
        Args:
            df: OHLCV DataFrame with columns: time, open, high, low, close, volume
            params: Strategy parameters
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Get components with current params
            detector = self._get_detector(params)
            engine = self._get_backtest_engine(params)
            
            # Detect patterns
            symbol = params.get("symbol", self.config.symbol)
            timeframe = params.get("timeframe", self.config.timeframe)
            
            patterns = detector.detect(
                symbol=symbol,
                timeframe=timeframe,
                df=df,
                lookback_bars=len(df)
            )
            
            if not patterns:
                # No patterns found - return default metrics
                return self._empty_metrics()
            
            # Run backtest
            price_data = {symbol: df}
            result = engine.run(patterns, price_data)
            
            # Extract metrics
            return self._extract_metrics(result)
            
        except Exception as e:
            logger.error(f"Strategy run failed: {e}")
            return self._empty_metrics()
    
    def _extract_metrics(self, result: Any) -> Dict[str, float]:
        """Extract metrics from BacktestResult."""
        trades = result.trades if hasattr(result, 'trades') else []
        
        if not trades:
            return self._empty_metrics()
        
        # Calculate metrics
        pnl_values = [t.pnl_pct for t in trades if hasattr(t, 'pnl_pct')]
        
        if not pnl_values:
            return self._empty_metrics()
        
        pnl_array = np.array(pnl_values)
        
        # Sharpe ratio (annualized, assuming daily returns)
        mean_return = np.mean(pnl_array)
        std_return = np.std(pnl_array) if len(pnl_array) > 1 else 0.01
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Win rate
        wins = np.sum(pnl_array > 0)
        win_rate = wins / len(pnl_array)
        
        # Profit factor
        gross_profit = np.sum(pnl_array[pnl_array > 0]) if np.any(pnl_array > 0) else 0
        gross_loss = abs(np.sum(pnl_array[pnl_array < 0])) if np.any(pnl_array < 0) else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Max drawdown
        equity = np.cumprod(1 + pnl_array / 100)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max * 100
        max_drawdown_pct = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Average win/loss
        avg_win_pct = np.mean(pnl_array[pnl_array > 0]) if np.any(pnl_array > 0) else 0
        avg_loss_pct = np.mean(pnl_array[pnl_array < 0]) if np.any(pnl_array < 0) else 0
        
        return {
            "sharpe_ratio": float(sharpe_ratio),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_drawdown_pct),
            "total_trades": len(trades),
            "total_return": float(np.sum(pnl_array)),
            "avg_win_pct": float(avg_win_pct),
            "avg_loss_pct": float(avg_loss_pct),
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty/default metrics when no trades."""
        return {
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "total_return": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }


def run_qml_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Convenience function to run QML strategy.
    
    This is the main function to pass to ValidationOrchestrator.
    
    Args:
        df: OHLCV DataFrame
        params: Strategy parameters including:
            - min_validity_score: Minimum pattern score (default 0.7)
            - risk_reward: Risk/reward ratio (default 3.0)
            - stop_loss_atr: Stop loss ATR multiplier (default 1.5)
            - risk_per_trade_pct: Risk per trade (default 1.0)
            
    Returns:
        Dictionary of metrics:
            - sharpe_ratio
            - win_rate
            - profit_factor
            - max_drawdown_pct
            - total_trades
            - total_return
    """
    adapter = QMLStrategyAdapter()
    return adapter.run(df, params)


def create_qml_adapter(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    initial_capital: float = 100000.0
) -> QMLStrategyAdapter:
    """Factory function for QMLStrategyAdapter."""
    config = QMLAdapterConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=initial_capital,
    )
    return QMLStrategyAdapter(config=config)
