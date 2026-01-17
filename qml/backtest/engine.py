"""
Backtest Engine
===============
Unified backtesting interface.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class BacktestResult:
    """Backtest result container."""
    trades: pd.DataFrame
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0


class BacktestEngine:
    """
    Unified backtesting engine.
    
    Wraps existing backtest infrastructure.
    
    Example:
        engine = BacktestEngine()
        result = engine.run(patterns, initial_capital=10000)
    """
    
    def __init__(self, config=None):
        """Initialize backtest engine."""
        self.config = config
    
    def run(
        self,
        patterns: List[Dict],
        initial_capital: float = 10000.0,
        position_size_pct: float = 2.0,
        commission_pct: float = 0.1
    ) -> BacktestResult:
        """
        Run backtest on patterns.
        
        Args:
            patterns: List of detected patterns
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            commission_pct: Commission per trade
            
        Returns:
            BacktestResult with metrics
        """
        if not patterns:
            return self._empty_result()
        
        # Simple backtest logic
        trades_data = []
        capital = initial_capital
        
        for pattern in patterns:
            entry_price = pattern.get("entry_price", pattern.get("p5_price", 0))
            exit_price = pattern.get("exit_price", entry_price)
            
            if entry_price <= 0:
                continue
            
            position_size = capital * (position_size_pct / 100)
            pnl = position_size * ((exit_price - entry_price) / entry_price)
            pnl -= position_size * (commission_pct / 100) * 2  # Entry + exit
            
            capital += pnl
            
            trades_data.append({
                "entry_time": pattern.get("detection_time"),
                "exit_time": pattern.get("exit_time"),
                "type": pattern.get("type"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": (exit_price - entry_price) / entry_price * 100
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        if trades_df.empty:
            return self._empty_result()
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        returns = trades_df["pnl_pct"].values
        
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # Max drawdown
        cumulative = (1 + returns / 100).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max * 100
        max_dd = np.min(drawdowns)
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / (losses + 1e-10)
        
        return BacktestResult(
            trades=trades_df,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(trades_df),
            profit_factor=profit_factor,
            avg_trade_return=np.mean(returns)
        )
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result."""
        return BacktestResult(
            trades=pd.DataFrame(),
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0
        )
