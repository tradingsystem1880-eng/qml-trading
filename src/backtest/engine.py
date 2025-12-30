"""
Backtesting Engine for QML Trading System
==========================================
Comprehensive backtesting with performance metrics, equity curves,
and detailed trade analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.models import PatternType, QMLPattern, TradeOutcome


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Capital
    initial_capital: float = 100000.0
    
    # Position sizing
    risk_per_trade_pct: float = 1.0  # 1% risk per trade
    max_positions: int = 5
    
    # Transaction costs
    commission_pct: float = 0.1  # 0.1% per trade
    slippage_pct: float = 0.05  # 0.05% slippage
    
    # Risk management
    max_drawdown_pct: float = 20.0  # Stop trading if hit
    
    # Trade management
    use_trailing_stop: bool = False
    trailing_stop_activation_rr: float = 1.0  # Activate at 1:1
    trailing_stop_distance_atr: float = 1.5
    
    # Filter settings
    min_validity_score: float = 0.5
    min_ml_confidence: float = 0.0  # Don't require ML confidence for backtest


@dataclass
class Trade:
    """Individual trade record."""
    
    id: int
    symbol: str
    timeframe: str
    pattern_type: PatternType
    
    entry_time: datetime
    entry_price: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    position_size: float = 0.0
    risk_amount: float = 0.0
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    outcome: Optional[TradeOutcome] = None
    exit_reason: str = ""
    
    validity_score: float = 0.0
    ml_confidence: Optional[float] = None


@dataclass
class BacktestResult:
    """Complete backtest results."""
    
    # Trade list
    trades: List[Trade] = field(default_factory=list)
    
    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_trade_pct: float = 0.0
    
    avg_holding_bars: float = 0.0
    
    # By category
    metrics_by_symbol: Dict[str, Dict] = field(default_factory=dict)
    metrics_by_timeframe: Dict[str, Dict] = field(default_factory=dict)
    metrics_by_pattern_type: Dict[str, Dict] = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine for QML patterns.
    
    Features:
    - Realistic transaction cost modeling
    - Position sizing based on risk
    - Multiple exit strategies
    - Comprehensive performance metrics
    - Equity curve and drawdown analysis
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
    
    def run(
        self,
        patterns: List[QMLPattern],
        price_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on detected patterns.
        
        Args:
            patterns: List of QML patterns to test
            price_data: Dict mapping symbol to OHLCV DataFrame
            start_date: Start of backtest period
            end_date: End of backtest period
            
        Returns:
            BacktestResult with complete analysis
        """
        # Filter patterns
        filtered_patterns = self._filter_patterns(patterns, start_date, end_date)
        
        if not filtered_patterns:
            logger.warning("No patterns to backtest after filtering")
            return BacktestResult()
        
        # Sort by detection time
        filtered_patterns.sort(key=lambda p: p.detection_time)
        
        logger.info(f"Running backtest on {len(filtered_patterns)} patterns")
        
        # Initialize state
        capital = self.config.initial_capital
        equity = [capital]
        equity_times = [filtered_patterns[0].detection_time]
        trades: List[Trade] = []
        open_positions: Dict[str, Trade] = {}
        trade_id = 0
        
        # Process each pattern
        for pattern in filtered_patterns:
            # Check if we can open new position
            if len(open_positions) >= self.config.max_positions:
                continue
            
            # Check max drawdown
            current_dd = (max(equity) - equity[-1]) / max(equity) * 100
            if current_dd >= self.config.max_drawdown_pct:
                logger.warning(f"Max drawdown reached: {current_dd:.2f}%")
                break
            
            # Get price data for symbol
            df = price_data.get(pattern.symbol)
            if df is None or df.empty:
                continue
            
            # Execute trade
            trade = self._execute_trade(
                pattern=pattern,
                df=df,
                capital=capital,
                trade_id=trade_id
            )
            
            if trade is None:
                continue
            
            trade_id += 1
            trades.append(trade)
            
            # Update capital
            capital += trade.pnl
            equity.append(capital)
            equity_times.append(trade.exit_time or pattern.detection_time)
        
        # Build equity curve
        equity_curve = pd.Series(equity, index=pd.to_datetime(equity_times))
        
        # Calculate metrics
        result = self._calculate_metrics(trades, equity_curve)
        
        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"Win Rate: {result.win_rate:.2%}, "
            f"Sharpe: {result.sharpe_ratio:.2f}"
        )
        
        return result
    
    def _filter_patterns(
        self,
        patterns: List[QMLPattern],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[QMLPattern]:
        """Filter patterns by date and quality."""
        filtered = []
        
        for p in patterns:
            # Date filter
            if start_date and p.detection_time < start_date:
                continue
            if end_date and p.detection_time > end_date:
                continue
            
            # Quality filter
            if p.validity_score < self.config.min_validity_score:
                continue
            if p.ml_confidence and p.ml_confidence < self.config.min_ml_confidence:
                continue
            
            # Must have trading levels
            if not p.trading_levels:
                continue
            
            filtered.append(p)
        
        return filtered
    
    def _execute_trade(
        self,
        pattern: QMLPattern,
        df: pd.DataFrame,
        capital: float,
        trade_id: int
    ) -> Optional[Trade]:
        """
        Execute a single trade based on pattern.
        
        Simulates entry, exit, and calculates P&L.
        """
        levels = pattern.trading_levels
        if not levels:
            return None
        
        # Find entry bar (detection time or next bar)
        entry_idx = self._find_bar_index(df, pattern.detection_time)
        if entry_idx is None or entry_idx >= len(df) - 1:
            return None
        
        # Calculate position size based on risk
        risk_per_trade = capital * (self.config.risk_per_trade_pct / 100)
        risk_amount = abs(levels.entry - levels.stop_loss)
        
        if risk_amount <= 0:
            return None
        
        position_size = risk_per_trade / risk_amount
        
        # Apply entry slippage
        entry_price = levels.entry
        if pattern.pattern_type == PatternType.BULLISH:
            entry_price *= (1 + self.config.slippage_pct / 100)
        else:
            entry_price *= (1 - self.config.slippage_pct / 100)
        
        # Create trade
        trade = Trade(
            id=trade_id,
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_type=pattern.pattern_type,
            entry_time=pattern.detection_time,
            entry_price=entry_price,
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit_3,
            position_size=position_size,
            risk_amount=risk_per_trade,
            validity_score=pattern.validity_score,
            ml_confidence=pattern.ml_confidence
        )
        
        # Simulate trade
        trade = self._simulate_trade(trade, df, entry_idx)
        
        # Calculate P&L
        trade = self._calculate_trade_pnl(trade)
        
        return trade
    
    def _simulate_trade(
        self,
        trade: Trade,
        df: pd.DataFrame,
        entry_idx: int
    ) -> Trade:
        """Simulate trade execution bar by bar."""
        
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        time = pd.to_datetime(df["time"].values)
        
        is_long = trade.pattern_type == PatternType.BULLISH
        max_bars = min(100, len(df) - entry_idx - 1)
        
        for i in range(1, max_bars):
            idx = entry_idx + i
            
            bar_high = high[idx]
            bar_low = low[idx]
            
            if is_long:
                # Check stop loss
                if bar_low <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss * (1 - self.config.slippage_pct / 100)
                    trade.exit_time = time[idx]
                    trade.outcome = TradeOutcome.LOSS
                    trade.exit_reason = "Stop Loss"
                    break
                
                # Check take profit
                if bar_high >= trade.take_profit:
                    trade.exit_price = trade.take_profit * (1 - self.config.slippage_pct / 100)
                    trade.exit_time = time[idx]
                    trade.outcome = TradeOutcome.WIN
                    trade.exit_reason = "Take Profit"
                    break
            else:
                # Short trade
                if bar_high >= trade.stop_loss:
                    trade.exit_price = trade.stop_loss * (1 + self.config.slippage_pct / 100)
                    trade.exit_time = time[idx]
                    trade.outcome = TradeOutcome.LOSS
                    trade.exit_reason = "Stop Loss"
                    break
                
                if bar_low <= trade.take_profit:
                    trade.exit_price = trade.take_profit * (1 + self.config.slippage_pct / 100)
                    trade.exit_time = time[idx]
                    trade.outcome = TradeOutcome.WIN
                    trade.exit_reason = "Take Profit"
                    break
        
        # Time exit if neither SL nor TP hit
        if trade.exit_time is None:
            final_idx = min(entry_idx + max_bars, len(df) - 1)
            trade.exit_price = close[final_idx]
            trade.exit_time = time[final_idx]
            trade.exit_reason = "Time Exit"
            
            # Determine outcome
            if is_long:
                trade.outcome = TradeOutcome.WIN if trade.exit_price > trade.entry_price else TradeOutcome.LOSS
            else:
                trade.outcome = TradeOutcome.WIN if trade.exit_price < trade.entry_price else TradeOutcome.LOSS
        
        return trade
    
    def _calculate_trade_pnl(self, trade: Trade) -> Trade:
        """Calculate trade P&L including costs."""
        
        if trade.exit_price is None:
            return trade
        
        # Gross P&L
        if trade.pattern_type == PatternType.BULLISH:
            gross_pnl = (trade.exit_price - trade.entry_price) * trade.position_size
        else:
            gross_pnl = (trade.entry_price - trade.exit_price) * trade.position_size
        
        # Commission
        entry_value = trade.entry_price * trade.position_size
        exit_value = trade.exit_price * trade.position_size
        commission = (entry_value + exit_value) * (self.config.commission_pct / 100)
        
        # Net P&L
        trade.pnl = gross_pnl - commission
        trade.pnl_pct = trade.pnl / trade.risk_amount * 100 if trade.risk_amount > 0 else 0
        
        return trade
    
    def _find_bar_index(
        self,
        df: pd.DataFrame,
        target_time: datetime
    ) -> Optional[int]:
        """Find bar index for timestamp."""
        times = pd.to_datetime(df["time"])
        target = pd.Timestamp(target_time)
        
        # Handle timezone comparison - convert both to UTC or remove tz
        if target.tz is not None:
            target = target.tz_convert('UTC').tz_localize(None)
        if len(times) > 0 and times.iloc[0].tz is not None:
            times = times.dt.tz_convert('UTC').dt.tz_localize(None)
        
        for i, t in enumerate(times):
            if t >= target:
                return i
        
        return None
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        
        result = BacktestResult(trades=trades, equity_curve=equity_curve)
        
        if not trades:
            return result
        
        # Basic counts
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.outcome == TradeOutcome.WIN)
        result.losing_trades = sum(1 for t in trades if t.outcome == TradeOutcome.LOSS)
        
        # Win rate
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        
        # P&L metrics
        wins = [t.pnl for t in trades if t.outcome == TradeOutcome.WIN]
        losses = [t.pnl for t in trades if t.outcome == TradeOutcome.LOSS]
        
        result.avg_win_pct = np.mean([t.pnl_pct for t in trades if t.outcome == TradeOutcome.WIN]) if wins else 0
        result.avg_loss_pct = np.mean([t.pnl_pct for t in trades if t.outcome == TradeOutcome.LOSS]) if losses else 0
        result.avg_trade_pct = np.mean([t.pnl_pct for t in trades])
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Returns
        if len(equity_curve) > 1:
            result.total_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
            
            # Annualized return
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days > 0:
                years = days / 365.25
                result.annualized_return_pct = ((1 + result.total_return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Sharpe and Sortino
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 1:
            # Annualized (assume daily returns, scale to annual)
            ann_factor = np.sqrt(252)
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            result.sharpe_ratio = mean_return / std_return * ann_factor if std_return > 0 else 0
            
            # Sortino (downside deviation)
            downside = returns[returns < 0]
            downside_std = downside.std() if len(downside) > 0 else std_return
            result.sortino_ratio = mean_return / downside_std * ann_factor if downside_std > 0 else 0
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        result.max_drawdown_pct = abs(drawdown.min())
        
        # Metrics by category
        result.metrics_by_symbol = self._metrics_by_category(trades, "symbol")
        result.metrics_by_timeframe = self._metrics_by_category(trades, "timeframe")
        
        return result
    
    def _metrics_by_category(
        self,
        trades: List[Trade],
        category: str
    ) -> Dict[str, Dict]:
        """Calculate metrics grouped by category."""
        
        metrics = {}
        
        # Group trades
        groups: Dict[str, List[Trade]] = {}
        for trade in trades:
            key = getattr(trade, category)
            if key not in groups:
                groups[key] = []
            groups[key].append(trade)
        
        # Calculate metrics per group
        for key, group_trades in groups.items():
            n = len(group_trades)
            wins = sum(1 for t in group_trades if t.outcome == TradeOutcome.WIN)
            
            metrics[key] = {
                "trades": n,
                "wins": wins,
                "win_rate": wins / n if n > 0 else 0,
                "total_pnl": sum(t.pnl for t in group_trades),
                "avg_pnl_pct": np.mean([t.pnl_pct for t in group_trades]),
            }
        
        return metrics
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate text report of backtest results."""
        
        lines = [
            "=" * 60,
            "QML PATTERN BACKTEST REPORT",
            "=" * 60,
            "",
            "PERFORMANCE SUMMARY",
            "-" * 40,
            f"Total Trades:           {result.total_trades}",
            f"Winning Trades:         {result.winning_trades}",
            f"Losing Trades:          {result.losing_trades}",
            f"Win Rate:               {result.win_rate:.2%}",
            "",
            f"Total Return:           {result.total_return_pct:.2f}%",
            f"Annualized Return:      {result.annualized_return_pct:.2f}%",
            "",
            f"Sharpe Ratio:           {result.sharpe_ratio:.2f}",
            f"Sortino Ratio:          {result.sortino_ratio:.2f}",
            f"Profit Factor:          {result.profit_factor:.2f}",
            "",
            f"Max Drawdown:           {result.max_drawdown_pct:.2f}%",
            "",
            f"Avg Win:                {result.avg_win_pct:.2f}%",
            f"Avg Loss:               {result.avg_loss_pct:.2f}%",
            f"Avg Trade:              {result.avg_trade_pct:.2f}%",
            "",
            "METRICS BY SYMBOL",
            "-" * 40,
        ]
        
        for symbol, metrics in result.metrics_by_symbol.items():
            lines.append(
                f"{symbol}: {metrics['trades']} trades, "
                f"{metrics['win_rate']:.2%} win rate, "
                f"${metrics['total_pnl']:.2f} P&L"
            )
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def create_backtest_engine(
    config: Optional[BacktestConfig] = None
) -> BacktestEngine:
    """Factory function for BacktestEngine."""
    return BacktestEngine(config=config)

