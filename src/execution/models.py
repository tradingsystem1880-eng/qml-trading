"""
Execution Module Data Models
============================
Data structures for exchange connectivity and paper trading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT_MARKET = "take_profit_market"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position side (long/short)."""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class ForwardTestPhase(Enum):
    """Forward testing phase."""
    PHASE1_PAPER = "phase1_paper"  # 50 trades @ 0.5% risk
    PHASE2_MICRO = "phase2_micro"  # 200 trades @ 0.75% risk
    PHASE3_FULL = "phase3_full"    # 500 trades @ 1.0% risk


@dataclass
class TradeSignal:
    """Signal from pattern detection."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    score: float
    timestamp: datetime
    pattern_id: str = ""
    tier: str = "C"
    validity_score: float = 0.5

    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0

    @property
    def side(self) -> OrderSide:
        """Get order side from direction."""
        return OrderSide.BUY if self.direction == "LONG" else OrderSide.SELL


@dataclass
class Order:
    """Exchange order."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: Optional[float]
    quantity: float
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    fee: float = 0.0
    fee_currency: str = "USDT"
    client_order_id: str = ""
    reduce_only: bool = False
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_open(self) -> bool:
        return self.status in [OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]


@dataclass
class Position:
    """Exchange position."""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: int = 1
    liquidation_price: Optional[float] = None
    mark_price: Optional[float] = None
    margin_type: str = "isolated"

    # Trade management
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    signal: Optional[TradeSignal] = None
    bars_held: int = 0
    highest_price: Optional[float] = None  # For trailing stop
    lowest_price: Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.side != PositionSide.NONE and self.quantity > 0

    @property
    def notional_value(self) -> float:
        """Position value in quote currency."""
        return self.quantity * self.entry_price

    @property
    def risk_amount(self) -> float:
        """Amount at risk (position to stop loss)."""
        if self.stop_loss_price is None:
            return 0.0
        if self.side == PositionSide.LONG:
            return (self.entry_price - self.stop_loss_price) * self.quantity
        else:
            return (self.stop_loss_price - self.entry_price) * self.quantity

    def update_trailing_extreme(self, current_price: float):
        """Update highest/lowest price for trailing stop."""
        if self.highest_price is None:
            self.highest_price = current_price
        if self.lowest_price is None:
            self.lowest_price = current_price

        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)


@dataclass
class AccountBalance:
    """Account balance information."""
    total_equity: float
    available_balance: float
    margin_used: float
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    currency: str = "USDT"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def margin_ratio(self) -> float:
        """Margin utilization ratio."""
        if self.total_equity <= 0:
            return 0.0
        return self.margin_used / self.total_equity


@dataclass
class PhaseConfig:
    """Configuration for a forward test phase."""
    phase: ForwardTestPhase
    min_trades: int
    risk_per_trade_pct: float
    max_daily_loss_pct: float = 2.0
    max_positions: int = 3
    min_pf_progress: float = 1.5
    min_wr_progress: float = 0.45
    pause_pf: float = 1.0
    pause_wr: float = 0.40
    shutdown_pf: float = 0.7
    shutdown_consecutive_losses: int = 5

    @classmethod
    def phase1(cls) -> 'PhaseConfig':
        """Phase 1: 50 trades @ 0.5% risk."""
        return cls(
            phase=ForwardTestPhase.PHASE1_PAPER,
            min_trades=50,
            risk_per_trade_pct=0.5,
            max_daily_loss_pct=2.0,
            max_positions=2,
            min_pf_progress=1.5,
            min_wr_progress=0.45,
            pause_pf=1.0,
            pause_wr=0.40,
            shutdown_pf=0.7,
            shutdown_consecutive_losses=5,
        )

    @classmethod
    def phase2(cls) -> 'PhaseConfig':
        """Phase 2: 200 trades @ 0.75% risk."""
        return cls(
            phase=ForwardTestPhase.PHASE2_MICRO,
            min_trades=200,
            risk_per_trade_pct=0.75,
            max_daily_loss_pct=2.5,
            max_positions=3,
            min_pf_progress=1.8,
            min_wr_progress=0.48,
            pause_pf=1.2,
            pause_wr=0.42,
            shutdown_pf=0.8,
            shutdown_consecutive_losses=6,
        )

    @classmethod
    def phase3(cls) -> 'PhaseConfig':
        """Phase 3: 500 trades @ 1.0% risk."""
        return cls(
            phase=ForwardTestPhase.PHASE3_FULL,
            min_trades=500,
            risk_per_trade_pct=1.0,
            max_daily_loss_pct=3.0,
            max_positions=4,
            min_pf_progress=2.0,
            min_wr_progress=0.50,
            pause_pf=1.5,
            pause_wr=0.45,
            shutdown_pf=1.0,
            shutdown_consecutive_losses=7,
        )


@dataclass
class CompletedTrade:
    """Record of a completed trade for forward testing."""
    id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl_usd: float
    pnl_r: float
    exit_reason: str  # "stop_loss", "take_profit", "time_exit", "manual"
    fees: float = 0.0
    bars_held: int = 0
    signal_score: float = 0.0
    pattern_tier: str = "C"

    @property
    def is_winner(self) -> bool:
        return self.pnl_r > 0


@dataclass
class ForwardTestState:
    """Current state of forward testing."""
    phase: ForwardTestPhase
    phase_config: PhaseConfig
    completed_trades: List[CompletedTrade] = field(default_factory=list)
    open_positions: List[Position] = field(default_factory=list)
    total_pnl_r: float = 0.0
    consecutive_losses: int = 0
    daily_pnl_r: float = 0.0
    last_trade_date: Optional[datetime] = None
    is_paused: bool = False
    pause_reason: Optional[str] = None
    is_shutdown: bool = False
    shutdown_reason: Optional[str] = None

    @property
    def trade_count(self) -> int:
        return len(self.completed_trades)

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        winners = sum(1 for t in self.completed_trades if t.is_winner)
        return winners / self.trade_count

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_r for t in self.completed_trades if t.pnl_r > 0)
        gross_loss = abs(sum(t.pnl_r for t in self.completed_trades if t.pnl_r <= 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def expectancy(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.total_pnl_r / self.trade_count

    def should_progress(self) -> bool:
        """Check if conditions met to progress to next phase."""
        if self.is_paused or self.is_shutdown:
            return False
        if self.trade_count < self.phase_config.min_trades:
            return False
        return (self.profit_factor >= self.phase_config.min_pf_progress and
                self.win_rate >= self.phase_config.min_wr_progress)

    def should_pause(self) -> bool:
        """Check if trading should be paused."""
        if self.trade_count < 10:
            return False
        return (self.profit_factor < self.phase_config.pause_pf or
                self.win_rate < self.phase_config.pause_wr)

    def should_shutdown(self) -> bool:
        """Check if forward test should be terminated."""
        if self.consecutive_losses >= self.phase_config.shutdown_consecutive_losses:
            return True
        if self.trade_count >= 20:
            return self.profit_factor < self.phase_config.shutdown_pf
        return False
