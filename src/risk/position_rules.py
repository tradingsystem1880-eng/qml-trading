"""
Position Sizing & Risk Rules for QML Trading System
=====================================================
Consolidated risk management rules for Phase 9.0 paper trading.

Based on:
- Phase 7.9 baseline performance (PF 1.23, WR 52%)
- Kelly criterion analysis (negative full Kelly -> use fixed fractional)
- Prop firm constraints (8% max DD, 4% daily limit)

Rules:
1. Fixed 1% risk per trade (conservative, ML rejected)
2. Max 3 concurrent positions
3. 4% daily loss limit (stop trading for day)
4. Pause after 7 consecutive losses
5. Reduce size 50% after 4 consecutive losses
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, date
import numpy as np


@dataclass
class RiskConfig:
    """Configuration for risk management rules."""

    # Position sizing
    risk_per_trade_pct: float = 0.01  # 1% per trade (fixed fractional)
    max_concurrent_positions: int = 3

    # Daily limits
    daily_loss_limit_pct: float = 0.04  # Stop trading after 4% daily loss
    daily_profit_target_pct: float = 0.06  # Optional: reduce size after 6% gain

    # Monthly limits (for prop firm)
    monthly_loss_limit_pct: float = 0.08  # Max 8% monthly drawdown

    # Consecutive loss management
    pause_after_consecutive_losses: int = 7  # Based on p=0.52, prob=0.48^7 ~ 0.5%
    reduce_size_after_losses: int = 4  # Cut size 50% after 4 losses
    size_reduction_factor: float = 0.5  # Reduce to 50%

    # Minimum and maximum bounds
    min_risk_pct: float = 0.005  # Never below 0.5%
    max_risk_pct: float = 0.02  # Never above 2%


@dataclass
class DailyStats:
    """Track daily trading statistics."""
    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_pct: float = 0.0
    pnl_r: float = 0.0
    consecutive_losses: int = 0


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    can_trade: bool
    risk_pct: float
    risk_dollars: float
    position_size_dollars: float
    warnings: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    rationale: str = ""


class PositionRulesManager:
    """
    Manages position sizing and risk rules for paper trading.

    Usage:
        manager = PositionRulesManager(RiskConfig())

        # Before each trade
        result = manager.calculate_position_size(
            account_equity=100_000,
            entry_price=50_000,
            stop_loss_price=49_000,
        )

        if result.can_trade:
            # Execute trade with result.position_size_dollars
            pass
        else:
            # Trade blocked, see result.blocks
            pass

        # After trade closes
        manager.record_trade_result(pnl_pct=0.02, pnl_r=1.5, is_win=True)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        """Initialize position rules manager."""
        self.config = config or RiskConfig()
        self.today_stats = DailyStats(date=date.today())
        self.consecutive_losses = 0
        self.current_positions = 0
        self.trade_history: List[Dict] = []

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> PositionSizeResult:
        """
        Calculate position size with all risk rule checks.

        Args:
            account_equity: Current account equity in dollars
            entry_price: Trade entry price
            stop_loss_price: Stop loss price

        Returns:
            PositionSizeResult with sizing details and any blocks/warnings
        """
        cfg = self.config
        result = PositionSizeResult(
            can_trade=True,
            risk_pct=0,
            risk_dollars=0,
            position_size_dollars=0,
        )

        # Check if new day, reset daily stats
        if date.today() != self.today_stats.date:
            self._reset_daily_stats()

        # === Rule 1: Check daily loss limit ===
        if self.today_stats.pnl_pct <= -cfg.daily_loss_limit_pct:
            result.can_trade = False
            result.blocks.append(
                f"Daily loss limit hit ({self.today_stats.pnl_pct:.1%} <= -{cfg.daily_loss_limit_pct:.1%})"
            )
            result.rationale = "BLOCKED: Daily loss limit"
            return result

        # === Rule 2: Check consecutive losses ===
        if self.consecutive_losses >= cfg.pause_after_consecutive_losses:
            result.can_trade = False
            result.blocks.append(
                f"{self.consecutive_losses} consecutive losses - pause required"
            )
            result.rationale = "BLOCKED: Too many consecutive losses"
            return result

        # === Rule 3: Check max concurrent positions ===
        if self.current_positions >= cfg.max_concurrent_positions:
            result.can_trade = False
            result.blocks.append(
                f"Max concurrent positions ({cfg.max_concurrent_positions}) reached"
            )
            result.rationale = "BLOCKED: Max positions"
            return result

        # === Rule 4: Validate stop loss ===
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            result.can_trade = False
            result.blocks.append("Invalid stop loss (zero distance)")
            result.rationale = "BLOCKED: Invalid SL"
            return result

        # === Calculate base risk ===
        base_risk_pct = cfg.risk_per_trade_pct

        # === Rule 5: Reduce size after consecutive losses ===
        if self.consecutive_losses >= cfg.reduce_size_after_losses:
            base_risk_pct *= cfg.size_reduction_factor
            result.warnings.append(
                f"Size reduced {(1-cfg.size_reduction_factor):.0%} after {self.consecutive_losses} losses"
            )

        # === Rule 6: Optional reduction after daily profit ===
        if self.today_stats.pnl_pct >= cfg.daily_profit_target_pct:
            base_risk_pct *= 0.75  # Reduce by 25% to protect gains
            result.warnings.append(
                f"Size reduced 25% to protect daily gains ({self.today_stats.pnl_pct:.1%})"
            )

        # === Clamp to bounds ===
        final_risk_pct = max(min(base_risk_pct, cfg.max_risk_pct), cfg.min_risk_pct)

        # === Calculate dollar amounts ===
        risk_dollars = account_equity * final_risk_pct
        stop_distance_pct = stop_distance / entry_price
        position_size = risk_dollars / stop_distance_pct

        result.risk_pct = final_risk_pct
        result.risk_dollars = risk_dollars
        result.position_size_dollars = position_size
        result.rationale = f"Fixed {final_risk_pct:.2%} risk"

        return result

    def record_trade_result(
        self,
        pnl_pct: float,
        pnl_r: float,
        is_win: bool,
    ):
        """
        Record a completed trade result.

        Args:
            pnl_pct: P&L as percentage of account
            pnl_r: P&L in R-multiples
            is_win: Whether trade was a winner
        """
        # Update daily stats
        self.today_stats.trades += 1
        self.today_stats.pnl_pct += pnl_pct
        self.today_stats.pnl_r += pnl_r

        if is_win:
            self.today_stats.wins += 1
            self.consecutive_losses = 0
        else:
            self.today_stats.losses += 1
            self.consecutive_losses += 1

        # Track history
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'pnl_pct': pnl_pct,
            'pnl_r': pnl_r,
            'is_win': is_win,
            'consecutive_losses': self.consecutive_losses,
        })

    def open_position(self):
        """Record that a new position was opened."""
        self.current_positions += 1

    def close_position(self):
        """Record that a position was closed."""
        self.current_positions = max(0, self.current_positions - 1)

    def reset_consecutive_losses(self):
        """Manually reset consecutive loss counter (e.g., after review/pause)."""
        self.consecutive_losses = 0

    def get_status(self) -> Dict:
        """Get current risk status."""
        cfg = self.config

        return {
            'date': self.today_stats.date.isoformat(),
            'daily_trades': self.today_stats.trades,
            'daily_pnl_pct': self.today_stats.pnl_pct,
            'daily_pnl_r': self.today_stats.pnl_r,
            'consecutive_losses': self.consecutive_losses,
            'current_positions': self.current_positions,
            'daily_limit_remaining': cfg.daily_loss_limit_pct + self.today_stats.pnl_pct,
            'can_trade': self._can_trade_now(),
            'warnings': self._get_warnings(),
        }

    def _can_trade_now(self) -> bool:
        """Check if trading is currently allowed."""
        cfg = self.config

        if self.today_stats.pnl_pct <= -cfg.daily_loss_limit_pct:
            return False
        if self.consecutive_losses >= cfg.pause_after_consecutive_losses:
            return False
        if self.current_positions >= cfg.max_concurrent_positions:
            return False

        return True

    def _get_warnings(self) -> List[str]:
        """Get list of current warnings."""
        cfg = self.config
        warnings = []

        # Approaching daily limit
        daily_remaining = cfg.daily_loss_limit_pct + self.today_stats.pnl_pct
        if 0 < daily_remaining < cfg.daily_loss_limit_pct * 0.5:
            warnings.append(f"Approaching daily limit ({daily_remaining:.1%} remaining)")

        # Consecutive losses building
        if cfg.reduce_size_after_losses <= self.consecutive_losses < cfg.pause_after_consecutive_losses:
            warnings.append(f"{self.consecutive_losses} consecutive losses (size reduced)")

        return warnings

    def _reset_daily_stats(self):
        """Reset stats for a new trading day."""
        self.today_stats = DailyStats(date=date.today())


def calculate_kelly_fraction(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
) -> float:
    """
    Calculate Kelly fraction from performance stats.

    Note: Phase 7.9 showed negative Kelly, so we use fixed fractional instead.
    This function is provided for reference.

    Args:
        win_rate: Historical win rate (0-1)
        avg_win_r: Average R-multiple on wins
        avg_loss_r: Average R-multiple on losses (absolute value)

    Returns:
        Kelly fraction (can be negative if no edge)
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    if avg_loss_r <= 0 or avg_win_r <= 0:
        return 0.0

    p = win_rate
    q = 1 - win_rate
    b = avg_win_r / avg_loss_r

    # Kelly formula: f* = (p * b - q) / b
    kelly = (p * b - q) / b

    return kelly


def print_status(manager: PositionRulesManager):
    """Print human-readable risk status."""
    status = manager.get_status()

    print("\n" + "=" * 50)
    print("RISK STATUS")
    print("=" * 50)
    print(f"Date: {status['date']}")
    print(f"Daily Trades: {status['daily_trades']}")
    print(f"Daily P&L: {status['daily_pnl_pct']:.2%} ({status['daily_pnl_r']:.1f}R)")
    print(f"Consecutive Losses: {status['consecutive_losses']}")
    print(f"Open Positions: {status['current_positions']}")
    print(f"Daily Limit Remaining: {status['daily_limit_remaining']:.2%}")
    print(f"Can Trade: {'✅ Yes' if status['can_trade'] else '❌ No'}")

    if status['warnings']:
        print("\nWarnings:")
        for w in status['warnings']:
            print(f"  ⚠️  {w}")

    print("=" * 50)
