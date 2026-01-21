"""
Prop Firm Compliance Tracker
============================
Real-time tracking for prop firm challenge compliance.

Tracks:
- Daily drawdown (resets each day)
- Total drawdown (trailing from peak)
- Profit progress toward target
- Consistency rule (no day > 30% of profits)
- Trading day count
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Dict, Any

from src.validation.monte_carlo import PropFirmRules


@dataclass
class DailyStats:
    """Stats for a single trading day."""
    date: date
    starting_equity: float
    ending_equity: float
    pnl: float
    pnl_pct: float
    trades: int
    max_intraday_dd: float = 0.0


@dataclass
class PropFirmStatus:
    """Current prop firm compliance status."""
    # Account state
    starting_balance: float
    current_equity: float
    peak_equity: float

    # Drawdown tracking
    daily_dd_pct: float
    daily_dd_limit_pct: float
    daily_dd_usage_pct: float  # How much of limit is used (0-100)

    total_dd_pct: float
    total_dd_limit_pct: float
    total_dd_usage_pct: float  # How much of limit is used (0-100)

    # Profit progress
    profit_pct: float
    profit_target_pct: float
    profit_progress_pct: float  # How close to target (0-100+)

    # Status
    status: str  # 'ON_TRACK', 'WARNING', 'VIOLATED', 'PASSED'
    days_traded: int
    min_days_required: int

    # Consistency (if rule enabled)
    consistency_ok: bool
    max_single_day_profit_pct: float

    # Alerts
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'starting_balance': self.starting_balance,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'daily_dd_pct': round(self.daily_dd_pct, 2),
            'daily_dd_limit_pct': self.daily_dd_limit_pct,
            'daily_dd_usage_pct': round(self.daily_dd_usage_pct, 1),
            'total_dd_pct': round(self.total_dd_pct, 2),
            'total_dd_limit_pct': self.total_dd_limit_pct,
            'total_dd_usage_pct': round(self.total_dd_usage_pct, 1),
            'profit_pct': round(self.profit_pct, 2),
            'profit_target_pct': self.profit_target_pct,
            'profit_progress_pct': round(self.profit_progress_pct, 1),
            'status': self.status,
            'days_traded': self.days_traded,
            'min_days_required': self.min_days_required,
            'consistency_ok': self.consistency_ok,
            'max_single_day_profit_pct': round(self.max_single_day_profit_pct, 1),
            'alerts': self.alerts,
        }


class PropFirmTracker:
    """
    Tracks prop firm compliance in real-time.

    Usage:
        rules = PropFirmRules(account_size=100000, max_daily_dd_pct=4.0)
        tracker = PropFirmTracker(rules)

        # Update throughout the day
        status = tracker.update(current_equity=99500)
        print(f"Status: {status.status}")
        print(f"Daily DD: {status.daily_dd_pct:.2f}%")

        # At end of day
        tracker.end_day(ending_equity=99800, trades=3)
    """

    def __init__(self, rules: PropFirmRules):
        """
        Initialize tracker with prop firm rules.

        Args:
            rules: PropFirmRules defining the challenge parameters
        """
        self.rules = rules
        self.starting_balance = rules.account_size
        self.peak_equity = rules.account_size
        self.daily_stats: List[DailyStats] = []
        self.current_day_pnl = 0.0
        self.current_day_start = rules.account_size
        self.current_day_trades = 0
        self.max_intraday_dd_today = 0.0

    def update(
        self,
        current_equity: float,
        trade_pnl: Optional[float] = None
    ) -> PropFirmStatus:
        """
        Update tracker with current equity.

        Call this after each trade or periodically throughout the day.

        Args:
            current_equity: Current account equity
            trade_pnl: P&L from a trade if just closed (for trade counting)

        Returns:
            PropFirmStatus with all compliance metrics
        """
        # Track trades
        if trade_pnl is not None:
            self.current_day_trades += 1

        # Update peak (high water mark)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate drawdowns
        daily_dd = (self.current_day_start - current_equity) / self.current_day_start * 100
        daily_dd = max(0, daily_dd)  # Only positive values

        # Track max intraday drawdown
        if daily_dd > self.max_intraday_dd_today:
            self.max_intraday_dd_today = daily_dd

        total_dd = (self.peak_equity - current_equity) / self.peak_equity * 100
        total_dd = max(0, total_dd)

        # Profit progress
        profit = (current_equity - self.starting_balance) / self.starting_balance * 100
        profit_progress = (profit / self.rules.profit_target_pct * 100
                          if self.rules.profit_target_pct > 0 else 0)

        # DD usage percentages (0-100 scale)
        daily_usage = (daily_dd / self.rules.daily_loss_limit_pct * 100
                       if self.rules.daily_loss_limit_pct > 0 else 0)
        total_usage = (total_dd / self.rules.total_loss_limit_pct * 100
                       if self.rules.total_loss_limit_pct > 0 else 0)

        # Determine status and generate alerts
        alerts = []
        status = self._determine_status(
            daily_dd, total_dd, profit, daily_usage, total_usage, alerts
        )

        # Consistency check
        consistency_ok, max_day_profit_pct = self._check_consistency(profit)
        if not consistency_ok and self.rules.consistency_rule:
            alerts.append(f"Consistency rule: single day is {max_day_profit_pct:.0f}% of profits")

        return PropFirmStatus(
            starting_balance=self.starting_balance,
            current_equity=current_equity,
            peak_equity=self.peak_equity,
            daily_dd_pct=daily_dd,
            daily_dd_limit_pct=self.rules.daily_loss_limit_pct,
            daily_dd_usage_pct=min(100, daily_usage),
            total_dd_pct=total_dd,
            total_dd_limit_pct=self.rules.total_loss_limit_pct,
            total_dd_usage_pct=min(100, total_usage),
            profit_pct=profit,
            profit_target_pct=self.rules.profit_target_pct,
            profit_progress_pct=min(200, max(0, profit_progress)),  # Cap at 200%
            status=status,
            days_traded=len(self.daily_stats),
            min_days_required=self.rules.min_trading_days,
            consistency_ok=consistency_ok,
            max_single_day_profit_pct=max_day_profit_pct,
            alerts=alerts
        )

    def _determine_status(
        self,
        daily_dd: float,
        total_dd: float,
        profit: float,
        daily_usage: float,
        total_usage: float,
        alerts: List[str]
    ) -> str:
        """Determine overall status based on metrics."""
        # Check for violations
        if daily_dd >= self.rules.daily_loss_limit_pct:
            alerts.append(f"DAILY DD LIMIT BREACHED: {daily_dd:.2f}%")
            return 'VIOLATED'

        if total_dd >= self.rules.total_loss_limit_pct:
            alerts.append(f"TOTAL DD LIMIT BREACHED: {total_dd:.2f}%")
            return 'VIOLATED'

        # Check for pass
        if (profit >= self.rules.profit_target_pct and
                len(self.daily_stats) >= self.rules.min_trading_days):
            alerts.append("Challenge PASSED!")
            return 'PASSED'

        # Check for warnings (approaching limits)
        if daily_usage > 75:
            alerts.append(f"Daily DD at {daily_usage:.0f}% of limit")
        if total_usage > 75:
            alerts.append(f"Total DD at {total_usage:.0f}% of limit")

        if daily_usage > 75 or total_usage > 75:
            return 'WARNING'

        return 'ON_TRACK'

    def _check_consistency(self, total_profit: float) -> tuple:
        """
        Check consistency rule: no single day > 30% of total profits.

        Returns:
            Tuple of (is_consistent, max_day_profit_percentage)
        """
        if not self.daily_stats or total_profit <= 0:
            return True, 0.0

        # Calculate total profit in dollars
        total_profit_dollars = total_profit / 100 * self.starting_balance

        # Find max single day profit
        daily_profits = [max(0, d.pnl) for d in self.daily_stats]
        if not daily_profits:
            return True, 0.0

        max_day_profit = max(daily_profits)
        max_day_profit_pct = (max_day_profit / total_profit_dollars * 100
                             if total_profit_dollars > 0 else 0)

        is_consistent = max_day_profit_pct <= 30
        return is_consistent, max_day_profit_pct

    def end_day(self, ending_equity: float, trades: Optional[int] = None) -> DailyStats:
        """
        Record end of trading day and reset for next day.

        Args:
            ending_equity: Account equity at end of day
            trades: Number of trades today (uses tracked count if None)

        Returns:
            DailyStats for the completed day
        """
        daily_pnl = ending_equity - self.current_day_start
        daily_pnl_pct = daily_pnl / self.current_day_start * 100

        trades_count = trades if trades is not None else self.current_day_trades

        stats = DailyStats(
            date=date.today(),
            starting_equity=self.current_day_start,
            ending_equity=ending_equity,
            pnl=daily_pnl,
            pnl_pct=daily_pnl_pct,
            trades=trades_count,
            max_intraday_dd=self.max_intraday_dd_today
        )
        self.daily_stats.append(stats)

        # Reset for next day
        self.current_day_start = ending_equity
        self.current_day_pnl = 0.0
        self.current_day_trades = 0
        self.max_intraday_dd_today = 0.0

        return stats

    def reset(self) -> None:
        """Reset tracker to initial state (start new challenge)."""
        self.peak_equity = self.rules.account_size
        self.daily_stats = []
        self.current_day_pnl = 0.0
        self.current_day_start = self.rules.account_size
        self.current_day_trades = 0
        self.max_intraday_dd_today = 0.0

    def get_daily_history(self) -> List[Dict[str, Any]]:
        """Get daily stats history as list of dicts."""
        return [
            {
                'date': d.date.isoformat(),
                'starting_equity': d.starting_equity,
                'ending_equity': d.ending_equity,
                'pnl': d.pnl,
                'pnl_pct': round(d.pnl_pct, 2),
                'trades': d.trades,
                'max_intraday_dd': round(d.max_intraday_dd, 2)
            }
            for d in self.daily_stats
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the challenge."""
        if not self.daily_stats:
            return {
                'days_traded': 0,
                'total_trades': 0,
                'best_day_pct': 0,
                'worst_day_pct': 0,
                'avg_daily_pnl_pct': 0,
                'winning_days': 0,
                'losing_days': 0,
            }

        pnl_pcts = [d.pnl_pct for d in self.daily_stats]
        total_trades = sum(d.trades for d in self.daily_stats)
        winning_days = sum(1 for d in self.daily_stats if d.pnl > 0)
        losing_days = sum(1 for d in self.daily_stats if d.pnl < 0)

        return {
            'days_traded': len(self.daily_stats),
            'total_trades': total_trades,
            'best_day_pct': max(pnl_pcts),
            'worst_day_pct': min(pnl_pcts),
            'avg_daily_pnl_pct': sum(pnl_pcts) / len(pnl_pcts),
            'winning_days': winning_days,
            'losing_days': losing_days,
        }
