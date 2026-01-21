"""
Kelly Criterion Position Sizer
==============================
Adapted for prop firm constraints.

Standard Kelly is often too aggressive for prop firm accounts.
This module:
1. Calculates base Kelly fraction
2. Caps it to prop firm drawdown limits
3. Adjusts based on current drawdown usage
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.validation.monte_carlo import PropFirmRules


@dataclass
class KellyResult:
    """Result from Kelly calculation."""
    base_kelly: float  # Standard Kelly % (before caps)
    adjusted_kelly: float  # After prop firm caps (decimal)
    position_size_pct: float  # Final recommended size (percentage)
    position_size_dollars: float  # Position in dollars
    risk_amount: float  # Risk in dollars if stopped out

    # Constraints applied
    capped_by_daily_dd: bool
    capped_by_total_dd: bool
    capped_by_max_position: bool

    # Risk metrics
    effective_leverage: float
    max_loss_if_stopped: float


class KellyPositionSizer:
    """
    Kelly Criterion with prop firm adaptations.

    Formula: f* = (p * b - q) / b
    Where:
        f* = Kelly fraction
        p = win probability
        b = win/loss ratio (avg win / avg loss)
        q = 1 - p (loss probability)

    Prop firm adaptations:
    1. Use Half-Kelly or less (reduce variance)
    2. Cap to max position size rule
    3. Never risk more than 20% of remaining daily DD buffer per trade
    4. Scale down as approaching total DD limit

    Usage:
        rules = PropFirmRules(account_size=100000, max_daily_dd_pct=4.0)
        sizer = KellyPositionSizer(rules, kelly_fraction=0.5)

        result = sizer.calculate(
            win_rate=0.55,
            avg_win=200,
            avg_loss=100,
            current_equity=100000,
            stop_loss_pct=0.02
        )
        print(f"Position size: ${result.position_size_dollars:,.0f}")
    """

    def __init__(self, rules: PropFirmRules, kelly_fraction: float = 0.5):
        """
        Args:
            rules: Prop firm rules
            kelly_fraction: Fraction of Kelly to use (0.5 = Half-Kelly)
        """
        self.rules = rules
        self.kelly_fraction = kelly_fraction

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_equity: float,
        daily_pnl: float = 0.0,
        peak_equity: Optional[float] = None,
        stop_loss_pct: float = 0.02
    ) -> KellyResult:
        """
        Calculate optimal position size.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade (absolute value in dollars)
            avg_loss: Average losing trade (absolute value in dollars)
            current_equity: Current account equity
            daily_pnl: Today's P&L so far (negative if losing)
            peak_equity: Peak equity for trailing DD (None = use account_size)
            stop_loss_pct: Stop loss as % of position (e.g., 0.02 = 2%)

        Returns:
            KellyResult with position sizing details
        """
        peak_equity = peak_equity or self.rules.account_size

        # === 1. Calculate Base Kelly ===
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            base_kelly = 0.0
        else:
            p = win_rate
            q = 1 - p
            b = avg_win / avg_loss  # Win/loss ratio

            base_kelly = (p * b - q) / b
            base_kelly = max(0, base_kelly)  # Can't be negative

        # Apply Kelly fraction (e.g., Half-Kelly)
        fractional_kelly = base_kelly * self.kelly_fraction

        # === 2. Apply Prop Firm Caps ===
        capped_by_daily = False
        capped_by_total = False
        capped_by_max = False

        # Cap 1: Max position size rule
        max_position = self.rules.max_position_size_pct / 100
        if fractional_kelly > max_position:
            fractional_kelly = max_position
            capped_by_max = True

        # Cap 2: Daily drawdown buffer
        # Never risk more than 20% of remaining daily buffer per trade
        daily_dd_limit = self.rules.daily_loss_limit_pct / 100 * self.rules.account_size
        daily_dd_remaining = daily_dd_limit + daily_pnl  # daily_pnl is negative if losing

        if daily_dd_remaining > 0:
            # Max risk is 20% of remaining daily buffer
            max_risk_daily = (daily_dd_remaining * 0.20) / current_equity
            if fractional_kelly > max_risk_daily:
                fractional_kelly = max_risk_daily
                capped_by_daily = True

        # Cap 3: Total drawdown buffer (trailing from peak)
        total_dd_limit = self.rules.total_loss_limit_pct / 100 * peak_equity
        current_dd = peak_equity - current_equity
        dd_remaining = total_dd_limit - current_dd

        # Scale down as approaching limit: reduce by ratio of DD used
        if total_dd_limit > 0:
            dd_usage_ratio = current_dd / total_dd_limit
            if dd_usage_ratio > 0.5:  # Start scaling after 50% DD used
                scale_factor = 1 - (dd_usage_ratio - 0.5) * 2  # Linear scale 1 -> 0
                scaled_kelly = fractional_kelly * max(0.1, scale_factor)
                if scaled_kelly < fractional_kelly:
                    fractional_kelly = scaled_kelly
                    capped_by_total = True

        # === 3. Calculate Final Position ===
        position_size_pct = max(0, fractional_kelly) * 100  # As percentage
        position_size_dollars = (position_size_pct / 100) * current_equity
        risk_amount = position_size_dollars * stop_loss_pct

        # Effective leverage (position / risk)
        effective_leverage = position_size_pct / (stop_loss_pct * 100) if stop_loss_pct > 0 else 0

        return KellyResult(
            base_kelly=base_kelly * 100,  # As percentage
            adjusted_kelly=fractional_kelly,
            position_size_pct=position_size_pct,
            position_size_dollars=position_size_dollars,
            risk_amount=risk_amount,
            capped_by_daily_dd=capped_by_daily,
            capped_by_total_dd=capped_by_total,
            capped_by_max_position=capped_by_max,
            effective_leverage=effective_leverage,
            max_loss_if_stopped=risk_amount
        )

    def calculate_from_trades(
        self,
        trades: list,
        current_equity: float,
        daily_pnl: float = 0.0,
        peak_equity: Optional[float] = None,
        stop_loss_pct: float = 0.02
    ) -> KellyResult:
        """
        Calculate position size from historical trades.

        Args:
            trades: List of trade dicts with 'pnl' or 'pnl_pct' field
            current_equity: Current account equity
            daily_pnl: Today's P&L so far
            peak_equity: Peak equity for trailing DD
            stop_loss_pct: Stop loss as % of position

        Returns:
            KellyResult with position sizing details
        """
        if not trades:
            return KellyResult(
                base_kelly=0, adjusted_kelly=0, position_size_pct=0,
                position_size_dollars=0, risk_amount=0,
                capped_by_daily_dd=False, capped_by_total_dd=False,
                capped_by_max_position=False,
                effective_leverage=0, max_loss_if_stopped=0
            )

        # Extract P&L values
        pnls = []
        for trade in trades:
            if isinstance(trade, dict):
                pnl = trade.get('pnl') or trade.get('pnl_pct', 0)
            else:
                pnl = getattr(trade, 'pnl', None) or getattr(trade, 'pnl_pct', 0)
            pnls.append(pnl)

        pnls = np.array(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0

        return self.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_equity=current_equity,
            daily_pnl=daily_pnl,
            peak_equity=peak_equity,
            stop_loss_pct=stop_loss_pct
        )
