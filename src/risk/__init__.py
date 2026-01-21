"""
Risk Management Module
======================
Prop firm compliant position sizing and compliance tracking.

Components:
- KellyPositionSizer: Kelly criterion adapted for prop firm limits
- PropFirmTracker: Real-time compliance tracking

Usage:
    from src.risk import KellyPositionSizer, PropFirmTracker
    from src.validation.monte_carlo import PropFirmRules

    rules = PropFirmRules(
        account_size=100000,
        daily_loss_limit_pct=4.0,
        total_loss_limit_pct=8.0,
        profit_target_pct=8.0
    )

    # Position sizing
    sizer = KellyPositionSizer(rules, kelly_fraction=0.5)
    result = sizer.calculate(win_rate=0.55, avg_win=200, avg_loss=100,
                             current_equity=100000, stop_loss_pct=0.02)
    print(f"Position: ${result.position_size_dollars:,.0f}")

    # Compliance tracking
    tracker = PropFirmTracker(rules)
    status = tracker.update(current_equity=99000)
    print(f"Status: {status.status}, Daily DD: {status.daily_dd_pct:.2f}%")
"""

from .kelly_sizer import KellyPositionSizer, KellyResult
from .prop_firm_tracker import PropFirmTracker, PropFirmStatus, DailyStats

__all__ = [
    'KellyPositionSizer',
    'KellyResult',
    'PropFirmTracker',
    'PropFirmStatus',
    'DailyStats',
]
