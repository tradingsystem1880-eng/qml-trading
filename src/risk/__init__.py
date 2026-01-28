"""
Risk Management Module
======================
Prop firm compliant position sizing and compliance tracking.

Components:
- KellyPositionSizer: Kelly criterion adapted for prop firm limits
- PropFirmTracker: Real-time compliance tracking
- PositionRulesManager: Phase 9.0 consolidated risk rules
- ForwardTestMonitor: Phase 9.0 edge degradation detection

Usage:
    from src.risk import KellyPositionSizer, PropFirmTracker
    from src.risk import PositionRulesManager, ForwardTestMonitor
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

    # Phase 9.0: Forward testing monitor
    monitor = ForwardTestMonitor()
    status = monitor.record_trade({'r_multiple': 1.5, 'symbol': 'BTC', ...})
    print(monitor.generate_report())
"""

from .kelly_sizer import KellyPositionSizer, KellyResult
from .prop_firm_tracker import PropFirmTracker, PropFirmStatus, DailyStats
from .position_rules import PositionRulesManager, RiskConfig, PositionSizeResult
from .forward_monitor import ForwardTestMonitor, ForwardTestConfig, ForwardTestStatus

__all__ = [
    # Existing
    'KellyPositionSizer',
    'KellyResult',
    'PropFirmTracker',
    'PropFirmStatus',
    'DailyStats',
    # Phase 9.0
    'PositionRulesManager',
    'RiskConfig',
    'PositionSizeResult',
    'ForwardTestMonitor',
    'ForwardTestConfig',
    'ForwardTestStatus',
]
