"""
Edge Degradation Monitor for QML Trading System
================================================
Real-time monitoring and alerting for edge degradation.

Based on Phase 9.4 validated metrics:
- Baseline WR: 55%
- Baseline PF: 4.49
- Expected expectancy: 1.15R+ (conservative estimate)

Alerts when:
- WR drops >10pp below baseline
- PF drops >50% below baseline
- Consecutive losses exceed threshold
- Rolling metrics show sustained degradation

Usage:
    from src.monitoring.edge_monitor import EdgeDegradationMonitor

    monitor = EdgeDegradationMonitor({
        'profit_factor': 4.49,
        'win_rate': 0.55,
        'expectancy': 1.15,
    })

    # After each trade
    monitor.add_trade({'pnl_r': 1.5, 'is_winner': True})

    # Check for alerts
    if monitor.alerts:
        for alert in monitor.alerts:
            print(f"[{alert.severity}] {alert.message}")

    # Get current status
    print(monitor.get_status())
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class EdgeAlert:
    """Single alert instance."""
    timestamp: datetime
    severity: AlertSeverity
    alert_type: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    baseline_value: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'alert_type': self.alert_type,
            'message': self.message,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'baseline_value': self.baseline_value,
        }


@dataclass
class EdgeMonitorConfig:
    """Configuration for edge monitoring."""
    # WR degradation thresholds
    wr_warning_delta: float = 0.08  # 8pp drop = warning
    wr_critical_delta: float = 0.10  # 10pp drop = critical

    # PF degradation thresholds
    pf_warning_pct: float = 0.30  # 30% drop = warning
    pf_critical_pct: float = 0.50  # 50% drop = critical

    # Consecutive loss thresholds
    consecutive_loss_warning: int = 5
    consecutive_loss_critical: int = 7

    # Rolling window settings
    rolling_window: int = 20
    min_trades_for_alert: int = 10

    # Recovery tracking
    trades_for_recovery: int = 10  # Trades without alerts to clear


@dataclass
class TradeRecord:
    """Individual trade record."""
    timestamp: datetime
    pnl_r: float
    is_winner: bool
    symbol: Optional[str] = None
    direction: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class EdgeDegradationMonitor:
    """
    Real-time edge degradation monitor.

    Tracks running metrics and generates alerts when performance
    degrades beyond acceptable thresholds.
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        config: Optional[EdgeMonitorConfig] = None,
    ):
        """
        Initialize edge monitor.

        Args:
            baseline_metrics: Dict with 'profit_factor', 'win_rate', 'expectancy'
            config: Monitoring configuration
        """
        self.baseline = baseline_metrics
        self.config = config or EdgeMonitorConfig()

        # Trade tracking
        self.trades: List[TradeRecord] = []
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # Running totals
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._winners = 0
        self._losers = 0

        # Alert tracking
        self.alerts: List[EdgeAlert] = []
        self.active_alerts: Dict[str, EdgeAlert] = {}  # alert_type -> most recent
        self._trades_since_last_alert = 0

        # State
        self.is_degraded = False
        self.degradation_severity = AlertSeverity.INFO

    def add_trade(self, trade: Dict) -> List[EdgeAlert]:
        """
        Add a completed trade and check for degradation.

        Args:
            trade: Dict with 'pnl_r' (required), optionally:
                   'is_winner', 'symbol', 'direction', 'timestamp'

        Returns:
            List of new alerts (empty if no new alerts)
        """
        pnl_r = trade['pnl_r']
        is_winner = trade.get('is_winner', pnl_r > 0)

        record = TradeRecord(
            timestamp=trade.get('timestamp', datetime.now()),
            pnl_r=pnl_r,
            is_winner=is_winner,
            symbol=trade.get('symbol'),
            direction=trade.get('direction'),
            metadata=trade.get('metadata', {}),
        )

        self.trades.append(record)

        # Update running totals
        if is_winner:
            self._gross_profit += pnl_r
            self._winners += 1
            self.consecutive_losses = 0
        else:
            self._gross_loss += abs(pnl_r)
            self._losers += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses,
                self.consecutive_losses
            )

        # Check for alerts
        new_alerts = self._check_for_alerts()

        if new_alerts:
            self._trades_since_last_alert = 0
        else:
            self._trades_since_last_alert += 1

            # Check for recovery
            if (self._trades_since_last_alert >= self.config.trades_for_recovery
                    and self.is_degraded):
                self._check_recovery()

        return new_alerts

    def _check_for_alerts(self) -> List[EdgeAlert]:
        """Check all alert conditions and return new alerts."""
        new_alerts = []
        cfg = self.config
        n = len(self.trades)

        if n < cfg.min_trades_for_alert:
            return new_alerts

        # Calculate current metrics
        current_wr = self._winners / n if n > 0 else 0
        current_pf = self._gross_profit / self._gross_loss if self._gross_loss > 0 else float('inf')

        baseline_wr = self.baseline.get('win_rate', 0.55)
        baseline_pf = self.baseline.get('profit_factor', 4.49)

        # Check win rate degradation
        wr_delta = baseline_wr - current_wr

        if wr_delta >= cfg.wr_critical_delta:
            alert = self._create_alert(
                AlertSeverity.CRITICAL,
                "wr_degradation",
                f"Win rate dropped {wr_delta:.1%} below baseline",
                "win_rate",
                current_wr,
                baseline_wr - cfg.wr_critical_delta,
                baseline_wr,
            )
            new_alerts.append(alert)
            self.is_degraded = True
            self.degradation_severity = AlertSeverity.CRITICAL

        elif wr_delta >= cfg.wr_warning_delta:
            alert = self._create_alert(
                AlertSeverity.WARNING,
                "wr_degradation",
                f"Win rate dropped {wr_delta:.1%} below baseline",
                "win_rate",
                current_wr,
                baseline_wr - cfg.wr_warning_delta,
                baseline_wr,
            )
            new_alerts.append(alert)
            self.is_degraded = True
            if self.degradation_severity != AlertSeverity.CRITICAL:
                self.degradation_severity = AlertSeverity.WARNING

        # Check PF degradation
        if baseline_pf > 0 and not np.isinf(current_pf):
            pf_drop_pct = (baseline_pf - current_pf) / baseline_pf

            if pf_drop_pct >= cfg.pf_critical_pct:
                alert = self._create_alert(
                    AlertSeverity.CRITICAL,
                    "pf_degradation",
                    f"Profit factor dropped {pf_drop_pct:.0%} below baseline",
                    "profit_factor",
                    current_pf,
                    baseline_pf * (1 - cfg.pf_critical_pct),
                    baseline_pf,
                )
                new_alerts.append(alert)
                self.is_degraded = True
                self.degradation_severity = AlertSeverity.CRITICAL

            elif pf_drop_pct >= cfg.pf_warning_pct:
                alert = self._create_alert(
                    AlertSeverity.WARNING,
                    "pf_degradation",
                    f"Profit factor dropped {pf_drop_pct:.0%} below baseline",
                    "profit_factor",
                    current_pf,
                    baseline_pf * (1 - cfg.pf_warning_pct),
                    baseline_pf,
                )
                new_alerts.append(alert)
                self.is_degraded = True
                if self.degradation_severity != AlertSeverity.CRITICAL:
                    self.degradation_severity = AlertSeverity.WARNING

        # Check consecutive losses
        if self.consecutive_losses >= cfg.consecutive_loss_critical:
            alert = self._create_alert(
                AlertSeverity.CRITICAL,
                "consecutive_losses",
                f"{self.consecutive_losses} consecutive losses - statistical anomaly",
                "consecutive_losses",
                self.consecutive_losses,
                cfg.consecutive_loss_critical,
                0,
            )
            new_alerts.append(alert)
            self.is_degraded = True
            self.degradation_severity = AlertSeverity.CRITICAL

        elif self.consecutive_losses >= cfg.consecutive_loss_warning:
            alert = self._create_alert(
                AlertSeverity.WARNING,
                "consecutive_losses",
                f"{self.consecutive_losses} consecutive losses",
                "consecutive_losses",
                self.consecutive_losses,
                cfg.consecutive_loss_warning,
                0,
            )
            new_alerts.append(alert)

        # Check rolling metrics (recent performance)
        if n >= cfg.rolling_window:
            rolling_alerts = self._check_rolling_metrics()
            new_alerts.extend(rolling_alerts)

        # Store alerts
        for alert in new_alerts:
            self.alerts.append(alert)
            self.active_alerts[alert.alert_type] = alert

        return new_alerts

    def _check_rolling_metrics(self) -> List[EdgeAlert]:
        """Check rolling window metrics for sustained degradation."""
        alerts = []
        cfg = self.config

        recent = self.trades[-cfg.rolling_window:]
        recent_winners = [t for t in recent if t.is_winner]
        recent_losers = [t for t in recent if not t.is_winner]

        rolling_wr = len(recent_winners) / len(recent)
        rolling_profit = sum(t.pnl_r for t in recent_winners) if recent_winners else 0
        rolling_loss = abs(sum(t.pnl_r for t in recent_losers)) if recent_losers else 0
        rolling_pf = rolling_profit / rolling_loss if rolling_loss > 0 else float('inf')

        baseline_wr = self.baseline.get('win_rate', 0.55)

        # Rolling WR below 45% is critical (near coin-flip)
        if rolling_wr < 0.45:
            alert = self._create_alert(
                AlertSeverity.CRITICAL,
                "rolling_wr_critical",
                f"Rolling {cfg.rolling_window} trades WR={rolling_wr:.0%} (< 45%)",
                "rolling_win_rate",
                rolling_wr,
                0.45,
                baseline_wr,
            )
            alerts.append(alert)
            self.is_degraded = True
            self.degradation_severity = AlertSeverity.CRITICAL

        # Rolling PF below 1.0 is critical (losing money)
        if rolling_pf < 1.0 and not np.isinf(rolling_pf):
            alert = self._create_alert(
                AlertSeverity.CRITICAL,
                "rolling_pf_critical",
                f"Rolling {cfg.rolling_window} trades PF={rolling_pf:.2f} (< 1.0 = losing)",
                "rolling_profit_factor",
                rolling_pf,
                1.0,
                self.baseline.get('profit_factor', 4.49),
            )
            alerts.append(alert)
            self.is_degraded = True
            self.degradation_severity = AlertSeverity.CRITICAL

        return alerts

    def _check_recovery(self):
        """Check if performance has recovered."""
        n = len(self.trades)
        if n < self.config.min_trades_for_alert:
            return

        current_wr = self._winners / n
        current_pf = self._gross_profit / self._gross_loss if self._gross_loss > 0 else float('inf')

        baseline_wr = self.baseline.get('win_rate', 0.55)
        baseline_pf = self.baseline.get('profit_factor', 4.49)

        # Check if metrics are back to acceptable levels
        wr_ok = (baseline_wr - current_wr) < self.config.wr_warning_delta
        pf_ok = current_pf >= baseline_pf * (1 - self.config.pf_warning_pct)

        if wr_ok and pf_ok and self.consecutive_losses < self.config.consecutive_loss_warning:
            self.is_degraded = False
            self.degradation_severity = AlertSeverity.INFO
            self.active_alerts.clear()

    def _create_alert(
        self,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        baseline_value: float,
    ) -> EdgeAlert:
        """Create an alert instance."""
        return EdgeAlert(
            timestamp=datetime.now(),
            severity=severity,
            alert_type=alert_type,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            baseline_value=baseline_value,
        )

    def get_current_metrics(self) -> Dict:
        """Get current running metrics."""
        n = len(self.trades)

        if n == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'consecutive_losses': 0,
            }

        r_returns = [t.pnl_r for t in self.trades]

        return {
            'total_trades': n,
            'winners': self._winners,
            'losers': self._losers,
            'win_rate': self._winners / n,
            'profit_factor': self._gross_profit / self._gross_loss if self._gross_loss > 0 else float('inf'),
            'expectancy': np.mean(r_returns),
            'gross_profit': self._gross_profit,
            'gross_loss': self._gross_loss,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
        }

    def get_status(self) -> Dict:
        """Get full monitor status."""
        metrics = self.get_current_metrics()

        return {
            'metrics': metrics,
            'baseline': self.baseline,
            'is_degraded': self.is_degraded,
            'degradation_severity': self.degradation_severity.value,
            'active_alerts': [a.to_dict() for a in self.active_alerts.values()],
            'total_alerts': len(self.alerts),
            'trades_since_last_alert': self._trades_since_last_alert,
        }

    def generate_report(self) -> str:
        """Generate human-readable status report."""
        metrics = self.get_current_metrics()
        status = self.get_status()

        lines = []
        lines.append("=" * 60)
        lines.append("EDGE DEGRADATION MONITOR STATUS")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Current Performance
        lines.append("-" * 60)
        lines.append("CURRENT PERFORMANCE vs BASELINE")
        lines.append("-" * 60)
        lines.append(f"{'Metric':<20} {'Current':<12} {'Baseline':<12} {'Delta':<12}")
        lines.append("-" * 60)

        baseline_wr = self.baseline.get('win_rate', 0.55)
        baseline_pf = self.baseline.get('profit_factor', 4.49)

        wr_delta = metrics['win_rate'] - baseline_wr
        pf_delta = (metrics['profit_factor'] - baseline_pf) / baseline_pf if baseline_pf > 0 else 0

        lines.append(f"{'Win Rate':<20} {metrics['win_rate']:.1%}        {baseline_wr:.1%}        {wr_delta:+.1%}")
        lines.append(f"{'Profit Factor':<20} {metrics['profit_factor']:.2f}         {baseline_pf:.2f}         {pf_delta:+.0%}")
        lines.append(f"{'Trades':<20} {metrics['total_trades']:<12}")
        lines.append("")

        # Alert Status
        lines.append("-" * 60)
        lines.append("ALERT STATUS")
        lines.append("-" * 60)

        if self.is_degraded:
            lines.append(f"Status: {self.degradation_severity.value} - DEGRADATION DETECTED")
        else:
            lines.append("Status: OK - Performance within acceptable range")

        lines.append(f"Consecutive Losses: {metrics['consecutive_losses']}")
        lines.append(f"Total Alerts: {len(self.alerts)}")
        lines.append("")

        # Active Alerts
        if self.active_alerts:
            lines.append("-" * 60)
            lines.append("ACTIVE ALERTS")
            lines.append("-" * 60)
            for alert in self.active_alerts.values():
                emoji = "🔴" if alert.severity == AlertSeverity.CRITICAL else "🟡"
                lines.append(f"{emoji} [{alert.severity.value}] {alert.message}")
        else:
            lines.append("No active alerts")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def clear_alerts(self):
        """Manually clear all alerts."""
        self.alerts = []
        self.active_alerts = {}
        self.is_degraded = False
        self.degradation_severity = AlertSeverity.INFO
        self._trades_since_last_alert = 0

    def reset(self):
        """Reset monitor to initial state."""
        self.trades = []
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._winners = 0
        self._losers = 0
        self.clear_alerts()
