"""
Monitoring Module
=================
Real-time monitoring and alerting for the QML trading system.

Phase 9.7 additions:
- EdgeDegradationMonitor: Real-time edge degradation detection
"""

from .edge_monitor import (
    EdgeDegradationMonitor,
    EdgeMonitorConfig,
    EdgeAlert,
    AlertSeverity,
)

__all__ = [
    'EdgeDegradationMonitor',
    'EdgeMonitorConfig',
    'EdgeAlert',
    'AlertSeverity',
]
