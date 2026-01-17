"""Dashboard Core Package"""
from src.dashboard.core.integration import (
    run_vrd_validation,
    run_backtest,
    scan_for_patterns,
    update_market_data,
    get_system_status,
)

__all__ = [
    "run_vrd_validation",
    "run_backtest", 
    "scan_for_patterns",
    "update_market_data",
    "get_system_status",
]
