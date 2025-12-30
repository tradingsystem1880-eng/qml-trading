"""
Alert System for QML Trading System
=====================================
Provides notifications via Telegram and webhooks.
"""

from src.alerts.telegram import TelegramAlerts, AlertMessage

__all__ = ["TelegramAlerts", "AlertMessage"]

