"""
QML Trading Execution Module
============================
Exchange connectivity and paper trading for forward testing.

Components:
- BybitTestnetClient: CCXT wrapper for Bybit testnet
- BybitPaperTrader: Paper trading engine with real detection
"""

from .models import (
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide,
    TradeSignal,
    Order,
    Position,
    AccountBalance,
    ForwardTestPhase,
    PhaseConfig,
    CompletedTrade,
    ForwardTestState,
)
from .bybit_client import BybitTestnetClient
from .paper_trader_bybit import BybitPaperTrader

__all__ = [
    # Models
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'PositionSide',
    'TradeSignal',
    'Order',
    'Position',
    'AccountBalance',
    'ForwardTestPhase',
    'PhaseConfig',
    'CompletedTrade',
    'ForwardTestState',
    # Clients
    'BybitTestnetClient',
    'BybitPaperTrader',
]
