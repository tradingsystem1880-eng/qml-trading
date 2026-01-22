"""Data pipeline module for fetching, storing, and managing OHLCV data."""

from src.data.fetcher import DataFetcher
from src.data.database import DatabaseManager
from src.data.models import OHLCV, SwingPoint, MarketStructure

# Phase 2: New data infrastructure
from src.data.schemas import (
    PatternDetection,
    TradeOutcome,
    FeatureVector,
    ExperimentRun,
    generate_id,
    hash_params,
)
from src.data.sqlite_manager import SQLiteManager, get_db

# Phase 8.6: Data loading utility
from src.data.loader import (
    load_ohlcv,
    quick_load,
    get_available_symbols,
    get_available_timeframes,
    check_data_availability,
)

__all__ = [
    # Original exports
    "DataFetcher",
    "DatabaseManager",
    "OHLCV",
    "SwingPoint",
    "MarketStructure",
    # Phase 2 exports
    "PatternDetection",
    "TradeOutcome",
    "FeatureVector",
    "ExperimentRun",
    "generate_id",
    "hash_params",
    "SQLiteManager",
    "get_db",
    # Phase 8.6 exports
    "load_ohlcv",
    "quick_load",
    "get_available_symbols",
    "get_available_timeframes",
    "check_data_availability",
]

