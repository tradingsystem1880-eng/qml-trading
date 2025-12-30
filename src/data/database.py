"""
Database Manager for QML Trading System
========================================
Handles all database operations with TimescaleDB.
Provides connection pooling, CRUD operations, and bulk inserts.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from config.settings import settings


class DatabaseManager:
    """
    Manages database connections and operations for the QML system.
    
    Features:
    - Connection pooling for performance
    - Bulk insert operations for OHLCV data
    - Pattern and event storage/retrieval
    - Error handling with automatic reconnection
    """
    
    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            connection_url: Database URL. Defaults to settings.
        """
        self.connection_url = connection_url or settings.database.url
        self._engine: Optional[Engine] = None
        self._connected = False
    
    @property
    def engine(self) -> Engine:
        """Get or create database engine with connection pooling."""
        if self._engine is None:
            self._engine = create_engine(
                self.connection_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,  # Recycle connections after 30 min
                echo=settings.debug,
            )
            self._connected = True
            logger.info("Database engine created successfully")
        return self._engine
    
    @contextmanager
    def connection(self) -> Generator:
        """
        Context manager for database connections.
        
        Yields:
            Connection object
        
        Usage:
            with db.connection() as conn:
                conn.execute(query)
        """
        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except SQLAlchemyError as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.connection() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    # =========================================================================
    # OHLCV Data Operations
    # =========================================================================
    
    def insert_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        exchange: str = "binance"
    ) -> int:
        """
        Bulk insert OHLCV data with upsert logic.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            exchange: Exchange name
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0
        
        # Prepare data
        df = df.copy()
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        df["exchange"] = exchange
        
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])
        
        # Use ON CONFLICT for upsert
        insert_query = text("""
            INSERT INTO ohlcv (time, symbol, exchange, timeframe, open, high, low, close, volume, quote_volume, trades)
            VALUES (:time, :symbol, :exchange, :timeframe, :open, :high, :low, :close, :volume, :quote_volume, :trades)
            ON CONFLICT (time, symbol, exchange, timeframe) 
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                quote_volume = EXCLUDED.quote_volume,
                trades = EXCLUDED.trades
        """)
        
        records = df.to_dict("records")
        
        # Fill optional columns
        for record in records:
            record.setdefault("quote_volume", None)
            record.setdefault("trades", None)
        
        try:
            with self.connection() as conn:
                conn.execute(insert_query, records)
            logger.debug(f"Inserted {len(records)} OHLCV records for {symbol} {timeframe}")
            return len(records)
        except SQLAlchemyError as e:
            logger.error(f"Failed to insert OHLCV data: {e}")
            raise
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        exchange: str = "binance"
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data from database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start of date range
            end_time: End of date range
            limit: Maximum number of candles (most recent)
            exchange: Exchange name
            
        Returns:
            DataFrame with OHLCV data
        """
        query_parts = [
            "SELECT time, open, high, low, close, volume, quote_volume, trades",
            "FROM ohlcv",
            "WHERE symbol = :symbol AND timeframe = :timeframe AND exchange = :exchange"
        ]
        params: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": exchange
        }
        
        if start_time:
            query_parts.append("AND time >= :start_time")
            params["start_time"] = start_time
        
        if end_time:
            query_parts.append("AND time <= :end_time")
            params["end_time"] = end_time
        
        query_parts.append("ORDER BY time DESC")
        
        if limit:
            query_parts.append("LIMIT :limit")
            params["limit"] = limit
        
        query = text(" ".join(query_parts))
        
        try:
            with self.connection() as conn:
                df = pd.read_sql(query, conn, params=params)
            
            # Sort chronologically
            df = df.sort_values("time").reset_index(drop=True)
            return df
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve OHLCV data: {e}")
            raise
    
    def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = "binance"
    ) -> Optional[datetime]:
        """
        Get the most recent timestamp for a symbol/timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            exchange: Exchange name
            
        Returns:
            Most recent timestamp or None if no data
        """
        query = text("""
            SELECT MAX(time) as latest
            FROM ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe AND exchange = :exchange
        """)
        
        try:
            with self.connection() as conn:
                result = conn.execute(
                    query,
                    {"symbol": symbol, "timeframe": timeframe, "exchange": exchange}
                )
                row = result.fetchone()
                return row[0] if row and row[0] else None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get latest timestamp: {e}")
            return None
    
    # =========================================================================
    # Swing Point Operations
    # =========================================================================
    
    def insert_swing_points(self, swing_points: List[Dict[str, Any]]) -> int:
        """
        Insert swing points into database.
        
        Args:
            swing_points: List of swing point dictionaries
            
        Returns:
            Number of rows inserted
        """
        if not swing_points:
            return 0
        
        insert_query = text("""
            INSERT INTO swing_points (time, symbol, timeframe, swing_type, price, significance, atr_at_point, confirmed)
            VALUES (:time, :symbol, :timeframe, :swing_type, :price, :significance, :atr_at_point, :confirmed)
            ON CONFLICT DO NOTHING
        """)
        
        try:
            with self.connection() as conn:
                conn.execute(insert_query, swing_points)
            return len(swing_points)
        except SQLAlchemyError as e:
            logger.error(f"Failed to insert swing points: {e}")
            raise
    
    def get_swing_points(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        swing_type: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Retrieve swing points from database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start of date range
            end_time: End of date range
            swing_type: Filter by 'high' or 'low'
            limit: Maximum number of points
            
        Returns:
            DataFrame with swing points
        """
        query_parts = [
            "SELECT time, symbol, timeframe, swing_type, price, significance, atr_at_point, confirmed",
            "FROM swing_points",
            "WHERE symbol = :symbol AND timeframe = :timeframe"
        ]
        params: Dict[str, Any] = {"symbol": symbol, "timeframe": timeframe}
        
        if start_time:
            query_parts.append("AND time >= :start_time")
            params["start_time"] = start_time
        
        if end_time:
            query_parts.append("AND time <= :end_time")
            params["end_time"] = end_time
        
        if swing_type:
            query_parts.append("AND swing_type = :swing_type")
            params["swing_type"] = swing_type
        
        query_parts.append("ORDER BY time DESC LIMIT :limit")
        params["limit"] = limit
        
        query = text(" ".join(query_parts))
        
        try:
            with self.connection() as conn:
                df = pd.read_sql(query, conn, params=params)
            return df.sort_values("time").reset_index(drop=True)
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve swing points: {e}")
            raise
    
    # =========================================================================
    # QML Pattern Operations
    # =========================================================================
    
    def insert_pattern(self, pattern: Dict[str, Any]) -> int:
        """
        Insert a QML pattern into database.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Pattern ID
        """
        insert_query = text("""
            INSERT INTO qml_patterns (
                detection_time, symbol, timeframe, pattern_type,
                left_shoulder_price, left_shoulder_time,
                head_price, head_time,
                right_shoulder_price, right_shoulder_time,
                neckline_start, neckline_end,
                entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
                validity_score, geometric_score, volume_score, context_score,
                ml_confidence, ml_model_version,
                status
            ) VALUES (
                :detection_time, :symbol, :timeframe, :pattern_type,
                :left_shoulder_price, :left_shoulder_time,
                :head_price, :head_time,
                :right_shoulder_price, :right_shoulder_time,
                :neckline_start, :neckline_end,
                :entry_price, :stop_loss, :take_profit_1, :take_profit_2, :take_profit_3,
                :validity_score, :geometric_score, :volume_score, :context_score,
                :ml_confidence, :ml_model_version,
                :status
            ) RETURNING id
        """)
        
        try:
            with self.connection() as conn:
                result = conn.execute(insert_query, pattern)
                row = result.fetchone()
                pattern_id = row[0] if row else 0
            logger.info(f"Inserted pattern {pattern_id} for {pattern['symbol']}")
            return pattern_id
        except SQLAlchemyError as e:
            logger.error(f"Failed to insert pattern: {e}")
            raise
    
    def get_active_patterns(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_validity: float = 0.7
    ) -> pd.DataFrame:
        """
        Get active (non-invalidated, non-completed) patterns.
        
        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            min_validity: Minimum validity score
            
        Returns:
            DataFrame with active patterns
        """
        query_parts = [
            "SELECT * FROM qml_patterns",
            "WHERE status IN ('forming', 'active', 'triggered')",
            "AND validity_score >= :min_validity"
        ]
        params: Dict[str, Any] = {"min_validity": min_validity}
        
        if symbol:
            query_parts.append("AND symbol = :symbol")
            params["symbol"] = symbol
        
        if timeframe:
            query_parts.append("AND timeframe = :timeframe")
            params["timeframe"] = timeframe
        
        query_parts.append("ORDER BY detection_time DESC")
        
        query = text(" ".join(query_parts))
        
        try:
            with self.connection() as conn:
                return pd.read_sql(query, conn, params=params)
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve active patterns: {e}")
            raise
    
    def update_pattern_status(
        self,
        pattern_id: int,
        pattern_time: datetime,
        status: str,
        outcome: Optional[str] = None,
        actual_return_pct: Optional[float] = None,
        invalidation_reason: Optional[str] = None
    ) -> bool:
        """
        Update pattern status and outcome.
        
        Args:
            pattern_id: Pattern ID
            pattern_time: Pattern detection time (for hypertable)
            status: New status
            outcome: Trade outcome
            actual_return_pct: Actual return percentage
            invalidation_reason: Reason for invalidation
            
        Returns:
            True if updated successfully
        """
        update_query = text("""
            UPDATE qml_patterns
            SET status = :status,
                outcome = COALESCE(:outcome, outcome),
                actual_return_pct = COALESCE(:actual_return_pct, actual_return_pct),
                invalidation_reason = COALESCE(:invalidation_reason, invalidation_reason),
                updated_at = NOW()
            WHERE id = :pattern_id AND detection_time = :pattern_time
        """)
        
        try:
            with self.connection() as conn:
                conn.execute(update_query, {
                    "pattern_id": pattern_id,
                    "pattern_time": pattern_time,
                    "status": status,
                    "outcome": outcome,
                    "actual_return_pct": actual_return_pct,
                    "invalidation_reason": invalidation_reason
                })
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to update pattern status: {e}")
            return False
    
    # =========================================================================
    # Statistics & Metrics
    # =========================================================================
    
    def get_pattern_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get pattern detection and performance statistics.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Dictionary with statistics
        """
        query = text("""
            SELECT
                COUNT(*) as total_patterns,
                COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
                COUNT(CASE WHEN outcome = 'breakeven' THEN 1 END) as breakeven,
                AVG(validity_score) as avg_validity,
                AVG(ml_confidence) as avg_ml_confidence,
                AVG(CASE WHEN outcome = 'win' THEN actual_return_pct END) as avg_win_return,
                AVG(CASE WHEN outcome = 'loss' THEN actual_return_pct END) as avg_loss_return,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT timeframe) as timeframes_used
            FROM qml_patterns
            WHERE (:start_date IS NULL OR detection_time >= :start_date)
              AND (:end_date IS NULL OR detection_time <= :end_date)
        """)
        
        try:
            with self.connection() as conn:
                result = conn.execute(query, {
                    "start_date": start_date,
                    "end_date": end_date
                })
                row = result.fetchone()
                
                if row:
                    total = row[0] or 0
                    wins = row[1] or 0
                    losses = row[2] or 0
                    completed = wins + losses + (row[3] or 0)
                    
                    return {
                        "total_patterns": total,
                        "wins": wins,
                        "losses": losses,
                        "breakeven": row[3] or 0,
                        "win_rate": wins / completed if completed > 0 else 0,
                        "avg_validity": row[4] or 0,
                        "avg_ml_confidence": row[5] or 0,
                        "avg_win_return": row[6] or 0,
                        "avg_loss_return": row[7] or 0,
                        "unique_symbols": row[8] or 0,
                        "timeframes_used": row[9] or 0
                    }
                return {}
        except SQLAlchemyError as e:
            logger.error(f"Failed to get pattern statistics: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._connected = False
            logger.info("Database connection closed")


# Singleton instance
_db_instance: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get singleton database manager instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance

