"""
SQLite Database Manager for QML Trading System - Phase 2
=========================================================
Unified database interface for patterns, trades, features, and experiments.

Extends the existing experiments.db with new tables while maintaining
backwards compatibility with ExperimentLogger.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from .schemas import (
    PatternDetection,
    TradeOutcome,
    FeatureVector,
    ExperimentRun,
    generate_id,
    hash_params,
)


class SQLiteManager:
    """
    Unified SQLite database manager for QML system.

    Manages:
    - patterns: Detected QML patterns
    - trades: Trade outcomes
    - features: Feature vectors for ML
    - experiments: Backtest runs (extends existing table)
    - parameters: Parameter combinations for A/B testing
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database. Defaults to results/experiments.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "results" / "experiments.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_schema()

    @contextmanager
    def connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self.connection() as conn:
            # Patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    detection_time DATETIME NOT NULL,

                    -- Swing points
                    p1_price REAL NOT NULL,
                    p1_time DATETIME NOT NULL,
                    p2_price REAL NOT NULL,
                    p2_time DATETIME NOT NULL,
                    p3_price REAL NOT NULL,
                    p3_time DATETIME NOT NULL,
                    p4_price REAL NOT NULL,
                    p4_time DATETIME NOT NULL,
                    p5_price REAL NOT NULL,
                    p5_time DATETIME NOT NULL,

                    -- Levels
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL NOT NULL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,

                    -- Params & scores
                    detection_params TEXT,
                    validity_score REAL DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE',

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    pattern_id TEXT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    direction TEXT NOT NULL,

                    -- Execution
                    entry_time DATETIME NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_time DATETIME,
                    exit_price REAL,

                    -- Result
                    status TEXT DEFAULT 'OPEN',
                    pnl_dollars REAL,
                    pnl_percent REAL,
                    r_multiple REAL,

                    -- Risk
                    position_size REAL DEFAULT 0,
                    risk_amount REAL DEFAULT 0,
                    stop_loss REAL DEFAULT 0,
                    take_profit REAL DEFAULT 0,

                    -- Metadata
                    exit_reason TEXT,
                    bars_held INTEGER,

                    experiment_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (pattern_id) REFERENCES patterns(id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(run_id)
                )
            """)

            # Features table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT NOT NULL,
                    calculation_time DATETIME NOT NULL,

                    -- Tier 1: Geometry
                    head_extension_atr REAL DEFAULT 0,
                    bos_depth_atr REAL DEFAULT 0,
                    shoulder_symmetry REAL DEFAULT 0,
                    amplitude_ratio REAL DEFAULT 0,
                    time_ratio REAL DEFAULT 0,
                    fib_retracement_p5 REAL DEFAULT 0,

                    -- Tier 2: Context
                    htf_trend_alignment REAL DEFAULT 0,
                    distance_to_sr_atr REAL DEFAULT 0,
                    volatility_percentile REAL DEFAULT 0,
                    regime_state TEXT DEFAULT 'UNKNOWN',
                    rsi_divergence REAL DEFAULT 0,

                    -- Tier 3: Volume
                    volume_spike_p3 REAL DEFAULT 0,
                    volume_spike_p4 REAL DEFAULT 0,
                    volume_trend_p1_p5 REAL DEFAULT 0,

                    -- Tier 4: Quality
                    noise_ratio REAL DEFAULT 0,
                    bos_candle_strength REAL DEFAULT 0,

                    -- Outcome
                    outcome TEXT,
                    r_multiple REAL,

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (pattern_id) REFERENCES patterns(id)
                )
            """)

            # Parameters table for A/B tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parameters (
                    hash TEXT PRIMARY KEY,
                    params_json TEXT NOT NULL,
                    first_tested DATETIME DEFAULT CURRENT_TIMESTAMP,
                    test_count INTEGER DEFAULT 1,
                    best_sharpe REAL,
                    best_experiment_id TEXT
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_time ON patterns(detection_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_experiment ON trades(experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_pattern ON features(pattern_id)")

    # =========================================================================
    # Pattern Operations
    # =========================================================================

    def save_pattern(self, pattern: PatternDetection) -> str:
        """
        Save a detected pattern.

        Args:
            pattern: PatternDetection object

        Returns:
            Pattern ID
        """
        d = pattern.to_dict()

        with self.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO patterns (
                    id, symbol, timeframe, direction, detection_time,
                    p1_price, p1_time, p2_price, p2_time,
                    p3_price, p3_time, p4_price, p4_time,
                    p5_price, p5_time,
                    entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
                    detection_params, validity_score, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                d['id'], d['symbol'], d['timeframe'], d['direction'], d['detection_time'],
                d['p1_price'], d['p1_time'], d['p2_price'], d['p2_time'],
                d['p3_price'], d['p3_time'], d['p4_price'], d['p4_time'],
                d['p5_price'], d['p5_time'],
                d['entry_price'], d['stop_loss'], d['take_profit_1'], d['take_profit_2'], d['take_profit_3'],
                d['detection_params'], d['validity_score'], d['confidence'], d['status']
            ))

        return pattern.id

    def get_pattern(self, pattern_id: str) -> Optional[PatternDetection]:
        """Get pattern by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM patterns WHERE id = ?", (pattern_id,)
            ).fetchone()

            if row:
                return PatternDetection.from_dict(dict(row))
            return None

    def get_patterns_by_date(
        self,
        start: datetime,
        end: datetime,
        symbol: Optional[str] = None
    ) -> List[PatternDetection]:
        """Get patterns within date range."""
        query = "SELECT * FROM patterns WHERE detection_time BETWEEN ? AND ?"
        params = [start.isoformat(), end.isoformat()]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY detection_time DESC"

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [PatternDetection.from_dict(dict(row)) for row in rows]

    def get_active_patterns(self, symbol: Optional[str] = None) -> List[PatternDetection]:
        """Get active (non-closed) patterns."""
        query = "SELECT * FROM patterns WHERE status = 'ACTIVE'"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY detection_time DESC"

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [PatternDetection.from_dict(dict(row)) for row in rows]

    def update_pattern_status(self, pattern_id: str, status: str) -> bool:
        """Update pattern status."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE patterns SET status = ? WHERE id = ?",
                (status, pattern_id)
            )
        return True

    # =========================================================================
    # Trade Operations
    # =========================================================================

    def save_trade(self, trade: TradeOutcome, experiment_id: Optional[str] = None) -> str:
        """
        Save a trade outcome.

        Args:
            trade: TradeOutcome object
            experiment_id: Optional experiment ID to link

        Returns:
            Trade ID
        """
        d = trade.to_dict()

        with self.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades (
                    id, pattern_id, symbol, timeframe, direction,
                    entry_time, entry_price, exit_time, exit_price,
                    status, pnl_dollars, pnl_percent, r_multiple,
                    position_size, risk_amount, stop_loss, take_profit,
                    exit_reason, bars_held, experiment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                d['id'], d['pattern_id'], d['symbol'], d['timeframe'], d['direction'],
                d['entry_time'], d['entry_price'], d['exit_time'], d['exit_price'],
                d['status'], d['pnl_dollars'], d['pnl_percent'], d['r_multiple'],
                d['position_size'], d['risk_amount'], d['stop_loss'], d['take_profit'],
                d['exit_reason'], d['bars_held'], experiment_id
            ))

        return trade.id

    def get_trades_by_experiment(self, experiment_id: str) -> List[TradeOutcome]:
        """Get all trades for an experiment."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE experiment_id = ? ORDER BY entry_time",
                (experiment_id,)
            ).fetchall()
            return [TradeOutcome.from_dict(dict(row)) for row in rows]

    def get_recent_trades(self, limit: int = 20) -> List[TradeOutcome]:
        """Get most recent trades."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [TradeOutcome.from_dict(dict(row)) for row in rows]

    # =========================================================================
    # Feature Operations
    # =========================================================================

    def save_features(self, features: FeatureVector) -> int:
        """
        Save a feature vector.

        Args:
            features: FeatureVector object

        Returns:
            Feature record ID
        """
        d = features.to_dict()

        with self.connection() as conn:
            cursor = conn.execute("""
                INSERT INTO features (
                    pattern_id, calculation_time,
                    head_extension_atr, bos_depth_atr, shoulder_symmetry,
                    amplitude_ratio, time_ratio, fib_retracement_p5,
                    htf_trend_alignment, distance_to_sr_atr, volatility_percentile,
                    regime_state, rsi_divergence,
                    volume_spike_p3, volume_spike_p4, volume_trend_p1_p5,
                    noise_ratio, bos_candle_strength,
                    outcome, r_multiple
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                d['pattern_id'], d['calculation_time'],
                d['head_extension_atr'], d['bos_depth_atr'], d['shoulder_symmetry'],
                d['amplitude_ratio'], d['time_ratio'], d['fib_retracement_p5'],
                d['htf_trend_alignment'], d['distance_to_sr_atr'], d['volatility_percentile'],
                d['regime_state'], d['rsi_divergence'],
                d['volume_spike_p3'], d['volume_spike_p4'], d['volume_trend_p1_p5'],
                d['noise_ratio'], d['bos_candle_strength'],
                d['outcome'], d['r_multiple']
            ))
            return cursor.lastrowid

    def get_features_for_training(self, with_outcomes_only: bool = True) -> List[Dict]:
        """Get features for ML training."""
        query = "SELECT * FROM features"
        if with_outcomes_only:
            query += " WHERE outcome IS NOT NULL"

        with self.connection() as conn:
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # Experiment Operations (extends existing table)
    # =========================================================================

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by run_id."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE run_id = ?", (exp_id,)
            ).fetchone()

            if row:
                return dict(row)
            return None

    def get_all_experiments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all experiments, most recent first."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_latest_experiment(self) -> Optional[Dict[str, Any]]:
        """Get the most recent experiment."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

            if row:
                return dict(row)
            return None

    def get_experiments_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiments for a symbol."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_best_experiment(
        self,
        symbol: Optional[str] = None,
        metric: str = 'sharpe_ratio'
    ) -> Optional[Dict[str, Any]]:
        """Get best experiment by metric."""
        query = f"SELECT * FROM experiments WHERE {metric} IS NOT NULL"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += f" ORDER BY {metric} DESC LIMIT 1"

        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                return dict(row)
            return None

    # =========================================================================
    # Parameter A/B Testing
    # =========================================================================

    def register_params(self, params: Dict[str, Any]) -> str:
        """
        Register a parameter combination.

        Args:
            params: Parameter dictionary

        Returns:
            Parameter hash
        """
        param_hash = hash_params(params)
        params_json = json.dumps(params, sort_keys=True, default=str)

        with self.connection() as conn:
            # Check if exists
            row = conn.execute(
                "SELECT hash, test_count FROM parameters WHERE hash = ?",
                (param_hash,)
            ).fetchone()

            if row:
                # Increment test count
                conn.execute(
                    "UPDATE parameters SET test_count = test_count + 1 WHERE hash = ?",
                    (param_hash,)
                )
            else:
                # Insert new
                conn.execute(
                    "INSERT INTO parameters (hash, params_json) VALUES (?, ?)",
                    (param_hash, params_json)
                )

        return param_hash

    def has_been_tested(self, params: Dict[str, Any]) -> bool:
        """Check if parameters have been tested."""
        param_hash = hash_params(params)

        with self.connection() as conn:
            row = conn.execute(
                "SELECT hash FROM parameters WHERE hash = ?",
                (param_hash,)
            ).fetchone()
            return row is not None

    def get_experiments_with_params(self, param_hash: str) -> List[Dict[str, Any]]:
        """Get experiments that used specific parameters."""
        with self.connection() as conn:
            # Get params JSON
            row = conn.execute(
                "SELECT params_json FROM parameters WHERE hash = ?",
                (param_hash,)
            ).fetchone()

            if not row:
                return []

            # Find experiments with matching config
            rows = conn.execute(
                "SELECT * FROM experiments WHERE config_json LIKE ?",
                (f"%{row['params_json'][:50]}%",)  # Partial match
            ).fetchall()
            return [dict(row) for row in rows]

    def update_best_params(self, param_hash: str, sharpe: float, experiment_id: str) -> None:
        """Update best result for parameters."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT best_sharpe FROM parameters WHERE hash = ?",
                (param_hash,)
            ).fetchone()

            if row and (row['best_sharpe'] is None or sharpe > row['best_sharpe']):
                conn.execute(
                    "UPDATE parameters SET best_sharpe = ?, best_experiment_id = ? WHERE hash = ?",
                    (sharpe, experiment_id, param_hash)
                )

    # =========================================================================
    # Aggregate Metrics
    # =========================================================================

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all data."""
        with self.connection() as conn:
            stats = {}

            # Experiment stats
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_experiments,
                    AVG(win_rate) as avg_win_rate,
                    AVG(sharpe_ratio) as avg_sharpe,
                    MAX(pnl_percent) as best_return,
                    AVG(total_trades) as avg_trades
                FROM experiments
            """).fetchone()
            stats['experiments'] = dict(row) if row else {}

            # Trade stats
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN status = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    AVG(pnl_percent) as avg_pnl,
                    SUM(pnl_dollars) as total_pnl
                FROM trades
            """).fetchone()
            stats['trades'] = dict(row) if row else {}

            # Pattern stats
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_patterns,
                    AVG(validity_score) as avg_validity,
                    SUM(CASE WHEN direction = 'BULLISH' THEN 1 ELSE 0 END) as bullish,
                    SUM(CASE WHEN direction = 'BEARISH' THEN 1 ELSE 0 END) as bearish
                FROM patterns
            """).fetchone()
            stats['patterns'] = dict(row) if row else {}

            return stats


# Singleton instance
_db_instance: Optional[SQLiteManager] = None


def get_db() -> SQLiteManager:
    """Get singleton database manager instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SQLiteManager()
    return _db_instance
