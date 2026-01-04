"""
VRD 2.0 - Versioned Research Database
======================================
SQLite-backed experiment tracking with comprehensive metadata storage.
Enables cross-experiment querying, comparison, and reproducibility.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


@dataclass
class ExperimentRecord:
    """Complete record of a single experiment/backtest run."""
    
    experiment_id: str
    timestamp: str  # ISO format
    git_hash: str
    strategy_name: str
    param_hash: str
    params: Dict[str, Any]
    data_start: str  # ISO date
    data_end: str    # ISO date
    random_seed: int
    fold_count: int
    
    # Core Performance Metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_holding_bars: float = 0.0
    
    # Statistical Significance
    sharpe_p_value: Optional[float] = None
    sharpe_percentile: Optional[float] = None
    
    # Monte Carlo Risk
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    kill_switch_prob: Optional[float] = None
    
    # Regime Breakdown (JSON blob)
    regime_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Fold-level Results (JSON blob)
    fold_results: List[Dict] = field(default_factory=list)
    
    # Paths and Status
    artifact_path: str = ""
    status: str = "running"  # running, completed, failed
    error_message: str = ""
    
    def to_db_tuple(self) -> tuple:
        """Convert to tuple for SQLite insertion."""
        return (
            self.experiment_id,
            self.timestamp,
            self.git_hash,
            self.strategy_name,
            self.param_hash,
            json.dumps(self.params),
            self.data_start,
            self.data_end,
            self.random_seed,
            self.fold_count,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.calmar_ratio,
            self.total_return_pct,
            self.annualized_return_pct,
            self.max_drawdown_pct,
            self.win_rate,
            self.profit_factor,
            self.total_trades,
            self.avg_trade_pnl,
            self.avg_holding_bars,
            self.sharpe_p_value,
            self.sharpe_percentile,
            self.var_95,
            self.var_99,
            self.kill_switch_prob,
            json.dumps(self.regime_metrics),
            json.dumps(self.fold_results),
            self.artifact_path,
            self.status,
            self.error_message,
        )


class VRDDatabase:
    """
    Versioned Research Database using SQLite.
    
    Provides:
    - Experiment metadata storage
    - Cross-experiment querying by metrics, regimes, parameters
    - Comparison and ranking capabilities
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        git_hash TEXT,
        strategy_name TEXT NOT NULL,
        param_hash TEXT NOT NULL,
        params TEXT NOT NULL,
        data_start TEXT,
        data_end TEXT,
        random_seed INTEGER,
        fold_count INTEGER,
        
        -- Performance Metrics
        sharpe_ratio REAL,
        sortino_ratio REAL,
        calmar_ratio REAL,
        total_return_pct REAL,
        annualized_return_pct REAL,
        max_drawdown_pct REAL,
        win_rate REAL,
        profit_factor REAL,
        total_trades INTEGER,
        avg_trade_pnl REAL,
        avg_holding_bars REAL,
        
        -- Statistical Significance
        sharpe_p_value REAL,
        sharpe_percentile REAL,
        
        -- Monte Carlo Risk
        var_95 REAL,
        var_99 REAL,
        kill_switch_prob REAL,
        
        -- JSON Blobs
        regime_metrics TEXT,
        fold_results TEXT,
        
        -- Paths and Status
        artifact_path TEXT,
        status TEXT DEFAULT 'running',
        error_message TEXT,
        
        -- Indexes
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_strategy ON experiments(strategy_name);
    CREATE INDEX IF NOT EXISTS idx_param_hash ON experiments(param_hash);
    CREATE INDEX IF NOT EXISTS idx_sharpe ON experiments(sharpe_ratio);
    CREATE INDEX IF NOT EXISTS idx_status ON experiments(status);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments(timestamp);
    """
    
    def __init__(self, db_path: str = "experiments.db"):
        """
        Initialize VRD database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self.create_tables()
        logger.info(f"VRD Database initialized at {self.db_path}")
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.executescript(self.SCHEMA)
        self.conn.commit()
        logger.debug("Database tables created/verified")
    
    def insert_experiment(self, record: ExperimentRecord) -> str:
        """
        Insert a new experiment record.
        
        Args:
            record: ExperimentRecord to insert
            
        Returns:
            experiment_id of inserted record
        """
        cursor = self.conn.cursor()
        
        columns = """
            experiment_id, timestamp, git_hash, strategy_name, param_hash,
            params, data_start, data_end, random_seed, fold_count,
            sharpe_ratio, sortino_ratio, calmar_ratio, total_return_pct,
            annualized_return_pct, max_drawdown_pct, win_rate, profit_factor,
            total_trades, avg_trade_pnl, avg_holding_bars,
            sharpe_p_value, sharpe_percentile,
            var_95, var_99, kill_switch_prob,
            regime_metrics, fold_results,
            artifact_path, status, error_message
        """
        placeholders = ", ".join(["?"] * 31)
        
        sql = f"INSERT OR REPLACE INTO experiments ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, record.to_db_tuple())
        self.conn.commit()
        
        logger.info(f"Inserted experiment: {record.experiment_id}")
        return record.experiment_id
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> None:
        """
        Update specific fields of an experiment.
        
        Args:
            experiment_id: ID of experiment to update
            updates: Dictionary of field -> value updates
        """
        cursor = self.conn.cursor()
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [experiment_id]
        
        # Serialize any dict/list values to JSON
        for i, v in enumerate(values[:-1]):
            if isinstance(v, (dict, list)):
                values[i] = json.dumps(v)
        
        sql = f"UPDATE experiments SET {set_clause} WHERE experiment_id = ?"
        cursor.execute(sql, values)
        self.conn.commit()
        
        logger.debug(f"Updated experiment {experiment_id}: {list(updates.keys())}")
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """
        Retrieve a single experiment by ID.
        
        Args:
            experiment_id: Experiment ID to retrieve
            
        Returns:
            ExperimentRecord or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_record(row)
    
    def query_experiments(
        self,
        strategy_name: Optional[str] = None,
        status: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: Optional[int] = None,
        data_start_after: Optional[str] = None,
        limit: int = 100,
        order_by: str = "timestamp DESC"
    ) -> List[ExperimentRecord]:
        """
        Query experiments with filters.
        
        Args:
            strategy_name: Filter by strategy name
            status: Filter by status (completed, running, failed)
            min_sharpe: Minimum Sharpe ratio
            max_drawdown: Maximum drawdown percentage
            min_trades: Minimum number of trades
            data_start_after: Data must start after this date
            limit: Maximum number of results
            order_by: SQL ORDER BY clause
            
        Returns:
            List of matching ExperimentRecords
        """
        conditions = []
        params = []
        
        if strategy_name:
            conditions.append("strategy_name = ?")
            params.append(strategy_name)
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if min_sharpe is not None:
            conditions.append("sharpe_ratio >= ?")
            params.append(min_sharpe)
        
        if max_drawdown is not None:
            conditions.append("max_drawdown_pct <= ?")
            params.append(max_drawdown)
        
        if min_trades is not None:
            conditions.append("total_trades >= ?")
            params.append(min_trades)
        
        if data_start_after:
            conditions.append("data_start >= ?")
            params.append(data_start_after)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
            SELECT * FROM experiments 
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ?
        """
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_best_experiments(
        self,
        metric: str = "sharpe_ratio",
        n: int = 10,
        strategy_name: Optional[str] = None
    ) -> List[ExperimentRecord]:
        """
        Get top N experiments by a metric.
        
        Args:
            metric: Metric to rank by (sharpe_ratio, profit_factor, etc.)
            n: Number of top experiments to return
            strategy_name: Optional filter by strategy
            
        Returns:
            List of top ExperimentRecords
        """
        where_clause = "status = 'completed'"
        params = []
        
        if strategy_name:
            where_clause += " AND strategy_name = ?"
            params.append(strategy_name)
        
        sql = f"""
            SELECT * FROM experiments
            WHERE {where_clause}
            ORDER BY {metric} DESC
            LIMIT ?
        """
        params.append(n)
        
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments side-by-side.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Optional list of metrics to include (all if None)
            
        Returns:
            DataFrame with experiments as rows, metrics as columns
        """
        if not experiment_ids:
            return pd.DataFrame()
        
        placeholders = ", ".join(["?"] * len(experiment_ids))
        sql = f"SELECT * FROM experiments WHERE experiment_id IN ({placeholders})"
        
        cursor = self.conn.cursor()
        cursor.execute(sql, experiment_ids)
        
        records = [self._row_to_record(row) for row in cursor.fetchall()]
        
        # Convert to DataFrame
        data = [asdict(r) for r in records]
        df = pd.DataFrame(data)
        
        # Filter metrics if specified
        if metrics:
            always_include = ["experiment_id", "strategy_name", "timestamp"]
            cols = always_include + [m for m in metrics if m in df.columns]
            df = df[cols]
        
        return df.set_index("experiment_id")
    
    def get_parameter_performance(
        self,
        param_name: str,
        strategy_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze how a specific parameter affects performance.
        
        Args:
            param_name: Name of parameter to analyze
            strategy_name: Optional filter by strategy
            
        Returns:
            DataFrame with parameter values and average metrics
        """
        records = self.query_experiments(
            strategy_name=strategy_name,
            status="completed",
            limit=1000
        )
        
        if not records:
            return pd.DataFrame()
        
        data = []
        for r in records:
            if param_name in r.params:
                data.append({
                    "param_value": r.params[param_name],
                    "sharpe_ratio": r.sharpe_ratio,
                    "profit_factor": r.profit_factor,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "win_rate": r.win_rate,
                })
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
        
        return df.groupby("param_value").agg({
            "sharpe_ratio": ["mean", "std", "count"],
            "profit_factor": ["mean", "std"],
            "max_drawdown_pct": ["mean", "std"],
            "win_rate": ["mean", "std"],
        })
    
    def _row_to_record(self, row: sqlite3.Row) -> ExperimentRecord:
        """Convert SQLite row to ExperimentRecord."""
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            timestamp=row["timestamp"],
            git_hash=row["git_hash"] or "",
            strategy_name=row["strategy_name"],
            param_hash=row["param_hash"],
            params=json.loads(row["params"]) if row["params"] else {},
            data_start=row["data_start"] or "",
            data_end=row["data_end"] or "",
            random_seed=row["random_seed"] or 0,
            fold_count=row["fold_count"] or 0,
            sharpe_ratio=row["sharpe_ratio"] or 0.0,
            sortino_ratio=row["sortino_ratio"] or 0.0,
            calmar_ratio=row["calmar_ratio"] or 0.0,
            total_return_pct=row["total_return_pct"] or 0.0,
            annualized_return_pct=row["annualized_return_pct"] or 0.0,
            max_drawdown_pct=row["max_drawdown_pct"] or 0.0,
            win_rate=row["win_rate"] or 0.0,
            profit_factor=row["profit_factor"] or 0.0,
            total_trades=row["total_trades"] or 0,
            avg_trade_pnl=row["avg_trade_pnl"] or 0.0,
            avg_holding_bars=row["avg_holding_bars"] or 0.0,
            sharpe_p_value=row["sharpe_p_value"],
            sharpe_percentile=row["sharpe_percentile"],
            var_95=row["var_95"],
            var_99=row["var_99"],
            kill_switch_prob=row["kill_switch_prob"],
            regime_metrics=json.loads(row["regime_metrics"]) if row["regime_metrics"] else {},
            fold_results=json.loads(row["fold_results"]) if row["fold_results"] else [],
            artifact_path=row["artifact_path"] or "",
            status=row["status"] or "unknown",
            error_message=row["error_message"] or "",
        )
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def generate_param_hash(params: Dict[str, Any]) -> str:
    """
    Generate deterministic hash for parameter dictionary.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        8-character MD5 hash
    """
    # Sort keys for deterministic ordering
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.md5(sorted_params.encode()).hexdigest()[:8]
