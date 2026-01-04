"""
Experiment Logger - Flight Recorder
====================================
Immutable storage for all backtest runs.

Every run is logged to SQLite with:
- Unique run_id (hash of config + timestamp)
- Full configuration as JSON
- All performance metrics

This enables:
- "Show me the best params for BTC"
- "What was the win rate when I used ATR=20?"
- Full experiment reproducibility
"""

import hashlib
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ExperimentLogger:
    """
    SQLite-based experiment storage.
    
    Logs every backtest run with full configuration and results
    for later analysis and comparison.
    
    Usage:
        logger = ExperimentLogger()
        run_id = logger.log_run(config, results)
        print(f"Logged as: {run_id}")
        
        # Query later
        best_runs = logger.get_top_runs(symbol='BTCUSDT', metric='pnl_percent', limit=10)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize experiment logger.
        
        Args:
            db_path: Path to SQLite database. Defaults to results/experiments.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "results" / "experiments.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT,
                    timeframe TEXT,
                    
                    -- Performance metrics
                    pnl_percent REAL,
                    pnl_usd REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    
                    -- Configuration
                    config_json TEXT NOT NULL,
                    
                    -- Data range
                    data_start TEXT,
                    data_end TEXT,
                    data_bars INTEGER,
                    
                    -- Notes
                    notes TEXT,
                    tags TEXT,
                    
                    -- Report path
                    report_path TEXT
                )
            """)
            
            # Create index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_symbol 
                ON experiments(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_pnl 
                ON experiments(pnl_percent DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_timestamp 
                ON experiments(timestamp DESC)
            """)
            
            conn.commit()
    
    def _generate_run_id(self, config: Dict[str, Any]) -> str:
        """
        Generate unique run ID from config + timestamp.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            8-character hex hash
        """
        timestamp = datetime.now().isoformat()
        config_str = json.dumps(config, sort_keys=True, default=str)
        combined = f"{timestamp}_{config_str}"
        
        hash_obj = hashlib.sha256(combined.encode())
        return hash_obj.hexdigest()[:8]
    
    def _serialize_config(self, config: Any) -> Dict[str, Any]:
        """Convert config to serializable dictionary."""
        if is_dataclass(config):
            return asdict(config)
        elif hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        elif isinstance(config, dict):
            return config
        else:
            return {'raw': str(config)}
    
    def log_run(
        self,
        config: Any,
        results: Dict[str, Any],
        strategy_name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        report_path: Optional[str] = None
    ) -> str:
        """
        Log a backtest run to the database.
        
        Args:
            config: Backtest configuration (dataclass or dict)
            results: Results dictionary from BacktestEngine
            strategy_name: Strategy name override
            notes: Optional notes about this run
            tags: Optional tags for filtering
            report_path: Path to generated HTML report
        
        Returns:
            run_id: Unique identifier for this run
        """
        config_dict = self._serialize_config(config)
        run_id = self._generate_run_id(config_dict)
        timestamp = datetime.now().isoformat()
        
        # Extract values from config
        symbol = config_dict.get('symbol', 'UNKNOWN')
        timeframe = config_dict.get('timeframe', 'UNKNOWN')
        strategy = strategy_name or config_dict.get('detector_method', 'qml_atr')
        
        # Extract metrics from results
        pnl_percent = results.get('net_profit_pct', 0)
        pnl_usd = results.get('net_profit', 0)
        max_drawdown = results.get('max_drawdown', 0)
        win_rate = results.get('win_rate', 0)
        profit_factor = results.get('profit_factor', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        sortino_ratio = results.get('sortino_ratio', 0)
        total_trades = results.get('total_trades', 0)
        winning_trades = results.get('winning_trades', 0)
        losing_trades = results.get('losing_trades', 0)
        
        # Data info
        equity_curve = results.get('equity_curve', [])
        if equity_curve:
            data_start = str(equity_curve[0][0]) if equity_curve else None
            data_end = str(equity_curve[-1][0]) if equity_curve else None
            data_bars = len(equity_curve)
        else:
            data_start = None
            data_end = None
            data_bars = 0
        
        # Serialize
        config_json = json.dumps(config_dict, default=str, indent=2)
        tags_str = ','.join(tags) if tags else None
        
        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments (
                    run_id, timestamp, strategy_name, symbol, timeframe,
                    pnl_percent, pnl_usd, max_drawdown, win_rate, profit_factor,
                    sharpe_ratio, sortino_ratio, total_trades, winning_trades, losing_trades,
                    config_json, data_start, data_end, data_bars, notes, tags, report_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, timestamp, strategy, symbol, timeframe,
                pnl_percent, pnl_usd, max_drawdown, win_rate, profit_factor,
                sharpe_ratio, sortino_ratio, total_trades, winning_trades, losing_trades,
                config_json, data_start, data_end, data_bars, notes, tags_str, report_path
            ))
            conn.commit()
        
        return run_id
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific run by ID.
        
        Args:
            run_id: Run identifier
        
        Returns:
            Run data as dictionary, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_top_runs(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        metric: str = 'pnl_percent',
        limit: int = 10,
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get top runs by a specific metric.
        
        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy
            metric: Metric to sort by (pnl_percent, sharpe_ratio, etc.)
            limit: Number of results
            ascending: Sort ascending (default: descending)
        
        Returns:
            List of run dictionaries
        """
        order = "ASC" if ascending else "DESC"
        
        query = f"SELECT * FROM experiments WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if strategy:
            query += " AND strategy_name = ?"
            params.append(strategy)
        
        query += f" ORDER BY {metric} {order} LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent runs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def count_runs(self) -> int:
        """Get total number of logged runs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM experiments")
            return cursor.fetchone()[0]
