"""
Results Manager
===============
Unified interface for backtest and experiment results.

Manages:
- Backtest results (trade-by-trade data) → Parquet
- Experiment metadata (params, metrics) → SQLite
- Reports (markdown, JSON, HTML)

Usage:
    from src.data.results_manager import ResultsManager
    
    results = ResultsManager()
    results.save_backtest(trades_df, name='qml_2023')
    results.save_experiment(params, metrics, name='high_vol_filter')
    
    # Load
    backtest = results.load_backtest('qml_2023')
    experiments = results.list_experiments()
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
import json
import sqlite3
from loguru import logger


class ResultsManager:
    """
    Unified interface for results management.
    
    Single source of truth for:
    - Backtest results (Parquet files)
    - Experiment metadata (SQLite database)
    - Reports and analysis outputs
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize results manager.
        
        Args:
            base_dir: Base directory for results (default: results/)
        """
        self.base_dir = base_dir or Path("results")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.backtests_dir = self.base_dir / "backtests"
        self.experiments_dir = self.base_dir / "experiments"
        self.reports_dir = self.base_dir / "reports"
        
        for dir_path in [self.backtests_dir, self.experiments_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Experiments database
        self.experiments_db = self.experiments_dir / "experiments.db"
        self._init_experiments_db()
        
        logger.info(f"ResultsManager initialized: {self.base_dir}")
    
    def _init_experiments_db(self):
        """Initialize experiments SQLite database."""
        conn = sqlite3.connect(self.experiments_db)
        cursor = conn.cursor()
        
        # Create experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT,  -- JSON
                metrics TEXT,     -- JSON
                backtest_path TEXT,
                status TEXT DEFAULT 'complete'
            )
        """)
        
        # Create experiment runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                run_number INTEGER,
                parameters TEXT,  -- JSON
                metrics TEXT,     -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_backtest_path(self, name: str) -> Path:
        """Get path to backtest parquet file."""
        return self.backtests_dir / f"{name}.parquet"
    
    def save_backtest(
        self,
        trades_df: pd.DataFrame,
        name: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save backtest results to Parquet.
        
        Args:
            trades_df: DataFrame with trade-by-trade results
            name: Backtest name (will be filename)
            metadata: Optional metadata to include
            
        Example:
            >>> results = ResultsManager()
            >>> results.save_backtest(trades_df, 'qml_2023_btc_4h')
        """
        path = self._get_backtest_path(name)
        
        # Add metadata columns if provided
        if metadata:
            for key, value in metadata.items():
                if key not in trades_df.columns:
                    trades_df[key] = value
        
        # Save to parquet
        trades_df.to_parquet(path, index=False)
        
        logger.info(f"✅ Saved backtest '{name}': {len(trades_df)} trades to {path}")
    
    def load_backtest(self, name: str) -> pd.DataFrame:
        """
        Load backtest results from Parquet.
        
        Args:
            name: Backtest name
            
        Returns:
            DataFrame with backtest results
        """
        path = self._get_backtest_path(name)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Backtest '{name}' not found\n"
                f"Expected: {path}\n"
                f"\n"
                f"Available backtests: {self.list_backtests()}"
            )
        
        df = pd.read_parquet(path)
        logger.debug(f"Loaded backtest '{name}': {len(df)} trades")
        return df
    
    def list_backtests(self) -> List[str]:
        """List available backtests."""
        return [p.stem for p in self.backtests_dir.glob("*.parquet")]
    
    def save_experiment(
        self,
        name: str,
        parameters: Dict,
        metrics: Dict,
        description: Optional[str] = None,
        backtest_name: Optional[str] = None
    ) -> int:
        """
        Save experiment metadata to SQLite.
        
        Args:
            name: Experiment name
            parameters: Strategy parameters (dict)
            metrics: Performance metrics (dict)
            description: Optional description
            backtest_name: Link to backtest parquet file
            
        Returns:
            Experiment ID
            
        Example:
            >>> results = ResultsManager()
            >>> exp_id = results.save_experiment(
            ...     name='qml_high_vol',
            ...     parameters={'vol_filter': 0.7},
            ...     metrics={'sharpe': 0.85, 'win_rate': 0.68}
            ... )
        """
        conn = sqlite3.connect(self.experiments_db)
        cursor = conn.cursor()
        
        # Prepare backtest path
        backtest_path = None
        if backtest_name:
            backtest_path = str(self._get_backtest_path(backtest_name))
        
        try:
            cursor.execute("""
                INSERT INTO experiments (name, description, parameters, metrics, backtest_path)
                VALUES (?, ?, ?, ?, ?)
            """, (
                name,
                description,
                json.dumps(parameters),
                json.dumps(metrics),
                backtest_path
            ))
            
            experiment_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"✅ Saved experiment '{name}' (ID: {experiment_id})")
            return experiment_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"Experiment '{name}' already exists, updating...")
            cursor.execute("""
                UPDATE experiments
                SET description = ?, parameters = ?, metrics = ?, backtest_path = ?
                WHERE name = ?
            """, (
                description,
                json.dumps(parameters),
                json.dumps(metrics),
                backtest_path,
                name
            ))
            conn.commit()
            
            cursor.execute("SELECT id FROM experiments WHERE name = ?", (name,))
            experiment_id = cursor.fetchone()[0]
            return experiment_id
        
        finally:
            conn.close()
    
    def load_experiment(self, name: str) -> Dict:
        """Load experiment by name."""
        conn = sqlite3.connect(self.experiments_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Experiment '{name}' not found")
        
        return {
            'id': row['id'],
            'name': row['name'],
            'description': row['description'],
            'created_at': row['created_at'],
            'parameters': json.loads(row['parameters']) if row['parameters'] else {},
            'metrics': json.loads(row['metrics']) if row['metrics'] else {},
            'backtest_path': row['backtest_path'],
            'status': row['status']
        }
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        conn = sqlite3.connect(self.experiments_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        experiments = []
        for row in rows:
            experiments.append({
                'id': row['id'],
                'name': row['name'],
                'description': row['description'],
                'created_at': row['created_at'],
                'status': row['status']
            })
        
        return experiments
    
    def save_report(
        self,
        content: str,
        name: str,
        format: str = 'md'
    ):
        """
        Save report to file.
        
        Args:
            content: Report content
            name: Report name
            format: Report format ('md', 'json', 'html')
        """
        path = self.reports_dir / f"{name}.{format}"
        
        with open(path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Saved report '{name}.{format}' to {path}")
    
    def get_stats(self) -> Dict:
        """Get statistics about stored results."""
        stats = {
            'backtests': len(self.list_backtests()),
            'experiments': len(self.list_experiments()),
            'reports': len(list(self.reports_dir.glob("*")))
        }
        
        return stats


# Convenience function
def get_results_manager() -> ResultsManager:
    """Get global ResultsManager instance."""
    return ResultsManager()
