"""
Database Migrations for ML Pattern Registry
============================================
Manages schema updates for the experiments.db database.

Usage:
    python -m src.ml.migrations
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from loguru import logger


# Migration definitions: (version, description, sql_statements)
MIGRATIONS: List[Tuple[int, str, List[str]]] = [
    (
        1,
        "Create ml_pattern_registry table",
        [
            """
            CREATE TABLE IF NOT EXISTS ml_pattern_registry (
                pattern_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                detection_time DATETIME NOT NULL,
                pattern_type TEXT NOT NULL,
                features_json TEXT NOT NULL,
                validity_score FLOAT,
                ml_confidence FLOAT,
                user_label TEXT,
                trade_outcome FLOAT,
                paper_traded BOOLEAN DEFAULT 0,
                live_traded BOOLEAN DEFAULT 0,
                regime_at_detection TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_pattern_symbol ON ml_pattern_registry(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_pattern_time ON ml_pattern_registry(detection_time)",
            "CREATE INDEX IF NOT EXISTS idx_pattern_label ON ml_pattern_registry(user_label)",
            "CREATE INDEX IF NOT EXISTS idx_pattern_type ON ml_pattern_registry(pattern_type)",
        ]
    ),
    (
        2,
        "Create ml_model_versions table",
        [
            """
            CREATE TABLE IF NOT EXISTS ml_model_versions (
                model_id TEXT PRIMARY KEY,
                trained_time DATETIME NOT NULL,
                features_used TEXT NOT NULL,
                test_accuracy FLOAT,
                model_path TEXT NOT NULL,
                active BOOLEAN DEFAULT 0,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_model_active ON ml_model_versions(active)",
        ]
    ),
    (
        3,
        "Create schema_migrations tracking table",
        [
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
    ),
]


class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, db_path: str = "results/experiments.db"):
        self.db_path = Path(db_path)
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def get_current_version(self) -> int:
        """Get current schema version."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Check if migrations table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_migrations'
            """)
            
            if not cursor.fetchone():
                return 0
            
            # Get max applied version
            cursor.execute("SELECT MAX(version) FROM schema_migrations")
            result = cursor.fetchone()
            return result[0] if result[0] else 0
            
        finally:
            conn.close()
    
    def run_migrations(self) -> List[str]:
        """Run all pending migrations."""
        current_version = self.get_current_version()
        applied = []
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            for version, description, statements in MIGRATIONS:
                if version <= current_version:
                    continue
                
                logger.info(f"Applying migration {version}: {description}")
                
                # Execute all statements for this migration
                for sql in statements:
                    cursor.execute(sql)
                
                # Record migration (skip for the migrations table itself)
                if version > 3 or self._table_exists(cursor, "schema_migrations"):
                    cursor.execute(
                        "INSERT OR REPLACE INTO schema_migrations (version, description) VALUES (?, ?)",
                        (version, description)
                    )
                
                applied.append(f"v{version}: {description}")
            
            conn.commit()
            
            if applied:
                logger.info(f"Applied {len(applied)} migrations")
            else:
                logger.info("Database schema is up to date")
                
            return applied
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            conn.close()
    
    def _table_exists(self, cursor: sqlite3.Cursor, table_name: str) -> bool:
        """Check if a table exists."""
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        return cursor.fetchone() is not None
    
    def get_schema_info(self) -> dict:
        """Get information about current schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get migration history
            migrations = []
            if self._table_exists(cursor, "schema_migrations"):
                cursor.execute("SELECT version, description, applied_at FROM schema_migrations ORDER BY version")
                migrations = [
                    {"version": row[0], "description": row[1], "applied_at": row[2]}
                    for row in cursor.fetchall()
                ]
            
            return {
                "db_path": str(self.db_path),
                "current_version": self.get_current_version(),
                "tables": tables,
                "migrations": migrations,
            }
            
        finally:
            conn.close()


def run_migrations(db_path: str = "results/experiments.db") -> List[str]:
    """Run all pending database migrations."""
    manager = MigrationManager(db_path)
    return manager.run_migrations()


def main():
    """CLI entry point for running migrations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--db-path",
        default="results/experiments.db",
        help="Path to database file"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show schema information instead of running migrations"
    )
    
    args = parser.parse_args()
    
    manager = MigrationManager(args.db_path)
    
    if args.info:
        info = manager.get_schema_info()
        print(f"\n📊 Database Schema Info")
        print(f"{'=' * 50}")
        print(f"Path: {info['db_path']}")
        print(f"Schema Version: {info['current_version']}")
        print(f"\nTables: {', '.join(info['tables'])}")
        
        if info['migrations']:
            print(f"\nMigration History:")
            for m in info['migrations']:
                print(f"  v{m['version']}: {m['description']} ({m['applied_at']})")
    else:
        print(f"\n🔄 Running Migrations...")
        print(f"{'=' * 50}")
        
        applied = manager.run_migrations()
        
        if applied:
            print(f"\n✅ Applied {len(applied)} migrations:")
            for m in applied:
                print(f"  • {m}")
        else:
            print(f"\n✅ Database is already up to date")
        
        # Show current state
        info = manager.get_schema_info()
        print(f"\nCurrent schema version: {info['current_version']}")
        print(f"Tables: {', '.join(info['tables'])}")


if __name__ == "__main__":
    main()
