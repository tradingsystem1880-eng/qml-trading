"""
Pattern Registry Manager
========================
Core data access layer for ML pattern storage and retrieval.

Provides persistent storage for detected patterns, their features,
labels, and outcomes. Enables:
- Pattern registration with full feature vectors
- Manual labeling for supervised learning
- Training data extraction for XGBoost
- Similar pattern retrieval for analysis

Usage:
    from src.ml.pattern_registry import PatternRegistry
    
    registry = PatternRegistry()
    
    # Register a pattern
    pattern_id = registry.register_pattern(pattern_data, features)
    
    # Label it after trade completes
    registry.label_pattern(pattern_id, 'win', outcome=2.5)
    
    # Get training data
    X, y = registry.get_training_data()
"""

# Use orjson for 20x faster JSON parsing (critical for pattern features)
try:
    import orjson as json
    USING_ORJSON = True
except ImportError:
    import json
    USING_ORJSON = False
    
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.ml.migrations import run_migrations


class PatternRegistry:
    """
    Manages pattern storage and retrieval for ML training.
    
    Central registry for all detected QML patterns. Stores:
    - Pattern metadata (symbol, timeframe, type)
    - Full feature vectors (170+ features as JSON)
    - Quality scores and ML predictions
    - User labels and trade outcomes
    - Trading status (paper/live)
    
    Supports:
    - CRUD operations for patterns
    - Training data extraction with filtering
    - Cosine similarity search for similar patterns
    """
    
    def __init__(self, db_path: str = "results/experiments.db"):
        """
        Initialize pattern registry.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure required tables exist."""
        run_migrations(str(self.db_path))
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def register_pattern(
        self,
        pattern_data: Dict[str, Any],
        features: Dict[str, float],
        ml_confidence: Optional[float] = None
    ) -> str:
        """
        Store a new detected pattern.
        
        Args:
            pattern_data: Pattern metadata dict with:
                - symbol: Trading pair symbol
                - timeframe: Candle timeframe
                - pattern_type: 'bullish' or 'bearish'
                - detection_time: Detection timestamp
                - validity_score: Pattern quality score
                - (optional) regime: Market regime at detection
            features: Feature dictionary (170+ features)
            ml_confidence: Optional ML model prediction
            
        Returns:
            pattern_id: Unique identifier for the pattern
        """
        pattern_id = self._generate_pattern_id(pattern_data)
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Extract fields
            symbol = pattern_data.get('symbol', 'UNKNOWN')
            timeframe = pattern_data.get('timeframe', '4h')
            pattern_type = pattern_data.get('pattern_type', 'unknown')
            
            # Normalize pattern type
            if pattern_type.lower() in ['bullish', 'bullish_qml']:
                pattern_type = 'bullish_qml'
            elif pattern_type.lower() in ['bearish', 'bearish_qml']:
                pattern_type = 'bearish_qml'
            
            detection_time = pattern_data.get('detection_time')
            if detection_time is None:
                detection_time = datetime.now()
            if isinstance(detection_time, str):
                detection_time = datetime.fromisoformat(detection_time)
            
            validity_score = pattern_data.get('validity_score', 0.0)
            regime = pattern_data.get('regime', None)
            
            # Serialize features (use orjson if available for speed)
            if USING_ORJSON:
                features_json = json.dumps(features).decode('utf-8')
            else:
                features_json = json.dumps(features)
            
            # Insert pattern
            cursor.execute("""
                INSERT OR REPLACE INTO ml_pattern_registry (
                    pattern_id, symbol, timeframe, detection_time, pattern_type,
                    features_json, validity_score, ml_confidence, regime_at_detection,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id,
                symbol,
                timeframe,
                detection_time.isoformat(),
                pattern_type,
                features_json,
                validity_score,
                ml_confidence,
                regime,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))
            
            conn.commit()
            logger.info(f"Registered pattern {pattern_id[:8]}... ({pattern_type})")
            
            return pattern_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to register pattern: {e}")
            raise
        finally:
            conn.close()
    
    def label_pattern(
        self,
        pattern_id: str,
        label: str,
        outcome: Optional[float] = None
    ) -> bool:
        """
        Update pattern with manual label and trade outcome.
        
        Args:
            pattern_id: Pattern identifier
            label: Label string - 'win', 'loss', or 'ignore'
            outcome: Optional PnL percentage (e.g., 2.5 for +2.5%)
            
        Returns:
            True if updated successfully
        """
        # Normalize label
        label = label.lower().strip()
        if label in ['w', 'win', '1']:
            label = 'win'
        elif label in ['l', 'loss', '0']:
            label = 'loss'
        elif label in ['i', 'ignore', 'skip']:
            label = 'ignore'
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE ml_pattern_registry
                SET user_label = ?,
                    trade_outcome = ?,
                    updated_at = ?
                WHERE pattern_id = ?
            """, (label, outcome, datetime.now().isoformat(), pattern_id))
            
            if cursor.rowcount == 0:
                logger.warning(f"Pattern {pattern_id} not found")
                return False
            
            conn.commit()
            logger.info(f"Labeled pattern {pattern_id[:8]}... as '{label}' (outcome: {outcome})")
            return True
            
        finally:
            conn.close()
    
    def update_label(self, pattern_id: str, label: str) -> bool:
        """
        Alias for label_pattern() - dashboard compatibility.
        
        Args:
            pattern_id: Pattern identifier
            label: Label string - 'win', 'loss', or 'unlabeled'
            
        Returns:
            True if updated successfully
        """
        return self.label_pattern(pattern_id, label)
    
    def update_trade_status(
        self,
        pattern_id: str,
        paper_traded: Optional[bool] = None,
        live_traded: Optional[bool] = None,
        outcome: Optional[float] = None
    ) -> bool:
        """
        Update pattern trading status.
        
        Args:
            pattern_id: Pattern identifier
            paper_traded: Whether pattern was paper traded
            live_traded: Whether pattern was live traded
            outcome: Trade PnL percentage
            
        Returns:
            True if updated successfully
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            updates = ["updated_at = ?"]
            params = [datetime.now().isoformat()]
            
            if paper_traded is not None:
                updates.append("paper_traded = ?")
                params.append(1 if paper_traded else 0)
            
            if live_traded is not None:
                updates.append("live_traded = ?")
                params.append(1 if live_traded else 0)
            
            if outcome is not None:
                updates.append("trade_outcome = ?")
                params.append(outcome)
                
                # Auto-label based on outcome if not already labeled
                label = 'win' if outcome > 0 else 'loss'
                updates.append("user_label = COALESCE(user_label, ?)")
                params.append(label)
            
            params.append(pattern_id)
            
            cursor.execute(f"""
                UPDATE ml_pattern_registry
                SET {', '.join(updates)}
                WHERE pattern_id = ?
            """, params)
            
            conn.commit()
            return cursor.rowcount > 0
            
        finally:
            conn.close()
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single pattern by ID.
        
        Args:
            pattern_id: Pattern identifier
            
        Returns:
            Pattern dict or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ml_pattern_registry WHERE pattern_id = ?", (pattern_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_dict(row)
            
        finally:
            conn.close()
    
    def get_patterns(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern_type: Optional[str] = None,
        label: Optional[str] = None,
        min_validity: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query patterns with optional filters.
        
        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            pattern_type: Filter by pattern type
            label: Filter by label ('win', 'loss', 'ignore', or None for unlabeled)
            min_validity: Minimum validity score
            limit: Maximum number of results
            
        Returns:
            List of pattern dicts
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if timeframe:
                conditions.append("timeframe = ?")
                params.append(timeframe)
            
            if pattern_type:
                conditions.append("pattern_type = ?")
                params.append(pattern_type)
            
            if label is not None:
                if label == 'unlabeled':
                    conditions.append("user_label IS NULL")
                else:
                    conditions.append("user_label = ?")
                    params.append(label)
            
            if min_validity is not None:
                conditions.append("validity_score >= ?")
                params.append(min_validity)
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            cursor.execute(f"""
                SELECT * FROM ml_pattern_registry
                {where_clause}
                ORDER BY detection_time DESC
                LIMIT ?
            """, params + [limit])
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
            
        finally:
            conn.close()
    
    def get_unlabeled_patterns(
        self,
        limit: int = 50,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get patterns that need labeling.
        
        Args:
            limit: Maximum number of patterns
            symbol: Optional symbol filter
            
        Returns:
            List of unlabeled patterns (oldest first)
        """
        return self.get_patterns(symbol=symbol, label='unlabeled', limit=limit)
    
    def get_training_data(
        self,
        min_labels: int = 30,
        label_filter: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Get labeled patterns as training data.
        
        Args:
            min_labels: Minimum number of labeled patterns required
            label_filter: Labels to include (default: ['win', 'loss'])
            feature_names: Optional list of features to include
            
        Returns:
            (X, y): Feature DataFrame and binary labels (1=win, 0=loss)
            
        Raises:
            ValueError: If not enough labeled patterns
        """
        if label_filter is None:
            label_filter = ['win', 'loss']
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in label_filter])
            cursor.execute(f"""
                SELECT pattern_id, features_json, user_label, trade_outcome
                FROM ml_pattern_registry
                WHERE user_label IN ({placeholders})
                ORDER BY detection_time
            """, label_filter)
            
            rows = cursor.fetchall()
            
            if len(rows) < min_labels:
                raise ValueError(
                    f"Not enough labeled patterns: {len(rows)} < {min_labels}. "
                    f"Use the labeling CLI to label more patterns."
                )
            
            # Extract features and labels
            feature_rows = []
            labels = []
            
            for row in rows:
                # Fast JSON parsing with orjson
                if USING_ORJSON:
                    features = json.loads(row['features_json'])
                else:
                    features = json.loads(row['features_json'])
                
                if feature_names:
                    # Filter to specified features
                    features = {k: features.get(k, 0.0) for k in feature_names}
                
                feature_rows.append(features)
                labels.append(1 if row['user_label'] == 'win' else 0)
            
            X = pd.DataFrame(feature_rows)
            y = np.array(labels)
            
            logger.info(f"Loaded training data: {len(X)} samples, {len(X.columns)} features")
            
            return X, y
            
        finally:
            conn.close()
    
    def find_similar_patterns(
        self,
        current_features: Dict[str, float],
        n: int = 5,
        label_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find most similar historical patterns using cosine similarity.
        
        Args:
            current_features: Feature vector to compare against
            n: Number of similar patterns to return
            label_filter: Optional filter for labeled patterns only
            
        Returns:
            List of similar patterns with similarity scores
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Build query
            if label_filter:
                placeholders = ','.join(['?' for _ in label_filter])
                cursor.execute(f"""
                    SELECT * FROM ml_pattern_registry
                    WHERE user_label IN ({placeholders})
                """, label_filter)
            else:
                cursor.execute("SELECT * FROM ml_pattern_registry")
            
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Compute similarities
            current_vec = self._features_to_vector(current_features)
            similarities = []
            
            for row in rows:
                stored_features = json.loads(row['features_json'])
                stored_vec = self._features_to_vector(stored_features, list(current_features.keys()))
                
                similarity = self._cosine_similarity(current_vec, stored_vec)
                
                pattern_dict = self._row_to_dict(row)
                pattern_dict['similarity'] = similarity
                similarities.append(pattern_dict)
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similarities[:n]
            
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with pattern counts and distributions
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Total patterns
            cursor.execute("SELECT COUNT(*) FROM ml_pattern_registry")
            total = cursor.fetchone()[0]
            
            # By label
            cursor.execute("""
                SELECT user_label, COUNT(*) 
                FROM ml_pattern_registry 
                GROUP BY user_label
            """)
            by_label = {row[0] or 'unlabeled': row[1] for row in cursor.fetchall()}
            
            # By type
            cursor.execute("""
                SELECT pattern_type, COUNT(*) 
                FROM ml_pattern_registry 
                GROUP BY pattern_type
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By symbol
            cursor.execute("""
                SELECT symbol, COUNT(*) 
                FROM ml_pattern_registry 
                GROUP BY symbol
            """)
            by_symbol = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Win rate (labeled only)
            wins = by_label.get('win', 0)
            losses = by_label.get('loss', 0)
            labeled = wins + losses
            win_rate = wins / labeled * 100 if labeled > 0 else 0
            
            return {
                'total_patterns': total,
                'labeled': labeled,
                'unlabeled': by_label.get('unlabeled', 0),
                'by_label': by_label,
                'by_type': by_type,
                'by_symbol': by_symbol,
                'win_rate': win_rate,
            }
            
        finally:
            conn.close()
    
    def _generate_pattern_id(self, pattern_data: Dict[str, Any]) -> str:
        """Generate unique pattern ID based on key attributes."""
        # Use combination of symbol, time, and type for deterministic ID
        symbol = pattern_data.get('symbol', 'UNK')
        detection_time = pattern_data.get('detection_time', datetime.now())
        
        if isinstance(detection_time, str):
            detection_time = datetime.fromisoformat(detection_time)
        
        pattern_type = pattern_data.get('pattern_type', 'unknown')
        
        # Create deterministic hash
        key = f"{symbol}_{detection_time.isoformat()}_{pattern_type}"
        
        # Use UUID5 for deterministic but unique ID
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert sqlite3.Row to dictionary."""
        d = dict(row)
        
        # Parse JSON fields
        if 'features_json' in d and d['features_json']:
            d['features'] = json.loads(d['features_json'])
        
        return d
    
    def _features_to_vector(
        self,
        features: Dict[str, float],
        keys: Optional[List[str]] = None
    ) -> np.ndarray:
        """Convert feature dict to numpy array."""
        if keys is None:
            keys = sorted(features.keys())
        
        return np.array([features.get(k, 0.0) for k in keys])
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))


def create_registry(db_path: str = "results/experiments.db") -> PatternRegistry:
    """Factory function for PatternRegistry."""
    return PatternRegistry(db_path)
