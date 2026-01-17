"""
ML Data Manager
===============
Unified interface for ML data (features, labels, training data).

Manages:
- Pattern features (170+ VRD features) → Parquet
- Pattern labels (win/loss/ignore) → Parquet
- Feature versioning (v1, v2, etc.)
- Train/test splits
- Data lineage tracking

Usage:
    from src.data.ml_manager import MLDataManager
    
    ml = MLDataManager()
    features = ml.load_features(version='v1')
    labels = ml.load_labels(version='v1')
    training_data = ml.get_training_data(version='v1', test_split=0.2)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import pandas as pd
import json
from loguru import logger


class MLDataManager:
    """
    Unified interface for ML data management.
    
    Single source of truth for:
    - Pattern features (Parquet files)
    - Pattern labels (Parquet files)
    - Feature schemas (JSON versioning)
    - Training dataset assembly
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize ML data manager.
        
        Args:
            base_dir: Base directory for ML data (default: data/ml)
        """
        self.base_dir = base_dir or Path("data/ml")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.features_dir = self.base_dir / "features"
        self.labels_dir = self.base_dir / "labels"
        self.models_dir = self.base_dir / "models"
        self.registry_dir = self.base_dir / "registry"
        
        for dir_path in [self.features_dir, self.labels_dir, self.models_dir, self.registry_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"MLDataManager initialized: {self.base_dir}")
    
    def _get_features_path(self, version: str) -> Path:
        """Get path to features parquet file."""
        return self.features_dir / f"qml_features_{version}.parquet"
    
    def _get_labels_path(self, version: str) -> Path:
        """Get path to labels parquet file."""
        return self.labels_dir / f"qml_labels_{version}.parquet"
    
    def _get_schema_path(self, version: str) -> Path:
        """Get path to feature schema JSON."""
        return self.features_dir / f"schema_{version}.json"
    
    def load_features(
        self,
        version: str = 'v1',
        columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Load pattern features.
        
        Args:
            version: Feature version (e.g., 'v1', 'v2')
            columns: Optional list of specific columns to load
            
        Returns:
            DataFrame with pattern features
            
        Example:
            >>> ml = MLDataManager()
            >>> features = ml.load_features('v1')
            >>> print(f"Loaded {len(features)} patterns with {len(features.columns)} features")
        """
        path = self._get_features_path(version)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Features version '{version}' not found\n"
                f"Expected: {path}\n"
                f"\n"
                f"Available versions: {self.get_available_versions()}"
            )
        
        # Load from parquet (columnar = only load needed columns)
        df = pd.read_parquet(path, columns=columns)
        
        logger.debug(f"Loaded {len(df)} patterns, {len(df.columns)} features (version: {version})")
        return df
    
    def load_labels(self, version: str = 'v1') -> pd.DataFrame:
        """
        Load pattern labels.
        
        Args:
            version: Label version
            
        Returns:
            DataFrame with pattern labels
        """
        path = self._get_labels_path(version)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Labels version '{version}' not found\n"
                f"Expected: {path}"
            )
        
        df = pd.read_parquet(path)
        logger.debug(f"Loaded {len(df)} labels (version: {version})")
        return df
    
    def save_features(
        self,
        df: pd.DataFrame,
        version: str,
        schema: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Save pattern features to Parquet.
        
        Args:
            df: Features DataFrame
            version: Version identifier (e.g., 'v1', 'v2')
            schema: Optional feature schema documentation
            metadata: Optional metadata about feature extraction
        """
        path = self._get_features_path(version)
        
        # Save features to parquet
        df.to_parquet(path, index=False)
        
        # Save schema if provided
        if schema or metadata:
            schema_data = {
                'version': version,
                'created_at': datetime.now().isoformat(),
                'num_patterns': len(df),
                'num_features': len(df.columns),
                'columns': list(df.columns),
                'schema': schema or {},
                'metadata': metadata or {}
            }
            
            schema_path = self._get_schema_path(version)
            with open(schema_path, 'w') as f:
                json.dump(schema_data, f, indent=2)
        
        logger.info(f"✅ Saved {len(df)} patterns with {len(df.columns)} features (version: {version})")
    
    def save_labels(
        self,
        df: pd.DataFrame,
        version: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save pattern labels to Parquet.
        
        Args:
            df: Labels DataFrame
            version: Version identifier
            metadata: Optional metadata about labeling
        """
        path = self._get_labels_path(version)
        
        # Add metadata columns
        if metadata:
            for key, value in metadata.items():
                if key not in df.columns:
                    df[key] = value
        
        df.to_parquet(path, index=False)
        logger.info(f"✅ Saved {len(df)} labels (version: {version})")
    
    def get_training_data(
        self,
        version: str = 'v1',
        test_split: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training data (features + labels merged).
        
        Args:
            version: Data version
            test_split: Fraction for test set (0.0-1.0)
            shuffle: Whether to shuffle before split
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df) with features and labels merged
            
        Example:
            >>> ml = MLDataManager()
            >>> train, test = ml.get_training_data('v1', test_split=0.2)
            >>> print(f"Train: {len(train)}, Test: {len(test)}")
        """
        # Load features and labels
        features = self.load_features(version)
        labels = self.load_labels(version)
        
        # Merge on pattern_id
        if 'pattern_id' not in features.columns or 'pattern_id' not in labels.columns:
            raise ValueError("Both features and labels must have 'pattern_id' column")
        
        merged = features.merge(labels, on='pattern_id', how='inner')
        
        # Shuffle if requested
        if shuffle:
            merged = merged.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split
        split_idx = int(len(merged) * (1 - test_split))
        train_df = merged.iloc[:split_idx]
        test_df = merged.iloc[split_idx:]
        
        logger.info(f"Training data: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df
    
    def get_available_versions(self) -> Dict[str, Dict]:
        """
        Get available feature and label versions.
        
        Returns:
            Dictionary with version info
        """
        versions = {}
        
        # Check features
        for path in self.features_dir.glob("qml_features_*.parquet"):
            version = path.stem.replace("qml_features_", "")
            
            # Load schema if exists
            schema_path = self._get_schema_path(version)
            schema_info = {}
            if schema_path.exists():
                with open(schema_path) as f:
                    schema_info = json.load(f)
            
            versions[version] = {
                'has_features': True,
                'has_labels': self._get_labels_path(version).exists(),
                'schema': schema_info
            }
        
        return versions
    
    def get_stats(self) -> Dict:
        """Get statistics about ML data."""
        stats = {
            'versions': self.get_available_versions(),
            'total_versions': 0,
            'latest_version': None
        }
        
        versions = list(stats['versions'].keys())
        if versions:
            stats['total_versions'] = len(versions)
            stats['latest_version'] = sorted(versions)[-1]
        
        return stats


# Convenience function
def get_ml_manager() -> MLDataManager:
    """Get global MLDataManager instance."""
    return MLDataManager()
