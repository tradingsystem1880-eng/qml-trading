"""
Pattern Registry Migration Script
=================================
Migrate ML pattern registry from SQLite to Parquet format.

This script:
1. Reads existing pattern_registry.db (SQLite)
2. Extracts features and labels
3. Saves to Parquet files in data/ml/
4. Creates schema documentation

Usage:
    python -m src.scripts.migrate_pattern_registry
"""

from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from loguru import logger

from src.ml.pattern_registry import PatternRegistry
from src.data.ml_manager import MLDataManager


def migrate_pattern_registry():
    """Migrate pattern registry from SQLite to Parquet."""
    
    print("\n" + "="*60)
    print("PATTERN REGISTRY MIGRATION")
    print("SQLite → Parquet Format")
    print("="*60)
    
    # Initialize managers
    old_registry = PatternRegistry()
    ml_manager = MLDataManager()
    
    # Get all patterns from SQLite
    print("\n📊 Reading patterns from SQLite...")
    patterns = old_registry.get_patterns(limit=10000)  # Get all patterns
    
    if not patterns:
        print("⚠️  No patterns found in registry")
        return
    
    print(f"✅ Found {len(patterns)} patterns")
    
    # Convert to DataFrames
    print("\n🔄 Converting to Parquet format...")
    
    # Features DataFrame
    features_data = []
    labels_data = []
    
    for pattern in patterns:
        pattern_id = pattern.get('id', pattern.get('pattern_id', ''))
        
        # Extract features
        features_dict = pattern.get('features', {})
        if features_dict:
            features_dict['pattern_id'] = pattern_id
            features_dict['symbol'] = pattern.get('symbol')
            features_dict['timeframe'] = pattern.get('timeframe')
            features_dict['timestamp'] = pattern.get('detection_time')
            features_data.append(features_dict)
        
        # Extract labels
        label = pattern.get('label')
        if label:
            labels_data.append({
                'pattern_id': pattern_id,
                'label': label,
                'outcome_pct': pattern.get('outcome_pct'),
                'bars_to_outcome': pattern.get('bars_to_outcome'),
                'labeled_at': pattern.get('labeled_at'),
                'labeled_by': pattern.get('labeled_by', 'backtest')
            })
    
    # Create DataFrames
    features_df = pd.DataFrame(features_data)
    labels_df = pd.DataFrame(labels_data)
    
    print(f"✅ Features: {len(features_df)} patterns, {len(features_df.columns)} features")
    print(f"✅ Labels: {len(labels_df)} labeled patterns")
    
    # Save to Parquet
    print("\n💾 Saving to Parquet...")
    
    # Schema metadata
    schema = {
        'description': 'QML Pattern Features v1',
        'source': 'Migrated from SQLite pattern_registry.db',
        'migrated_at': datetime.now().isoformat(),
        'feature_groups': {
            'geometric': 'Pattern geometry features (depth, symmetry, etc.)',
            'volume': 'Volume-based features (OBV, VWAP, etc.)',
            'context': 'Market context (RSI, MACD, ATR, etc.)',
            'regime': 'Market regime classification'
        }
    }
    
    metadata = {
        'patterns_total': len(features_df),
        'labeled_patterns': len(labels_df),
        'symbols': features_df['symbol'].nunique() if 'symbol' in features_df.columns else 0,
        'timeframes': features_df['timeframe'].unique().tolist() if 'timeframe' in features_df.columns else []
    }
    
    # Save features
    ml_manager.save_features(
        features_df,
        version='v1',
        schema=schema,
        metadata=metadata
    )
    
    # Save labels
    ml_manager.save_labels(
        labels_df,
        version='v1',
        metadata={'source': 'Migrated from SQLite'}
    )
    
    print("\n" + "="*60)
    print("✅ MIGRATION COMPLETE")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - data/ml/features/qml_features_v1.parquet")
    print(f"  - data/ml/labels/qml_labels_v1.parquet")
    print(f"  - data/ml/features/schema_v1.json")
    print()
    
    # Show stats
    stats = ml_manager.get_stats()
    print("📊 ML Data Stats:")
    print(f"  Versions: {stats['total_versions']}")
    print(f"  Latest: {stats['latest_version']}")
    print()
    
    return features_df, labels_df


def verify_migration():
    """Verify migration was successful."""
    print("\n🔍 Verifying migration...")
    
    ml_manager = MLDataManager()
    
    try:
        # Load features
        features = ml_manager.load_features('v1')
        print(f"✅ Features loaded: {len(features)} patterns")
        
        # Load labels
        labels = ml_manager.load_labels('v1')
        print(f"✅ Labels loaded: {len(labels)} patterns")
        
        # Get training data
        train, test = ml_manager.get_training_data('v1', test_split=0.2)
        print(f"✅ Training data: {len(train)} train, {len(test)} test")
        
        print("\n✅ Migration verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    try:
        features_df, labels_df = migrate_pattern_registry()
        verify_migration()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
