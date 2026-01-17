#!/usr/bin/env python3
"""
Data Migration Script: BTC -> BTCUSDT
======================================
One-time migration to establish the new multi-symbol folder structure.

This script:
1. Finds the legacy data/processed/BTC/ folder
2. Moves all .parquet files to data/processed/BTCUSDT/
3. Removes the empty legacy folder

Usage:
    python scripts/migrate_data_to_symbol_folders.py
    
    # Dry run (preview only)
    python scripts/migrate_data_to_symbol_folders.py --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Legacy and new paths
LEGACY_PATH = DATA_DIR / "BTC"
NEW_PATH = DATA_DIR / "BTCUSDT"


def migrate(dry_run: bool = False) -> bool:
    """
    Migrate BTC folder to BTCUSDT.
    
    Args:
        dry_run: If True, only print what would be done
        
    Returns:
        True if migration successful or no migration needed
    """
    print("=" * 60)
    print("  DATA MIGRATION: BTC -> BTCUSDT")
    print("=" * 60)
    print()
    
    # Check if legacy folder exists
    if not LEGACY_PATH.exists():
        print(f"✓ No legacy folder found at: {LEGACY_PATH}")
        
        if NEW_PATH.exists():
            print(f"✓ New format folder already exists: {NEW_PATH}")
            print()
            print("✅ No migration needed - already using new format!")
            return True
        else:
            print(f"✗ Neither legacy nor new folder exists")
            print()
            print("ℹ️  No data to migrate. Run build_master_store() to fetch data.")
            return True
    
    # Check if new folder already exists
    if NEW_PATH.exists():
        files_in_new = list(NEW_PATH.glob("*.parquet"))
        if files_in_new:
            print(f"⚠️  Both legacy and new folders exist!")
            print(f"   Legacy: {LEGACY_PATH}")
            print(f"   New:    {NEW_PATH} ({len(files_in_new)} files)")
            print()
            print("   Please manually resolve this conflict.")
            return False
    
    # Find files to migrate
    parquet_files = list(LEGACY_PATH.glob("*.parquet"))
    
    if not parquet_files:
        print(f"✓ Legacy folder exists but is empty: {LEGACY_PATH}")
        if not dry_run:
            LEGACY_PATH.rmdir()
            print(f"  Removed empty folder")
        return True
    
    print(f"📁 Legacy folder: {LEGACY_PATH}")
    print(f"📁 Target folder: {NEW_PATH}")
    print()
    print(f"Files to migrate ({len(parquet_files)}):")
    
    total_size = 0
    for f in parquet_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  • {f.name} ({size_mb:.1f} MB)")
    
    print()
    print(f"Total size: {total_size:.1f} MB")
    print()
    
    if dry_run:
        print("🔍 DRY RUN - No changes made")
        print()
        print("Run without --dry-run to perform migration.")
        return True
    
    # Perform migration
    print("🚀 Migrating...")
    
    # Create new directory
    NEW_PATH.mkdir(parents=True, exist_ok=True)
    
    # Move files
    for f in parquet_files:
        dest = NEW_PATH / f.name
        shutil.move(str(f), str(dest))
        print(f"  ✓ Moved: {f.name}")
    
    # Remove legacy folder
    try:
        LEGACY_PATH.rmdir()
        print(f"  ✓ Removed empty legacy folder: {LEGACY_PATH.name}/")
    except OSError:
        print(f"  ⚠️  Could not remove legacy folder (not empty?)")
    
    print()
    print("=" * 60)
    print("✅ MIGRATION COMPLETE!")
    print()
    print(f"New data location: {NEW_PATH}")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate BTC folder to BTCUSDT format"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without making them'
    )
    
    args = parser.parse_args()
    
    success = migrate(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
