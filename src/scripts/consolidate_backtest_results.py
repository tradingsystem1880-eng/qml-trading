"""
Backtest Results Consolidation Script
=====================================
Consolidate backtest results from various formats into Parquet.

This script:
1. Finds all backtest CSV files in results/
2. Consolidates into standardized Parquet format
3. Saves metadata to experiments.db
4. Cleans up old files

Usage:
    python -m src.scripts.consolidate_backtest_results
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
from loguru import logger

from src.data.results_manager import ResultsManager


def find_backtest_csvs(results_dir: Path) -> list:
    """Find all backtest CSV files."""
    csv_files = []
    
    # Look for CSV files
    for pattern in ['backtest*.csv', '*_validation*.csv', 'diagnostic*.csv']:
        csv_files.extend(results_dir.glob(pattern))
    
    # Also check subdirectories
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            for pattern in ['*.csv', '*/trades.csv']:
                csv_files.extend(subdir.glob(pattern))
    
    return list(set(csv_files))


def standardize_backtest_df(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
   """Standardize backtest DataFrame columns."""
    
    # Required columns mapping
    column_mapping = {
        'entry_time': ['entry_time', 'open_time', 'timestamp', 'time'],
        'exit_time': ['exit_time', 'close_time'],
        'symbol': ['symbol', 'pair'],
        'entry_price': ['entry_price', 'entry', 'open_price'],
        'exit_price': ['exit_price', 'exit', 'close_price'],
        'pnl': ['pnl', 'profit', 'return'],
        'pnl_pct': ['pnl_pct', 'profit_pct', 'return_pct'],
    }
    
    standardized = pd.DataFrame()
    
    # Map columns
    for std_col, possible_cols in column_mapping.items():
        for col in possible_cols:
            if col in df.columns:
                standardized[std_col] = df[col]
                break
    
    # Add metadata
    standardized['source_file'] = source_file
    standardized['migrated_at'] = datetime.now()
    
    return standardized


def consolidate_backtest_results():
    """Consolidate all backtest results."""
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS CONSOLIDATION")
    print("CSV → Parquet Format")
    print("="*60)
    
    results_manager = ResultsManager()
    results_dir = Path("results")
    
    # Find CSV files
    print("\n📁 Searching for backtest CSV files...")
    csv_files = find_backtest_csvs(results_dir)
    
    if not csv_files:
        print("⚠️  No CSV files found")
        return
    
    print(f"✅ Found {len(csv_files)} CSV files")
    
    # Process each file
    consolidated_count = 0
    
    for csv_file in csv_files:
        try:
            print(f"\n📊 Processing: {csv_file.name}")
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            if df.empty:
                print(f"  ⚠️  Empty file, skipping")
                continue
            
            # Standardize
            std_df = standardize_backtest_df(df, csv_file.name)
            
            # Generate name (remove .csv, sanitize)
            name = csv_file.stem.replace(' ', '_').replace('(', '').replace(')', '').lower()
            
            # Save to Parquet
            results_manager.save_backtest(
                std_df,
                name=name,
                metadata={
                    'original_file': str(csv_file),
                    'rows': len(df),
                    'migrated_at': datetime.now().isoformat()
                }
            )
            
            print(f"  ✅ Saved {len(df)} trades as '{name}.parquet'")
            consolidated_count += 1
            
            # Save experiment metadata
            metrics = {}
            if 'pnl' in std_df.columns:
                metrics['total_pnl'] = std_df['pnl'].sum()
            if 'pnl_pct' in std_df.columns:
                metrics['avg_return'] = std_df['pnl_pct'].mean()
                metrics['win_rate'] = (std_df['pnl_pct'] > 0).mean()
            
            if metrics:
                results_manager.save_experiment(
                    name=name,
                    parameters={'source': 'consolidated_from_csv'},
                    metrics=metrics,
                    description=f"Migrated from {csv_file.name}",
                    backtest_name=name
                )
            
        except Exception as e:
            print(f"  ❌ Error processing {csv_file.name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("✅ CONSOLIDATION COMPLETE")
    print("="*60)
    print(f"\nConsolidated: {consolidated_count}/{len(csv_files)} files")
    print(f"Location: {results_manager.backtests_dir}")
    print()
    
    # Show stats
    stats = results_manager.get_stats()
    print("📊 Results Stats:")
    print(f"  Backtests: {stats['backtests']}")
    print(f"  Experiments: {stats['experiments']}")
    print()


def verify_consolidation():
    """Verify consolidation was successful."""
    print("\n🔍 Verifying consolidation...")
    
    results_manager = ResultsManager()
    
    try:
        # List backtests
        backtests = results_manager.list_backtests()
        print(f"✅ Backtests available: {len(backtests)}")
        
        # List experiments
        experiments = results_manager.list_experiments()
        print(f"✅ Experiments available: {len(experiments)}")
        
        # Try loading one
        if backtests:
            test_bt = results_manager.load_backtest(backtests[0])
            print(f"✅ Sample backtest loaded: {len(test_bt)} trades")
        
        print("\n✅ Consolidation verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    try:
        consolidate_backtest_results()
        verify_consolidation()
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        import traceback
        traceback.print_exc()
