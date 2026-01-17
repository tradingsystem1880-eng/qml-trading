"""
Pattern Labeling CLI
====================
Interactive command-line interface for reviewing and labeling patterns.

Enables human labeling of detected patterns for supervised ML training.
Labels are used to train the XGBoost classifier for pattern quality prediction.

Usage:
    python -m src.ml.label_patterns
    python -m src.ml.label_patterns --symbol BTC/USDT --limit 10
    python -m src.ml.label_patterns --stats
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger

from src.ml.pattern_registry import PatternRegistry


def format_pattern_summary(pattern: Dict[str, Any]) -> str:
    """Format pattern for display."""
    lines = []
    
    # Header
    lines.append("\n" + "=" * 60)
    lines.append(f"  📊 Pattern: {pattern['pattern_id'][:12]}...")
    lines.append("=" * 60)
    
    # Metadata
    lines.append(f"  Symbol:     {pattern.get('symbol', 'N/A')}")
    lines.append(f"  Timeframe:  {pattern.get('timeframe', 'N/A')}")
    lines.append(f"  Type:       {pattern.get('pattern_type', 'N/A')}")
    lines.append(f"  Detected:   {pattern.get('detection_time', 'N/A')}")
    lines.append(f"  Validity:   {pattern.get('validity_score', 0):.2f}")
    
    if pattern.get('ml_confidence'):
        lines.append(f"  ML Score:   {pattern['ml_confidence']:.2f}")
    
    if pattern.get('regime_at_detection'):
        lines.append(f"  Regime:     {pattern['regime_at_detection']}")
    
    # Key features (if available)
    features = pattern.get('features', {})
    if features:
        lines.append("\n  Key Features:")
        
        # Show most relevant features
        key_features = [
            ('geo_head_depth_atr', 'Head Depth (ATR)'),
            ('geo_shoulder_symmetry', 'Shoulder Symmetry'),
            ('geo_rr_ratio', 'R:R Ratio'),
            ('ctx_rsi_14', 'RSI(14)'),
            ('vol_atr_14', 'ATR(14)'),
            ('mom_return_20', 'Return 20 bars'),
        ]
        
        for feat_key, feat_name in key_features:
            if feat_key in features:
                lines.append(f"    • {feat_name}: {features[feat_key]:.3f}")
    
    lines.append("")
    
    return "\n".join(lines)


def prompt_for_label() -> str:
    """Prompt user for pattern label."""
    print("\n  Label this pattern:")
    print("    [w] Win     - Pattern led to profitable trade")
    print("    [l] Loss    - Pattern led to losing trade")
    print("    [i] Ignore  - Invalid pattern or unclear outcome")
    print("    [s] Skip    - Skip this pattern for now")
    print("    [q] Quit    - Exit labeling session")
    print()
    
    while True:
        try:
            choice = input("  Enter label (w/l/i/s/q): ").lower().strip()
            
            if choice in ['w', 'l', 'i', 's', 'q']:
                return choice
            
            print("  ❌ Invalid choice. Please enter w, l, i, s, or q.")
            
        except KeyboardInterrupt:
            return 'q'


def prompt_for_outcome() -> Optional[float]:
    """Prompt for trade outcome percentage."""
    print("\n  Enter trade outcome (PnL %):")
    print("    • Enter percentage (e.g., 2.5 or -1.2)")
    print("    • Press Enter to skip")
    print()
    
    try:
        value = input("  PnL %: ").strip()
        
        if not value:
            return None
        
        return float(value)
        
    except (ValueError, KeyboardInterrupt):
        return None


def run_labeling_session(
    registry: PatternRegistry,
    symbol: Optional[str] = None,
    limit: int = 50,
    show_outcome_prompt: bool = True
) -> Dict[str, int]:
    """
    Run interactive labeling session.
    
    Args:
        registry: Pattern registry instance
        symbol: Optional symbol filter
        limit: Maximum patterns to show
        show_outcome_prompt: Whether to ask for PnL outcome
        
    Returns:
        Dictionary of label counts
    """
    # Get unlabeled patterns
    patterns = registry.get_unlabeled_patterns(limit=limit, symbol=symbol)
    
    if not patterns:
        print("\n✅ No unlabeled patterns found!")
        return {}
    
    print(f"\n🏷️  Found {len(patterns)} unlabeled patterns")
    print("Starting labeling session...\n")
    
    counts = {'win': 0, 'loss': 0, 'ignore': 0, 'skip': 0}
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\n[{i}/{len(patterns)}]")
        print(format_pattern_summary(pattern))
        
        label = prompt_for_label()
        
        if label == 'q':
            print("\n👋 Exiting labeling session")
            break
        
        if label == 's':
            counts['skip'] += 1
            continue
        
        # Get outcome if labeling as win/loss
        outcome = None
        if show_outcome_prompt and label in ['w', 'l']:
            outcome = prompt_for_outcome()
        
        # Apply label
        label_map = {'w': 'win', 'l': 'loss', 'i': 'ignore'}
        full_label = label_map[label]
        
        success = registry.label_pattern(
            pattern['pattern_id'],
            full_label,
            outcome=outcome
        )
        
        if success:
            counts[full_label] += 1
            print(f"  ✅ Labeled as '{full_label}'")
        else:
            print(f"  ❌ Failed to label pattern")
    
    # Summary
    print("\n" + "=" * 60)
    print("  📊 Labeling Session Summary")
    print("=" * 60)
    print(f"  Wins labeled:    {counts['win']}")
    print(f"  Losses labeled:  {counts['loss']}")
    print(f"  Ignored:         {counts['ignore']}")
    print(f"  Skipped:         {counts['skip']}")
    print("=" * 60)
    
    return counts


def show_statistics(registry: PatternRegistry):
    """Display registry statistics."""
    stats = registry.get_statistics()
    
    print("\n" + "=" * 60)
    print("  📊 Pattern Registry Statistics")
    print("=" * 60)
    print(f"  Total Patterns:    {stats['total_patterns']}")
    print(f"  Labeled:           {stats['labeled']}")
    print(f"  Unlabeled:         {stats['unlabeled']}")
    print(f"  Win Rate:          {stats['win_rate']:.1f}%")
    print()
    print("  By Label:")
    for label, count in stats['by_label'].items():
        print(f"    • {label or 'unlabeled'}: {count}")
    print()
    print("  By Type:")
    for ptype, count in stats['by_type'].items():
        print(f"    • {ptype}: {count}")
    print()
    print("  By Symbol:")
    for symbol, count in stats['by_symbol'].items():
        print(f"    • {symbol}: {count}")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive pattern labeling CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--db-path',
        default='results/experiments.db',
        help='Path to database file'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        default=None,
        help='Filter patterns by symbol'
    )
    
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=50,
        help='Maximum number of patterns to label'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show registry statistics and exit'
    )
    
    parser.add_argument(
        '--no-outcome',
        action='store_true',
        help='Skip outcome (PnL) prompts'
    )
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = PatternRegistry(args.db_path)
    
    if args.stats:
        show_statistics(registry)
        return 0
    
    try:
        run_labeling_session(
            registry,
            symbol=args.symbol,
            limit=args.limit,
            show_outcome_prompt=not args.no_outcome
        )
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Labeling session interrupted")
        return 1
    except Exception as e:
        logger.error(f"Error during labeling: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
