#!/usr/bin/env python3
"""
Phase 9.6: SHAP Feature Analysis
================================
Train XGBoost model to predict trade outcomes, then explain with SHAP.

This tells us which features drive the edge and if we're relying on robust
features or noise.

Outputs:
- shap_summary_beeswarm.png - Feature importance with direction
- shap_importance_bar.png - Absolute importance ranking
- shap_dependence_*.png - Top 5 feature partial dependence plots
- feature_importance.json - Ranked feature importance data

Usage:
    python scripts/run_shap_analysis.py
    python scripts/run_shap_analysis.py --trades-file results/report_trades.parquet
    python scripts/run_shap_analysis.py --output-dir reports/shap
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Feature columns to analyze
FEATURE_COLUMNS = [
    'total_score',
    'head_extension_score',
    'bos_efficiency_score',
    'shoulder_symmetry_score',
    'swing_significance_score',
    'atr_at_signal',
    'atr_percentile',
    'adx',
    'volume_ratio',
    'pattern_bars',
    'entry_hour',
    'entry_day_of_week',
    'regime_score',
]


def run_shap_analysis(
    trades_df: pd.DataFrame,
    feature_columns: List[str],
    output_dir: str = 'reports/shap/',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train a model to predict trade outcome, then explain with SHAP.

    Args:
        trades_df: DataFrame with trades and features
        feature_columns: List of feature column names
        output_dir: Directory to save plots
        verbose: Print progress

    Returns:
        DataFrame with feature importance ranking
    """
    # Lazy imports for optional dependencies
    try:
        import shap
        import xgboost as xgb
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"ERROR: Required library not installed: {e}")
        print("Install with: pip install shap xgboost matplotlib")
        return pd.DataFrame()

    # Set matplotlib backend for non-interactive use
    import matplotlib
    matplotlib.use('Agg')

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter to available features
    available_features = [f for f in feature_columns if f in trades_df.columns]
    missing_features = [f for f in feature_columns if f not in trades_df.columns]

    if missing_features and verbose:
        print(f"Warning: Missing features: {missing_features}")

    if len(available_features) < 3:
        print("ERROR: Not enough features available for SHAP analysis")
        return pd.DataFrame()

    if verbose:
        print(f"Using {len(available_features)} features: {available_features}")

    # Prepare data
    X = trades_df[available_features].copy()
    y = (trades_df['pnl_r'] > 0).astype(int)  # Binary: win/loss

    # Handle missing values
    X = X.fillna(X.median())

    if verbose:
        print(f"\nTraining data: {len(X)} samples")
        print(f"Win rate: {y.mean():.1%}")

    # Train XGBoost (just for explanation, not prediction)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    model.fit(X, y)

    if verbose:
        print("Model trained. Calculating SHAP values...")

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. Summary plot (beeswarm)
    if verbose:
        print("Generating beeswarm plot...")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False, plot_size=(12, 8))
    plt.title("SHAP Feature Importance - Beeswarm Plot", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'shap_summary_beeswarm.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    # 2. Feature importance bar chart
    if verbose:
        print("Generating importance bar chart...")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance - Bar Chart", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'shap_importance_bar.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    # 3. Calculate feature importance ranking
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    if verbose:
        print("\nFeature Importance Ranking:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    # 4. Partial dependence for top 5 features
    top_features = feature_importance.head(5)['feature'].tolist()

    if verbose:
        print(f"\nGenerating dependence plots for top 5 features...")

    for feature in top_features:
        try:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(feature, shap_values, X, show=False)
            plt.title(f"SHAP Dependence: {feature}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            safe_name = feature.replace('/', '_').replace(' ', '_')
            plt.savefig(output_path / f'shap_dependence_{safe_name}.png', dpi=150,
                        bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate dependence plot for {feature}: {e}")

    # 5. Save feature importance to JSON
    importance_dict = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(X),
        "num_features": len(available_features),
        "win_rate": float(y.mean()),
        "feature_importance": feature_importance.to_dict('records'),
        "top_5_features": top_features,
    }

    with open(output_path / 'feature_importance.json', 'w') as f:
        json.dump(importance_dict, f, indent=2)

    if verbose:
        print(f"\nSHAP analysis complete. Files saved to: {output_path}")

    return feature_importance


def main():
    parser = argparse.ArgumentParser(description="Run SHAP feature analysis")
    parser.add_argument('--trades-file', type=str, default='results/report_trades.parquet',
                        help='Path to trades parquet file')
    parser.add_argument('--output-dir', type=str, default='reports/shap',
                        help='Output directory for plots')
    args = parser.parse_args()

    trades_path = PROJECT_ROOT / args.trades_file

    print("=" * 70)
    print("PHASE 9.6: SHAP FEATURE ANALYSIS")
    print("=" * 70)

    if not trades_path.exists():
        print(f"\nTrades file not found: {trades_path}")
        print("Run collect_report_data.py first to generate trade data.")
        sys.exit(1)

    print(f"\nLoading trades from: {trades_path}")
    trades_df = pd.read_parquet(trades_path)
    print(f"Loaded {len(trades_df)} trades")

    feature_importance = run_shap_analysis(
        trades_df,
        FEATURE_COLUMNS,
        output_dir=args.output_dir,
        verbose=True,
    )

    if not feature_importance.empty:
        print("\n" + "=" * 70)
        print("TOP 3 MOST IMPORTANT FEATURES")
        print("=" * 70)
        for i, row in feature_importance.head(3).iterrows():
            print(f"\n{row['feature']}:")
            print(f"  SHAP Importance: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
