#!/usr/bin/env python3
"""
Phase 9.6: Feature Scatter Plots
================================
Generate scatter plots for every feature vs trade R-multiple outcome.
Helps identify which features have predictive power.

Outputs:
- feature_scatter_matrix.png - All features vs R-multiple
- feature_correlation_heatmap.png - Correlation matrix

Usage:
    python scripts/generate_feature_scatter.py
    python scripts/generate_feature_scatter.py --trades-file results/report_trades.parquet
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Feature columns to plot
FEATURE_COLUMNS = [
    'total_score',
    'head_extension_score',
    'bos_efficiency_score',
    'atr_at_signal',
    'atr_percentile',
    'adx',
    'volume_ratio',
    'pattern_bars',
    'entry_hour',
    'regime_score',
]


def generate_scatter_matrix(
    trades_df: pd.DataFrame,
    feature_columns: List[str],
    output_path: str,
    verbose: bool = True,
) -> None:
    """
    Generate scatter plots for all features vs R-multiple.

    Args:
        trades_df: DataFrame with trades and features
        feature_columns: List of feature column names
        output_path: Path to save output image
        verbose: Print progress
    """
    # Filter to available features
    available_features = [f for f in feature_columns if f in trades_df.columns]

    if len(available_features) == 0:
        print("ERROR: No features available for plotting")
        return

    if verbose:
        print(f"Plotting {len(available_features)} features")

    # Set up figure
    n_features = len(available_features)
    fig, axes = plt.subplots(
        nrows=n_features,
        ncols=1,
        figsize=(12, 4 * n_features)
    )

    if n_features == 1:
        axes = [axes]

    for idx, feature in enumerate(available_features):
        ax = axes[idx]

        # Get data
        x = trades_df[feature].fillna(trades_df[feature].median())
        y = trades_df['pnl_r']

        # Color by win/loss
        colors = ['#22C55E' if pnl > 0 else '#EF4444' for pnl in y]

        # Scatter plot
        ax.scatter(x, y, c=colors, alpha=0.5, s=30, edgecolors='none')

        # Add regression line
        valid_mask = ~(np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y))
        if valid_mask.sum() > 10:
            try:
                z = np.polyfit(x[valid_mask], y[valid_mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x[valid_mask])
                ax.plot(x_sorted, p(x_sorted), "#3B82F6", linestyle='--',
                        alpha=0.8, linewidth=2, label='Trend')
            except:
                pass

        # Correlation coefficient
        corr = x.corr(y)

        # Styling
        ax.set_facecolor('#0B1426')
        ax.set_title(f'{feature} vs R-Multiple (r={corr:.3f})',
                     fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel(feature, color='white')
        ax.set_ylabel('R-Multiple', color='white')
        ax.axhline(y=0, color='#64748B', linestyle='-', alpha=0.5)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['right'].set_color('#334155')

        # Add correlation annotation
        corr_color = '#22C55E' if corr > 0.1 else '#EF4444' if corr < -0.1 else '#F59E0B'
        ax.text(0.02, 0.98, f'Correlation: {corr:+.3f}',
                transform=ax.transAxes, fontsize=10, color=corr_color,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#162032', alpha=0.8))

    plt.tight_layout()
    fig.patch.set_facecolor('#0B1426')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0B1426', edgecolor='none')
    plt.close()

    if verbose:
        print(f"Saved scatter matrix to: {output_path}")


def generate_correlation_heatmap(
    trades_df: pd.DataFrame,
    feature_columns: List[str],
    output_path: str,
    verbose: bool = True,
) -> None:
    """
    Generate correlation heatmap between features and outcome.

    Args:
        trades_df: DataFrame with trades and features
        feature_columns: List of feature column names
        output_path: Path to save output image
        verbose: Print progress
    """
    # Filter to available features
    available_features = [f for f in feature_columns if f in trades_df.columns]
    available_features.append('pnl_r')

    if len(available_features) < 3:
        print("ERROR: Not enough features for correlation heatmap")
        return

    # Calculate correlation matrix
    corr_data = trades_df[available_features].fillna(0)
    corr_matrix = corr_data.corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use imshow for heatmap
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Set ticks
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', color='white', fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, color='white', fontsize=9)

    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontsize=8)

    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold',
                 color='white', pad=20)
    ax.set_facecolor('#0B1426')
    fig.patch.set_facecolor('#0B1426')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0B1426', edgecolor='none')
    plt.close()

    if verbose:
        print(f"Saved correlation heatmap to: {output_path}")

    # Print top correlations with outcome
    outcome_corrs = corr_matrix['pnl_r'].drop('pnl_r').sort_values(key=abs, ascending=False)

    if verbose:
        print("\nTop Feature Correlations with R-Multiple:")
        for feature, corr in outcome_corrs.head(5).items():
            direction = "positive" if corr > 0 else "negative"
            print(f"  {feature}: {corr:+.3f} ({direction})")


def main():
    parser = argparse.ArgumentParser(description="Generate feature scatter plots")
    parser.add_argument('--trades-file', type=str, default='results/report_trades.parquet',
                        help='Path to trades parquet file')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Output directory for plots')
    args = parser.parse_args()

    trades_path = PROJECT_ROOT / args.trades_file
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 70)
    print("PHASE 9.6: FEATURE SCATTER PLOTS")
    print("=" * 70)

    if not trades_path.exists():
        print(f"\nTrades file not found: {trades_path}")
        print("Run collect_report_data.py first to generate trade data.")
        sys.exit(1)

    print(f"\nLoading trades from: {trades_path}")
    trades_df = pd.read_parquet(trades_path)
    print(f"Loaded {len(trades_df)} trades")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate scatter matrix
    print("\n" + "=" * 70)
    print("GENERATING SCATTER PLOTS")
    print("=" * 70)

    generate_scatter_matrix(
        trades_df,
        FEATURE_COLUMNS,
        str(output_dir / 'feature_scatter_matrix.png'),
        verbose=True,
    )

    # Generate correlation heatmap
    print("\n" + "=" * 70)
    print("GENERATING CORRELATION HEATMAP")
    print("=" * 70)

    generate_correlation_heatmap(
        trades_df,
        FEATURE_COLUMNS,
        str(output_dir / 'feature_correlation_heatmap.png'),
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
