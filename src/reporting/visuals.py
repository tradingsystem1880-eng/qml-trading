"""
Visualization Engine for Strategy Reports
==========================================
Publication-ready charts for the Deployment Dossier.

Includes:
1. Regime-Colored Equity Curves
2. Monte Carlo Confidence Cones
3. Parameter Sensitivity Surfaces
4. Feature Importance Charts
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for notebook compatibility
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - plotting disabled")


@dataclass
class VisualizationConfig:
    """Configuration for report visualizations."""
    
    # Figure settings
    figsize_large: Tuple[int, int] = (14, 8)
    figsize_medium: Tuple[int, int] = (12, 6)
    figsize_small: Tuple[int, int] = (10, 5)
    
    # DPI for saved images
    dpi: int = 150
    
    # Style
    style: str = "dark_background"
    
    # Colors
    color_bull: str = "#2ecc71"      # Green
    color_bear: str = "#e74c3c"      # Red
    color_neutral: str = "#3498db"   # Blue
    color_equity: str = "#00d4ff"    # Cyan
    color_ci_95: str = "#3498db"     # Blue
    color_ci_99: str = "#9b59b6"     # Purple
    
    # Regime colors
    regime_colors: Dict[str, str] = field(default_factory=lambda: {
        "bull_quiet": "#27ae60",     # Green
        "bull_volatile": "#f1c40f",  # Yellow
        "bear_quiet": "#e67e22",     # Orange
        "bear_volatile": "#c0392b",  # Red
    })
    
    # Output directory
    output_dir: str = "reports"


class ReportVisualizer:
    """
    Professional Visualization Engine for Strategy Reports.
    
    Creates publication-ready charts for the Deployment Dossier.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.style.use(self.config.style)
        
        logger.info(f"ReportVisualizer initialized with style: {self.config.style}")
    
    def plot_regime_equity_curve(
        self,
        equity_curve: np.ndarray,
        regime_labels: np.ndarray,
        regime_mapping: Dict[int, str],
        timestamps: Optional[pd.DatetimeIndex] = None,
        title: str = "Equity Curve by Market Regime",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot equity curve with background shading by regime.
        
        Args:
            equity_curve: Equity values over time
            regime_labels: Regime label for each bar
            regime_mapping: Label -> regime name mapping
            timestamps: Optional timestamps for x-axis
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure or None
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize_large)
        
        # Create x-axis
        n = len(equity_curve)
        x = np.arange(n) if timestamps is None else timestamps
        
        # Plot regime backgrounds
        current_regime = None
        start_idx = 0
        
        for i, label in enumerate(regime_labels):
            if np.isnan(label):
                continue
            
            regime = regime_mapping.get(int(label), "unknown")
            
            if current_regime is not None and regime != current_regime:
                # Draw background for previous regime
                color = self.config.regime_colors.get(current_regime, "#808080")
                ax.axvspan(x[start_idx] if timestamps is not None else start_idx,
                          x[i] if timestamps is not None else i,
                          alpha=0.3, color=color, label=current_regime if i == 1 else None)
                start_idx = i
            
            current_regime = regime
        
        # Final regime
        if current_regime:
            color = self.config.regime_colors.get(current_regime, "#808080")
            ax.axvspan(x[start_idx] if timestamps is not None else start_idx,
                      x[-1] if timestamps is not None else n - 1,
                      alpha=0.3, color=color)
        
        # Plot equity curve
        ax.plot(x, equity_curve, color=self.config.color_equity, linewidth=2, label="Equity")
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Equity ($)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Legend
        legend_patches = [mpatches.Patch(color=c, alpha=0.5, label=n) 
                         for n, c in self.config.regime_colors.items()]
        ax.legend(handles=legend_patches, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved regime equity curve to {save_path}")
        
        return fig
    
    def plot_monte_carlo_cones(
        self,
        equity_paths: np.ndarray,
        initial_capital: float = 100000,
        title: str = "Monte Carlo Simulation - Equity Cones",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot Monte Carlo equity cones with 95% and 99% confidence bands.
        
        Args:
            equity_paths: Array of equity paths (n_simulations x n_steps)
            initial_capital: Initial capital for labeling
            title: Plot title
            save_path: Optional save path
            
        Returns:
            matplotlib Figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize_large)
        
        n_steps = equity_paths.shape[1]
        x = np.arange(n_steps)
        
        # Calculate percentiles
        p1 = np.percentile(equity_paths, 0.5, axis=0)   # 99% lower
        p5 = np.percentile(equity_paths, 2.5, axis=0)   # 95% lower
        p50 = np.percentile(equity_paths, 50, axis=0)   # Median
        p95 = np.percentile(equity_paths, 97.5, axis=0) # 95% upper
        p99 = np.percentile(equity_paths, 99.5, axis=0) # 99% upper
        
        # Plot 99% confidence cone
        ax.fill_between(x, p1, p99, alpha=0.2, color=self.config.color_ci_99, 
                        label='99% Confidence')
        
        # Plot 95% confidence cone
        ax.fill_between(x, p5, p95, alpha=0.3, color=self.config.color_ci_95,
                        label='95% Confidence')
        
        # Plot median
        ax.plot(x, p50, color=self.config.color_equity, linewidth=2.5, 
                label='Median Path')
        
        # Plot a few sample paths for context (faded)
        n_sample = min(50, equity_paths.shape[0])
        sample_indices = np.random.choice(equity_paths.shape[0], n_sample, replace=False)
        for idx in sample_indices:
            ax.plot(x, equity_paths[idx], alpha=0.05, color='white', linewidth=0.5)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Trade Number", fontsize=12)
        ax.set_ylabel("Equity ($)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Add initial capital reference line
        ax.axhline(y=initial_capital, color='white', linestyle='--', alpha=0.5, 
                   label=f'Initial: ${initial_capital:,.0f}')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved Monte Carlo cones to {save_path}")
        
        return fig
    
    def plot_parameter_surface(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        param1_name: str = "Parameter 1",
        param2_name: str = "Parameter 2",
        metric_name: str = "Sharpe Ratio",
        title: str = "Parameter Sensitivity Surface",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot 3D parameter sensitivity surface.
        
        Args:
            X, Y, Z: Meshgrid arrays for surface
            param1_name, param2_name: Parameter names
            metric_name: Metric being plotted (Z axis)
            title: Plot title
            save_path: Optional save path
            
        Returns:
            matplotlib Figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig = plt.figure(figsize=self.config.figsize_large)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create custom colormap (red -> yellow -> green)
        colors = ['#c0392b', '#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
        cmap = LinearSegmentedColormap.from_list('stability', colors)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, 
                               edgecolor='white', linewidth=0.3)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=metric_name)
        
        # Formatting
        ax.set_xlabel(param1_name, fontsize=11)
        ax.set_ylabel(param2_name, fontsize=11)
        ax.set_zlabel(metric_name, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Adjust viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved parameter surface to {save_path}")
        
        return fig
    
    def plot_parameter_heatmap(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        param1_name: str = "Parameter 1",
        param2_name: str = "Parameter 2",
        metric_name: str = "Sharpe Ratio",
        title: str = "Parameter Stability Heatmap",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot 2D heatmap of parameter sensitivity (alternative to 3D surface).
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize_medium)
        
        # Create heatmap
        colors = ['#c0392b', '#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
        cmap = LinearSegmentedColormap.from_list('stability', colors)
        
        im = ax.imshow(Z.T, cmap=cmap, aspect='auto', origin='lower',
                       extent=[X.min(), X.max(), Y.min(), Y.max()])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name, fontsize=11)
        
        # Mark optimal point
        max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
        opt_x = X[max_idx[0], max_idx[1]]
        opt_y = Y[max_idx[0], max_idx[1]]
        ax.plot(opt_x, opt_y, 'w*', markersize=15, markeredgecolor='black',
                label=f'Optimal: ({opt_x:.3f}, {opt_y:.3f})')
        
        # Formatting
        ax.set_xlabel(param1_name, fontsize=12)
        ax.set_ylabel(param2_name, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved parameter heatmap to {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 20,
        title: str = "Top Features Driving Edge",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot horizontal bar chart of feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            top_n: Number of top features to show
            title: Plot title
            save_path: Optional save path
            
        Returns:
            matplotlib Figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Sort and get top N
        indices = np.argsort(importance_scores)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        fig, ax = plt.subplots(figsize=self.config.figsize_medium)
        
        # Create gradient colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
        
        # Horizontal bar chart (reversed for top to bottom)
        y_pos = np.arange(top_n)
        bars = ax.barh(y_pos, top_scores[::-1], color=colors[::-1])
        
        # Feature labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features[::-1], fontsize=10)
        
        # Value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores[::-1])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', fontsize=9)
        
        # Formatting
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved feature importance to {save_path}")
        
        return fig
    
    def plot_drawdown_chart(
        self,
        equity_curve: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        title: str = "Drawdown Analysis",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot drawdown chart with underwater plot.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize_large, 
                                        height_ratios=[2, 1], sharex=True)
        
        n = len(equity_curve)
        x = np.arange(n) if timestamps is None else timestamps
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max * 100
        
        # Top panel: Equity curve
        ax1.plot(x, equity_curve, color=self.config.color_equity, linewidth=1.5)
        ax1.plot(x, running_max, color='white', linewidth=1, linestyle='--', alpha=0.5)
        ax1.fill_between(x, equity_curve, running_max, alpha=0.3, color='#e74c3c')
        ax1.set_ylabel("Equity ($)", fontsize=11)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Bottom panel: Drawdown
        ax2.fill_between(x, 0, -drawdown, color='#e74c3c', alpha=0.7)
        ax2.plot(x, -drawdown, color='#c0392b', linewidth=1)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Time", fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Max drawdown annotation
        max_dd_idx = np.argmax(drawdown)
        max_dd = drawdown[max_dd_idx]
        ax2.annotate(f'Max DD: {max_dd:.1f}%', 
                    xy=(x[max_dd_idx], -max_dd),
                    xytext=(x[max_dd_idx], -max_dd - 5),
                    fontsize=10, color='white',
                    arrowprops=dict(arrowstyle='->', color='white'))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved drawdown chart to {save_path}")
        
        return fig
    
    def plot_permutation_histogram(
        self,
        permutation_sharpes: np.ndarray,
        actual_sharpe: float,
        p_value: float,
        title: str = "Permutation Test - Skill vs Luck",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot histogram of permutation test results.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize_medium)
        
        # Histogram
        ax.hist(permutation_sharpes, bins=50, alpha=0.7, color=self.config.color_ci_95,
                edgecolor='white', linewidth=0.5, label='Random Shuffles')
        
        # Actual Sharpe line
        ax.axvline(x=actual_sharpe, color=self.config.color_equity, linewidth=3,
                   label=f'Actual Sharpe: {actual_sharpe:.3f}')
        
        # Significance threshold (95th percentile)
        threshold_95 = np.percentile(permutation_sharpes, 95)
        ax.axvline(x=threshold_95, color='#f1c40f', linewidth=2, linestyle='--',
                   label=f'95th Percentile: {threshold_95:.3f}')
        
        # Formatting
        ax.set_xlabel("Sharpe Ratio", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add p-value annotation
        significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
        color = self.config.color_bull if p_value < 0.05 else self.config.color_bear
        ax.text(0.02, 0.98, f'p-value: {p_value:.4f}\n{significance}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved permutation histogram to {save_path}")
        
        return fig
    
    def plot_regime_performance(
        self,
        regime_stats: pd.DataFrame,
        title: str = "Performance by Market Regime",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot bar chart of performance metrics by regime.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=self.config.figsize_large)
        
        regimes = regime_stats["regime"].tolist()
        colors = [self.config.regime_colors.get(r, "#808080") for r in regimes]
        
        # Sharpe by regime
        axes[0].bar(regimes, regime_stats["sharpe"], color=colors, edgecolor='white')
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].set_title("Sharpe by Regime")
        axes[0].axhline(y=0, color='white', linewidth=0.5)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Win rate by regime
        if "win_rate" in regime_stats.columns:
            axes[1].bar(regimes, regime_stats["win_rate"] * 100, color=colors, edgecolor='white')
            axes[1].set_ylabel("Win Rate (%)")
            axes[1].set_title("Win Rate by Regime")
            axes[1].axhline(y=50, color='white', linestyle='--', linewidth=0.5)
            axes[1].tick_params(axis='x', rotation=45)
        
        # % of data by regime
        axes[2].bar(regimes, regime_stats["pct_of_data"], color=colors, edgecolor='white')
        axes[2].set_ylabel("% of Data")
        axes[2].set_title("Data Distribution")
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved regime performance to {save_path}")
        
        return fig
    
    def create_all_visuals(
        self,
        result: Any,  # PipelineResult
        output_dir: str
    ) -> Dict[str, str]:
        """
        Create all visualizations for a pipeline result.
        
        Args:
            result: PipelineResult from ValidationOrchestrator
            output_dir: Directory to save visuals
            
        Returns:
            Dictionary of visual name -> file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visuals = {}
        
        # Monte Carlo cones
        if result.monte_carlo_result and hasattr(result.monte_carlo_result, 'equity_paths'):
            path = str(output_path / "monte_carlo_cones.png")
            self.plot_monte_carlo_cones(
                result.monte_carlo_result.equity_paths,
                save_path=path
            )
            visuals["monte_carlo"] = path
        
        # Permutation histogram
        if result.permutation_result:
            path = str(output_path / "permutation_test.png")
            self.plot_permutation_histogram(
                result.permutation_result.permutation_sharpes,
                result.permutation_result.actual_sharpe,
                result.permutation_result.sharpe_p_value,
                save_path=path
            )
            visuals["permutation"] = path
        
        logger.info(f"Created {len(visuals)} visualizations in {output_dir}")
        
        return visuals


def create_visualizer(
    style: str = "dark_background",
    output_dir: str = "reports"
) -> ReportVisualizer:
    """Factory function for ReportVisualizer."""
    config = VisualizationConfig(style=style, output_dir=output_dir)
    return ReportVisualizer(config=config)
