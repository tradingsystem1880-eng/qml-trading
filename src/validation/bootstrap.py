"""
Block Bootstrap Confidence Intervals
=====================================
Computes confidence intervals for strategy metrics while
preserving temporal dependencies via block bootstrapping.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis for a single metric."""
    
    metric_name: str
    point_estimate: float
    ci_lower: float           # Lower bound of CI
    ci_upper: float           # Upper bound of CI
    confidence_level: float   # e.g., 0.95
    
    # Full distribution
    bootstrap_distribution: np.ndarray
    
    # Hypothesis testing
    p_value_vs_null: float    # P-value against null hypothesis
    null_value: float         # Null value tested against
    
    # Standard error
    standard_error: float
    
    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower
    
    @property
    def is_significant(self) -> bool:
        """Is the metric significantly different from null?"""
        return self.p_value_vs_null < (1 - self.confidence_level)


class BlockBootstrap:
    """
    Block Bootstrap for Confidence Intervals.
    
    Uses block bootstrapping to preserve temporal dependencies
    in the trade sequence when computing confidence intervals.
    
    This is critical for time-series data where trades may be
    serially correlated.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 5000,
        block_size: int = 5,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None
    ):
        """
        Initialize block bootstrap.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            block_size: Size of blocks for preserving autocorrelation
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            random_seed: Optional seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.confidence_level = confidence_level
        self.rng = np.random.default_rng(random_seed)
        
        logger.info(
            f"BlockBootstrap initialized: {n_bootstrap} samples, "
            f"block_size={block_size}, CI={confidence_level:.0%}"
        )
    
    def confidence_interval(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float]
    ) -> BootstrapResult:
        """
        Compute bootstrap confidence interval for a statistic.
        
        Args:
            data: Data array (e.g., trade returns)
            statistic_fn: Function that computes statistic from data
            
        Returns:
            BootstrapResult with CI and distribution
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        if n < self.block_size * 2:
            logger.warning(
                f"Data length ({n}) is small relative to block_size ({self.block_size}). "
                "Results may be unreliable."
            )
        
        # Calculate point estimate
        point_estimate = statistic_fn(data)
        
        # Generate bootstrap samples
        boot_stats = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            boot_sample = self._block_resample(data)
            boot_stats[i] = statistic_fn(boot_sample)
        
        # Calculate confidence interval (percentile method)
        alpha = 1 - self.confidence_level
        ci_lower = float(np.percentile(boot_stats, alpha / 2 * 100))
        ci_upper = float(np.percentile(boot_stats, (1 - alpha / 2) * 100))
        
        # Standard error
        standard_error = float(np.std(boot_stats, ddof=1))
        
        return BootstrapResult(
            metric_name="custom",
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            bootstrap_distribution=boot_stats,
            p_value_vs_null=np.nan,
            null_value=np.nan,
            standard_error=standard_error,
        )
    
    def hypothesis_test(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        null_value: float,
        alternative: str = "greater"
    ) -> BootstrapResult:
        """
        Bootstrap hypothesis test against a null value.
        
        Args:
            data: Data array
            statistic_fn: Function that computes statistic
            null_value: Null hypothesis value (e.g., 0.5 for win rate)
            alternative: "greater", "less", or "two-sided"
            
        Returns:
            BootstrapResult with p-value
        """
        # First get confidence interval
        result = self.confidence_interval(data, statistic_fn)
        
        # Calculate p-value
        if alternative == "greater":
            p_value = float(np.mean(result.bootstrap_distribution <= null_value))
        elif alternative == "less":
            p_value = float(np.mean(result.bootstrap_distribution >= null_value))
        else:  # two-sided
            dist = result.bootstrap_distribution
            centered = np.abs(dist - np.mean(dist))
            observed_diff = np.abs(result.point_estimate - null_value)
            p_value = float(np.mean(centered >= observed_diff))
        
        result.p_value_vs_null = p_value
        result.null_value = null_value
        
        return result
    
    def all_metrics_ci(
        self,
        trades_df: pd.DataFrame,
        pnl_column: str = "pnl_pct"
    ) -> Dict[str, BootstrapResult]:
        """
        Compute confidence intervals for all standard metrics.
        
        Args:
            trades_df: DataFrame with trade data
            pnl_column: Column containing trade P&L (percentage)
            
        Returns:
            Dictionary mapping metric names to BootstrapResults
        """
        returns = trades_df[pnl_column].values
        
        # Define metric functions
        def sharpe_fn(r):
            if len(r) == 0 or np.std(r) == 0:
                return 0.0
            return float(np.mean(r) / np.std(r))
        
        def win_rate_fn(r):
            if len(r) == 0:
                return 0.0
            return float(np.mean(r > 0))
        
        def profit_factor_fn(r):
            wins = r[r > 0].sum()
            losses = np.abs(r[r < 0].sum())
            if losses == 0:
                return float('inf') if wins > 0 else 0.0
            return float(wins / losses)
        
        def avg_win_fn(r):
            wins = r[r > 0]
            return float(np.mean(wins)) if len(wins) > 0 else 0.0
        
        def avg_loss_fn(r):
            losses = r[r < 0]
            return float(np.mean(losses)) if len(losses) > 0 else 0.0
        
        def max_dd_fn(r):
            equity = np.cumprod(1 + r / 100)
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max * 100
            return float(np.max(drawdowns))
        
        def total_return_fn(r):
            return float(np.sum(r))
        
        # Compute all CIs
        results = {}
        
        logger.info("Computing bootstrap CIs for all metrics...")
        
        # Sharpe Ratio
        result = self.confidence_interval(returns, sharpe_fn)
        result.metric_name = "sharpe_ratio"
        results["sharpe_ratio"] = result
        
        # Win Rate (with hypothesis test against 50%)
        result = self.hypothesis_test(returns, win_rate_fn, null_value=0.5, alternative="greater")
        result.metric_name = "win_rate"
        results["win_rate"] = result
        
        # Profit Factor (with hypothesis test against 1.0)
        result = self.hypothesis_test(returns, profit_factor_fn, null_value=1.0, alternative="greater")
        result.metric_name = "profit_factor"
        results["profit_factor"] = result
        
        # Average Win
        result = self.confidence_interval(returns, avg_win_fn)
        result.metric_name = "avg_win"
        results["avg_win"] = result
        
        # Average Loss
        result = self.confidence_interval(returns, avg_loss_fn)
        result.metric_name = "avg_loss"
        results["avg_loss"] = result
        
        # Max Drawdown
        result = self.confidence_interval(returns, max_dd_fn)
        result.metric_name = "max_drawdown"
        results["max_drawdown"] = result
        
        # Total Return
        result = self.confidence_interval(returns, total_return_fn)
        result.metric_name = "total_return"
        results["total_return"] = result
        
        logger.info(f"Computed CIs for {len(results)} metrics")
        
        return results
    
    def _block_resample(self, data: np.ndarray) -> np.ndarray:
        """
        Resample data using moving block bootstrap.
        
        Args:
            data: Original data array
            
        Returns:
            Resampled array of same length
        """
        n = len(data)
        block_size = min(self.block_size, n)
        n_blocks = int(np.ceil(n / block_size))
        
        # Generate random block starts
        max_start = n - block_size
        if max_start <= 0:
            # Data too short for blocks, fall back to regular bootstrap
            indices = self.rng.integers(0, n, size=n)
            return data[indices]
        
        block_starts = self.rng.integers(0, max_start + 1, size=n_blocks)
        
        # Build resampled array
        resampled = []
        for start in block_starts:
            block = data[start:start + block_size]
            resampled.extend(block)
        
        return np.array(resampled[:n])
    
    def generate_report(self, results: Dict[str, BootstrapResult]) -> str:
        """
        Generate text report of bootstrap results.
        
        Args:
            results: Dictionary of BootstrapResults
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "BOOTSTRAP CONFIDENCE INTERVALS",
            "=" * 60,
            "",
            f"Bootstrap samples: {self.n_bootstrap:,}",
            f"Block size: {self.block_size}",
            f"Confidence level: {self.confidence_level:.0%}",
            "",
            "Metric Confidence Intervals:",
            "-" * 60,
        ]
        
        for name, res in results.items():
            ci_str = f"[{res.ci_lower:.4f}, {res.ci_upper:.4f}]"
            
            line = f"  {name:20s}: {res.point_estimate:8.4f}  CI: {ci_str}"
            
            if not np.isnan(res.p_value_vs_null):
                sig = "***" if res.p_value_vs_null < 0.01 else (
                    "**" if res.p_value_vs_null < 0.05 else (
                        "*" if res.p_value_vs_null < 0.10 else ""
                    )
                )
                line += f"  p={res.p_value_vs_null:.4f}{sig}"
            
            lines.append(line)
        
        lines.extend([
            "",
            "-" * 60,
            "Significance: *** p<0.01, ** p<0.05, * p<0.10",
            "",
            "Key Statistical Tests:",
        ])
        
        if "win_rate" in results:
            wr = results["win_rate"]
            if wr.p_value_vs_null < 0.01:
                lines.append(f"  ✓ Win rate ({wr.point_estimate:.1%}) is significantly > 50% (p={wr.p_value_vs_null:.4f})")
            else:
                lines.append(f"  ✗ Win rate ({wr.point_estimate:.1%}) is NOT significantly > 50% (p={wr.p_value_vs_null:.4f})")
        
        if "profit_factor" in results:
            pf = results["profit_factor"]
            if pf.p_value_vs_null < 0.01:
                lines.append(f"  ✓ Profit factor ({pf.point_estimate:.2f}) is significantly > 1.0 (p={pf.p_value_vs_null:.4f})")
            else:
                lines.append(f"  ✗ Profit factor ({pf.point_estimate:.2f}) is NOT significantly > 1.0 (p={pf.p_value_vs_null:.4f})")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def compute_all_confidence_intervals(
    trades_df: pd.DataFrame,
    pnl_column: str = "pnl_pct",
    n_bootstrap: int = 5000,
    block_size: int = 5,
    confidence_level: float = 0.95,
    seed: Optional[int] = None
) -> Dict[str, BootstrapResult]:
    """
    Convenience function to compute all metric CIs.
    
    Args:
        trades_df: DataFrame with trade data
        pnl_column: P&L column name
        n_bootstrap: Number of bootstrap samples
        block_size: Block size for block bootstrap
        confidence_level: Confidence level
        seed: Random seed
        
    Returns:
        Dictionary of BootstrapResults
    """
    bootstrap = BlockBootstrap(
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        confidence_level=confidence_level,
        random_seed=seed
    )
    return bootstrap.all_metrics_ci(trades_df, pnl_column)
