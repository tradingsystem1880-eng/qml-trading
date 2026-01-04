"""
Purged Walk-Forward Optimization Engine
========================================
Rolling walk-forward validation with purge/embargo periods
to eliminate information leakage between train and test sets.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.validation.tracker import ExperimentTracker


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""
    
    n_folds: int = 10
    purge_bars: int = 5        # Gap after train before test (prevent leakage)
    embargo_bars: int = 5      # Gap after test before next train
    train_ratio: float = 0.7   # Train/test split within each fold
    min_trades_per_fold: int = 10  # Minimum trades for statistical validity
    min_fold_bars: int = 100   # Minimum bars per fold
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if not 0.5 <= self.train_ratio <= 0.9:
            raise ValueError("train_ratio must be between 0.5 and 0.9")
        if self.purge_bars < 0 or self.embargo_bars < 0:
            raise ValueError("purge_bars and embargo_bars must be non-negative")


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""
    
    fold_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Parameters
    optimal_params: Dict[str, Any] = field(default_factory=dict)
    
    # In-sample metrics
    in_sample_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Out-of-sample metrics
    out_of_sample_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Trade counts
    in_sample_trades: int = 0
    out_of_sample_trades: int = 0
    
    # Stability
    param_stability_score: float = 0.0


@dataclass
class WalkForwardResult:
    """Complete walk-forward optimization result."""
    
    fold_results: List[FoldResult]
    
    # Aggregated OOS metrics
    aggregate_sharpe: float = 0.0
    aggregate_return: float = 0.0
    aggregate_max_dd: float = 0.0
    aggregate_win_rate: float = 0.0
    aggregate_profit_factor: float = 0.0
    total_oos_trades: int = 0
    
    # Stability analysis
    param_stability_score: float = 0.0
    sharpe_stability: float = 0.0  # Std of fold Sharpes
    is_to_oos_ratio: float = 0.0   # Average IS/OOS performance ratio
    
    # Configuration used
    config: WalkForwardConfig = field(default_factory=WalkForwardConfig)


class PurgedWalkForwardEngine:
    """
    Walk-Forward Optimization with Purge and Embargo Periods.
    
    Implements rolling train/test splits with:
    - PURGE: Gap between end of training and start of test
      (prevents information leakage from overlapping patterns)
    - EMBARGO: Gap between end of test and start of next training
      (prevents serial correlation contamination)
    
    Timeline for each fold:
    |------- TRAIN -------|-- PURGE --|------- TEST -------|-- EMBARGO --|
    """
    
    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        tracker: Optional[ExperimentTracker] = None
    ):
        """
        Initialize walk-forward engine.
        
        Args:
            config: Walk-forward configuration
            tracker: Experiment tracker for logging results
        """
        self.config = config or WalkForwardConfig()
        self.config.validate()
        self.tracker = tracker
        
        logger.info(
            f"WalkForwardEngine initialized: {self.config.n_folds} folds, "
            f"purge={self.config.purge_bars}, embargo={self.config.embargo_bars}"
        )
    
    def generate_folds(
        self,
        df: pd.DataFrame,
        time_column: str = "time"
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/test fold indices with purge/embargo gaps.
        
        Args:
            df: DataFrame with time index or column
            time_column: Name of time column if not in index
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        # Get time series
        if time_column in df.columns:
            times = pd.to_datetime(df[time_column])
        else:
            times = pd.to_datetime(df.index)
        
        n_bars = len(df)
        n_folds = self.config.n_folds
        purge = self.config.purge_bars
        embargo = self.config.embargo_bars
        
        # Calculate fold size (total bars / n_folds, accounting for gaps)
        total_gap_per_fold = purge + embargo
        usable_bars = n_bars - (n_folds * total_gap_per_fold)
        
        if usable_bars < self.config.min_fold_bars * n_folds:
            raise ValueError(
                f"Insufficient data for {n_folds} folds with "
                f"purge={purge}, embargo={embargo}. "
                f"Need at least {self.config.min_fold_bars * n_folds + n_folds * total_gap_per_fold} bars."
            )
        
        fold_size = usable_bars // n_folds
        train_size = int(fold_size * self.config.train_ratio)
        test_size = fold_size - train_size
        
        folds = []
        
        for i in range(n_folds):
            # Calculate fold boundaries
            fold_start = i * (fold_size + total_gap_per_fold)
            train_start = fold_start
            train_end = train_start + train_size
            
            # Apply purge gap
            test_start = train_end + purge
            test_end = test_start + test_size
            
            # Clip to valid range
            if test_end > n_bars:
                test_end = n_bars
            if test_start >= n_bars:
                break
            
            train_idx = df.index[train_start:train_end]
            test_idx = df.index[test_start:test_end]
            
            folds.append((train_idx, test_idx))
            
            logger.debug(
                f"Fold {i}: train [{train_start}:{train_end}], "
                f"test [{test_start}:{test_end}]"
            )
        
        return folds
    
    def run(
        self,
        df: pd.DataFrame,
        objective_fn: Callable[[pd.DataFrame, Dict], Dict[str, float]],
        param_grid: Dict[str, List[Any]],
        optimization_metric: str = "sharpe_ratio",
        time_column: str = "time"
    ) -> WalkForwardResult:
        """
        Run purged walk-forward optimization.
        
        Args:
            df: Full DataFrame with OHLCV data
            objective_fn: Function that takes (df, params) and returns metrics dict
            param_grid: Parameter grid to search
            optimization_metric: Metric to optimize for
            time_column: Name of time column
            
        Returns:
            WalkForwardResult with fold-by-fold and aggregate results
        """
        folds = self.generate_folds(df, time_column)
        fold_results: List[FoldResult] = []
        param_history: List[Dict] = []
        
        logger.info(f"Starting walk-forward with {len(folds)} folds")
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"Processing fold {fold_idx + 1}/{len(folds)}")
            
            # Get train/test data
            train_df = df.loc[train_idx]
            test_df = df.loc[test_idx]
            
            # Optimize on training data
            best_params, is_metrics = self._optimize_fold(
                train_df, objective_fn, param_grid, optimization_metric
            )
            param_history.append(best_params)
            
            # Validate on test data
            oos_metrics = objective_fn(test_df, best_params)
            
            # Get timestamps
            train_start = train_df.index[0] if isinstance(train_df.index[0], datetime) else train_df[time_column].iloc[0]
            train_end = train_df.index[-1] if isinstance(train_df.index[-1], datetime) else train_df[time_column].iloc[-1]
            test_start = test_df.index[0] if isinstance(test_df.index[0], datetime) else test_df[time_column].iloc[0]
            test_end = test_df.index[-1] if isinstance(test_df.index[-1], datetime) else test_df[time_column].iloc[-1]
            
            # Create fold result
            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimal_params=best_params,
                in_sample_metrics=is_metrics,
                out_of_sample_metrics=oos_metrics,
                in_sample_trades=int(is_metrics.get("total_trades", 0)),
                out_of_sample_trades=int(oos_metrics.get("total_trades", 0)),
            )
            
            fold_results.append(fold_result)
            
            # Log to tracker if available
            if self.tracker:
                self.tracker.log_fold_result(
                    fold_idx=fold_idx,
                    train_start=str(train_start),
                    train_end=str(train_end),
                    test_start=str(test_start),
                    test_end=str(test_end),
                    optimal_params=best_params,
                    in_sample_metrics=is_metrics,
                    out_of_sample_metrics=oos_metrics,
                )
            
            logger.info(
                f"Fold {fold_idx}: IS Sharpe={is_metrics.get('sharpe_ratio', 0):.3f}, "
                f"OOS Sharpe={oos_metrics.get('sharpe_ratio', 0):.3f}"
            )
        
        # Calculate aggregate results
        result = self._aggregate_results(fold_results, param_history)
        result.config = self.config
        
        return result
    
    def _optimize_fold(
        self,
        train_df: pd.DataFrame,
        objective_fn: Callable,
        param_grid: Dict[str, List],
        optimization_metric: str
    ) -> Tuple[Dict, Dict[str, float]]:
        """
        Find optimal parameters for a single fold.
        
        Args:
            train_df: Training data
            objective_fn: Objective function
            param_grid: Parameter grid
            optimization_metric: Metric to optimize
            
        Returns:
            (best_params, best_metrics)
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_metric = -np.inf
        best_params = {k: v[0] for k, v in param_grid.items()}
        best_metrics = {}
        
        # Grid search
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            
            try:
                metrics = objective_fn(train_df, params)
                metric_value = metrics.get(optimization_metric, -np.inf)
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_metrics = metrics
                    
            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue
        
        return best_params, best_metrics
    
    def _aggregate_results(
        self,
        fold_results: List[FoldResult],
        param_history: List[Dict]
    ) -> WalkForwardResult:
        """
        Aggregate results across all folds.
        
        Args:
            fold_results: List of fold results
            param_history: History of optimal parameters per fold
            
        Returns:
            Aggregated WalkForwardResult
        """
        if not fold_results:
            return WalkForwardResult(fold_results=[])
        
        # Extract OOS metrics
        oos_sharpes = [fr.out_of_sample_metrics.get("sharpe_ratio", 0) for fr in fold_results]
        oos_returns = [fr.out_of_sample_metrics.get("total_return_pct", 0) for fr in fold_results]
        oos_drawdowns = [fr.out_of_sample_metrics.get("max_drawdown_pct", 0) for fr in fold_results]
        oos_win_rates = [fr.out_of_sample_metrics.get("win_rate", 0) for fr in fold_results]
        oos_profit_factors = [fr.out_of_sample_metrics.get("profit_factor", 0) for fr in fold_results]
        oos_trades = [fr.out_of_sample_trades for fr in fold_results]
        
        # Extract IS metrics for ratio calculation
        is_sharpes = [fr.in_sample_metrics.get("sharpe_ratio", 0) for fr in fold_results]
        
        # Calculate parameter stability
        param_stability = self._calculate_param_stability(param_history)
        
        # Calculate IS/OOS ratio (measure of overfitting)
        is_to_oos_ratios = []
        for is_s, oos_s in zip(is_sharpes, oos_sharpes):
            if is_s > 0 and oos_s > 0:
                is_to_oos_ratios.append(oos_s / is_s)
        
        return WalkForwardResult(
            fold_results=fold_results,
            aggregate_sharpe=float(np.mean(oos_sharpes)),
            aggregate_return=float(np.sum(oos_returns)),  # Cumulative
            aggregate_max_dd=float(np.max(oos_drawdowns)),  # Worst case
            aggregate_win_rate=float(np.mean(oos_win_rates)),
            aggregate_profit_factor=float(np.mean(oos_profit_factors)),
            total_oos_trades=sum(oos_trades),
            param_stability_score=param_stability,
            sharpe_stability=float(np.std(oos_sharpes)),
            is_to_oos_ratio=float(np.mean(is_to_oos_ratios)) if is_to_oos_ratios else 0.0,
        )
    
    def _calculate_param_stability(self, param_history: List[Dict]) -> float:
        """
        Calculate parameter stability score (0-1).
        
        Higher score = more stable parameters across folds.
        
        Args:
            param_history: List of optimal params per fold
            
        Returns:
            Stability score (0-1)
        """
        if len(param_history) < 2:
            return 1.0
        
        # For each parameter, calculate coefficient of variation
        param_names = set()
        for ph in param_history:
            param_names.update(ph.keys())
        
        stability_scores = []
        
        for param in param_names:
            values = [ph.get(param) for ph in param_history if param in ph]
            
            # Skip non-numeric parameters
            try:
                values = [float(v) for v in values]
            except (TypeError, ValueError):
                continue
            
            if len(values) < 2:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)  # Coefficient of variation
                # Convert to stability score (lower CV = higher stability)
                stability = max(0, 1 - cv)
                stability_scores.append(stability)
        
        if not stability_scores:
            return 1.0
        
        return float(np.mean(stability_scores))
    
    def generate_report(self, result: WalkForwardResult) -> str:
        """
        Generate text report of walk-forward results.
        
        Args:
            result: WalkForwardResult to report
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "PURGED WALK-FORWARD OPTIMIZATION REPORT",
            "=" * 60,
            "",
            f"Configuration:",
            f"  - Folds: {result.config.n_folds}",
            f"  - Purge bars: {result.config.purge_bars}",
            f"  - Embargo bars: {result.config.embargo_bars}",
            f"  - Train ratio: {result.config.train_ratio:.0%}",
            "",
            "Aggregate OOS Performance:",
            f"  - Sharpe Ratio: {result.aggregate_sharpe:.3f}",
            f"  - Total Return: {result.aggregate_return:.2f}%",
            f"  - Max Drawdown: {result.aggregate_max_dd:.2f}%",
            f"  - Win Rate: {result.aggregate_win_rate:.1%}",
            f"  - Profit Factor: {result.aggregate_profit_factor:.2f}",
            f"  - Total Trades: {result.total_oos_trades}",
            "",
            "Stability Analysis:",
            f"  - Parameter Stability: {result.param_stability_score:.3f}",
            f"  - Sharpe Std (across folds): {result.sharpe_stability:.3f}",
            f"  - IS/OOS Ratio: {result.is_to_oos_ratio:.3f}",
            "",
            "Fold-by-Fold Results:",
            "-" * 60,
        ]
        
        for fr in result.fold_results:
            lines.append(
                f"  Fold {fr.fold_idx}: "
                f"IS={fr.in_sample_metrics.get('sharpe_ratio', 0):.3f}, "
                f"OOS={fr.out_of_sample_metrics.get('sharpe_ratio', 0):.3f}, "
                f"Trades={fr.out_of_sample_trades}"
            )
        
        lines.extend([
            "",
            "=" * 60,
            "Interpretation:",
            f"  - IS/OOS < 1.0 indicates potential underfit",
            f"  - IS/OOS > 1.5 indicates potential overfit",
            f"  - Current: {result.is_to_oos_ratio:.2f}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def create_walk_forward_engine(
    n_folds: int = 10,
    purge_bars: int = 5,
    embargo_bars: int = 5,
    tracker: Optional[ExperimentTracker] = None
) -> PurgedWalkForwardEngine:
    """
    Factory function for PurgedWalkForwardEngine.
    
    Args:
        n_folds: Number of folds
        purge_bars: Purge period in bars
        embargo_bars: Embargo period in bars
        tracker: Optional experiment tracker
        
    Returns:
        PurgedWalkForwardEngine instance
    """
    config = WalkForwardConfig(
        n_folds=n_folds,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    return PurgedWalkForwardEngine(config=config, tracker=tracker)
