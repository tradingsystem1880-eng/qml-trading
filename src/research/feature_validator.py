"""
Feature Validator for QML Trading System
=========================================
Rigorous validation pipeline for new features.

Based on DeepSeek analysis: With PF 4.49, adding features is more likely to hurt than help.
Every feature must pass ALL validation steps or be REJECTED.

Validation Pipeline:
1. Apply filter → check sample size (min 100 trades)
2. Calculate metrics on filtered subset
3. Walk-forward validation (5 folds)
4. Permutation test (1000 iterations)

Auto-fail criteria:
- Trade reduction > 30%
- Walk-forward inconsistent (any fold PF < 1.0)
- p-value > 0.05
- PF degradation vs baseline

Usage:
    from src.research.feature_validator import FeatureValidator

    validator = FeatureValidator(baseline_metrics={'profit_factor': 4.49, 'win_rate': 0.55})
    result = validator.validate_filter(
        feature_name='funding_rate_filter',
        trades=all_trades,
        filter_func=lambda t: t.funding_rate < 0.0001,
    )
    print(result.verdict)  # 'PASS' or 'FAIL'
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationStepResult:
    """Result of a single validation step."""
    step_name: str
    passed: bool
    metric_value: Any
    threshold: Any
    details: str


@dataclass
class FeatureValidationResult:
    """Complete validation result for a feature."""
    feature_name: str
    timestamp: str
    verdict: str  # 'PASS', 'FAIL'
    fail_reason: Optional[str]

    # Step results
    sample_size_check: ValidationStepResult
    baseline_comparison: ValidationStepResult
    walk_forward_check: ValidationStepResult
    permutation_check: ValidationStepResult

    # Summary metrics
    original_trades: int
    filtered_trades: int
    trade_reduction_pct: float
    baseline_pf: float
    filtered_pf: float
    pf_change_pct: float
    p_value: float

    # Detailed results
    fold_results: List[Dict]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'feature_name': self.feature_name,
            'timestamp': self.timestamp,
            'verdict': self.verdict,
            'fail_reason': self.fail_reason,
            'original_trades': self.original_trades,
            'filtered_trades': self.filtered_trades,
            'trade_reduction_pct': self.trade_reduction_pct,
            'baseline_pf': self.baseline_pf,
            'filtered_pf': self.filtered_pf,
            'pf_change_pct': self.pf_change_pct,
            'p_value': self.p_value,
            'steps': {
                'sample_size': {
                    'passed': self.sample_size_check.passed,
                    'value': self.sample_size_check.metric_value,
                    'threshold': self.sample_size_check.threshold,
                },
                'baseline_comparison': {
                    'passed': self.baseline_comparison.passed,
                    'value': self.baseline_comparison.metric_value,
                    'threshold': self.baseline_comparison.threshold,
                },
                'walk_forward': {
                    'passed': self.walk_forward_check.passed,
                    'details': self.walk_forward_check.details,
                },
                'permutation': {
                    'passed': self.permutation_check.passed,
                    'p_value': self.permutation_check.metric_value,
                    'threshold': self.permutation_check.threshold,
                },
            },
            'fold_results': self.fold_results,
        }


@dataclass
class FeatureValidatorConfig:
    """Configuration for feature validation."""
    min_trades: int = 100
    max_trade_reduction_pct: float = 0.30
    min_pf_ratio: float = 1.0  # Filtered PF must be >= baseline * ratio
    n_folds: int = 5
    min_fold_pf: float = 1.0  # Each fold must have PF >= this
    n_permutations: int = 1000
    significance_threshold: float = 0.05
    random_seed: int = 42


class FeatureValidator:
    """
    Rigorous validation pipeline for new features.

    Every feature must pass ALL validation steps:
    1. Sample size (min 100 trades after filtering)
    2. Baseline comparison (no PF degradation)
    3. Walk-forward consistency (all 5 folds PF > 1.0)
    4. Permutation test (p < 0.05)
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        config: Optional[FeatureValidatorConfig] = None,
    ):
        """
        Initialize feature validator.

        Args:
            baseline_metrics: Dict with 'profit_factor', 'win_rate', 'expectancy'
            config: Validation configuration
        """
        self.baseline = baseline_metrics
        self.config = config or FeatureValidatorConfig()

    def validate_filter(
        self,
        feature_name: str,
        trades: List[Any],
        filter_func: Callable[[Any], bool],
        get_r_return: Optional[Callable[[Any], float]] = None,
        get_timestamp: Optional[Callable[[Any], datetime]] = None,
    ) -> FeatureValidationResult:
        """
        Run full validation pipeline on a filter.

        Args:
            feature_name: Name of feature being tested
            trades: List of trade objects
            filter_func: Function that returns True for trades to KEEP
            get_r_return: Function to extract R-multiple from trade (default: t.pnl_r)
            get_timestamp: Function to extract timestamp from trade (default: t.exit_time)

        Returns:
            FeatureValidationResult with verdict and details
        """
        cfg = self.config

        # Default extractors
        if get_r_return is None:
            get_r_return = lambda t: getattr(t, 'pnl_r', t.get('pnl_r', 0) if isinstance(t, dict) else 0)
        if get_timestamp is None:
            get_timestamp = lambda t: getattr(t, 'exit_time', t.get('exit_time') if isinstance(t, dict) else None)

        # Apply filter
        filtered_trades = [t for t in trades if filter_func(t)]

        original_count = len(trades)
        filtered_count = len(filtered_trades)
        reduction_pct = 1 - (filtered_count / original_count) if original_count > 0 else 1.0

        # Extract R returns
        original_returns = np.array([get_r_return(t) for t in trades])
        filtered_returns = np.array([get_r_return(t) for t in filtered_trades])

        # Calculate baseline PF
        baseline_pf = self._calculate_pf(original_returns)

        # Step 1: Sample size check
        sample_check = self._check_sample_size(filtered_count, reduction_pct)
        if not sample_check.passed:
            return self._create_fail_result(
                feature_name, "SAMPLE_SIZE", sample_check,
                original_count, filtered_count, reduction_pct, baseline_pf
            )

        # Step 2: Baseline comparison
        filtered_pf = self._calculate_pf(filtered_returns)
        baseline_check = self._check_baseline_comparison(filtered_pf, baseline_pf)
        if not baseline_check.passed:
            return self._create_fail_result(
                feature_name, "BASELINE_DEGRADATION", baseline_check,
                original_count, filtered_count, reduction_pct, baseline_pf, filtered_pf
            )

        # Step 3: Walk-forward validation
        wf_check, fold_results = self._check_walk_forward(
            filtered_trades, get_r_return, get_timestamp
        )
        if not wf_check.passed:
            return self._create_fail_result(
                feature_name, "WALK_FORWARD_INCONSISTENT", wf_check,
                original_count, filtered_count, reduction_pct, baseline_pf, filtered_pf,
                fold_results=fold_results
            )

        # Step 4: Permutation test
        perm_check = self._check_permutation(filtered_returns)
        if not perm_check.passed:
            return self._create_fail_result(
                feature_name, "NOT_SIGNIFICANT", perm_check,
                original_count, filtered_count, reduction_pct, baseline_pf, filtered_pf,
                fold_results=fold_results
            )

        # All checks passed!
        return FeatureValidationResult(
            feature_name=feature_name,
            timestamp=datetime.now().isoformat(),
            verdict="PASS",
            fail_reason=None,
            sample_size_check=sample_check,
            baseline_comparison=baseline_check,
            walk_forward_check=wf_check,
            permutation_check=perm_check,
            original_trades=original_count,
            filtered_trades=filtered_count,
            trade_reduction_pct=reduction_pct,
            baseline_pf=baseline_pf,
            filtered_pf=filtered_pf,
            pf_change_pct=(filtered_pf - baseline_pf) / baseline_pf if baseline_pf > 0 else 0,
            p_value=perm_check.metric_value,
            fold_results=fold_results,
        )

    def _calculate_pf(self, returns: np.ndarray) -> float:
        """Calculate profit factor from returns."""
        if len(returns) == 0:
            return 0.0
        winners = returns[returns > 0]
        losers = returns[returns <= 0]
        gross_profit = np.sum(winners) if len(winners) > 0 else 0
        gross_loss = abs(np.sum(losers)) if len(losers) > 0 else 0
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def _calculate_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate full metrics from returns."""
        if len(returns) == 0:
            return {'pf': 0, 'wr': 0, 'expectancy': 0, 'trades': 0}

        winners = returns[returns > 0]
        losers = returns[returns <= 0]

        pf = self._calculate_pf(returns)
        wr = len(winners) / len(returns)
        exp = np.mean(returns)

        return {'pf': pf, 'wr': wr, 'expectancy': exp, 'trades': len(returns)}

    def _check_sample_size(self, filtered_count: int, reduction_pct: float) -> ValidationStepResult:
        """Check sample size requirements."""
        cfg = self.config

        # Check minimum trades
        if filtered_count < cfg.min_trades:
            return ValidationStepResult(
                step_name="sample_size",
                passed=False,
                metric_value=filtered_count,
                threshold=cfg.min_trades,
                details=f"Only {filtered_count} trades after filter (min {cfg.min_trades})"
            )

        # Check reduction percentage
        if reduction_pct > cfg.max_trade_reduction_pct:
            return ValidationStepResult(
                step_name="sample_size",
                passed=False,
                metric_value=reduction_pct,
                threshold=cfg.max_trade_reduction_pct,
                details=f"Trade reduction {reduction_pct:.1%} exceeds {cfg.max_trade_reduction_pct:.1%} limit"
            )

        return ValidationStepResult(
            step_name="sample_size",
            passed=True,
            metric_value=filtered_count,
            threshold=cfg.min_trades,
            details=f"{filtered_count} trades, {reduction_pct:.1%} reduction - OK"
        )

    def _check_baseline_comparison(self, filtered_pf: float, baseline_pf: float) -> ValidationStepResult:
        """Check if filtered PF is not degraded vs baseline."""
        cfg = self.config

        # Filtered PF must be at least baseline * min_ratio
        min_required = baseline_pf * cfg.min_pf_ratio

        if filtered_pf < min_required:
            return ValidationStepResult(
                step_name="baseline_comparison",
                passed=False,
                metric_value=filtered_pf,
                threshold=min_required,
                details=f"Filtered PF {filtered_pf:.2f} < required {min_required:.2f}"
            )

        pf_change = (filtered_pf - baseline_pf) / baseline_pf if baseline_pf > 0 else 0
        return ValidationStepResult(
            step_name="baseline_comparison",
            passed=True,
            metric_value=filtered_pf,
            threshold=min_required,
            details=f"Filtered PF {filtered_pf:.2f} ({pf_change:+.1%} vs baseline) - OK"
        )

    def _check_walk_forward(
        self,
        trades: List[Any],
        get_r_return: Callable,
        get_timestamp: Callable,
    ) -> Tuple[ValidationStepResult, List[Dict]]:
        """Check walk-forward consistency across folds."""
        cfg = self.config

        # Sort trades by timestamp
        trades_with_time = []
        for t in trades:
            ts = get_timestamp(t)
            if ts is not None:
                trades_with_time.append((ts, t))

        # If no timestamps, fall back to order
        if len(trades_with_time) < len(trades) * 0.5:
            # Use index-based splitting
            fold_size = len(trades) // cfg.n_folds
            fold_results = []

            for i in range(cfg.n_folds):
                start = i * fold_size
                end = start + fold_size if i < cfg.n_folds - 1 else len(trades)
                fold_trades = trades[start:end]

                returns = np.array([get_r_return(t) for t in fold_trades])
                metrics = self._calculate_metrics(returns)
                fold_results.append({
                    'fold': i + 1,
                    'trades': len(fold_trades),
                    'pf': metrics['pf'],
                    'wr': metrics['wr'],
                })
        else:
            # Time-based splitting
            trades_with_time.sort(key=lambda x: x[0])
            sorted_trades = [t for _, t in trades_with_time]

            fold_size = len(sorted_trades) // cfg.n_folds
            fold_results = []

            for i in range(cfg.n_folds):
                start = i * fold_size
                end = start + fold_size if i < cfg.n_folds - 1 else len(sorted_trades)
                fold_trades = sorted_trades[start:end]

                returns = np.array([get_r_return(t) for t in fold_trades])
                metrics = self._calculate_metrics(returns)
                fold_results.append({
                    'fold': i + 1,
                    'trades': len(fold_trades),
                    'pf': metrics['pf'],
                    'wr': metrics['wr'],
                })

        # Check if all folds have PF >= min
        failing_folds = [f for f in fold_results if f['pf'] < cfg.min_fold_pf]

        if failing_folds:
            return ValidationStepResult(
                step_name="walk_forward",
                passed=False,
                metric_value=len(failing_folds),
                threshold=0,
                details=f"{len(failing_folds)} fold(s) have PF < {cfg.min_fold_pf}"
            ), fold_results

        return ValidationStepResult(
            step_name="walk_forward",
            passed=True,
            metric_value=0,
            threshold=0,
            details=f"All {cfg.n_folds} folds have PF >= {cfg.min_fold_pf}"
        ), fold_results

    def _check_permutation(self, returns: np.ndarray) -> ValidationStepResult:
        """Run permutation test for statistical significance."""
        cfg = self.config
        np.random.seed(cfg.random_seed)

        real_pf = self._calculate_pf(returns)

        # Permute by randomizing signs
        abs_returns = np.abs(returns)
        permuted_pfs = []

        for _ in range(cfg.n_permutations):
            random_signs = np.random.choice([-1, 1], size=len(returns))
            randomized = abs_returns * random_signs
            perm_pf = self._calculate_pf(randomized)
            permuted_pfs.append(perm_pf)

        permuted_pfs = np.array(permuted_pfs)

        # P-value: proportion of permuted PFs >= real PF
        p_value = np.mean(permuted_pfs >= real_pf)

        if p_value > cfg.significance_threshold:
            return ValidationStepResult(
                step_name="permutation",
                passed=False,
                metric_value=p_value,
                threshold=cfg.significance_threshold,
                details=f"p-value {p_value:.4f} > {cfg.significance_threshold} - not significant"
            )

        return ValidationStepResult(
            step_name="permutation",
            passed=True,
            metric_value=p_value,
            threshold=cfg.significance_threshold,
            details=f"p-value {p_value:.4f} < {cfg.significance_threshold} - significant"
        )

    def _create_fail_result(
        self,
        feature_name: str,
        fail_reason: str,
        failed_check: ValidationStepResult,
        original_count: int,
        filtered_count: int,
        reduction_pct: float,
        baseline_pf: float,
        filtered_pf: float = 0,
        fold_results: List[Dict] = None,
    ) -> FeatureValidationResult:
        """Create a FAIL result with appropriate defaults."""
        # Create dummy passing checks for steps that weren't reached
        dummy_pass = ValidationStepResult(
            step_name="not_reached",
            passed=True,
            metric_value=None,
            threshold=None,
            details="Not evaluated - failed earlier step"
        )

        return FeatureValidationResult(
            feature_name=feature_name,
            timestamp=datetime.now().isoformat(),
            verdict="FAIL",
            fail_reason=fail_reason,
            sample_size_check=failed_check if "SAMPLE" in fail_reason else dummy_pass,
            baseline_comparison=failed_check if "BASELINE" in fail_reason else dummy_pass,
            walk_forward_check=failed_check if "WALK" in fail_reason else dummy_pass,
            permutation_check=failed_check if "SIGNIFICANT" in fail_reason else dummy_pass,
            original_trades=original_count,
            filtered_trades=filtered_count,
            trade_reduction_pct=reduction_pct,
            baseline_pf=baseline_pf,
            filtered_pf=filtered_pf,
            pf_change_pct=(filtered_pf - baseline_pf) / baseline_pf if baseline_pf > 0 and filtered_pf > 0 else 0,
            p_value=failed_check.metric_value if "SIGNIFICANT" in fail_reason else 1.0,
            fold_results=fold_results or [],
        )

    # =========================================================================
    # Additional Validation Methods (Phase 9.7 DeepSeek Enhancements)
    # =========================================================================

    def permutation_significance_test(
        self,
        all_trades: pd.DataFrame,
        kept_mask: pd.Series,
        n_permutations: int = 10000,
        random_seed: int = None
    ) -> Dict:
        """
        Permutation test for filter effectiveness.

        Tests null hypothesis: "The filter doesn't select better trades than random"

        This is the CORRECT test for evaluating a trade filter. It shuffles which
        trades get "filtered" and measures if our actual filter outperforms random
        filtering of the same number of trades.

        Args:
            all_trades: DataFrame with ALL trades (before filtering), must have 'pnl_r'
            kept_mask: Boolean Series - True for trades that PASSED the filter
            n_permutations: Number of random permutations
            random_seed: For reproducibility (defaults to config value)

        Returns:
            Dict with p_value, actual_improvement, null_distribution stats
        """
        if random_seed is None:
            random_seed = self.config.random_seed

        np.random.seed(random_seed)

        def calc_pf(trades_df):
            if len(trades_df) == 0:
                return 0.0
            wins = trades_df[trades_df['pnl_r'] > 0]['pnl_r'].sum()
            losses = abs(trades_df[trades_df['pnl_r'] < 0]['pnl_r'].sum())
            if losses == 0:
                return 100.0  # Cap at 100 to avoid inf
            return wins / losses

        # Actual results
        baseline_pf = calc_pf(all_trades)
        kept_trades = all_trades[kept_mask]
        filtered_pf = calc_pf(kept_trades)
        actual_improvement = filtered_pf - baseline_pf

        n_total = len(all_trades)
        n_kept = kept_mask.sum()

        # Null distribution: randomly filter out same number of trades
        null_improvements = []

        for _ in range(n_permutations):
            # Randomly select which trades to "keep" (same count as actual filter)
            random_kept_idx = np.random.choice(n_total, size=n_kept, replace=False)
            random_kept = all_trades.iloc[random_kept_idx]
            random_pf = calc_pf(random_kept)
            null_improvements.append(random_pf - baseline_pf)

        null_improvements = np.array(null_improvements)

        # P-value: proportion of random filters that do as well or better
        # One-tailed test (we only care if our filter is BETTER)
        p_value = np.mean(null_improvements >= actual_improvement)

        # Handle edge case where actual improvement is best
        if p_value == 0:
            p_value = 1 / (n_permutations + 1)  # Conservative upper bound

        return {
            'test_type': 'permutation',
            'n_permutations': n_permutations,
            'baseline_pf': round(baseline_pf, 3),
            'filtered_pf': round(filtered_pf, 3),
            'actual_improvement': round(actual_improvement, 3),
            'p_value': round(p_value, 4),
            'null_mean': round(np.mean(null_improvements), 3),
            'null_std': round(np.std(null_improvements), 3),
            'null_95th_percentile': round(np.percentile(null_improvements, 95), 3),
            'significant_at_95': p_value < 0.05,
            'significant_at_99': p_value < 0.01,
            'n_total_trades': n_total,
            'n_kept_trades': int(n_kept),
            'n_filtered_out': n_total - int(n_kept)
        }

    def _bootstrap_significance_test_deprecated(
        self,
        baseline_trades: pd.DataFrame,
        filtered_trades: pd.DataFrame,
        n_bootstrap: int = 10000
    ) -> Dict:
        """
        DEPRECATED: Use permutation_significance_test instead.

        This test is methodologically flawed - it compares two different populations
        (baseline resamples vs filtered resamples), which tests "are these distributions
        different?" not "does our filter select better trades than random?"

        Kept for backwards compatibility only.
        """
        import warnings
        warnings.warn(
            "bootstrap_significance_test is deprecated. Use permutation_significance_test instead.",
            DeprecationWarning,
            stacklevel=2
        )
        def calc_pf(trades_df):
            wins = trades_df[trades_df['pnl_r'] > 0]['pnl_r'].sum()
            losses = abs(trades_df[trades_df['pnl_r'] < 0]['pnl_r'].sum())
            return wins / losses if losses > 0 else float('inf')

        n_filtered = len(filtered_trades)
        baseline_pfs = []
        filtered_pfs = []

        np.random.seed(self.config.random_seed)

        for _ in range(n_bootstrap):
            # Resample with replacement, same size as filtered
            b_sample = baseline_trades.sample(n=n_filtered, replace=True)
            f_sample = filtered_trades.sample(n=n_filtered, replace=True)

            baseline_pfs.append(calc_pf(b_sample))
            filtered_pfs.append(calc_pf(f_sample))

        # Handle inf values
        baseline_pfs = np.array([p if not np.isinf(p) else 100.0 for p in baseline_pfs])
        filtered_pfs = np.array([p if not np.isinf(p) else 100.0 for p in filtered_pfs])

        baseline_median = np.median(baseline_pfs)
        filtered_ci_lower = np.percentile(filtered_pfs, 2.5)
        filtered_ci_upper = np.percentile(filtered_pfs, 97.5)

        # Significant if filtered 95% CI entirely above baseline median
        significant = filtered_ci_lower > baseline_median
        p_value = np.mean(filtered_pfs <= baseline_median)

        return {
            'significant': significant,
            'p_value': round(p_value, 4),
            'baseline_median_pf': round(baseline_median, 3),
            'filtered_ci': (round(filtered_ci_lower, 3), round(filtered_ci_upper, 3)),
            'improvement_significant': significant and p_value < 0.05
        }

    # Backwards compatibility alias
    bootstrap_significance_test = _bootstrap_significance_test_deprecated

    def check_feature_correlation(
        self,
        trades_df: pd.DataFrame,
        new_feature_col: str,
        existing_features: List[str] = None
    ) -> Dict:
        """
        Check if new feature is correlated with existing features.
        High correlation (>0.5) means new feature may be redundant.

        Args:
            trades_df: DataFrame with feature columns
            new_feature_col: Column name of new feature
            existing_features: List of existing feature column names

        Returns:
            Dict with correlations and redundancy warnings
        """
        if existing_features is None:
            existing_features = ['adx', 'regime_score']

        correlations = {}
        for feat in existing_features:
            if feat in trades_df.columns and new_feature_col in trades_df.columns:
                valid_mask = trades_df[feat].notna() & trades_df[new_feature_col].notna()
                if valid_mask.sum() > 30:
                    corr, p_val = stats.spearmanr(
                        trades_df.loc[valid_mask, feat],
                        trades_df.loc[valid_mask, new_feature_col]
                    )
                    correlations[feat] = {
                        'correlation': round(corr, 3),
                        'p_value': round(p_val, 4),
                        'is_redundant': abs(corr) > 0.5
                    }

        any_redundant = any(c['is_redundant'] for c in correlations.values()) if correlations else False
        return {
            'correlations': correlations,
            'any_redundant': any_redundant,
            'warning': 'Feature may be redundant with existing features' if any_redundant else None
        }

    def check_directional_balance(
        self,
        baseline_trades: pd.DataFrame,
        filtered_trades: pd.DataFrame
    ) -> Dict:
        """
        Check if filter creates directional bias (too many LONGs or SHORTs).
        Severe imbalance (>70% one direction) is a red flag.

        Args:
            baseline_trades: DataFrame with 'direction' column
            filtered_trades: DataFrame with 'direction' column

        Returns:
            Dict with balance metrics and warnings
        """
        def calc_balance(df):
            if 'direction' not in df.columns:
                return None
            total = len(df)
            longs = (df['direction'] == 'LONG').sum()
            return longs / total if total > 0 else 0.5

        baseline_balance = calc_balance(baseline_trades)
        filtered_balance = calc_balance(filtered_trades)

        if baseline_balance is None or filtered_balance is None:
            return {'error': 'direction column not found'}

        balance_shift = abs(filtered_balance - baseline_balance)
        severe_imbalance = filtered_balance < 0.3 or filtered_balance > 0.7

        return {
            'baseline_long_pct': round(baseline_balance * 100, 1),
            'filtered_long_pct': round(filtered_balance * 100, 1),
            'balance_shift_pct': round(balance_shift * 100, 1),
            'severe_imbalance': severe_imbalance,
            'warning': 'Filter creates severe directional bias' if severe_imbalance else None
        }

    def compare_max_drawdown(
        self,
        baseline_trades: pd.DataFrame,
        filtered_trades: pd.DataFrame
    ) -> Dict:
        """
        Compare max drawdown between baseline and filtered.
        Filter may primarily reduce DD even if PF unchanged.

        Args:
            baseline_trades: DataFrame with 'pnl_r' column
            filtered_trades: DataFrame with 'pnl_r' column

        Returns:
            Dict with drawdown comparison metrics
        """
        def calc_max_dd(trades_df):
            if 'pnl_r' not in trades_df.columns:
                return None
            equity = trades_df['pnl_r'].cumsum()
            running_max = equity.cummax()
            drawdown = running_max - equity
            return drawdown.max()

        baseline_dd = calc_max_dd(baseline_trades)
        filtered_dd = calc_max_dd(filtered_trades)

        if baseline_dd is None or filtered_dd is None:
            return {'error': 'pnl_r column not found'}

        dd_reduction = baseline_dd - filtered_dd
        dd_reduction_pct = (dd_reduction / baseline_dd * 100) if baseline_dd > 0 else 0

        return {
            'baseline_max_dd_r': round(baseline_dd, 2),
            'filtered_max_dd_r': round(filtered_dd, 2),
            'dd_reduction_r': round(dd_reduction, 2),
            'dd_reduction_pct': round(dd_reduction_pct, 1),
            'significant_improvement': dd_reduction_pct > 15  # >15% is meaningful
        }

    def generate_report(self, result: FeatureValidationResult) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"FEATURE VALIDATION REPORT: {result.feature_name}")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append("")

        # Verdict
        if result.verdict == "PASS":
            lines.append("VERDICT: ✅ PASS - Feature approved for deployment")
        else:
            lines.append(f"VERDICT: ❌ FAIL - {result.fail_reason}")
        lines.append("")

        # Summary
        lines.append("-" * 70)
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"Original Trades:    {result.original_trades}")
        lines.append(f"Filtered Trades:    {result.filtered_trades}")
        lines.append(f"Trade Reduction:    {result.trade_reduction_pct:.1%}")
        lines.append(f"Baseline PF:        {result.baseline_pf:.2f}")
        lines.append(f"Filtered PF:        {result.filtered_pf:.2f}")
        lines.append(f"PF Change:          {result.pf_change_pct:+.1%}")
        lines.append(f"P-value:            {result.p_value:.4f}")
        lines.append("")

        # Validation Steps
        lines.append("-" * 70)
        lines.append("VALIDATION STEPS")
        lines.append("-" * 70)

        steps = [
            ("1. Sample Size", result.sample_size_check),
            ("2. Baseline Comparison", result.baseline_comparison),
            ("3. Walk-Forward", result.walk_forward_check),
            ("4. Permutation Test", result.permutation_check),
        ]

        for name, step in steps:
            status = "✅" if step.passed else "❌"
            lines.append(f"{status} {name}: {step.details}")

        # Fold Results
        if result.fold_results:
            lines.append("")
            lines.append("-" * 70)
            lines.append("WALK-FORWARD FOLD RESULTS")
            lines.append("-" * 70)
            lines.append(f"{'Fold':<8} {'Trades':<10} {'PF':<10} {'WR':<10}")
            for fold in result.fold_results:
                lines.append(f"{fold['fold']:<8} {fold['trades']:<10} {fold['pf']:<10.2f} {fold['wr']:<10.1%}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
