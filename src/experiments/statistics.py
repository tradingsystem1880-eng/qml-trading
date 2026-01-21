"""
Statistical Analysis for A/B Testing - Phase 6
===============================================
Multiple testing correction and significance analysis.

Features:
- Benjamini-Hochberg FDR correction
- Significance analysis for experiment results
- Ranking with statistical filtering

CRITICAL: With 210K parameter combinations, false discovery rate correction
is essential. Without BH correction, at alpha=0.05, we'd expect ~10,500
false positives purely by chance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SignificanceResult:
    """
    Result of significance testing for a single experiment.
    """
    experiment_id: str
    raw_p_value: float
    adjusted_p_value: float
    is_significant: bool
    rank: int


def benjamini_hochberg_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[int, float, bool]]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.

    The BH procedure controls the False Discovery Rate (FDR) - the expected
    proportion of false positives among all significant results.

    Algorithm:
    1. Sort p-values in ascending order
    2. For each p-value at rank i (1-indexed), calculate threshold: (i/n) * alpha
    3. Find the largest rank k where p-value <= threshold
    4. All ranks 1..k are significant

    Args:
        p_values: List of raw p-values from experiments
        alpha: Significance level (default 0.05 = 5% FDR)

    Returns:
        List of (original_index, adjusted_p_value, is_significant)
        sorted by original index

    Example:
        >>> p_values = [0.01, 0.04, 0.03, 0.005, 0.50]
        >>> results = benjamini_hochberg_correction(p_values, alpha=0.05)
        >>> significant = [r for r in results if r[2]]
    """
    n = len(p_values)
    if n == 0:
        return []

    # Create indexed p-values and sort by p-value
    indexed = [(i, p) for i, p in enumerate(p_values)]
    sorted_by_p = sorted(indexed, key=lambda x: x[1])

    # Calculate BH threshold for each rank
    # Threshold at rank i (1-indexed) = (i / n) * alpha
    results = []
    max_significant_rank = -1

    for rank, (original_idx, p_value) in enumerate(sorted_by_p, start=1):
        threshold = (rank / n) * alpha
        if p_value <= threshold:
            max_significant_rank = rank

    # Calculate adjusted p-values using BH formula
    # Adjusted p-value = min(p * n / rank, 1.0), taking cumulative min from bottom
    adjusted_p_values = []
    for rank, (original_idx, p_value) in enumerate(sorted_by_p, start=1):
        adjusted = min(p_value * n / rank, 1.0)
        adjusted_p_values.append((original_idx, adjusted, rank))

    # Apply cumulative minimum from the end (larger p-values to smaller)
    # This ensures monotonicity: if p_i < p_j, then adjusted_i <= adjusted_j
    cumulative_min = 1.0
    final_results = []
    for i in range(len(adjusted_p_values) - 1, -1, -1):
        original_idx, adjusted, rank = adjusted_p_values[i]
        cumulative_min = min(cumulative_min, adjusted)
        is_significant = rank <= max_significant_rank
        final_results.append((original_idx, cumulative_min, is_significant))

    # Reverse to get ascending rank order, then sort by original index
    final_results.reverse()
    final_results.sort(key=lambda x: x[0])

    return final_results


def analyze_experiment_significance(
    experiments: List[Dict[str, Any]],
    alpha: float = 0.05,
    min_trades: int = 30,
    p_value_key: str = 'p_value',
    id_key: str = 'param_hash',
) -> List[SignificanceResult]:
    """
    Analyze statistical significance across experiments with BH correction.

    Args:
        experiments: List of experiment dicts with p_values
        alpha: Significance level for FDR
        min_trades: Minimum trades required for valid analysis
        p_value_key: Key for p-value in experiment dict
        id_key: Key for experiment identifier

    Returns:
        List of SignificanceResult sorted by adjusted p-value
    """
    # Filter experiments with valid p-values and sufficient trades
    valid_experiments = []
    for exp in experiments:
        p_value = exp.get(p_value_key)
        trades = exp.get('total_trades', 0)

        if p_value is not None and trades >= min_trades:
            valid_experiments.append(exp)

    if not valid_experiments:
        return []

    # Extract p-values
    p_values = [exp[p_value_key] for exp in valid_experiments]

    # Apply BH correction
    corrected = benjamini_hochberg_correction(p_values, alpha)

    # Build results
    results = []
    for i, (original_idx, adjusted_p, is_significant) in enumerate(corrected):
        exp = valid_experiments[original_idx]
        result = SignificanceResult(
            experiment_id=str(exp.get(id_key, f'exp_{i}')),
            raw_p_value=exp[p_value_key],
            adjusted_p_value=adjusted_p,
            is_significant=is_significant,
            rank=i + 1,
        )
        results.append(result)

    # Sort by adjusted p-value (most significant first)
    results.sort(key=lambda r: r.adjusted_p_value)

    return results


def rank_experiments(
    experiments: List[Dict[str, Any]],
    primary_metric: str = 'sharpe_ratio',
    min_trades: int = 30,
    require_significant: bool = False,
    significance_results: Optional[List[SignificanceResult]] = None,
) -> List[Dict[str, Any]]:
    """
    Rank experiments by metric with optional significance filtering.

    Args:
        experiments: List of experiment dicts
        primary_metric: Metric to rank by (descending)
        min_trades: Minimum trades required
        require_significant: Only include significant experiments
        significance_results: Optional pre-computed significance results

    Returns:
        Sorted list of experiment dicts with 'rank' added
    """
    # Build significance lookup
    sig_lookup = {}
    if significance_results:
        sig_lookup = {r.experiment_id: r.is_significant for r in significance_results}

    # Filter experiments
    filtered = []
    for exp in experiments:
        trades = exp.get('total_trades', 0)
        if trades < min_trades:
            continue

        exp_id = str(exp.get('param_hash', exp.get('id', '')))

        # Check significance if required
        if require_significant:
            if exp_id not in sig_lookup or not sig_lookup[exp_id]:
                continue

        # Add significance info if available
        if exp_id in sig_lookup:
            exp['is_significant'] = sig_lookup[exp_id]

        filtered.append(exp)

    # Sort by primary metric (descending)
    sorted_exps = sorted(
        filtered,
        key=lambda e: e.get(primary_metric, 0) or 0,
        reverse=True
    )

    # Add ranks
    for i, exp in enumerate(sorted_exps, start=1):
        exp['rank'] = i

    return sorted_exps


def calculate_experiment_p_value(
    sharpe_ratio: float,
    total_trades: int,
    baseline_sharpe: float = 0.0,
) -> float:
    """
    Calculate approximate p-value for experiment Sharpe ratio.

    Uses the formula for standard error of Sharpe ratio:
    SE(SR) ≈ sqrt((1 + 0.5 * SR^2) / n)

    Then calculates z-score and p-value for one-sided test.

    Args:
        sharpe_ratio: Observed Sharpe ratio
        total_trades: Number of trades
        baseline_sharpe: Baseline Sharpe to test against (default 0)

    Returns:
        P-value for one-sided test (H1: SR > baseline)
    """
    if total_trades < 2:
        return 1.0

    # Standard error of Sharpe ratio (Lo, 2002)
    se = np.sqrt((1 + 0.5 * sharpe_ratio ** 2) / total_trades)

    if se == 0:
        return 0.0 if sharpe_ratio > baseline_sharpe else 1.0

    # Z-score
    z = (sharpe_ratio - baseline_sharpe) / se

    # One-sided p-value (testing if SR > baseline)
    from scipy.stats import norm
    p_value = 1 - norm.cdf(z)

    return p_value


def add_p_values_to_experiments(
    experiments: List[Dict[str, Any]],
    baseline_sharpe: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Add p-values to experiment results.

    Args:
        experiments: List of experiment dicts with sharpe_ratio and total_trades
        baseline_sharpe: Baseline Sharpe to test against

    Returns:
        Experiments with 'p_value' added
    """
    for exp in experiments:
        sharpe = exp.get('sharpe_ratio', 0) or 0
        trades = exp.get('total_trades', 0) or 0

        exp['p_value'] = calculate_experiment_p_value(
            sharpe_ratio=sharpe,
            total_trades=trades,
            baseline_sharpe=baseline_sharpe,
        )

    return experiments


def get_significant_discoveries(
    experiments: List[Dict[str, Any]],
    alpha: float = 0.05,
    min_trades: int = 30,
    baseline_sharpe: float = 0.0,
) -> Dict[str, Any]:
    """
    Complete pipeline: add p-values, apply BH correction, return discoveries.

    Args:
        experiments: List of experiment dicts
        alpha: FDR significance level
        min_trades: Minimum trades required
        baseline_sharpe: Baseline Sharpe for p-value calculation

    Returns:
        Dict with 'significant', 'not_significant', 'summary' keys
    """
    # Add p-values
    experiments_with_p = add_p_values_to_experiments(
        experiments.copy(),
        baseline_sharpe=baseline_sharpe
    )

    # Apply BH correction
    significance_results = analyze_experiment_significance(
        experiments_with_p,
        alpha=alpha,
        min_trades=min_trades,
    )

    # Create lookup
    sig_lookup = {r.experiment_id: r for r in significance_results}

    # Separate significant and non-significant
    significant = []
    not_significant = []

    for exp in experiments_with_p:
        exp_id = str(exp.get('param_hash', ''))
        if exp_id in sig_lookup:
            sig_result = sig_lookup[exp_id]
            exp['adjusted_p_value'] = sig_result.adjusted_p_value
            exp['is_significant'] = sig_result.is_significant

            if sig_result.is_significant:
                significant.append(exp)
            else:
                not_significant.append(exp)
        else:
            exp['is_significant'] = False
            not_significant.append(exp)

    # Sort significant by Sharpe
    significant.sort(key=lambda e: e.get('sharpe_ratio', 0) or 0, reverse=True)

    return {
        'significant': significant,
        'not_significant': not_significant,
        'summary': {
            'total_experiments': len(experiments),
            'tested_for_significance': len(significance_results),
            'significant_discoveries': len(significant),
            'fdr_alpha': alpha,
            'min_trades': min_trades,
            'discovery_rate': len(significant) / len(significance_results) * 100 if significance_results else 0,
        }
    }
