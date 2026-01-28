#!/usr/bin/env python3
"""
Phase 9.7: Validate Funding Rate Filter
========================================
Comprehensive validation with DeepSeek-recommended statistical methodology.

Implements:
- Bootstrap significance test (10,000 iterations)
- Directional balance check
- Max drawdown comparison
- Sensitivity analysis across threshold range
- Filter attribution (winners vs losers removed)

Prerequisites:
1. Historical funding data (run fetch_historical_funding.py first)
2. Validated trades from Phase 9.5 (or backtest)

Usage:
    python scripts/validate_funding_filter.py
    python scripts/validate_funding_filter.py --threshold 0.0001
    python scripts/validate_funding_filter.py --symbols BTC/USDT,ETH/USDT
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Optional matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.research_journal import ResearchJournal
from src.research.feature_validator import FeatureValidator, FeatureValidatorConfig
from src.data.funding_rates import FundingRateFetcher, FundingRateFetcherConfig

# Configuration
FUNDING_THRESHOLD = 0.0001  # ±0.01%
MIN_TRADES_AFTER_FILTER = 150  # For statistical power
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'funding_filter_validation'


def load_validated_trades() -> pd.DataFrame:
    """
    Load trades from Phase 9.5 validation.

    Fallback order: parquet → CSV → helpful error
    """
    # Try parquet first
    parquet_path = DATA_DIR / 'backtest' / 'validated_trades.parquet'
    if parquet_path.exists():
        print(f"  Loading from {parquet_path}")
        return pd.read_parquet(parquet_path)

    # Try CSV fallback
    csv_path = DATA_DIR / 'backtest' / 'validated_trades.csv'
    if csv_path.exists():
        print(f"  Loading from {csv_path}")
        return pd.read_csv(csv_path, parse_dates=['entry_time', 'exit_time'])

    # Helpful error message
    raise FileNotFoundError(
        f"No validated trades found.\n"
        f"Expected locations:\n"
        f"  - {parquet_path}\n"
        f"  - {csv_path}\n\n"
        f"Run phase95 validation first to generate trades, or place a trades file at one of these locations.\n"
        f"Required columns: symbol, direction, entry_time, exit_time, pnl_r"
    )


def load_historical_funding() -> pd.DataFrame:
    """Load pre-fetched historical funding data."""
    funding_dir = DATA_DIR / 'funding_rates'
    all_funding = []

    if not funding_dir.exists():
        raise FileNotFoundError(
            f"Funding data directory not found: {funding_dir}\n"
            f"Run: python scripts/fetch_historical_funding.py first"
        )

    for fp in funding_dir.glob('*_funding.parquet'):
        df = pd.read_parquet(fp)
        all_funding.append(df)

    if not all_funding:
        raise FileNotFoundError(
            f"No funding data found in {funding_dir}\n"
            f"Run: python scripts/fetch_historical_funding.py first"
        )

    return pd.concat(all_funding, ignore_index=True)


def align_funding_to_trades(trades_df: pd.DataFrame,
                            funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join funding rate to trades using most recent funding BEFORE trade entry.
    This prevents look-ahead bias.
    """
    trades_df = trades_df.copy()
    trades_df['funding_rate'] = None
    trades_df['funding_timestamp'] = None

    # Ensure timestamps are timezone-aware
    if trades_df['entry_time'].dt.tz is None:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.tz_localize('UTC')

    for idx, trade in trades_df.iterrows():
        entry_time = trade['entry_time']
        symbol = trade['symbol']

        # Normalize symbol for matching
        symbol_base = symbol.replace('USDT', '').replace('/', '').replace('-', '')

        # Get funding rates for this symbol before entry time
        symbol_funding = funding_df[
            (funding_df['symbol'].str.contains(symbol_base, case=False)) &
            (funding_df['timestamp'] <= entry_time)
        ]

        if len(symbol_funding) > 0:
            # Most recent funding before entry (REALIZED rate, not predicted)
            latest = symbol_funding.loc[symbol_funding['timestamp'].idxmax()]
            trades_df.at[idx, 'funding_rate'] = latest['funding_rate']
            trades_df.at[idx, 'funding_timestamp'] = latest['timestamp']

    return trades_df


def apply_funding_filter(trades_df: pd.DataFrame,
                         threshold: float = FUNDING_THRESHOLD) -> tuple:
    """
    Apply funding rate filter with detailed logging.

    Returns: (filtered_df, filter_log_df)
    """
    filter_log = []
    mask = []

    for idx, trade in trades_df.iterrows():
        funding = trade['funding_rate']
        direction = trade.get('direction', 'UNKNOWN')

        # Missing data = reject (conservative)
        if pd.isna(funding):
            mask.append(False)
            filter_log.append({
                'trade_id': idx,
                'symbol': trade.get('symbol', 'UNKNOWN'),
                'direction': direction,
                'filtered': True,
                'reason': 'MISSING_DATA',
                'funding_rate': None,
                'pnl_r': trade.get('pnl_r', 0),
                'was_winner': trade.get('pnl_r', 0) > 0
            })
            continue

        # Filter logic (include equality as per DeepSeek)
        if direction == 'LONG' and funding >= threshold:
            mask.append(False)
            reason = f'LONG_OVERCROWDED: funding {funding:.6f} >= {threshold}'
            filtered = True
        elif direction == 'SHORT' and funding <= -threshold:
            mask.append(False)
            reason = f'SHORT_OVERCROWDED: funding {funding:.6f} <= {-threshold}'
            filtered = True
        else:
            mask.append(True)
            reason = 'PASSED'
            filtered = False

        filter_log.append({
            'trade_id': idx,
            'symbol': trade.get('symbol', 'UNKNOWN'),
            'direction': direction,
            'filtered': filtered,
            'reason': reason,
            'funding_rate': funding,
            'pnl_r': trade.get('pnl_r', 0),
            'was_winner': trade.get('pnl_r', 0) > 0
        })

    return trades_df[mask].copy(), pd.DataFrame(filter_log)


def analyze_filtered_trades(filter_log: pd.DataFrame) -> dict:
    """Analyze which trades were filtered - winners or losers?"""
    filtered = filter_log[filter_log['filtered'] == True]
    passed = filter_log[filter_log['filtered'] == False]

    if len(filtered) == 0:
        return {'error': 'No trades were filtered'}

    # Filtered trade analysis
    filtered_winners = filtered['was_winner'].sum()
    filtered_losers = len(filtered) - filtered_winners
    filtered_total_pnl = filtered['pnl_r'].sum()

    # Passed trade analysis
    passed_winners = passed['was_winner'].sum()
    passed_losers = len(passed) - passed_winners

    # By reason
    by_reason = filtered.groupby('reason').agg({
        'trade_id': 'count',
        'was_winner': 'sum',
        'pnl_r': 'sum'
    }).rename(columns={'trade_id': 'count', 'was_winner': 'winners', 'pnl_r': 'total_pnl'})

    return {
        'filtered_count': len(filtered),
        'filtered_winners': int(filtered_winners),
        'filtered_losers': int(filtered_losers),
        'filtered_win_pct': round(filtered_winners / len(filtered) * 100, 1) if len(filtered) > 0 else 0,
        'filtered_total_pnl_r': round(filtered_total_pnl, 2),
        'passed_count': len(passed),
        'passed_winners': int(passed_winners),
        'passed_losers': int(passed_losers),
        'by_reason': by_reason.to_dict() if len(by_reason) > 0 else {},
        'good_filter': filtered_losers > filtered_winners  # Filter should remove more losers
    }


def run_sensitivity_analysis(trades_df: pd.DataFrame,
                              thresholds: list = None) -> pd.DataFrame:
    """
    Test filter performance across threshold range.
    NOT for optimization - just to verify we're not on a cliff edge.
    """
    if thresholds is None:
        thresholds = [0.00005, 0.00008, 0.0001, 0.00012, 0.00015, 0.0002]

    results = []
    for thresh in thresholds:
        filtered_df, _ = apply_funding_filter(trades_df, threshold=thresh)

        if len(filtered_df) < 50:
            continue

        wins = (filtered_df['pnl_r'] > 0).sum()
        losses = (filtered_df['pnl_r'] <= 0).sum()
        total_wins = filtered_df[filtered_df['pnl_r'] > 0]['pnl_r'].sum()
        total_losses = abs(filtered_df[filtered_df['pnl_r'] < 0]['pnl_r'].sum())

        pf = total_wins / total_losses if total_losses > 0 else float('inf')
        wr = wins / len(filtered_df) * 100

        results.append({
            'threshold': thresh,
            'threshold_pct': f'{thresh*100:.3f}%',
            'trades_remaining': len(filtered_df),
            'trade_reduction_pct': round((1 - len(filtered_df)/len(trades_df)) * 100, 1),
            'win_rate': round(wr, 1),
            'profit_factor': round(pf, 2) if not np.isinf(pf) else 99.99
        })

    return pd.DataFrame(results)


def calculate_data_quality(trades_df: pd.DataFrame) -> dict:
    """Calculate data quality metrics."""
    total = len(trades_df)
    missing = trades_df['funding_rate'].isna().sum()
    missing_pct = missing / total * 100

    return {
        'total_trades': total,
        'trades_with_funding': total - missing,
        'trades_missing_funding': missing,
        'missing_pct': round(missing_pct, 1),
        'data_quality_warning': missing_pct > 10
    }


def calc_metrics(df: pd.DataFrame) -> dict:
    """Calculate trading metrics from DataFrame with pnl_r column."""
    if len(df) == 0:
        return {'trades': 0, 'win_rate': 0, 'profit_factor': 0, 'expectancy': 0}

    wins = df[df['pnl_r'] > 0]
    losses = df[df['pnl_r'] <= 0]
    total_win = wins['pnl_r'].sum()
    total_loss = abs(losses['pnl_r'].sum())

    return {
        'trades': len(df),
        'win_rate': len(wins) / len(df) * 100 if len(df) > 0 else 0,
        'profit_factor': total_win / total_loss if total_loss > 0 else float('inf'),
        'avg_win': wins['pnl_r'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl_r'].mean() if len(losses) > 0 else 0,
        'expectancy': df['pnl_r'].mean() if len(df) > 0 else 0
    }


# =============================================================================
# Phase 9.7 Corrections: New Validation Checks (DeepSeek Methodology)
# =============================================================================

def check_entry_time_distribution(trades_df: pd.DataFrame) -> dict:
    """
    Check if trade entries cluster near funding times (00:00, 08:00, 16:00 UTC).
    Clustering could create spurious correlation with funding rates.

    Args:
        trades_df: DataFrame with 'entry_time' column

    Returns:
        dict with clustering analysis and warning if >50% near funding times
    """
    if 'entry_time' not in trades_df.columns:
        return {'error': 'entry_time column not found'}

    # Extract hour of entry (UTC)
    entry_times = pd.to_datetime(trades_df['entry_time'])
    if entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_convert('UTC')
    entry_hours = entry_times.dt.hour

    # Funding windows: 2 hours before each funding time (00, 08, 16 UTC)
    funding_hours = [0, 8, 16]
    near_funding_hours = set()
    for fh in funding_hours:
        near_funding_hours.add((fh - 2) % 24)
        near_funding_hours.add((fh - 1) % 24)
        near_funding_hours.add(fh)

    entries_near_funding = entry_hours.isin(near_funding_hours).sum()
    total_entries = len(trades_df)
    pct_near_funding = entries_near_funding / total_entries * 100 if total_entries > 0 else 0

    # Expected if uniform: 9 hours / 24 hours = 37.5%
    expected_pct = len(near_funding_hours) / 24 * 100

    # Flag if significantly clustered (>50% when expecting ~37.5%)
    is_clustered = pct_near_funding > 50

    return {
        'entries_near_funding': int(entries_near_funding),
        'total_entries': total_entries,
        'pct_near_funding': round(pct_near_funding, 1),
        'expected_pct': round(expected_pct, 1),
        'is_clustered': is_clustered,
        'warning': 'Entries cluster near funding times - correlation may be spurious' if is_clustered else None
    }


def check_economic_significance(
    baseline_trades: pd.DataFrame,
    filtered_trades: pd.DataFrame,
    transaction_cost_pct: float = 0.0006
) -> dict:
    """
    Check if filter improvement is economically meaningful after costs.

    Args:
        baseline_trades: All trades DataFrame with 'pnl_r'
        filtered_trades: Filtered trades DataFrame with 'pnl_r'
        transaction_cost_pct: Round-trip cost (0.06% = 6bps typical for crypto futures)

    Returns:
        dict with economic analysis comparing R earned per 100 signals
    """
    # Convert cost to R-multiple (assuming 1R = 1% risk)
    cost_in_r = transaction_cost_pct / 0.01

    # Calculate net expectancy (gross expectancy - transaction cost)
    baseline_expectancy = baseline_trades['pnl_r'].mean() if len(baseline_trades) > 0 else 0
    filtered_expectancy = filtered_trades['pnl_r'].mean() if len(filtered_trades) > 0 else 0

    baseline_net = baseline_expectancy - cost_in_r
    filtered_net = filtered_expectancy - cost_in_r

    # Trade reduction impact
    trade_reduction = 1 - len(filtered_trades) / len(baseline_trades) if len(baseline_trades) > 0 else 0

    # Expected total R per 100 signals
    baseline_total_r_per_100 = baseline_net * 100
    filtered_total_r_per_100 = filtered_net * (100 * (1 - trade_reduction))

    # Is filtered strategy better in absolute R terms?
    economically_beneficial = filtered_total_r_per_100 > baseline_total_r_per_100

    return {
        'baseline_expectancy_gross': round(baseline_expectancy, 3),
        'filtered_expectancy_gross': round(filtered_expectancy, 3),
        'baseline_expectancy_net': round(baseline_net, 3),
        'filtered_expectancy_net': round(filtered_net, 3),
        'expectancy_improvement': round(filtered_net - baseline_net, 3),
        'trade_reduction_pct': round(trade_reduction * 100, 1),
        'baseline_r_per_100_signals': round(baseline_total_r_per_100, 1),
        'filtered_r_per_100_signals': round(filtered_total_r_per_100, 1),
        'economically_beneficial': economically_beneficial,
        'verdict': 'Filter improves total R' if economically_beneficial else 'Filter reduces total R (trade reduction outweighs quality improvement)'
    }


def check_walkforward_consistency(
    all_trades: pd.DataFrame,
    kept_mask: pd.Series,
    n_folds: int = 5
) -> dict:
    """
    Check if filter improvement is consistent across time periods.

    Args:
        all_trades: All trades DataFrame with 'entry_time' and 'pnl_r'
        kept_mask: Boolean Series - True for trades that passed filter
        n_folds: Number of time-based folds

    Returns:
        dict with fold-by-fold analysis and consistency check
    """
    if 'entry_time' not in all_trades.columns:
        return {'error': 'entry_time column not found'}

    # Sort by entry time and assign fold numbers
    all_trades = all_trades.sort_values('entry_time').reset_index(drop=True)
    kept_mask = kept_mask.reindex(all_trades.index)

    # Assign folds based on time (each fold is 1/n of data chronologically)
    n_trades = len(all_trades)
    fold_size = n_trades // n_folds
    all_trades['_fold'] = [min(i // fold_size, n_folds - 1) for i in range(n_trades)]

    def calc_pf(df):
        if len(df) == 0:
            return 0.0
        wins = df[df['pnl_r'] > 0]['pnl_r'].sum()
        losses = abs(df[df['pnl_r'] < 0]['pnl_r'].sum())
        return wins / losses if losses > 0 else 100.0

    fold_results = []
    for fold in range(n_folds):
        fold_mask = all_trades['_fold'] == fold
        fold_trades = all_trades[fold_mask]
        fold_kept_mask = kept_mask[fold_mask]
        fold_kept = fold_trades[fold_kept_mask]

        baseline_pf = calc_pf(fold_trades)
        filtered_pf = calc_pf(fold_kept)
        improvement = filtered_pf - baseline_pf

        fold_results.append({
            'fold': fold,
            'baseline_pf': round(baseline_pf, 2),
            'filtered_pf': round(filtered_pf, 2),
            'improvement': round(improvement, 2),
            'improved': improvement > 0,
            'n_trades': len(fold_trades),
            'n_kept': len(fold_kept)
        })

    # Clean up temp column
    all_trades.drop('_fold', axis=1, inplace=True)

    # Consistency: filter should improve in majority of folds
    folds_improved = sum(1 for f in fold_results if f['improved'])
    is_consistent = folds_improved >= 3  # At least 3/5 folds show improvement
    is_strongly_consistent = folds_improved >= 4  # 4/5 or 5/5

    return {
        'fold_results': fold_results,
        'folds_improved': folds_improved,
        'total_folds': n_folds,
        'is_consistent': is_consistent,
        'is_strongly_consistent': is_strongly_consistent,
        'consistency_pct': round(folds_improved / n_folds * 100, 0)
    }


def calculate_verdict(validation_results: dict) -> dict:
    """
    Determine PASS/INCONCLUSIVE/FAIL based on mandatory and weighted criteria.

    MANDATORY (must pass or verdict is FAIL):
    - Statistical significance (p < 0.05)
    - Walk-forward consistency (>=3/5 folds improved)

    WEIGHTED CRITERIA (need 4+ to pass):
    - PF improved
    - Trade reduction < 30%
    - Removes more losers than winners
    - No severe directional bias (30-70% either direction)
    - Economically beneficial
    - Max drawdown improved

    Args:
        validation_results: dict containing all test results

    Returns:
        dict with verdict and detailed breakdown
    """
    # Extract results
    sig_test = validation_results.get('significance_test', {})
    wf_test = validation_results.get('walkforward_consistency', {})
    filter_stats = validation_results.get('filter_statistics', {})
    attribution = validation_results.get('filter_attribution', {})
    balance = validation_results.get('directional_balance', {})
    economic = validation_results.get('economic_significance', {})
    drawdown = validation_results.get('drawdown_comparison', {})

    # MANDATORY CRITERIA
    mandatory_passed = True
    mandatory_failures = []

    # 1. Statistical significance
    if not sig_test.get('significant_at_95', False):
        mandatory_passed = False
        mandatory_failures.append(f"Not statistically significant (p={sig_test.get('p_value', 'N/A')})")

    # 2. Walk-forward consistency
    if not wf_test.get('is_consistent', False):
        mandatory_passed = False
        mandatory_failures.append(f"Walk-forward inconsistent ({wf_test.get('folds_improved', 0)}/{wf_test.get('total_folds', 5)} folds improved)")

    if not mandatory_passed:
        return {
            'verdict': 'FAIL',
            'reason': 'Mandatory criteria failed',
            'mandatory_criteria': {
                'statistical_significance': sig_test.get('significant_at_95', False),
                'walkforward_consistency': wf_test.get('is_consistent', False)
            },
            'mandatory_failures': mandatory_failures,
            'weighted_criteria': {},
            'weighted_passed': 0,
            'weighted_total': 6,
            'recommendation': 'Do NOT integrate filter. Move to Priority 2 (Liquidation Clusters).'
        }

    # WEIGHTED CRITERIA
    weighted_checks = {
        'pf_improved': filter_stats.get('filtered_pf', 0) > filter_stats.get('baseline_pf', 0),
        'trade_reduction_acceptable': filter_stats.get('trade_reduction_pct', 100) < 30,
        'removes_more_losers': attribution.get('filtered_losers', 0) > attribution.get('filtered_winners', 0),
        'no_directional_bias': not balance.get('severe_imbalance', True),
        'economically_beneficial': economic.get('economically_beneficial', False),
        'drawdown_improved': drawdown.get('dd_reduction_pct', 0) > 0
    }

    weighted_passed = sum(weighted_checks.values())
    total_weighted = len(weighted_checks)

    # Determine verdict
    if weighted_passed >= 5:
        verdict = 'STRONG_PASS'
        recommendation = 'Integrate into paper trading immediately. Run 50-trade A/B test.'
    elif weighted_passed >= 4:
        verdict = 'PASS'
        recommendation = 'Integrate into paper trading with monitoring. Compare to unfiltered baseline.'
    elif weighted_passed >= 3:
        verdict = 'INCONCLUSIVE'
        recommendation = 'Paper trade BASE system (no filter). Re-evaluate after 200 paper trades with more data.'
    else:
        verdict = 'FAIL'
        recommendation = 'Do NOT integrate filter. Move to Priority 2 (Liquidation Clusters).'

    return {
        'verdict': verdict,
        'reason': f'{weighted_passed}/{total_weighted} weighted criteria passed',
        'mandatory_criteria': {
            'statistical_significance': sig_test.get('significant_at_95', False),
            'walkforward_consistency': wf_test.get('is_consistent', False)
        },
        'mandatory_failures': [],
        'weighted_criteria': weighted_checks,
        'weighted_passed': weighted_passed,
        'weighted_total': total_weighted,
        'recommendation': recommendation
    }


def report_edge_cases(validation_results: dict, trades_df: pd.DataFrame,
                      filter_log: pd.DataFrame, wf_test: dict) -> dict:
    """
    Report edge cases identified by DeepSeek review.
    Called after main validation, before final output.

    Args:
        validation_results: Dict containing all test results
        trades_df: Original trades DataFrame
        filter_log: Filter log DataFrame with 'filtered' and 'was_winner' columns
        wf_test: Walk-forward test results

    Returns:
        dict with edge case analysis
    """
    print("\n" + "="*60)
    print("EDGE CASE ANALYSIS")
    print("="*60)

    edge_cases = {}

    # 1. Trade reduction severity (>40% is concerning)
    total = len(trades_df)
    filtered_out = filter_log['filtered'].sum()
    reduction_pct = filtered_out / total * 100 if total > 0 else 0
    edge_cases['trade_reduction'] = {
        'value': round(reduction_pct, 1),
        'threshold': 40,
        'warning': reduction_pct > 40
    }
    if reduction_pct > 40:
        print(f"⚠️  HIGH TRADE REDUCTION: {reduction_pct:.1f}% (>40% threshold)")
    else:
        print(f"✓ Trade reduction: {reduction_pct:.1f}%")

    # 2. One-fold dominance (any fold >50% of total improvement)
    if 'fold_results' in wf_test:
        improvements = [f['improvement'] for f in wf_test['fold_results']]
        positive_improvements = [i for i in improvements if i > 0]
        if positive_improvements and sum(positive_improvements) > 0:
            max_contribution = max(positive_improvements) / sum(positive_improvements)
            edge_cases['fold_dominance'] = {
                'max_contribution': round(max_contribution, 2),
                'warning': max_contribution > 0.5
            }
            if max_contribution > 0.5:
                print(f"⚠️  ONE FOLD DOMINANCE: {max_contribution:.0%} of improvement from single fold")
            else:
                print(f"✓ Balanced fold contributions: max {max_contribution:.0%}")
        else:
            print("✓ No positive improvements to analyze for fold dominance")
            edge_cases['fold_dominance'] = {'max_contribution': 0, 'warning': False}
    else:
        edge_cases['fold_dominance'] = {'max_contribution': 0, 'warning': False}
        print("  (Walk-forward results not available)")

    # 3. Counter-intuitive attribution (more winners filtered than losers)
    filtered_trades = filter_log[filter_log['filtered'] == True]
    winners_filtered = filtered_trades['was_winner'].sum() if len(filtered_trades) > 0 else 0
    losers_filtered = len(filtered_trades) - winners_filtered
    edge_cases['attribution'] = {
        'winners_filtered': int(winners_filtered),
        'losers_filtered': int(losers_filtered),
        'warning': winners_filtered > losers_filtered
    }
    if winners_filtered > losers_filtered:
        print(f"⚠️  COUNTER-INTUITIVE: Filtered {winners_filtered} winners > {losers_filtered} losers")
    else:
        print(f"✓ Attribution correct: {losers_filtered} losers > {winners_filtered} winners filtered")

    # 4. Time clustering already checked in step 2 (entry time distribution)

    # Summary
    warnings_count = sum(1 for v in edge_cases.values() if isinstance(v, dict) and v.get('warning'))
    print(f"\nEdge case warnings: {warnings_count}/3")
    edge_cases['total_warnings'] = warnings_count

    return edge_cases


def create_sensitivity_plot(sensitivity_df: pd.DataFrame, output_path: Path):
    """Create sensitivity analysis plot if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping plot (matplotlib not installed)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # PF vs Threshold
    axes[0].plot(sensitivity_df['threshold'] * 100, sensitivity_df['profit_factor'], 'b-o')
    axes[0].axvline(x=0.01, color='r', linestyle='--', label='Selected threshold')
    axes[0].set_xlabel('Threshold (%)')
    axes[0].set_ylabel('Profit Factor')
    axes[0].set_title('PF vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Trade Reduction vs Threshold
    axes[1].plot(sensitivity_df['threshold'] * 100, sensitivity_df['trade_reduction_pct'], 'g-o')
    axes[1].axvline(x=0.01, color='r', linestyle='--', label='Selected threshold')
    axes[1].set_xlabel('Threshold (%)')
    axes[1].set_ylabel('Trade Reduction (%)')
    axes[1].set_title('Trade Reduction vs Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Run complete funding filter validation with DeepSeek methodology fixes."""
    parser = argparse.ArgumentParser(description="Validate funding rate filter")
    parser.add_argument('--threshold', type=float, default=FUNDING_THRESHOLD,
                        help=f'Funding rate threshold (default {FUNDING_THRESHOLD})')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--no-journal', action='store_true', help='Skip logging to journal')
    args = parser.parse_args()

    threshold = args.threshold

    print("=" * 70)
    print("FUNDING RATE FILTER VALIDATION (Phase 9.7 DeepSeek Methodology)")
    print("=" * 70)
    print(f"Threshold: ±{threshold*100:.3f}%")
    print(f"Min trades after filter: {MIN_TRADES_AFTER_FILTER}")
    print("=" * 70)

    # Initialize output directory
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize validator
    validator = FeatureValidator(
        baseline_metrics={'profit_factor': 4.49, 'win_rate': 0.55},
        config=FeatureValidatorConfig()
    )

    # [1/10] Load trades
    print("\n[1/10] Loading trades...")
    try:
        trades_df = load_validated_trades()
        print(f"  Loaded {len(trades_df)} trades")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # [2/10] Check entry time distribution (NEW - DeepSeek correction)
    print("\n[2/10] Checking entry time distribution...")
    time_check = check_entry_time_distribution(trades_df)
    if 'error' not in time_check:
        print(f"  Entries near funding times: {time_check['pct_near_funding']:.1f}% (expected: {time_check['expected_pct']:.1f}%)")
        if time_check['warning']:
            print(f"  ⚠️  WARNING: {time_check['warning']}")
        else:
            print("  ✓ Entries distributed normally")
    else:
        print(f"  {time_check['error']}")
        time_check = {'is_clustered': False, 'pct_near_funding': 0}

    # [3/10] Load funding data
    print("\n[3/10] Loading funding data...")
    try:
        funding_df = load_historical_funding()
        print(f"  Loaded {len(funding_df)} funding records")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # [4/10] Align funding to trades
    print("\n[4/10] Aligning funding rates to trades (preventing look-ahead bias)...")
    trades_with_funding = align_funding_to_trades(trades_df, funding_df)

    # [5/10] Data quality check
    print("\n[5/10] Checking data quality...")
    data_quality = calculate_data_quality(trades_with_funding)
    print(f"  Trades with funding: {data_quality['trades_with_funding']}")
    print(f"  Missing funding: {data_quality['trades_missing_funding']} ({data_quality['missing_pct']}%)")
    if data_quality['data_quality_warning']:
        print("  ⚠️  WARNING: >10% missing data may affect results")

    # [6/10] Apply filter and get mask for permutation test
    print(f"\n[6/10] Applying funding filter (threshold: ±{threshold*100:.3f}%)...")
    filtered_trades, filter_log = apply_funding_filter(trades_with_funding, threshold=threshold)

    # Create kept_mask from filter_log (True = kept, False = filtered out)
    kept_mask = pd.Series(~filter_log['filtered'].values, index=trades_with_funding.index)

    trade_reduction = (1 - len(filtered_trades) / len(trades_df)) * 100
    print(f"  Baseline trades: {len(trades_df)}")
    print(f"  Kept trades: {len(filtered_trades)}")
    print(f"  Trade reduction: {trade_reduction:.1f}%")

    # Check minimum sample size
    if len(filtered_trades) < MIN_TRADES_AFTER_FILTER:
        print(f"\n❌ FAIL: Insufficient trades after filter ({len(filtered_trades)} < {MIN_TRADES_AFTER_FILTER})")
        print("   Result: INCONCLUSIVE - Cannot validate with this sample size")
        sys.exit(1)

    # [7/10] Analyze filtered trades (attribution)
    print("\n[7/10] Analyzing filter attribution (winners vs losers)...")
    filter_analysis = analyze_filtered_trades(filter_log)
    print(f"  Filtered out {filter_analysis['filtered_count']} trades:")
    print(f"    - Winners removed: {filter_analysis['filtered_winners']}")
    print(f"    - Losers removed: {filter_analysis['filtered_losers']}")
    good_filter_status = '✓ Good (removes more losers)' if filter_analysis['good_filter'] else '⚠️ Removes more winners'
    print(f"    - Filter quality: {good_filter_status}")

    # Calculate baseline and filtered metrics
    baseline_metrics = calc_metrics(trades_with_funding)
    filtered_metrics = calc_metrics(filtered_trades)

    print(f"\n  Baseline: PF={baseline_metrics['profit_factor']:.2f}, WR={baseline_metrics['win_rate']:.1f}%")
    print(f"  Filtered: PF={filtered_metrics['profit_factor']:.2f}, WR={filtered_metrics['win_rate']:.1f}%")

    # a) Directional balance
    balance = validator.check_directional_balance(trades_with_funding, filtered_trades)
    if 'error' not in balance:
        print(f"  Directional balance: {balance['baseline_long_pct']:.1f}% LONG -> {balance['filtered_long_pct']:.1f}% LONG")
        if balance.get('severe_imbalance'):
            print("  ⚠️  WARNING: Severe directional imbalance")
    else:
        print(f"  Directional balance: {balance['error']}")
        balance['severe_imbalance'] = False

    # b) Max drawdown comparison
    dd_compare = validator.compare_max_drawdown(trades_with_funding, filtered_trades)
    if 'error' not in dd_compare:
        print(f"  Max drawdown: {dd_compare['baseline_max_dd_r']:.2f}R -> {dd_compare['filtered_max_dd_r']:.2f}R ({dd_compare['dd_reduction_pct']:.1f}% reduction)")
    else:
        print(f"  Max drawdown: {dd_compare['error']}")
        dd_compare['dd_reduction_pct'] = 0

    # [8/10] Permutation significance test (NEW - replaces flawed bootstrap)
    print("\n[8/10] Running permutation significance test (10,000 iterations)...")
    print("  (Tests: 'Does filter select better trades than random filtering?')")
    sig_test = validator.permutation_significance_test(
        trades_with_funding, kept_mask, n_permutations=10000
    )
    print(f"  Actual PF improvement: {sig_test['actual_improvement']}")
    print(f"  Null distribution mean: {sig_test['null_mean']} (std: {sig_test['null_std']})")
    print(f"  P-value: {sig_test['p_value']}")
    if sig_test['significant_at_95']:
        print("  ✓ Statistically significant at 95% confidence")
    else:
        print("  ✗ NOT statistically significant")

    # [9/10] Walk-forward consistency check (NEW - DeepSeek correction)
    print("\n[9/10] Checking walk-forward consistency (5 folds)...")
    wf_test = check_walkforward_consistency(trades_with_funding, kept_mask, n_folds=5)
    if 'error' not in wf_test:
        print(f"  Folds where filter improved: {wf_test['folds_improved']}/{wf_test['total_folds']}")
        for fold in wf_test['fold_results']:
            status = "✓" if fold['improved'] else "✗"
            print(f"    Fold {fold['fold']}: baseline={fold['baseline_pf']:.2f}, filtered={fold['filtered_pf']:.2f} {status}")
        if wf_test['is_consistent']:
            print("  ✓ Walk-forward consistent (>=3/5 folds improved)")
        else:
            print("  ✗ Walk-forward INCONSISTENT")
    else:
        print(f"  {wf_test['error']}")
        wf_test = {'is_consistent': False, 'folds_improved': 0, 'total_folds': 5}

    # [10/10] Economic significance check (NEW - DeepSeek correction)
    print("\n[10/10] Checking economic significance...")
    economic = check_economic_significance(trades_with_funding, filtered_trades)
    print(f"  Baseline R per 100 signals: {economic['baseline_r_per_100_signals']:.1f}")
    print(f"  Filtered R per 100 signals: {economic['filtered_r_per_100_signals']:.1f}")
    print(f"  {economic['verdict']}")

    # Sensitivity analysis
    print("\n" + "-" * 70)
    print("Sensitivity Analysis")
    print("-" * 70)
    sensitivity = run_sensitivity_analysis(trades_with_funding)
    print(sensitivity.to_string(index=False))

    # Save sensitivity plot
    create_sensitivity_plot(sensitivity, output_dir / 'sensitivity_analysis.png')

    # =========================================================================
    # FINAL VERDICT (using new mandatory + weighted criteria)
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION VERDICT")
    print("=" * 70)

    # Compile all results for verdict calculation
    validation_results = {
        'significance_test': sig_test,
        'walkforward_consistency': wf_test,
        'filter_statistics': {
            'baseline_pf': baseline_metrics['profit_factor'],
            'filtered_pf': filtered_metrics['profit_factor'],
            'trade_reduction_pct': trade_reduction
        },
        'filter_attribution': filter_analysis,
        'directional_balance': balance,
        'economic_significance': economic,
        'drawdown_comparison': dd_compare,
        'time_distribution': time_check,
        'data_quality': data_quality
    }

    verdict_result = calculate_verdict(validation_results)

    # Report edge cases (DeepSeek review requirement)
    edge_cases = report_edge_cases(validation_results, trades_with_funding, filter_log, wf_test)
    validation_results['edge_cases'] = edge_cases

    # Print mandatory criteria
    print("\nMANDATORY CRITERIA:")
    for criterion, passed in verdict_result['mandatory_criteria'].items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  [{status}] {criterion}")

    if verdict_result['mandatory_failures']:
        print("\n  Mandatory failures:")
        for failure in verdict_result['mandatory_failures']:
            print(f"    - {failure}")

    # Print weighted criteria
    print(f"\nWEIGHTED CRITERIA ({verdict_result['weighted_passed']}/{verdict_result['weighted_total']}):")
    for criterion, passed in verdict_result['weighted_criteria'].items():
        status = '✓' if passed else '✗'
        print(f"  [{status}] {criterion}")

    # Final verdict
    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict_result['verdict']}")
    print(f"{'=' * 70}")
    print(f"\nRecommendation: {verdict_result['recommendation']}")

    # Log to research journal
    if not args.no_journal:
        print("\nLogging to research journal...")
        journal = ResearchJournal()

        experiment = {
            'hypothesis': f'Extreme funding rates (±{threshold*100:.3f}%) predict poor trade outcomes due to overcrowded positioning',
            'feature_name': 'funding_rate_filter',
            'methodology': 'Permutation test (10k) + walk-forward consistency (5 folds) + economic significance + mandatory/weighted criteria',
            'results': {
                'baseline_pf': round(baseline_metrics['profit_factor'], 3),
                'filtered_pf': round(filtered_metrics['profit_factor'], 3),
                'pf_improvement': round(filtered_metrics['profit_factor'] - baseline_metrics['profit_factor'], 3),
                'trade_reduction_pct': round(trade_reduction, 1),
                'permutation_p_value': sig_test['p_value'],
                'walkforward_folds_improved': f"{wf_test['folds_improved']}/{wf_test['total_folds']}",
                'economically_beneficial': bool(economic['economically_beneficial']),
                'dd_reduction_pct': round(dd_compare.get('dd_reduction_pct', 0), 1),
                'weighted_score': f"{verdict_result['weighted_passed']}/{verdict_result['weighted_total']}",
                'threshold': threshold
            },
            'conclusion': 'PASS' if verdict_result['verdict'] in ['PASS', 'STRONG_PASS'] else 'FAIL' if verdict_result['verdict'] == 'FAIL' else 'INCONCLUSIVE',
            'notes': f"Permutation p={sig_test['p_value']}, WF {wf_test['folds_improved']}/5 folds improved. Filter removes {filter_analysis['filtered_losers']} losers vs {filter_analysis['filtered_winners']} winners. {verdict_result['recommendation']}",
            'tags': ['funding_rate', 'phase97', 'deepseek_methodology']
        }

        data_files = [
            str(DATA_DIR / 'backtest'),
            str(DATA_DIR / 'funding_rates')
        ]

        entry = journal.log_experiment(experiment, data_files=data_files)
        print(f"  Logged experiment: {entry['id']}")

    # Save detailed results
    results = {
        'verdict': verdict_result['verdict'],
        'recommendation': verdict_result['recommendation'],
        'threshold': threshold,
        'baseline_metrics': baseline_metrics,
        'filtered_metrics': filtered_metrics,
        'data_quality': data_quality,
        'time_distribution': time_check,
        'filter_analysis': filter_analysis,
        'significance_test': sig_test,
        'walkforward_consistency': wf_test,
        'edge_cases': edge_cases,
        'economic_significance': economic,
        'dd_comparison': dd_compare,
        'directional_balance': balance,
        'verdict_breakdown': verdict_result
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f'validation_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    filter_log.to_csv(output_dir / f'filter_log_{timestamp}.csv', index=False)
    sensitivity.to_csv(output_dir / f'sensitivity_analysis_{timestamp}.csv', index=False)

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
