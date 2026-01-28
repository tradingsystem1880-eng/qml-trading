# Walk-Forward Analysis Skill

Rolling train/test validation for trading strategy robustness.

## When to Use
- Validating strategy robustness over time
- Detecting parameter overfitting
- Comparing in-sample vs out-of-sample performance
- QML Phase 9.4 validation

## Walk-Forward Concepts

### Basic Structure

```
|----Train----|--Test--|----Train----|--Test--|----Train----|--Test--|
     Fold 1              Fold 2                   Fold 3
```

- **Anchored**: Train always starts from beginning
- **Rolling**: Train window slides forward
- **Purge**: Gap between train and test (prevents leakage)
- **Embargo**: Gap after test before next train (prevents overlap)

## Implementation

### Basic Walk-Forward

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class WFResult:
    fold: int
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    train_trades: int
    test_trades: int
    train_pf: float
    test_pf: float
    train_wr: float
    test_wr: float
    is_oos_degradation: float  # (test_pf - train_pf) / train_pf

def walk_forward_analysis(
    df: pd.DataFrame,
    strategy_func,
    n_folds: int = 5,
    train_pct: float = 0.7,
    purge_bars: int = 20,
    embargo_bars: int = 10
) -> List[WFResult]:
    """
    Perform walk-forward analysis.

    Args:
        df: OHLCV data
        strategy_func: Function(df, is_optimize=bool) -> trades
        n_folds: Number of folds
        train_pct: Fraction of each fold for training
        purge_bars: Bars to skip between train and test
        embargo_bars: Bars to skip after test
    """
    results = []
    total_bars = len(df)
    fold_size = total_bars // n_folds

    for fold in range(n_folds):
        # Calculate boundaries
        fold_start = fold * fold_size
        fold_end = min((fold + 1) * fold_size, total_bars)
        train_end_idx = fold_start + int((fold_end - fold_start) * train_pct)

        # Apply purge
        test_start_idx = train_end_idx + purge_bars

        if test_start_idx >= fold_end:
            continue  # Skip if purge consumes entire test set

        # Split data
        train_df = df.iloc[fold_start:train_end_idx]
        test_df = df.iloc[test_start_idx:fold_end]

        # Run strategy
        train_trades = strategy_func(train_df, is_optimize=True)
        test_trades = strategy_func(test_df, is_optimize=False)

        # Calculate metrics
        train_pf = calculate_pf(train_trades)
        test_pf = calculate_pf(test_trades)
        train_wr = calculate_wr(train_trades)
        test_wr = calculate_wr(test_trades)

        degradation = (test_pf - train_pf) / train_pf if train_pf > 0 else 0

        results.append(WFResult(
            fold=fold,
            train_period=(str(train_df.index[0]), str(train_df.index[-1])),
            test_period=(str(test_df.index[0]), str(test_df.index[-1])),
            train_trades=len(train_trades),
            test_trades=len(test_trades),
            train_pf=train_pf,
            test_pf=test_pf,
            train_wr=train_wr,
            test_wr=test_wr,
            is_oos_degradation=degradation
        ))

    return results

def calculate_pf(trades):
    wins = sum(t.pnl for t in trades if t.pnl > 0)
    losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
    return wins / losses if losses > 0 else float('inf')

def calculate_wr(trades):
    if not trades:
        return 0
    return sum(1 for t in trades if t.pnl > 0) / len(trades)
```

### Anchored Walk-Forward

```python
def anchored_walk_forward(
    df: pd.DataFrame,
    strategy_func,
    min_train_bars: int = 1000,
    test_bars: int = 200,
    step_bars: int = 200
) -> List[WFResult]:
    """
    Walk-forward where training always starts from beginning.
    More data as time progresses.
    """
    results = []
    fold = 0
    train_end = min_train_bars

    while train_end + test_bars <= len(df):
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:train_end + test_bars]

        train_trades = strategy_func(train_df, is_optimize=True)
        test_trades = strategy_func(test_df, is_optimize=False)

        results.append(WFResult(
            fold=fold,
            train_period=(str(df.index[0]), str(df.index[train_end-1])),
            test_period=(str(df.index[train_end]), str(df.index[train_end + test_bars - 1])),
            train_trades=len(train_trades),
            test_trades=len(test_trades),
            train_pf=calculate_pf(train_trades),
            test_pf=calculate_pf(test_trades),
            train_wr=calculate_wr(train_trades),
            test_wr=calculate_wr(test_trades),
            is_oos_degradation=0  # Calculate after
        ))

        train_end += step_bars
        fold += 1

    return results
```

## Robustness Assessment

### Stability Metrics

```python
def assess_walk_forward_stability(results: List[WFResult]) -> dict:
    """Evaluate strategy stability across folds."""
    test_pfs = [r.test_pf for r in results if r.test_pf < float('inf')]
    test_wrs = [r.test_wr for r in results]

    # Coefficient of Variation (lower = more stable)
    pf_cv = np.std(test_pfs) / np.mean(test_pfs) if test_pfs else float('inf')
    wr_cv = np.std(test_wrs) / np.mean(test_wrs) if test_wrs else float('inf')

    # Degradation analysis
    degradations = [r.is_oos_degradation for r in results]
    avg_degradation = np.mean(degradations)

    # Consistency
    profitable_folds = sum(1 for r in results if r.test_pf > 1.0)
    consistency = profitable_folds / len(results)

    return {
        "mean_test_pf": np.mean(test_pfs),
        "std_test_pf": np.std(test_pfs),
        "pf_cv": pf_cv,
        "mean_test_wr": np.mean(test_wrs),
        "wr_cv": wr_cv,
        "avg_is_oos_degradation": avg_degradation,
        "profitable_folds": profitable_folds,
        "total_folds": len(results),
        "consistency": consistency,
        "stability_grade": grade_stability(pf_cv, consistency, avg_degradation)
    }

def grade_stability(pf_cv: float, consistency: float, degradation: float) -> str:
    """Grade overall stability A-F."""
    score = 0

    # PF stability (lower CV = better)
    if pf_cv < 0.1: score += 3
    elif pf_cv < 0.2: score += 2
    elif pf_cv < 0.3: score += 1

    # Consistency
    if consistency >= 0.9: score += 3
    elif consistency >= 0.7: score += 2
    elif consistency >= 0.5: score += 1

    # Degradation (closer to 0 = better)
    if abs(degradation) < 0.1: score += 2
    elif abs(degradation) < 0.2: score += 1

    grades = {8: "A", 7: "A-", 6: "B+", 5: "B", 4: "B-", 3: "C+", 2: "C", 1: "D"}
    return grades.get(score, "F")
```

## QML Integration

### Phase 9.4 Walk-Forward Script

```python
# scripts/walk_forward_validation.py usage
"""
Walk-forward validation for QML strategy.

Usage:
    python scripts/walk_forward_validation.py --folds 5
"""

from src.detection import get_detector
from src.optimization.trade_simulator import simulate_trades

def qml_strategy(df, is_optimize=False):
    """QML pattern detection + simulation."""
    detector = get_detector("atr")
    signals = detector.detect(df, config)

    if is_optimize:
        # Could tune parameters here
        pass

    trades = simulate_trades(signals, df, trade_config)
    return trades

# Run validation
results = walk_forward_analysis(
    df=full_data,
    strategy_func=qml_strategy,
    n_folds=5,
    train_pct=0.7
)

stability = assess_walk_forward_stability(results)
print(f"Stability Grade: {stability['stability_grade']}")
print(f"Consistency: {stability['consistency']:.1%} profitable folds")
print(f"Mean OOS PF: {stability['mean_test_pf']:.2f}")
```

## Pass/Fail Criteria

| Metric | Pass | Marginal | Fail |
|--------|------|----------|------|
| OOS PF Mean | > 1.5 | 1.0-1.5 | < 1.0 |
| PF CV | < 0.2 | 0.2-0.4 | > 0.4 |
| Consistency | > 80% | 60-80% | < 60% |
| IS-OOS Degradation | < 20% | 20-40% | > 40% |

## Visualization

```python
import matplotlib.pyplot as plt

def plot_walk_forward_results(results: List[WFResult]):
    """Visualize walk-forward results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Train vs Test PF by fold
    ax1 = axes[0, 0]
    folds = [r.fold for r in results]
    ax1.bar([f-0.2 for f in folds], [r.train_pf for r in results], 0.4, label='Train')
    ax1.bar([f+0.2 for f in folds], [r.test_pf for r in results], 0.4, label='Test')
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Breakeven')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Profit Factor')
    ax1.legend()
    ax1.set_title('Train vs Test PF')

    # 2. Win Rate comparison
    ax2 = axes[0, 1]
    ax2.plot(folds, [r.train_wr for r in results], 'b-o', label='Train WR')
    ax2.plot(folds, [r.test_wr for r in results], 'g-o', label='Test WR')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Win Rate')
    ax2.legend()
    ax2.set_title('Win Rate by Fold')

    # 3. Degradation
    ax3 = axes[1, 0]
    ax3.bar(folds, [r.is_oos_degradation * 100 for r in results])
    ax3.axhline(y=0, color='k', linestyle='-')
    ax3.axhline(y=-20, color='r', linestyle='--', label='-20% threshold')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('IS→OOS Degradation %')
    ax3.set_title('Performance Degradation')

    # 4. Trade counts
    ax4 = axes[1, 1]
    ax4.bar([f-0.2 for f in folds], [r.train_trades for r in results], 0.4, label='Train')
    ax4.bar([f+0.2 for f in folds], [r.test_trades for r in results], 0.4, label='Test')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Trade Count')
    ax4.legend()
    ax4.set_title('Trades per Fold')

    plt.tight_layout()
    plt.savefig('results/walk_forward_analysis.png')
    plt.show()
```
