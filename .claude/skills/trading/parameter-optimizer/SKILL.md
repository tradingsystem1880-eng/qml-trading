# Parameter Optimizer Skill

Bayesian optimization for trading strategy parameters.

## When to Use
- Tuning detection parameters
- Optimizing risk/reward settings
- Multi-objective optimization
- QML Phase 7.7 optimization

## Bayesian Optimization

### Basic Setup with Optuna

```python
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Any, Callable

def create_optimizer(
    objective_func: Callable,
    param_space: Dict[str, Any],
    n_trials: int = 100,
    direction: str = "maximize"
) -> optuna.Study:
    """
    Create Bayesian optimizer with TPE sampler.

    Args:
        objective_func: Function(trial) -> score
        param_space: Dict of parameter definitions
        n_trials: Number of optimization iterations
        direction: "maximize" or "minimize"
    """
    sampler = TPESampler(seed=42, n_startup_trials=20)

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name="qml_optimization"
    )

    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

    return study
```

### QML Parameter Space

```python
def qml_param_space(trial: optuna.Trial) -> dict:
    """Define QML optimization parameter space."""
    return {
        # Detection parameters
        "swing_lookback": trial.suggest_int("swing_lookback", 3, 10),
        "min_swing_pct": trial.suggest_float("min_swing_pct", 0.5, 3.0),
        "bos_threshold": trial.suggest_float("bos_threshold", 0.3, 0.8),

        # Quality scoring weights
        "head_extension_weight": trial.suggest_float("head_extension_weight", 0.15, 0.30),
        "bos_efficiency_weight": trial.suggest_float("bos_efficiency_weight", 0.10, 0.25),
        "shoulder_symmetry_weight": trial.suggest_float("shoulder_symmetry_weight", 0.05, 0.15),

        # Risk parameters
        "sl_atr_mult": trial.suggest_float("sl_atr_mult", 1.0, 2.5),
        "tp_atr_mult": trial.suggest_float("tp_atr_mult", 2.0, 6.0),

        # Filter parameters
        "min_quality_score": trial.suggest_float("min_quality_score", 0.4, 0.7),
        "regime_weight": trial.suggest_float("regime_weight", 0.05, 0.15),
        "volume_spike_weight": trial.suggest_float("volume_spike_weight", 0.05, 0.15),
    }
```

### Objective Functions

```python
def profit_factor_objective(trial: optuna.Trial, df, detector) -> float:
    """Optimize for profit factor."""
    params = qml_param_space(trial)

    # Run detection with params
    signals = detector.detect(df, params)

    # Simulate trades
    trades = simulate_trades(signals, df, params)

    if len(trades) < 20:
        return 0.0  # Penalize low trade count

    # Calculate PF
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

    if gross_loss == 0:
        return 0.0

    pf = gross_win / gross_loss

    # Penalize extreme values (likely overfit)
    if pf > 10:
        return pf * 0.5

    return pf

def sharpe_objective(trial: optuna.Trial, df, detector) -> float:
    """Optimize for Sharpe ratio."""
    params = qml_param_space(trial)
    signals = detector.detect(df, params)
    trades = simulate_trades(signals, df, params)

    if len(trades) < 20:
        return -10.0

    returns = pd.Series([t.pnl_pct for t in trades])
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    return sharpe

def composite_objective(trial: optuna.Trial, df, detector) -> float:
    """Multi-objective composite score."""
    params = qml_param_space(trial)
    signals = detector.detect(df, params)
    trades = simulate_trades(signals, df, params)

    if len(trades) < 20:
        return 0.0

    # Components
    pf = calculate_pf(trades)
    wr = calculate_wr(trades)
    sharpe = calculate_sharpe(trades)
    trade_count = len(trades)

    # Normalize to 0-1 range
    pf_score = min(pf / 5.0, 1.0)  # Cap at PF=5
    wr_score = wr
    sharpe_score = (sharpe + 1) / 3  # Assume sharpe range [-1, 2]
    count_score = min(trade_count / 200, 1.0)  # Target 200 trades

    # Weighted composite
    composite = (
        0.35 * pf_score +
        0.25 * wr_score +
        0.25 * sharpe_score +
        0.15 * count_score
    )

    return composite
```

## Multi-Objective Optimization

```python
def multi_objective_optimization(df, detector, n_trials=200):
    """Optimize multiple objectives simultaneously."""

    def multi_objective(trial):
        params = qml_param_space(trial)
        signals = detector.detect(df, params)
        trades = simulate_trades(signals, df, params)

        if len(trades) < 20:
            return 0.0, 0.0, 100.0  # Bad defaults

        pf = calculate_pf(trades)
        sharpe = calculate_sharpe(trades)
        max_dd = calculate_max_dd(trades)

        return pf, sharpe, max_dd

    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=TPESampler(seed=42)
    )

    study.optimize(multi_objective, n_trials=n_trials)

    # Get Pareto front
    pareto_front = study.best_trials
    print(f"Found {len(pareto_front)} Pareto-optimal solutions")

    return pareto_front
```

## Hyperparameter Importance

```python
def analyze_parameter_importance(study: optuna.Study) -> dict:
    """Analyze which parameters matter most."""
    importance = optuna.importance.get_param_importances(study)

    print("\n📊 PARAMETER IMPORTANCE:")
    print("-" * 40)
    for param, score in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"{param:25s} {score:.3f} {bar}")

    return importance
```

## Optimization Best Practices

### 1. Use Pruning

```python
def objective_with_pruning(trial: optuna.Trial, df, detector):
    """Early stopping for unpromising trials."""
    params = qml_param_space(trial)

    # Check intermediate results
    signals = detector.detect(df[:len(df)//2], params)
    interim_trades = simulate_trades(signals, df[:len(df)//2], params)

    interim_pf = calculate_pf(interim_trades)
    trial.report(interim_pf, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    # Full evaluation
    signals = detector.detect(df, params)
    trades = simulate_trades(signals, df, params)
    return calculate_pf(trades)

# Enable pruning
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
```

### 2. Warm Start from Previous

```python
def warm_start_optimization(study_name: str, additional_trials: int = 50):
    """Continue optimization from saved study."""
    storage = f"sqlite:///results/optuna/{study_name}.db"

    study = optuna.load_study(
        study_name=study_name,
        storage=storage
    )

    print(f"Resuming from {len(study.trials)} trials")
    study.optimize(objective, n_trials=additional_trials)

    return study
```

### 3. Cross-Validation Objective

```python
def cv_objective(trial: optuna.Trial, df, detector, n_folds=3):
    """Objective with cross-validation to reduce overfitting."""
    params = qml_param_space(trial)

    fold_scores = []
    fold_size = len(df) // n_folds

    for i in range(n_folds):
        # Leave one fold out
        test_start = i * fold_size
        test_end = (i + 1) * fold_size

        train_df = pd.concat([df[:test_start], df[test_end:]])
        test_df = df[test_start:test_end]

        # Optimize on train (or use fixed params)
        signals = detector.detect(test_df, params)
        trades = simulate_trades(signals, test_df, params)

        if trades:
            fold_scores.append(calculate_pf(trades))

    if not fold_scores:
        return 0.0

    # Return mean - penalty for variance
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    return mean_score - 0.5 * std_score  # Penalize inconsistency
```

## QML Integration

```python
# scripts/run_phase77_optimization.py
"""
Run Bayesian optimization for QML parameters.

Usage:
    python scripts/run_phase77_optimization.py --objective composite --iterations 500
"""

OBJECTIVES = {
    "profit_factor": profit_factor_objective,
    "sharpe": sharpe_objective,
    "composite": composite_objective,
    "count_quality": count_quality_objective,
    "expectancy": expectancy_objective,
    "max_drawdown": max_drawdown_objective,
}

def main(objective_name: str, iterations: int):
    study = create_optimizer(
        objective_func=lambda trial: OBJECTIVES[objective_name](trial, df, detector),
        param_space=qml_param_space,
        n_trials=iterations
    )

    # Save results
    best_params = study.best_params
    with open(f"results/phase77_optimization/{objective_name}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Analyze importance
    importance = analyze_parameter_importance(study)

    print(f"\n🏆 BEST RESULT: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
```

## Results Summary

From Phase 7.7 optimization:

| Objective | Best Score | Key Parameters |
|-----------|------------|----------------|
| COUNT_QUALITY | 0.9138 | swing_lookback=5, min_quality=0.55 |
| SHARPE | 0.0366 | sl_atr=1.4, tp_atr=4.2 |
| PROFIT_FACTOR | 0.2402 | bos_threshold=0.45 |
| COMPOSITE | 0.3348 | Balanced settings |

Key finding: Pattern detection works well, but profitability requires additional filters (Phase 7.8).
