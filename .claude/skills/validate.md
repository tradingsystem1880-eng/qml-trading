# Validate

Run the full validation suite on backtest results.

## Usage
```
/validate [experiment_id]
```

## Examples
- `/validate` - Validate most recent backtest
- `/validate exp_123` - Validate specific experiment

## Instructions

When the user invokes this skill:

1. If no experiment_id provided, find the most recent experiment in `results/experiments.db`

2. Load the backtest results and trades

3. Run the validation suite:
   ```python
   from src.validation import run_validation_suite
   suite = run_validation_suite(results, trades=trade_list)
   ```

4. Report results for each validator:
   - **PermutationTest**: p-value and significance
   - **MonteCarloSim**: VaR, CVaR, Risk of Ruin
   - **BootstrapResample**: Confidence intervals
   - **PurgedWalkForward**: Out-of-sample performance

5. Give overall verdict:
   - PASS: All validators pass (p < 0.05, positive edge in walk-forward)
   - CAUTION: Mixed results
   - FAIL: Statistical evidence of no edge
