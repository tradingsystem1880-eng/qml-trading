# Optimize

Run parameter optimization for the detection system.

## Usage
```
/optimize [objective] [iterations]
```

## Objectives
- `composite` - Balanced multi-objective (recommended)
- `profit_factor` - Maximize profit factor
- `sharpe` - Maximize Sharpe ratio
- `expectancy` - Maximize per-trade expectancy
- `count_quality` - Maximize pattern detection quality
- `max_drawdown` - Minimize maximum drawdown

## Examples
- `/optimize` - Run composite optimization with 200 iterations
- `/optimize sharpe 500` - Optimize for Sharpe with 500 iterations
- `/optimize profit_factor 100` - Quick optimization for profit factor

## Instructions

When the user invokes this skill:

1. Parse arguments:
   - objective: Default "composite"
   - iterations: Default 200

2. Warn user this can take a long time (estimate ~5-10 min per 100 iterations)

3. Run optimization:
   ```bash
   python scripts/run_phase77_optimization.py --objective {objective} --iterations {iterations}
   ```

4. Report best parameters found:
   - Detection thresholds
   - TP/SL ATR multipliers
   - Filter weights
   - Resulting metrics (PF, Sharpe, Win Rate)

5. Results saved to `results/phase77_optimization/{objective}/`
