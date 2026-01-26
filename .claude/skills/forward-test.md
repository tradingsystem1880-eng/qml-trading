# Forward Test

Run forward testing / out-of-sample validation.

## Usage
```
/forward-test [days]
```

## Examples
- `/forward-test` - Test on last 30 days of data
- `/forward-test 60` - Test on last 60 days

## Instructions

When the user invokes this skill:

1. This tests the Phase 7.9 baseline on recent data NOT used in optimization

2. Parse arguments:
   - days: Default 30 (data from last N days)

3. Run forward test:
   ```bash
   python scripts/run_phase90_forward.py --days {days}
   ```

4. Report key metrics:
   - Number of signals generated
   - Win rate on forward period
   - Profit factor
   - Comparison to historical backtest metrics

5. Interpretation:
   - **Consistent**: Forward metrics within 20% of backtest = good
   - **Degraded**: Forward metrics significantly worse = possible overfit
   - **Improved**: Forward metrics better = possibly lucky or regime change

6. This is critical for validating the edge is real before live trading
