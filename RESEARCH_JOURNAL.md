# Research Journal

> **Purpose**: Document hypotheses and experimental rationale BEFORE running backtests.
> 
> This creates a forensic audit trail of your research process (VRD 2.0 compliant).

---

## How to Use This Journal

1. **Before** running a new experiment, create an entry below
2. State your **hypothesis** clearly
3. Record the **parameters** you're testing
4. After the run, record **observations** and **conclusions**

---

## Entry Template

```markdown
### YYYY-MM-DD: [Descriptive Title]

**Run ID**: [To be filled after run]

**Hypothesis**:
[What do you expect to happen and why?]

**Parameters Changed**:
- `param_name`: old_value → new_value
- `param_name`: old_value → new_value

**Expected Outcome**:
[What metrics do you expect to improve/worsen?]

**Command**:
```bash
python -m cli.run_backtest --symbol BTCUSDT --timeframe 4h [other flags]
```

**Observations**:
[What actually happened? Record key metrics.]

**Conclusions**:
[Was the hypothesis validated? What did you learn?]

**Next Steps**:
[What will you try next based on these results?]

---
```

---

## Journal Entries

### 2026-01-04: Initial Baseline (Default Parameters)

**Run ID**: 25cc57df

**Hypothesis**:
Default ATR detection parameters will produce a baseline result that we can improve upon.

**Parameters**:
- `min_validity_score`: 0.5
- `atr_period`: 14
- `stop_loss_atr_mult`: 0.5
- `shoulder_tolerance`: 10%

**Expected Outcome**:
Unknown - establishing baseline.

**Command**:
```bash
python -m cli.run_backtest --symbol BTCUSDT --timeframe 4h --min-validity 0.5
```

**Observations**:
- Net P&L: -34.40%
- Win Rate: 31.58%
- Total Trades: 114
- Profit Factor: 0.48
- Sharpe: -0.32

**Conclusions**:
- Default parameters produce too many false signals
- Win rate is too low for 1:1 R:R
- Need to investigate: Is validity score filtering too loose? Are shoulder/head constraints correct?

**Next Steps**:
1. Increase `min_validity_score` to 0.7 and compare
2. Analyze losing trades to understand pattern quality issues
3. Consider stricter head depth ATR requirements

---

<!-- Add new entries below -->
