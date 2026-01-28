# Code Reviewer Skill

Automated code review checklist for trading systems.

## When to Use
- Reviewing pull requests
- Pre-commit validation
- Trading code audits
- Identifying bugs before production

## Trading Code Review Checklist

### 1. Look-Ahead Bias

```python
# ❌ BAD: Using future data
def calculate_signal(df, i):
    future_vol = df['close'].iloc[i:i+20].std()  # LOOK-AHEAD!
    return future_vol > threshold

# ✅ GOOD: Only past data
def calculate_signal(df, i):
    past_vol = df['close'].iloc[max(0, i-20):i+1].std()
    return past_vol > threshold
```

**Check**: Every `iloc[i:]` or `loc[date:]` that looks forward.

### 2. Off-by-One Errors

```python
# ❌ BAD: Includes current bar in "previous" calculation
previous_high = df['high'].iloc[i-5:i].max()  # Missing current bar

# ✅ GOOD: Includes current bar
previous_high = df['high'].iloc[i-5:i+1].max()

# ❌ BAD: Entry on same bar as signal
signal_bar = df.index.get_loc(signal.time)
entry_bar = signal_bar  # Same bar!

# ✅ GOOD: Entry on next bar
signal_bar = df.index.get_loc(signal.time)
entry_bar = signal_bar + 1  # Next bar open
```

### 3. Division by Zero

```python
# ❌ BAD: No zero check
profit_factor = gross_profit / gross_loss

# ✅ GOOD: Handle zero case
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

# ❌ BAD: ATR could be zero
sl_price = entry - (atr * sl_mult)

# ✅ GOOD: Check ATR
if atr <= 0:
    raise ValueError("ATR must be positive")
sl_price = entry - (atr * sl_mult)
```

### 4. Timezone Issues

```python
# ❌ BAD: Mixing timezones
signal_time = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
df_time = df.index[0]  # Might be timezone-naive

if signal_time == df_time:  # Comparison may fail!
    ...

# ✅ GOOD: Normalize timezones
signal_time = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
df_time = df.index[0].tz_localize("UTC") if df.index[0].tz is None else df.index[0]

if signal_time == df_time:
    ...
```

### 5. Float Comparison

```python
# ❌ BAD: Direct float equality
if price == stop_loss:
    trigger_stop()

# ✅ GOOD: Use tolerance
if abs(price - stop_loss) < 1e-8:
    trigger_stop()

# Or for price comparisons
if price <= stop_loss:  # Use inequality
    trigger_stop()
```

### 6. Mutable Default Arguments

```python
# ❌ BAD: Mutable default
def process_trades(trades=[]):
    trades.append(new_trade)  # Modifies default!
    return trades

# ✅ GOOD: None default
def process_trades(trades=None):
    if trades is None:
        trades = []
    trades.append(new_trade)
    return trades
```

### 7. Exception Handling

```python
# ❌ BAD: Bare except
try:
    execute_trade()
except:
    pass  # Swallows ALL exceptions

# ❌ BAD: Too broad
try:
    execute_trade()
except Exception:
    log("error")  # No context

# ✅ GOOD: Specific exceptions
try:
    execute_trade()
except InsufficientFundsError as e:
    log(f"Insufficient funds: {e.required} needed, {e.available} available")
    reduce_position_size()
except NetworkError as e:
    log(f"Network error: {e}")
    retry_with_backoff()
```

### 8. Resource Leaks

```python
# ❌ BAD: File not closed on error
f = open("trades.json", "w")
json.dump(trades, f)  # If this fails, file stays open
f.close()

# ✅ GOOD: Context manager
with open("trades.json", "w") as f:
    json.dump(trades, f)

# ❌ BAD: Database connection leak
conn = sqlite3.connect("results.db")
cursor = conn.cursor()
cursor.execute(query)  # If fails, connection not closed

# ✅ GOOD: Context manager
with sqlite3.connect("results.db") as conn:
    cursor = conn.cursor()
    cursor.execute(query)
```

## Review Automation

### Pre-Commit Hook

```python
#!/usr/bin/env python
"""Pre-commit hook for trading code review."""

import ast
import sys
from pathlib import Path

class TradingCodeChecker(ast.NodeVisitor):
    def __init__(self):
        self.issues = []

    def visit_Subscript(self, node):
        """Check for potential look-ahead bias in slicing."""
        if isinstance(node.slice, ast.Slice):
            # Check for iloc[i:i+n] patterns
            if node.slice.upper and isinstance(node.slice.upper, ast.BinOp):
                if isinstance(node.slice.upper.op, ast.Add):
                    self.issues.append({
                        "line": node.lineno,
                        "type": "POTENTIAL_LOOKAHEAD",
                        "msg": "Forward-looking slice detected - verify no look-ahead bias"
                    })
        self.generic_visit(node)

    def visit_Compare(self, node):
        """Check for float equality."""
        for op in node.ops:
            if isinstance(op, (ast.Eq, ast.NotEq)):
                # Check if comparing floats
                self.issues.append({
                    "line": node.lineno,
                    "type": "FLOAT_EQUALITY",
                    "msg": "Direct equality comparison - consider using tolerance"
                })
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Check for bare except."""
        if node.type is None:
            self.issues.append({
                "line": node.lineno,
                "type": "BARE_EXCEPT",
                "msg": "Bare except clause - specify exception type"
            })
        self.generic_visit(node)

def check_file(filepath: Path) -> list:
    """Run checks on a Python file."""
    with open(filepath) as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError as e:
            return [{"line": e.lineno, "type": "SYNTAX_ERROR", "msg": str(e)}]

    checker = TradingCodeChecker()
    checker.visit(tree)
    return checker.issues

def main():
    """Run on staged files."""
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True
    )

    files = [f for f in result.stdout.strip().split("\n")
             if f.endswith(".py") and f.startswith("src/")]

    all_issues = []
    for filepath in files:
        issues = check_file(Path(filepath))
        for issue in issues:
            all_issues.append(f"{filepath}:{issue['line']}: [{issue['type']}] {issue['msg']}")

    if all_issues:
        print("Trading code review issues found:")
        for issue in all_issues:
            print(f"  {issue}")
        sys.exit(1)

    print("Trading code review passed")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

## Code Review Template

```markdown
## Code Review: [PR Title]

### Trading Logic
- [ ] No look-ahead bias in feature calculation
- [ ] Entry/exit on correct bars (not signal bar)
- [ ] SL/TP calculated from entry-time data only
- [ ] Slippage and fees accounted for

### Risk Management
- [ ] Position size limits enforced
- [ ] Daily/weekly loss limits checked
- [ ] No divide-by-zero in risk calculations
- [ ] Negative position sizes impossible

### Data Handling
- [ ] Timezone handling consistent
- [ ] NaN/missing data handled
- [ ] Data sorted chronologically
- [ ] No duplicate timestamps

### Error Handling
- [ ] Specific exceptions caught
- [ ] Errors logged with context
- [ ] Graceful degradation on API failures
- [ ] No silent exception swallowing

### Testing
- [ ] Unit tests for new logic
- [ ] Edge cases covered
- [ ] Mock external APIs
- [ ] Integration test if needed

### Performance
- [ ] No N+1 queries
- [ ] Large data handled efficiently
- [ ] Memory usage reasonable

### Security
- [ ] No hardcoded API keys
- [ ] Sensitive data not logged
- [ ] Input validation on external data
```

## Quick Review Commands

```bash
# Find potential look-ahead bias
grep -rn "iloc\[.*:.*\+" src/

# Find bare excepts
grep -rn "except:" src/

# Find hardcoded credentials
grep -rn "api_key\s*=" src/
grep -rn "secret\s*=" src/

# Find TODO/FIXME
grep -rn "TODO\|FIXME\|HACK\|XXX" src/

# Find print statements (should use logging)
grep -rn "print(" src/ --include="*.py"
```
