# Code Audit: Custom vs Library Code

## Summary
- **Total files audited**: 25+
- **Custom implementations found**: 8
- **Already using libraries**: 6 (excellent!)
- **Recommended replacements**: 5

## Overall Assessment

The codebase is **well-designed** with good library usage in critical areas:
- Technical indicators use `ta` library (battle-tested)
- Data validation uses `pydantic` (industry standard)
- ML uses `xgboost` (proven library)

However, there are several opportunities to reduce custom code and improve reliability.

---

## Findings

### 1. Technical Indicators - ALREADY USING LIBRARY

| File | Implementation | Status | Notes |
|------|----------------|--------|-------|
| `src/utils/indicators.py` | Uses `ta` library | **GOOD** | Wraps ta.volatility, ta.momentum, ta.trend |
| `src/features/calculator.py` | Uses `ta` library | **GOOD** | Direct ta library usage |
| `src/features/library.py` | Uses wrappers | **GOOD** | Imports from utils/indicators.py |

**Verdict**: No changes needed. The `ta` library is a solid choice (3.7k+ stars, TradingView parity).

**Note**: The codebase uses `ta` instead of `pandas-ta`. Both are valid:
- `ta`: More comprehensive, class-based API
- `pandas-ta`: DataFrame method extensions

Current choice is fine.

---

### 2. Swing Point Detection - CUSTOM (Appropriate)

| File | Custom Code | Status | Notes |
|------|-------------|--------|-------|
| `src/detection/swing.py` | SwingDetector class | **APPROPRIATE** | QML-specific logic |
| `src/detection/structure.py` | StructureAnalyzer class | **APPROPRIATE** | Market structure analysis |

**Analysis**: The swing detection is intentionally custom because it includes:
- ATR-adaptive significance thresholds
- Confirmation-based validation
- Timeframe-specific multipliers
- QML pattern geometry requirements

**Could consider**: `scipy.signal.argrelextrema` as base algorithm, but the current implementation is well-documented and tailored to QML needs.

**Verdict**: Keep custom implementation. It's appropriate for this domain.

---

### 3. Statistical Tests - CUSTOM (Should Improve)

| File | Custom Code | Recommended Library | Priority |
|------|-------------|---------------------|----------|
| `src/validation/permutation.py` | Custom shuffle test | scipy.stats.permutation_test | MEDIUM |
| `src/validation/bootstrap.py` | Custom resampling | scipy.stats.bootstrap | MEDIUM |
| `src/validation/monte_carlo.py` | Custom Monte Carlo | Keep custom | LOW |

**Details**:

#### Permutation Test (lines 120-135)
```python
# Current (custom)
for _ in range(n_perms):
    shuffled = np.random.permutation(returns)
    perm_sharpes.append(self._calculate_sharpe(shuffled))
p_value = np.mean(perm_sharpes >= real_sharpe)
```

**Could use**: `scipy.stats.permutation_test` (added in scipy 1.8.0)

#### Bootstrap Resampling (lines 113-127)
```python
# Current (custom)
for _ in range(n_resamples):
    sample = np.random.choice(returns, size=len(returns), replace=True)
    bootstrap_sharpes.append(self._calculate_sharpe(sample))
```

**Could use**: `scipy.stats.bootstrap` (added in scipy 1.7.0)

**Verdict**: Current implementations work but could benefit from scipy for:
- Statistical rigor (proper CI calculation methods)
- BCa confidence intervals
- Parallel processing support

---

### 4. Data Validation - ALREADY USING LIBRARY

| File | Implementation | Status | Notes |
|------|----------------|--------|-------|
| `src/data/models.py` | pydantic BaseModel | **GOOD** | Full pydantic v2 usage |
| `src/data/schemas.py` | dataclasses | **GOOD** | Appropriate for simple DTOs |

**Verdict**: Excellent. Using pydantic for validation and dataclasses for simple structures.

---

### 5. Time Series Operations - MOSTLY GOOD

| File | Operation | Status | Notes |
|------|-----------|--------|-------|
| `src/features/library.py` | Rolling calculations | **GOOD** | Uses pd.Series.rolling() |
| `src/utils/indicators.py:337` | Rolling percentile | **CUSTOM** | Could use pandas |

**One improvement**:
```python
# Current (custom for-loop at line 337)
for i in range(lookback, len(atr)):
    window = atr_pct[i - lookback:i]
    percentiles[i] = (np.sum(window < atr_pct[i]) / lookback) * 100

# Better (vectorized)
percentiles = pd.Series(atr_pct).rolling(lookback).apply(
    lambda x: scipy.stats.percentileofscore(x, x.iloc[-1])
)
```

**Priority**: LOW - works correctly, performance impact minimal

---

### 6. ML Pipeline - NEEDS IMPROVEMENT

| File | Custom Code | Recommended Library | Priority |
|------|-------------|---------------------|----------|
| `src/ml/predictor.py:196-204` | Manual train/test split | sklearn.model_selection.train_test_split | **HIGH** |
| `src/ml/predictor.py:239-248` | Custom AUC calculation | sklearn.metrics.roc_auc_score | **HIGH** |

**Details**:

#### Train/Test Split (lines 196-204)
```python
# Current (custom)
n_test = int(len(X) * test_size)
indices = np.random.permutation(len(X))
test_idx = indices[:n_test]
train_idx = indices[n_test:]
```

**Replace with**:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
```

#### AUC Calculation (lines 239-248)
```python
# Current (approximation)
pos_probs = y_prob[y_test == 1]
neg_probs = y_prob[y_test == 0]
auc = np.mean([p > n for p in pos_probs for n in neg_probs])
```

**Replace with**:
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_prob)
```

**Why this matters**: The current AUC calculation is O(n²) and an approximation. sklearn's implementation is exact and O(n log n).

---

### 7. Backtesting Engine - CUSTOM (Appropriate)

| File | Implementation | Status | Notes |
|------|----------------|--------|-------|
| `src/backtest/engine.py` | BacktestEngine class | **APPROPRIATE** | QML-specific |

**Analysis**: The backtest engine is custom but appropriate because:
- Tightly integrated with QML pattern detection
- Uses specific trade management rules
- Calculates QML-specific metrics

**Could consider**: `vectorbt` for performance-critical scenarios, but:
- Current implementation is readable and maintainable
- vectorbt has a steep learning curve
- Integration with QML patterns would require significant refactoring

**Verdict**: Keep custom. Consider vectorbt only if performance becomes an issue with large datasets (>100k trades).

---

## Recommended Actions

### 1. HIGH PRIORITY - COMPLETED ✅

| Item | File | Action | Status |
|------|------|--------|--------|
| Train/test split | `src/ml/predictor.py` | Use sklearn.train_test_split | **DONE** |
| AUC calculation | `src/ml/predictor.py` | Use sklearn.roc_auc_score | **DONE** |

*Fixed on 2026-01-20: Added sklearn imports and replaced custom implementations.*

### 2. MEDIUM PRIORITY (Replace Soon)

| Item | File | Action | Effort |
|------|------|--------|--------|
| Permutation test | `src/validation/permutation.py` | Use scipy.stats.permutation_test | 30 min |
| Bootstrap CI | `src/validation/bootstrap.py` | Use scipy.stats.bootstrap | 30 min |

### 3. LOW PRIORITY (Consider Later)

| Item | File | Action | Effort |
|------|------|--------|--------|
| Rolling percentile | `src/utils/indicators.py` | Vectorize with pandas | 15 min |
| Vectorized backtest | `src/backtest/engine.py` | Consider vectorbt | 2-4 hours |

---

## Code Snippets to Replace

### Example 1: Train/Test Split

**BEFORE** (custom in `src/ml/predictor.py:196-204`):
```python
# Simple train/test split
n_test = int(len(X) * test_size)
indices = np.random.permutation(len(X))
test_idx = indices[:n_test]
train_idx = indices[n_test:]

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]
```

**AFTER** (sklearn):
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
```

Benefits:
- Stratified sampling preserves class distribution
- Reproducible with random_state
- Well-tested edge case handling

---

### Example 2: AUC Score

**BEFORE** (custom approximation in `src/ml/predictor.py:239-248`):
```python
# AUC calculation (approximation)
if len(np.unique(y_test)) > 1:
    pos_probs = y_prob[y_test == 1]
    neg_probs = y_prob[y_test == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        auc = np.mean([p > n for p in pos_probs for n in neg_probs])
    else:
        auc = 0.5
else:
    auc = 0.5
```

**AFTER** (sklearn):
```python
from sklearn.metrics import roc_auc_score

try:
    auc = roc_auc_score(y_test, y_prob)
except ValueError:
    # Only one class in y_test
    auc = 0.5
```

Benefits:
- Exact calculation (not approximation)
- O(n log n) vs O(n²)
- Handles edge cases properly

---

### Example 3: Bootstrap Confidence Intervals

**BEFORE** (custom in `src/validation/bootstrap.py`):
```python
# Custom bootstrap
bootstrap_sharpes = []
for _ in range(n_resamples):
    sample = np.random.choice(returns, size=len(returns), replace=True)
    bootstrap_sharpes.append(self._calculate_sharpe(sample))

# Percentile CI
lower = np.percentile(bootstrap_sharpes, alpha * 100)
upper = np.percentile(bootstrap_sharpes, (1 - alpha) * 100)
```

**AFTER** (scipy):
```python
from scipy import stats

def sharpe_statistic(x, axis):
    return np.mean(x, axis=axis) / np.std(x, axis=axis)

result = stats.bootstrap(
    (returns,),
    sharpe_statistic,
    n_resamples=n_resamples,
    confidence_level=conf_level,
    method='BCa'  # Bias-corrected and accelerated
)
sharpe_ci = (result.confidence_interval.low, result.confidence_interval.high)
```

Benefits:
- BCa method (more accurate than percentile)
- Parallel processing support
- Standard error estimation

---

## Libraries Reference

| Purpose | Current | Recommended | Install |
|---------|---------|-------------|---------|
| Technical indicators | `ta` | Keep `ta` | Already installed |
| ML pipeline | Custom | `sklearn` | `pip install scikit-learn` |
| Statistical tests | Custom | `scipy.stats` | Already installed |
| Data validation | `pydantic` | Keep `pydantic` | Already installed |
| Backtesting | Custom | Keep custom (consider `vectorbt` later) | `pip install vectorbt` |

---

## Do NOT Replace

The following custom code is appropriate and should be kept:

1. **QML Pattern Geometry** (`src/detection/v2_atr.py`)
   - Pattern-specific, no library exists
   - Core business logic

2. **Swing Point Detection** (`src/detection/swing.py`)
   - ATR-adaptive thresholds
   - QML-specific validation rules

3. **Market Structure Analysis** (`src/detection/structure.py`)
   - HH/HL/LH/LL classification
   - Domain-specific logic

4. **Feature Calculator Tier 1** (`src/features/calculator.py`)
   - Pattern geometry features
   - QML-specific calculations

5. **Dashboard UI Components** (`qml/dashboard/`)
   - Streamlit-specific
   - Visual presentation logic

6. **Monte Carlo Simulator** (`src/validation/monte_carlo.py`)
   - Custom risk metrics (VaR, CVaR, Risk of Ruin)
   - Trading-specific implementation

---

## Next Steps

1. **Immediate**: Add sklearn imports to `src/ml/predictor.py`
2. **This Week**: Refactor validation modules to use scipy.stats
3. **Later**: Consider vectorbt if backtesting performance becomes an issue

**Total estimated effort**: 2-3 hours for high/medium priority items
