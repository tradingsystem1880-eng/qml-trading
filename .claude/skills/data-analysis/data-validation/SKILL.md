# Data Validation Skill

Quality scoring and validation for financial data.

## When to Use
- Validating OHLCV data quality
- Checking parquet file integrity
- Detecting data anomalies
- Pre-backtest data verification

## Data Quality Checks

### Comprehensive Validator

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'

@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    total_rows: int
    date_range: tuple
    quality_score: float
    quality_level: QualityLevel
    checks: List[ValidationResult]

def validate_ohlcv(df: pd.DataFrame, symbol: str = "UNKNOWN", timeframe: str = "4h") -> DataQualityReport:
    """
    Comprehensive OHLCV data validation.

    Returns quality score 0-100 and detailed check results.
    """
    checks = []
    score_deductions = 0

    # Check 1: Required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        checks.append(ValidationResult(
            "required_columns", False,
            f"Missing columns: {missing}", "error"
        ))
        score_deductions += 50
    else:
        checks.append(ValidationResult(
            "required_columns", True,
            "All required columns present", "info"
        ))

    # Check 2: Index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        checks.append(ValidationResult(
            "datetime_index", False,
            "Index is not DatetimeIndex", "error"
        ))
        score_deductions += 30
    else:
        checks.append(ValidationResult(
            "datetime_index", True,
            "Index is valid DatetimeIndex", "info"
        ))

    # Check 3: Sorted chronologically
    if not df.index.is_monotonic_increasing:
        checks.append(ValidationResult(
            "sorted_index", False,
            "Index is not sorted ascending", "error"
        ))
        score_deductions += 20
    else:
        checks.append(ValidationResult(
            "sorted_index", True,
            "Index is properly sorted", "info"
        ))

    # Check 4: No duplicate timestamps
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        checks.append(ValidationResult(
            "no_duplicates", False,
            f"Found {duplicates} duplicate timestamps", "error"
        ))
        score_deductions += 15
    else:
        checks.append(ValidationResult(
            "no_duplicates", True,
            "No duplicate timestamps", "info"
        ))

    # Check 5: OHLC relationship (high >= open/close >= low)
    ohlc_valid = (
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    )
    ohlc_violations = (~ohlc_valid).sum()
    if ohlc_violations > 0:
        checks.append(ValidationResult(
            "ohlc_relationship", False,
            f"{ohlc_violations} rows violate OHLC relationship", "warning"
        ))
        score_deductions += min(10, ohlc_violations * 0.1)
    else:
        checks.append(ValidationResult(
            "ohlc_relationship", True,
            "All OHLC relationships valid", "info"
        ))

    # Check 6: No NaN values
    nan_counts = df[required].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        checks.append(ValidationResult(
            "no_nan_values", False,
            f"Found NaN values: {nan_counts.to_dict()}", "warning"
        ))
        score_deductions += min(15, total_nans * 0.5)
    else:
        checks.append(ValidationResult(
            "no_nan_values", True,
            "No NaN values in OHLCV", "info"
        ))

    # Check 7: Positive prices
    negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
    if negative_prices > 0:
        checks.append(ValidationResult(
            "positive_prices", False,
            f"{negative_prices} non-positive price values", "error"
        ))
        score_deductions += 25
    else:
        checks.append(ValidationResult(
            "positive_prices", True,
            "All prices positive", "info"
        ))

    # Check 8: Non-negative volume
    negative_volume = (df['volume'] < 0).sum()
    if negative_volume > 0:
        checks.append(ValidationResult(
            "non_negative_volume", False,
            f"{negative_volume} negative volume values", "warning"
        ))
        score_deductions += 5

    # Check 9: Data gaps
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        freq_map = {'1h': 'h', '4h': '4h', '1d': 'D'}
        expected_freq = freq_map.get(timeframe, '4h')

        try:
            expected_index = pd.date_range(df.index[0], df.index[-1], freq=expected_freq)
            gap_count = len(expected_index) - len(df)
            gap_pct = gap_count / len(expected_index) * 100

            if gap_pct > 5:
                checks.append(ValidationResult(
                    "data_continuity", False,
                    f"{gap_pct:.1f}% missing bars ({gap_count} gaps)", "warning"
                ))
                score_deductions += min(15, gap_pct)
            else:
                checks.append(ValidationResult(
                    "data_continuity", True,
                    f"Data continuity good ({gap_pct:.1f}% gaps)", "info"
                ))
        except:
            pass

    # Check 10: Price spike detection
    returns = df['close'].pct_change().abs()
    spikes = (returns > 0.5).sum()  # >50% move in one bar
    if spikes > 0:
        checks.append(ValidationResult(
            "no_price_spikes", False,
            f"{spikes} extreme price spikes (>50%)", "warning"
        ))
        score_deductions += min(10, spikes * 2)

    # Calculate final score
    quality_score = max(0, 100 - score_deductions)

    # Determine quality level
    if quality_score >= 95:
        level = QualityLevel.EXCELLENT
    elif quality_score >= 80:
        level = QualityLevel.GOOD
    elif quality_score >= 60:
        level = QualityLevel.ACCEPTABLE
    elif quality_score >= 40:
        level = QualityLevel.POOR
    else:
        level = QualityLevel.INVALID

    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        total_rows=len(df),
        date_range=(str(df.index[0]), str(df.index[-1])),
        quality_score=quality_score,
        quality_level=level,
        checks=checks
    )
```

### Quick Validation

```python
def quick_validate(df: pd.DataFrame) -> bool:
    """Quick pass/fail validation for OHLCV data."""
    try:
        assert len(df) > 0, "Empty DataFrame"
        assert 'close' in df.columns, "Missing close column"
        assert isinstance(df.index, pd.DatetimeIndex), "Not datetime index"
        assert df.index.is_monotonic_increasing, "Not sorted"
        assert not df['close'].isna().any(), "NaN in close"
        assert (df['close'] > 0).all(), "Non-positive prices"
        return True
    except AssertionError as e:
        print(f"Validation failed: {e}")
        return False
```

## Parquet File Validation

```python
from pathlib import Path

def validate_parquet_file(filepath: Path) -> DataQualityReport:
    """Validate a parquet file containing OHLCV data."""
    # Extract symbol and timeframe from filename
    # Expected format: BTCUSDT_4h.parquet
    stem = filepath.stem
    parts = stem.split('_')
    symbol = parts[0] if len(parts) > 0 else "UNKNOWN"
    timeframe = parts[1] if len(parts) > 1 else "4h"

    try:
        df = pd.read_parquet(filepath)
        report = validate_ohlcv(df, symbol, timeframe)
    except Exception as e:
        report = DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_rows=0,
            date_range=("N/A", "N/A"),
            quality_score=0,
            quality_level=QualityLevel.INVALID,
            checks=[ValidationResult("file_read", False, str(e), "error")]
        )

    return report

def validate_data_directory(data_dir: str = "data/") -> pd.DataFrame:
    """Validate all parquet files in a directory."""
    from pathlib import Path

    results = []
    for filepath in Path(data_dir).glob("*.parquet"):
        report = validate_parquet_file(filepath)
        results.append({
            'file': filepath.name,
            'symbol': report.symbol,
            'timeframe': report.timeframe,
            'rows': report.total_rows,
            'score': report.quality_score,
            'level': report.quality_level.value,
            'errors': sum(1 for c in report.checks if c.severity == 'error'),
            'warnings': sum(1 for c in report.checks if c.severity == 'warning')
        })

    df = pd.DataFrame(results).sort_values('score')
    return df
```

## Trade Data Validation

```python
def validate_trades(trades: list) -> List[ValidationResult]:
    """Validate trade data integrity."""
    checks = []

    if not trades:
        return [ValidationResult("has_trades", False, "No trades to validate", "error")]

    # Check 1: Required fields
    required_fields = ['entry_time', 'exit_time', 'direction', 'pnl']
    for trade in trades[:5]:  # Check first 5
        missing = [f for f in required_fields if not hasattr(trade, f)]
        if missing:
            checks.append(ValidationResult(
                "trade_fields", False,
                f"Trade missing fields: {missing}", "error"
            ))
            break
    else:
        checks.append(ValidationResult(
            "trade_fields", True, "All trades have required fields", "info"
        ))

    # Check 2: Valid directions
    directions = set(t.direction for t in trades)
    invalid = directions - {'LONG', 'SHORT'}
    if invalid:
        checks.append(ValidationResult(
            "valid_directions", False,
            f"Invalid directions found: {invalid}", "error"
        ))
    else:
        checks.append(ValidationResult(
            "valid_directions", True, "All directions valid", "info"
        ))

    # Check 3: Entry before exit
    wrong_order = sum(1 for t in trades if t.entry_time >= t.exit_time)
    if wrong_order > 0:
        checks.append(ValidationResult(
            "entry_before_exit", False,
            f"{wrong_order} trades with entry >= exit time", "error"
        ))

    # Check 4: Chronological order
    entry_times = [t.entry_time for t in trades]
    is_sorted = all(entry_times[i] <= entry_times[i+1] for i in range(len(entry_times)-1))
    if not is_sorted:
        checks.append(ValidationResult(
            "chronological_order", False,
            "Trades not in chronological order", "warning"
        ))

    # Check 5: Reasonable PnL values
    extreme_pnl = sum(1 for t in trades if abs(t.pnl_r) > 20)
    if extreme_pnl > 0:
        checks.append(ValidationResult(
            "reasonable_pnl", False,
            f"{extreme_pnl} trades with |PnL| > 20R", "warning"
        ))

    return checks
```

## Usage

```bash
# Validate single file
python -c "
from src.data.validation import validate_parquet_file
from pathlib import Path
report = validate_parquet_file(Path('data/BTCUSDT_4h.parquet'))
print(f'Score: {report.quality_score}/100 ({report.quality_level.value})')
for check in report.checks:
    status = '✅' if check.passed else '❌'
    print(f'{status} {check.check_name}: {check.message}')
"

# Validate all data files
python -c "
from src.data.validation import validate_data_directory
df = validate_data_directory('data/')
print(df.to_string())
"
```
