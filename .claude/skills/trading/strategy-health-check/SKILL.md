# Strategy Health Check Skill

Pre-live diagnostics and validation suite for trading strategies.

## When to Use
- Before deploying to forward testing
- After parameter changes
- Periodic strategy health monitoring
- QML Phase 9.5 validation

## Health Check Suite

### Quick Diagnostic

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class HealthStatus(Enum):
    PASS = "✅ PASS"
    WARN = "⚠️ WARN"
    FAIL = "❌ FAIL"

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    value: float
    threshold: float
    message: str

def quick_health_check(trades: list) -> List[HealthCheckResult]:
    """Fast diagnostic checks before trading."""
    results = []

    # Check 1: Minimum trade count
    n_trades = len(trades)
    results.append(HealthCheckResult(
        name="Trade Count",
        status=HealthStatus.PASS if n_trades >= 50 else HealthStatus.FAIL,
        value=n_trades,
        threshold=50,
        message=f"{n_trades} trades (min 50 required)"
    ))

    # Check 2: Win rate sanity
    wr = sum(1 for t in trades if t.pnl > 0) / n_trades if n_trades > 0 else 0
    status = HealthStatus.PASS if 0.35 <= wr <= 0.75 else HealthStatus.WARN
    results.append(HealthCheckResult(
        name="Win Rate",
        status=status,
        value=wr,
        threshold=0.35,
        message=f"{wr:.1%} (expect 35-75%)"
    ))

    # Check 3: Profit factor
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    status = HealthStatus.PASS if pf >= 1.5 else (HealthStatus.WARN if pf >= 1.0 else HealthStatus.FAIL)
    results.append(HealthCheckResult(
        name="Profit Factor",
        status=status,
        value=pf,
        threshold=1.5,
        message=f"{pf:.2f} (target >= 1.5)"
    ))

    # Check 4: Consecutive losses
    max_consec = calculate_max_consecutive_losses(trades)
    status = HealthStatus.PASS if max_consec <= 8 else HealthStatus.WARN
    results.append(HealthCheckResult(
        name="Max Consecutive Losses",
        status=status,
        value=max_consec,
        threshold=8,
        message=f"{max_consec} (warn if > 8)"
    ))

    # Check 5: Average win vs loss
    avg_win = sum(t.pnl_r for t in trades if t.pnl > 0) / len([t for t in trades if t.pnl > 0]) if any(t.pnl > 0 for t in trades) else 0
    avg_loss = abs(sum(t.pnl_r for t in trades if t.pnl < 0)) / len([t for t in trades if t.pnl < 0]) if any(t.pnl < 0 for t in trades) else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    status = HealthStatus.PASS if rr >= 2.0 else (HealthStatus.WARN if rr >= 1.5 else HealthStatus.FAIL)
    results.append(HealthCheckResult(
        name="Reward:Risk",
        status=status,
        value=rr,
        threshold=2.0,
        message=f"{rr:.2f}:1 (target >= 2:1)"
    ))

    return results

def calculate_max_consecutive_losses(trades):
    max_streak = current_streak = 0
    for t in trades:
        if t.pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak
```

### Full Validation Suite

```python
class StrategyHealthChecker:
    """Comprehensive strategy health validation."""

    def __init__(self, trades: list, equity_curve=None):
        self.trades = trades
        self.equity_curve = equity_curve
        self.results: Dict[str, HealthCheckResult] = {}

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run complete validation suite."""
        self.check_statistical_significance()
        self.check_robustness()
        self.check_risk_metrics()
        self.check_trade_quality()
        self.check_data_integrity()
        return self.results

    def check_statistical_significance(self):
        """Permutation test for edge validity."""
        from src.validation import PermutationTest

        perm = PermutationTest(n_iterations=1000)
        result = perm.validate(self.trades)

        self.results["Statistical Significance"] = HealthCheckResult(
            name="Statistical Significance",
            status=HealthStatus.PASS if result.p_value < 0.05 else HealthStatus.FAIL,
            value=result.p_value,
            threshold=0.05,
            message=f"p-value: {result.p_value:.4f} (need < 0.05)"
        )

    def check_robustness(self):
        """Parameter sensitivity check."""
        from scripts.phase95_parameter_sensitivity import run_sensitivity

        sensitivity = run_sensitivity(self.trades, variation=0.2)
        pf_range = sensitivity['pf_max'] - sensitivity['pf_min']

        self.results["Parameter Robustness"] = HealthCheckResult(
            name="Parameter Robustness",
            status=HealthStatus.PASS if pf_range < 1.5 else HealthStatus.WARN,
            value=pf_range,
            threshold=1.5,
            message=f"PF range: {pf_range:.2f} under ±20% param changes"
        )

    def check_risk_metrics(self):
        """Drawdown and risk analysis."""
        if self.equity_curve is None:
            return

        from src.quant.metrics import max_drawdown, risk_of_ruin

        dd = max_drawdown(self.equity_curve)
        wr = sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)
        avg_win = sum(t.pnl_r for t in self.trades if t.pnl > 0) / max(1, sum(1 for t in self.trades if t.pnl > 0))
        avg_loss = abs(sum(t.pnl_r for t in self.trades if t.pnl < 0)) / max(1, sum(1 for t in self.trades if t.pnl < 0))

        ror = risk_of_ruin(wr, avg_win, avg_loss, risk_per_trade=0.01)

        self.results["Max Drawdown"] = HealthCheckResult(
            name="Max Drawdown",
            status=HealthStatus.PASS if dd['max_dd_pct'] < 20 else HealthStatus.WARN,
            value=dd['max_dd_pct'],
            threshold=20,
            message=f"{dd['max_dd_pct']:.1f}% (target < 20%)"
        )

        self.results["Risk of Ruin"] = HealthCheckResult(
            name="Risk of Ruin",
            status=HealthStatus.PASS if ror < 0.01 else HealthStatus.FAIL,
            value=ror,
            threshold=0.01,
            message=f"{ror:.2%} (need < 1%)"
        )

    def check_trade_quality(self):
        """Analyze trade distribution and quality."""
        # Dust wins check (wins < 0.5R)
        dust_wins = [t for t in self.trades if 0 < t.pnl_r < 0.5]
        dust_pct = len(dust_wins) / len(self.trades) if self.trades else 0

        self.results["Dust Wins"] = HealthCheckResult(
            name="Dust Wins",
            status=HealthStatus.PASS if dust_pct < 0.1 else HealthStatus.WARN,
            value=dust_pct,
            threshold=0.1,
            message=f"{dust_pct:.1%} wins < 0.5R (target < 10%)"
        )

        # Direction balance
        longs = [t for t in self.trades if t.direction == "LONG"]
        shorts = [t for t in self.trades if t.direction == "SHORT"]
        long_wr = sum(1 for t in longs if t.pnl > 0) / len(longs) if longs else 0
        short_wr = sum(1 for t in shorts if t.pnl > 0) / len(shorts) if shorts else 0
        wr_diff = abs(long_wr - short_wr)

        self.results["Direction Balance"] = HealthCheckResult(
            name="Direction Balance",
            status=HealthStatus.PASS if wr_diff < 0.15 else HealthStatus.WARN,
            value=wr_diff,
            threshold=0.15,
            message=f"Long WR: {long_wr:.1%}, Short WR: {short_wr:.1%}"
        )

    def check_data_integrity(self):
        """Verify data quality."""
        # Check for duplicate timestamps
        timestamps = [t.entry_time for t in self.trades]
        duplicates = len(timestamps) - len(set(timestamps))

        self.results["No Duplicate Entries"] = HealthCheckResult(
            name="No Duplicate Entries",
            status=HealthStatus.PASS if duplicates == 0 else HealthStatus.FAIL,
            value=duplicates,
            threshold=0,
            message=f"{duplicates} duplicate entry times"
        )

        # Check chronological order
        is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

        self.results["Chronological Order"] = HealthCheckResult(
            name="Chronological Order",
            status=HealthStatus.PASS if is_sorted else HealthStatus.FAIL,
            value=1 if is_sorted else 0,
            threshold=1,
            message="Trades in time order" if is_sorted else "Trades out of order!"
        )

    def summary(self) -> str:
        """Generate health report summary."""
        passed = sum(1 for r in self.results.values() if r.status == HealthStatus.PASS)
        warned = sum(1 for r in self.results.values() if r.status == HealthStatus.WARN)
        failed = sum(1 for r in self.results.values() if r.status == HealthStatus.FAIL)

        lines = [
            "=" * 50,
            "STRATEGY HEALTH CHECK REPORT",
            "=" * 50,
            f"Total Checks: {len(self.results)}",
            f"Passed: {passed} | Warnings: {warned} | Failed: {failed}",
            "-" * 50,
        ]

        for name, result in self.results.items():
            lines.append(f"{result.status.value} {name}: {result.message}")

        lines.append("=" * 50)

        overall = "READY FOR LIVE" if failed == 0 and warned <= 2 else "NEEDS REVIEW"
        lines.append(f"VERDICT: {overall}")

        return "\n".join(lines)
```

## Pre-Live Checklist

```python
def pre_live_checklist(trades, equity_curve):
    """Final checks before live trading."""
    print("🔍 PRE-LIVE VALIDATION CHECKLIST")
    print("=" * 40)

    checks = [
        ("1. Statistical edge verified (p < 0.05)?", lambda: permutation_p_value(trades) < 0.05),
        ("2. Walk-forward consistency > 80%?", lambda: walk_forward_consistency(trades) > 0.8),
        ("3. Monte Carlo 95% DD < 20%?", lambda: monte_carlo_dd_95(trades) < 0.20),
        ("4. No look-ahead bias confirmed?", lambda: no_lookahead_bias(trades)),
        ("5. Risk of ruin < 1%?", lambda: risk_of_ruin_check(trades) < 0.01),
        ("6. Min 50 out-of-sample trades?", lambda: len(trades) >= 50),
        ("7. PF > 1.5 on OOS data?", lambda: calculate_pf(trades) > 1.5),
        ("8. API connectivity tested?", lambda: test_api_connection()),
        ("9. Kill switch configured?", lambda: verify_kill_switch()),
        ("10. Risk limits set?", lambda: verify_risk_limits()),
    ]

    all_passed = True
    for check_name, check_func in checks:
        try:
            passed = check_func()
            status = "✅" if passed else "❌"
            if not passed:
                all_passed = False
        except Exception as e:
            status = "⚠️"
            all_passed = False
        print(f"{status} {check_name}")

    print("=" * 40)
    if all_passed:
        print("🚀 ALL CHECKS PASSED - Ready for live trading")
    else:
        print("🛑 CHECKS FAILED - Do not proceed to live")

    return all_passed
```

## Integration with QML

```bash
# Run health check
python scripts/run_phase95_validation.py

# Quick check before trading session
python -c "
from scripts.run_phase95_validation import quick_validate
quick_validate()
"
```
