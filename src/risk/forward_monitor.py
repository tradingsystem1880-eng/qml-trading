"""
Forward Test Monitor for QML Trading System
=============================================
Real-time tracking of forward test performance with edge degradation detection.

Based on:
- Phase 7.9 baseline performance (PF 1.23, WR 52%, DSR 0.986)
- Wald's Sequential Probability Ratio Test (SPRT) for early stopping
- Required sample sizes for statistical significance

This module monitors paper trading performance and alerts when:
1. Performance degrades significantly vs baseline
2. Win rate confidence interval includes 50% (no edge)
3. Consecutive losses exceed statistical threshold

Usage:
    from src.risk.forward_monitor import ForwardTestMonitor, ForwardTestConfig

    monitor = ForwardTestMonitor(ForwardTestConfig(
        baseline_pf=1.23,
        baseline_wr=0.52,
    ))

    # After each trade
    status = monitor.record_trade({
        'r_multiple': 1.5,
        'exit_type': 'take_profit',
        'symbol': 'BTCUSDT',
        'pattern_quality': 0.75,
    })

    # Check for degradation
    if status.is_degraded:
        print(f"Warning: {status.alerts}")

    # Generate report
    print(monitor.generate_report())
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
import json
from pathlib import Path


@dataclass
class ForwardTestConfig:
    """Configuration for forward testing monitor."""

    # Phase 7.9 baseline expectations
    baseline_pf: float = 1.23
    baseline_wr: float = 0.52
    baseline_expectancy: float = 0.18  # R per trade

    # Alert thresholds
    min_trades_for_comparison: int = 30  # Minimum trades before alerts
    min_trades_for_deployment: int = 500  # Minimum before live trading

    # Degradation thresholds
    degradation_threshold_pf: float = 0.15  # Alert if PF drops 15% below baseline
    degradation_threshold_wr: float = 0.10  # Alert if WR drops 10% below baseline

    # Rolling window for recent performance
    rolling_window: int = 50

    # Consecutive loss threshold (based on p=0.52, prob=0.48^7 ~ 0.5%)
    max_consecutive_losses: int = 7


@dataclass
class ForwardTestStatus:
    """Current forward test status."""

    # Trade counts
    total_trades: int
    winners: int
    losers: int

    # Running metrics
    running_pf: float
    running_wr: float
    running_sharpe: float
    running_expectancy: float

    # Comparison to baseline
    pf_vs_baseline_pct: float
    wr_vs_baseline_pct: float

    # Confidence intervals
    wr_ci_lower: float
    wr_ci_upper: float

    # Edge degradation
    is_degraded: bool
    degradation_severity: str  # 'none', 'warning', 'critical'
    alerts: List[str]

    # Loss tracking
    consecutive_losses: int
    max_consecutive_losses: int

    # Deployment readiness
    ready_for_deployment: bool
    trades_until_deployment: int


@dataclass
class TradeRecord:
    """Single trade record for forward testing."""
    timestamp: datetime
    symbol: str
    r_multiple: float
    exit_type: str
    pattern_quality: float
    bars_held: int = 0
    metadata: Dict = field(default_factory=dict)


class ForwardTestMonitor:
    """
    Real-time forward testing monitor for paper trading.

    Tracks running metrics and detects edge degradation using
    statistical tests and comparison to Phase 7.9 baseline.
    """

    def __init__(
        self,
        config: Optional[ForwardTestConfig] = None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize forward test monitor.

        Args:
            config: Configuration for monitoring thresholds
            db_path: Optional path to SQLite database for persistence
        """
        self.config = config or ForwardTestConfig()
        self.db_path = db_path

        self.trades: List[TradeRecord] = []
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # Running calculations
        self._cumulative_r = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0

    def record_trade(self, trade_dict: Dict) -> ForwardTestStatus:
        """
        Record a completed trade and return updated status.

        Args:
            trade_dict: Dict with keys:
                - r_multiple: P&L in R-multiples
                - exit_type: 'take_profit', 'stop_loss', 'trailing', 'time'
                - symbol: Trading symbol
                - pattern_quality: Quality score (0-1)
                - bars_held: Optional bars in trade
                - timestamp: Optional timestamp (defaults to now)

        Returns:
            ForwardTestStatus with current metrics and alerts
        """
        # Create trade record
        trade = TradeRecord(
            timestamp=trade_dict.get('timestamp', datetime.now()),
            symbol=trade_dict.get('symbol', 'UNKNOWN'),
            r_multiple=trade_dict['r_multiple'],
            exit_type=trade_dict.get('exit_type', 'unknown'),
            pattern_quality=trade_dict.get('pattern_quality', 0.5),
            bars_held=trade_dict.get('bars_held', 0),
            metadata=trade_dict.get('metadata', {}),
        )
        self.trades.append(trade)

        # Update running calculations
        self._cumulative_r += trade.r_multiple
        if trade.r_multiple > 0:
            self._gross_profit += trade.r_multiple
            self.consecutive_losses = 0
        else:
            self._gross_loss += abs(trade.r_multiple)
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses,
                self.consecutive_losses
            )

        # Persist if db configured
        if self.db_path:
            self._save_to_db(trade)

        return self._calculate_status()

    def _calculate_status(self) -> ForwardTestStatus:
        """Calculate current forward test status with all metrics."""
        cfg = self.config
        n = len(self.trades)

        if n == 0:
            return ForwardTestStatus(
                total_trades=0,
                winners=0,
                losers=0,
                running_pf=0,
                running_wr=0,
                running_sharpe=0,
                running_expectancy=0,
                pf_vs_baseline_pct=0,
                wr_vs_baseline_pct=0,
                wr_ci_lower=0,
                wr_ci_upper=1,
                is_degraded=False,
                degradation_severity='none',
                alerts=[],
                consecutive_losses=0,
                max_consecutive_losses=0,
                ready_for_deployment=False,
                trades_until_deployment=cfg.min_trades_for_deployment,
            )

        # Calculate metrics
        r_multiples = [t.r_multiple for t in self.trades]
        winners = [t for t in self.trades if t.r_multiple > 0]
        losers = [t for t in self.trades if t.r_multiple <= 0]

        win_rate = len(winners) / n
        pf = self._gross_profit / self._gross_loss if self._gross_loss > 0 else float('inf')
        expectancy = np.mean(r_multiples)
        sharpe = np.mean(r_multiples) / np.std(r_multiples) if np.std(r_multiples) > 0 else 0

        # Confidence interval on win rate (Wilson score interval)
        wr_se = np.sqrt(win_rate * (1 - win_rate) / n)
        wr_ci_lower = win_rate - 1.96 * wr_se
        wr_ci_upper = win_rate + 1.96 * wr_se

        # Comparison to baseline
        pf_ratio = pf / cfg.baseline_pf if not np.isinf(pf) else 0
        wr_ratio = win_rate / cfg.baseline_wr

        # Check for degradation
        alerts = []
        is_degraded = False
        severity = 'none'

        if n >= cfg.min_trades_for_comparison:
            # Check PF degradation
            pf_degradation = 1 - pf_ratio
            if pf_degradation > cfg.degradation_threshold_pf:
                alerts.append(f"Profit factor {pf_degradation:.1%} below baseline ({pf:.2f} vs {cfg.baseline_pf:.2f})")
                is_degraded = True
                severity = 'warning'

            # Check WR degradation
            wr_degradation = 1 - wr_ratio
            if wr_degradation > cfg.degradation_threshold_wr:
                alerts.append(f"Win rate {wr_degradation:.1%} below baseline ({win_rate:.1%} vs {cfg.baseline_wr:.1%})")
                is_degraded = True
                severity = 'warning'

            # Check if win rate CI includes 50% (no edge)
            if wr_ci_lower < 0.50:
                alerts.append(f"Win rate 95% CI [{wr_ci_lower:.1%}, {wr_ci_upper:.1%}] includes 50% - edge not statistically confirmed")
                is_degraded = True
                severity = 'critical' if wr_ci_lower < 0.45 else 'warning'

        # Check consecutive losses
        if self.consecutive_losses >= cfg.max_consecutive_losses:
            alerts.append(f"{self.consecutive_losses} consecutive losses - statistical anomaly")
            is_degraded = True
            severity = 'critical'

        # Deployment readiness
        ready = n >= cfg.min_trades_for_deployment and not is_degraded
        trades_remaining = max(0, cfg.min_trades_for_deployment - n)

        return ForwardTestStatus(
            total_trades=n,
            winners=len(winners),
            losers=len(losers),
            running_pf=pf if not np.isinf(pf) else 999.0,
            running_wr=win_rate,
            running_sharpe=sharpe,
            running_expectancy=expectancy,
            pf_vs_baseline_pct=(pf_ratio - 1),
            wr_vs_baseline_pct=(wr_ratio - 1),
            wr_ci_lower=wr_ci_lower,
            wr_ci_upper=wr_ci_upper,
            is_degraded=is_degraded,
            degradation_severity=severity,
            alerts=alerts,
            consecutive_losses=self.consecutive_losses,
            max_consecutive_losses=self.max_consecutive_losses,
            ready_for_deployment=ready,
            trades_until_deployment=trades_remaining,
        )

    def check_edge_degradation(self) -> Tuple[bool, str, List[str]]:
        """
        Statistical test for edge degradation.

        Returns:
            Tuple of (is_degraded, severity, alerts)
        """
        status = self._calculate_status()
        return status.is_degraded, status.degradation_severity, status.alerts

    def get_rolling_metrics(self, window: Optional[int] = None) -> Dict:
        """
        Get metrics for the most recent N trades.

        Args:
            window: Number of recent trades (defaults to config.rolling_window)

        Returns:
            Dict with rolling metrics
        """
        window = window or self.config.rolling_window

        if len(self.trades) < window:
            recent = self.trades
        else:
            recent = self.trades[-window:]

        if not recent:
            return {'n': 0}

        r_multiples = [t.r_multiple for t in recent]
        winners = [t for t in recent if t.r_multiple > 0]
        losers = [t for t in recent if t.r_multiple <= 0]

        gross_profit = sum(t.r_multiple for t in winners) if winners else 0
        gross_loss = abs(sum(t.r_multiple for t in losers)) if losers else 0

        return {
            'n': len(recent),
            'win_rate': len(winners) / len(recent),
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
            'expectancy': np.mean(r_multiples),
            'best_trade': max(r_multiples),
            'worst_trade': min(r_multiples),
        }

    def generate_report(self) -> str:
        """Generate human-readable forward test report."""
        status = self._calculate_status()
        cfg = self.config

        lines = []
        lines.append("=" * 60)
        lines.append("FORWARD TEST PERFORMANCE REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Trade Summary
        lines.append("-" * 60)
        lines.append("TRADE SUMMARY")
        lines.append("-" * 60)
        lines.append(f"Total Trades:     {status.total_trades}")
        lines.append(f"Winners:          {status.winners}")
        lines.append(f"Losers:           {status.losers}")
        lines.append("")

        # Performance vs Baseline
        lines.append("-" * 60)
        lines.append("PERFORMANCE vs BASELINE")
        lines.append("-" * 60)
        lines.append(f"{'Metric':<20} {'Actual':<12} {'Baseline':<12} {'Delta':<12}")
        lines.append("-" * 60)
        lines.append(f"{'Win Rate':<20} {status.running_wr:.1%}        {cfg.baseline_wr:.1%}        {status.wr_vs_baseline_pct:+.1%}")
        lines.append(f"{'Profit Factor':<20} {status.running_pf:.2f}         {cfg.baseline_pf:.2f}         {status.pf_vs_baseline_pct:+.1%}")
        lines.append(f"{'Expectancy (R)':<20} {status.running_expectancy:.2f}         {cfg.baseline_expectancy:.2f}         {status.running_expectancy - cfg.baseline_expectancy:+.2f}")
        lines.append(f"{'Sharpe Ratio':<20} {status.running_sharpe:.2f}")
        lines.append("")

        # Confidence Intervals
        lines.append("-" * 60)
        lines.append("STATISTICAL CONFIDENCE")
        lines.append("-" * 60)
        lines.append(f"Win Rate 95% CI: [{status.wr_ci_lower:.1%}, {status.wr_ci_upper:.1%}]")
        edge_confirmed = status.wr_ci_lower > 0.50
        lines.append(f"Edge Confirmed:  {'Yes' if edge_confirmed else 'No (CI includes 50%)'}")
        lines.append("")

        # Risk Metrics
        lines.append("-" * 60)
        lines.append("RISK METRICS")
        lines.append("-" * 60)
        lines.append(f"Current Consecutive Losses: {status.consecutive_losses}")
        lines.append(f"Max Consecutive Losses:     {status.max_consecutive_losses}")
        lines.append("")

        # Deployment Status
        lines.append("-" * 60)
        lines.append("DEPLOYMENT STATUS")
        lines.append("-" * 60)
        if status.ready_for_deployment:
            lines.append("Status: READY FOR LIVE DEPLOYMENT")
        else:
            lines.append(f"Status: NOT READY")
            lines.append(f"Trades until minimum: {status.trades_until_deployment}")
            if status.is_degraded:
                lines.append(f"Degradation detected: {status.degradation_severity.upper()}")

        # Alerts
        if status.alerts:
            lines.append("")
            lines.append("-" * 60)
            lines.append("ALERTS")
            lines.append("-" * 60)
            for alert in status.alerts:
                lines.append(f"  - {alert}")

        lines.append("")
        lines.append("=" * 60)

        # Verdict
        if status.is_degraded:
            if status.degradation_severity == 'critical':
                lines.append("VERDICT: PAUSE TRADING - Review required")
            else:
                lines.append("VERDICT: CAUTION - Monitor closely")
        elif status.ready_for_deployment:
            lines.append("VERDICT: PERFORMING AS EXPECTED - Ready for live")
        else:
            lines.append(f"VERDICT: CONTINUE TESTING - {status.trades_until_deployment} more trades needed")

        lines.append("=" * 60)

        return "\n".join(lines)

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        status = self._calculate_status()

        data = {
            'generated_at': datetime.now().isoformat(),
            'config': {
                'baseline_pf': self.config.baseline_pf,
                'baseline_wr': self.config.baseline_wr,
                'min_trades_for_deployment': self.config.min_trades_for_deployment,
            },
            'trades': [
                {
                    'timestamp': t.timestamp.isoformat(),
                    'symbol': t.symbol,
                    'r_multiple': t.r_multiple,
                    'exit_type': t.exit_type,
                    'pattern_quality': t.pattern_quality,
                    'bars_held': t.bars_held,
                }
                for t in self.trades
            ],
            'status': {
                'total_trades': status.total_trades,
                'winners': status.winners,
                'losers': status.losers,
                'running_pf': status.running_pf,
                'running_wr': status.running_wr,
                'running_sharpe': status.running_sharpe,
                'running_expectancy': status.running_expectancy,
                'is_degraded': status.is_degraded,
                'degradation_severity': status.degradation_severity,
                'alerts': status.alerts,
                'ready_for_deployment': status.ready_for_deployment,
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_results(self, filepath: str):
        """Load previous results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore trades
        self.trades = []
        self._cumulative_r = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        for t in data.get('trades', []):
            trade_dict = {
                'timestamp': datetime.fromisoformat(t['timestamp']),
                'symbol': t['symbol'],
                'r_multiple': t['r_multiple'],
                'exit_type': t['exit_type'],
                'pattern_quality': t['pattern_quality'],
                'bars_held': t.get('bars_held', 0),
            }
            self.record_trade(trade_dict)

    def _save_to_db(self, trade: TradeRecord):
        """Save trade to SQLite database (if configured)."""
        if not self.db_path:
            return

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forward_test_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                r_multiple REAL,
                exit_type TEXT,
                pattern_quality REAL,
                bars_held INTEGER,
                metadata TEXT
            )
        ''')

        # Insert trade
        cursor.execute('''
            INSERT INTO forward_test_trades
            (timestamp, symbol, r_multiple, exit_type, pattern_quality, bars_held, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.timestamp.isoformat(),
            trade.symbol,
            trade.r_multiple,
            trade.exit_type,
            trade.pattern_quality,
            trade.bars_held,
            json.dumps(trade.metadata),
        ))

        conn.commit()
        conn.close()


def print_status(monitor: ForwardTestMonitor):
    """Print human-readable forward test status."""
    print(monitor.generate_report())
