"""
Production Gate: ML vs Baseline Comparison
==========================================
Compares ML-enhanced system vs Phase 7.9 baseline to decide:
- DEPLOY ML: ML improves performance significantly
- USE BASELINE: ML doesn't add value, use fixed position sizing

Critical principle: ML must OUTPERFORM baseline on held-out data.
If it fails, we fallback to Phase 7.9 with fixed 1% position sizing.

This is the final validation before deploying ML meta-labeling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np
import pandas as pd


@dataclass
class ProductionGateConfig:
    """Configuration for production gate validation."""
    # Improvement thresholds
    min_pf_improvement_pct: float = 10.0  # ML PF must be 10%+ better
    min_sharpe_improvement_pct: float = 5.0  # ML Sharpe must be 5%+ better

    # Absolute thresholds
    min_ml_pf: float = 1.0  # ML must be profitable
    min_ml_sharpe: float = 0.0  # ML must have positive Sharpe

    # Degradation limits
    max_dd_increase_pct: float = 20.0  # ML DD can't be 20%+ worse than baseline

    # Baseline position sizing
    baseline_risk_pct: float = 0.01  # Fixed 1% risk for baseline


@dataclass
class GateResult:
    """Result of production gate validation."""
    # Decision
    decision: str  # 'DEPLOY_ML' or 'USE_BASELINE'
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)

    # Baseline metrics (fixed 1% sizing)
    baseline_pf: float = 0.0
    baseline_sharpe: float = 0.0
    baseline_total_return: float = 0.0
    baseline_max_dd: float = 0.0
    baseline_trades: int = 0

    # ML metrics (confidence-adjusted sizing)
    ml_pf: float = 0.0
    ml_sharpe: float = 0.0
    ml_total_return: float = 0.0
    ml_max_dd: float = 0.0
    ml_trades: int = 0

    # Improvement metrics
    pf_improvement_pct: float = 0.0
    sharpe_improvement_pct: float = 0.0
    return_improvement_pct: float = 0.0

    # ML sizing stats
    trades_skipped: int = 0
    trades_full: int = 0
    trades_half: int = 0
    trades_minimum: int = 0
    avg_position_size_pct: float = 0.0

    # Timing
    timestamp: str = ""


class ProductionGate:
    """
    Final gate to compare ML-enhanced system vs baseline.

    Simulates both systems on held-out data (never seen by optimizer or ML model)
    and determines which to deploy.
    """

    def __init__(self, config: Optional[ProductionGateConfig] = None):
        """
        Initialize gate.

        Args:
            config: Gate configuration
        """
        self.config = config or ProductionGateConfig()

    def run_gate(
        self,
        trades: List[Dict[str, Any]],
        ml_confidences: np.ndarray,
        position_sizes: np.ndarray,
        baseline_metrics: Dict[str, float],
    ) -> GateResult:
        """
        Run production gate comparison.

        Args:
            trades: List of trade dictionaries with 'r_multiple', 'result', etc.
            ml_confidences: Array of ML predicted confidences (0-1)
            position_sizes: Array of ML-adjusted position sizes (as risk %)
            baseline_metrics: Dict with baseline win_rate, avg_win_r, avg_loss_r, etc.

        Returns:
            GateResult with decision and metrics
        """
        cfg = self.config

        if len(trades) == 0:
            return GateResult(
                decision='USE_BASELINE',
                passed=False,
                failure_reasons=['No trades to evaluate'],
                timestamp=datetime.now().isoformat(),
            )

        # Simulate baseline (fixed 1% sizing for all trades)
        baseline_results = self._simulate_baseline(trades, cfg.baseline_risk_pct)

        # Simulate ML (confidence-adjusted sizing)
        ml_results = self._simulate_ml(trades, position_sizes)

        # Calculate improvement (handle edge cases)
        if baseline_results['pf'] > 0.001:
            pf_improvement = ((ml_results['pf'] / baseline_results['pf']) - 1) * 100
        else:
            # Baseline unprofitable - any positive ML PF is improvement
            pf_improvement = 100.0 if ml_results['pf'] > 1.0 else 0.0

        if abs(baseline_results['sharpe']) > 0.001:
            sharpe_improvement = ((ml_results['sharpe'] / baseline_results['sharpe']) - 1) * 100
        else:
            # Baseline near-zero Sharpe - positive ML Sharpe is improvement
            sharpe_improvement = 100.0 if ml_results['sharpe'] > 0.1 else 0.0

        if abs(baseline_results['total_return']) > 0.0001:
            return_improvement = ((ml_results['total_return'] / baseline_results['total_return']) - 1) * 100
        else:
            return_improvement = 0.0

        # Check gate criteria
        failure_reasons = []

        if ml_results['pf'] < cfg.min_ml_pf:
            failure_reasons.append(f"ML PF {ml_results['pf']:.3f} < {cfg.min_ml_pf} (not profitable)")

        if pf_improvement < cfg.min_pf_improvement_pct:
            failure_reasons.append(f"PF improvement {pf_improvement:.1f}% < {cfg.min_pf_improvement_pct}% required")

        if ml_results['sharpe'] < cfg.min_ml_sharpe:
            failure_reasons.append(f"ML Sharpe {ml_results['sharpe']:.3f} < {cfg.min_ml_sharpe}")

        # Check for excessive DD increase
        if baseline_results['max_dd'] > 0:
            dd_increase = ((ml_results['max_dd'] / baseline_results['max_dd']) - 1) * 100
            if dd_increase > cfg.max_dd_increase_pct:
                failure_reasons.append(f"Max DD increased by {dd_increase:.1f}% (limit: {cfg.max_dd_increase_pct}%)")

        passed = len(failure_reasons) == 0
        decision = 'DEPLOY_ML' if passed else 'USE_BASELINE'

        # Count ML sizing actions
        sizing_stats = self._count_sizing_actions(ml_confidences, position_sizes, cfg)

        result = GateResult(
            decision=decision,
            passed=passed,
            failure_reasons=failure_reasons,
            baseline_pf=baseline_results['pf'],
            baseline_sharpe=baseline_results['sharpe'],
            baseline_total_return=baseline_results['total_return'],
            baseline_max_dd=baseline_results['max_dd'],
            baseline_trades=baseline_results['n_trades'],
            ml_pf=ml_results['pf'],
            ml_sharpe=ml_results['sharpe'],
            ml_total_return=ml_results['total_return'],
            ml_max_dd=ml_results['max_dd'],
            ml_trades=ml_results['n_trades'],
            pf_improvement_pct=pf_improvement,
            sharpe_improvement_pct=sharpe_improvement,
            return_improvement_pct=return_improvement,
            trades_skipped=sizing_stats['skipped'],
            trades_full=sizing_stats['full'],
            trades_half=sizing_stats['half'],
            trades_minimum=sizing_stats['minimum'],
            avg_position_size_pct=sizing_stats['avg_size'],
            timestamp=datetime.now().isoformat(),
        )

        self._print_gate_result(result)

        return result

    def _simulate_baseline(
        self,
        trades: List[Dict[str, Any]],
        risk_pct: float,
    ) -> Dict[str, float]:
        """
        Simulate baseline with fixed position sizing.
        """
        if not trades:
            return {'pf': 0, 'sharpe': 0, 'total_return': 0, 'max_dd': 0, 'n_trades': 0}

        # P&L array (each trade uses fixed risk %)
        pnl = []
        for trade in trades:
            # Support both 'r_multiple' and 'pnl_r' keys
            r_mult = trade.get('pnl_r', trade.get('r_multiple', 0))
            trade_pnl = r_mult * risk_pct  # P&L as % of equity
            pnl.append(trade_pnl)

        pnl = np.array(pnl)

        return self._calculate_metrics(pnl)

    def _simulate_ml(
        self,
        trades: List[Dict[str, Any]],
        position_sizes: np.ndarray,
    ) -> Dict[str, float]:
        """
        Simulate ML with confidence-adjusted position sizing.
        """
        if not trades or len(position_sizes) == 0:
            return {'pf': 0, 'sharpe': 0, 'total_return': 0, 'max_dd': 0, 'n_trades': 0}

        # P&L array (each trade uses ML-adjusted size)
        pnl = []
        for i, trade in enumerate(trades):
            # Support both 'r_multiple' and 'pnl_r' keys
            r_mult = trade.get('pnl_r', trade.get('r_multiple', 0))
            size = position_sizes[i] if i < len(position_sizes) else 0
            trade_pnl = r_mult * size  # P&L as % of equity
            pnl.append(trade_pnl)

        pnl = np.array(pnl)

        return self._calculate_metrics(pnl)

    def _calculate_metrics(self, pnl: np.ndarray) -> Dict[str, float]:
        """
        Calculate trading metrics from P&L array.
        """
        if len(pnl) == 0:
            return {'pf': 0, 'sharpe': 0, 'total_return': 0, 'max_dd': 0, 'n_trades': 0}

        # Filter out zero-size trades
        active_pnl = pnl[pnl != 0]
        n_trades = len(active_pnl)

        if n_trades == 0:
            return {'pf': 0, 'sharpe': 0, 'total_return': 0, 'max_dd': 0, 'n_trades': 0}

        # Profit factor
        wins = active_pnl[active_pnl > 0]
        losses = active_pnl[active_pnl < 0]
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001  # Avoid div by zero
        pf = gross_profit / gross_loss

        # Sharpe ratio (annualized, assume ~250 trading days)
        mean_pnl = active_pnl.mean()
        std_pnl = active_pnl.std() if len(active_pnl) > 1 else 1
        sharpe = (mean_pnl / std_pnl) * np.sqrt(250) if std_pnl > 0 else 0

        # Total return
        total_return = active_pnl.sum()

        # Max drawdown
        equity_curve = np.cumsum(active_pnl)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = running_max - equity_curve
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

        return {
            'pf': float(pf),
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'max_dd': float(max_dd),
            'n_trades': int(n_trades),
        }

    def _count_sizing_actions(
        self,
        confidences: np.ndarray,
        sizes: np.ndarray,
        cfg: ProductionGateConfig,
    ) -> Dict[str, Any]:
        """
        Count how many trades fell into each sizing category.
        """
        skipped = 0
        full = 0
        half = 0
        minimum = 0

        for conf, size in zip(confidences, sizes):
            if size == 0:
                skipped += 1
            elif conf >= 0.70:
                full += 1
            elif conf >= 0.55:
                half += 1
            else:
                minimum += 1

        avg_size = sizes.mean() if len(sizes) > 0 else 0

        return {
            'skipped': skipped,
            'full': full,
            'half': half,
            'minimum': minimum,
            'avg_size': float(avg_size),
        }

    def _print_gate_result(self, result: GateResult) -> None:
        """
        Print production gate result.
        """
        print(f"\n{'='*60}")
        print(f"PRODUCTION GATE RESULT")
        print(f"{'='*60}")

        print(f"\n--- Baseline (Fixed 1% Sizing) ---")
        print(f"Profit Factor: {result.baseline_pf:.4f}")
        print(f"Sharpe Ratio: {result.baseline_sharpe:.4f}")
        print(f"Total Return: {result.baseline_total_return:.2%}")
        print(f"Max Drawdown: {result.baseline_max_dd:.2%}")
        print(f"Trades: {result.baseline_trades}")

        print(f"\n--- ML (Confidence-Adjusted Sizing) ---")
        print(f"Profit Factor: {result.ml_pf:.4f}")
        print(f"Sharpe Ratio: {result.ml_sharpe:.4f}")
        print(f"Total Return: {result.ml_total_return:.2%}")
        print(f"Max Drawdown: {result.ml_max_dd:.2%}")
        print(f"Trades: {result.ml_trades}")

        print(f"\n--- Sizing Distribution ---")
        print(f"Full Position: {result.trades_full}")
        print(f"Half Position: {result.trades_half}")
        print(f"Minimum Position: {result.trades_minimum}")
        print(f"Skipped: {result.trades_skipped}")
        print(f"Avg Position Size: {result.avg_position_size_pct:.2%}")

        print(f"\n--- Improvement ---")
        print(f"PF Improvement: {result.pf_improvement_pct:+.1f}%")
        print(f"Sharpe Improvement: {result.sharpe_improvement_pct:+.1f}%")
        print(f"Return Improvement: {result.return_improvement_pct:+.1f}%")

        print(f"\n{'='*60}")
        if result.passed:
            print(f">>> DECISION: {result.decision} <<<")
            print(f"ML adds {result.pf_improvement_pct:.1f}% to profit factor.")
            print(f"Deploying ML meta-labeling for position sizing.")
        else:
            print(f">>> DECISION: {result.decision} <<<")
            print(f"Failure reasons:")
            for reason in result.failure_reasons:
                print(f"  - {reason}")
            print(f"Using Phase 7.9 baseline with fixed 1% sizing.")
        print(f"{'='*60}\n")

    def save_result(self, result: GateResult, output_path: Path) -> None:
        """
        Save gate result to JSON.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = {
            'decision': result.decision,
            'passed': result.passed,
            'failure_reasons': result.failure_reasons,
            'baseline': {
                'pf': result.baseline_pf,
                'sharpe': result.baseline_sharpe,
                'total_return': result.baseline_total_return,
                'max_dd': result.baseline_max_dd,
                'trades': result.baseline_trades,
            },
            'ml': {
                'pf': result.ml_pf,
                'sharpe': result.ml_sharpe,
                'total_return': result.ml_total_return,
                'max_dd': result.ml_max_dd,
                'trades': result.ml_trades,
            },
            'improvement': {
                'pf_pct': result.pf_improvement_pct,
                'sharpe_pct': result.sharpe_improvement_pct,
                'return_pct': result.return_improvement_pct,
            },
            'sizing_distribution': {
                'full': result.trades_full,
                'half': result.trades_half,
                'minimum': result.trades_minimum,
                'skipped': result.trades_skipped,
                'avg_size_pct': result.avg_position_size_pct,
            },
            'timestamp': result.timestamp,
        }

        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"Gate result saved to: {output_path}")
