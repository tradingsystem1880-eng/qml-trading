"""
Validation Service
==================
Orchestrates all validation components into a single interface.

Combines:
- Permutation testing (statistical significance)
- Monte Carlo simulation (risk metrics)
- Bootstrap resampling (confidence intervals)
- PBO calculation (overfitting detection)
- Prop firm simulation (challenge probability)
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.validation.base import ValidationStatus, ValidationResult
from src.validation.permutation import PermutationTest
from src.validation.monte_carlo import MonteCarloSim, PropFirmSimulator, PropFirmRules, PropFirmResult
from src.validation.bootstrap import BootstrapResample
from src.validation.pbo import PBOCalculator


@dataclass
class ValidationReport:
    """
    Comprehensive validation report combining all validators.

    Provides overall verdict and actionable recommendations.
    """
    permutation_result: Optional[ValidationResult] = None
    monte_carlo_result: Optional[ValidationResult] = None
    bootstrap_result: Optional[ValidationResult] = None
    pbo_result: Optional[ValidationResult] = None
    prop_firm_result: Optional[PropFirmResult] = None

    overall_verdict: str = "UNKNOWN"  # "PASS", "WARN", "FAIL"
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_verdict': self.overall_verdict,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat(),
            'permutation': self.permutation_result.to_dict() if self.permutation_result else None,
            'monte_carlo': self.monte_carlo_result.to_dict() if self.monte_carlo_result else None,
            'bootstrap': self.bootstrap_result.to_dict() if self.bootstrap_result else None,
            'pbo': self.pbo_result.to_dict() if self.pbo_result else None,
            'prop_firm': {
                'pass_rate': self.prop_firm_result.pass_rate,
                'avg_days_to_pass': self.prop_firm_result.avg_days_to_pass,
                'fail_reasons': self.prop_firm_result.fail_reasons,
                'profit_on_pass': self.prop_firm_result.profit_on_pass,
            } if self.prop_firm_result else None,
        }

    def __str__(self) -> str:
        icon = {
            'PASS': '✅',
            'WARN': '⚠️',
            'FAIL': '❌',
        }.get(self.overall_verdict, '❓')

        lines = [
            f"\n{'='*60}",
            f"{icon} VALIDATION REPORT: {self.overall_verdict}",
            f"{'='*60}",
        ]

        if self.permutation_result:
            lines.append(f"\n📊 Permutation Test: {self.permutation_result.status.value.upper()}")
            lines.append(f"   {self.permutation_result.interpretation}")

        if self.monte_carlo_result:
            lines.append(f"\n🎲 Monte Carlo: {self.monte_carlo_result.status.value.upper()}")
            lines.append(f"   {self.monte_carlo_result.interpretation}")

        if self.bootstrap_result:
            lines.append(f"\n📈 Bootstrap: {self.bootstrap_result.status.value.upper()}")
            lines.append(f"   {self.bootstrap_result.interpretation}")

        if self.pbo_result:
            lines.append(f"\n🔬 PBO: {self.pbo_result.status.value.upper()}")
            lines.append(f"   {self.pbo_result.interpretation}")

        if self.prop_firm_result:
            lines.append(f"\n💰 Prop Firm Challenge:")
            lines.append(f"   Pass Rate: {self.prop_firm_result.pass_rate:.1%}")
            if self.prop_firm_result.pass_rate > 0:
                lines.append(f"   Avg Days to Pass: {self.prop_firm_result.avg_days_to_pass:.1f}")

        if self.recommendations:
            lines.append(f"\n📝 Recommendations:")
            for rec in self.recommendations:
                lines.append(f"   • {rec}")

        lines.append(f"\n{'='*60}\n")

        return '\n'.join(lines)


class ValidationService:
    """
    One-stop validation for trading strategies.

    Orchestrates all validation components and provides a unified interface
    for comprehensive strategy validation.

    Usage:
        service = ValidationService()
        report = service.validate_strategy(trades, returns_matrix)
        print(report.overall_verdict)
        print(report.recommendations)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validation service.

        Args:
            config: Optional configuration dict with keys for each validator:
                - permutation: dict for PermutationTest config
                - monte_carlo: dict for MonteCarloSim config
                - bootstrap: dict for BootstrapResample config
                - pbo: dict for PBOCalculator config
        """
        self.config = config or {}

        # Initialize validators
        self.permutation = PermutationTest(
            config=self.config.get('permutation', {})
        )
        self.monte_carlo = MonteCarloSim(
            config=self.config.get('monte_carlo', {})
        )
        self.bootstrap = BootstrapResample(
            config=self.config.get('bootstrap', {})
        )
        self.pbo = PBOCalculator(
            config=self.config.get('pbo', {})
        )
        self.prop_firm = PropFirmSimulator()

    def validate_strategy(
        self,
        trades: List[Dict[str, Any]],
        returns_matrix: Optional[np.ndarray] = None,
        prop_firm_rules: Optional[PropFirmRules] = None,
        run_all: bool = True,
        validators: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Run all validators and return comprehensive report.

        Args:
            trades: List of trade dictionaries with 'pnl_pct' field
            returns_matrix: Optional matrix for PBO (n_periods, n_strategies)
            prop_firm_rules: Optional PropFirmRules for challenge simulation
            run_all: If True, run all validators (default)
            validators: If specified, only run these validators
                       Options: ['permutation', 'monte_carlo', 'bootstrap', 'pbo', 'prop_firm']

        Returns:
            ValidationReport with all results and overall verdict
        """
        report = ValidationReport()

        # Determine which validators to run
        if validators is None:
            validators = ['permutation', 'monte_carlo', 'bootstrap', 'pbo', 'prop_firm']

        backtest_result = {'trades': trades}

        # Extract returns for prop firm simulation
        returns = self._extract_returns(trades)

        # Run requested validators
        if 'permutation' in validators:
            report.permutation_result = self.permutation.validate(
                backtest_result, trades=trades
            )

        if 'monte_carlo' in validators:
            report.monte_carlo_result = self.monte_carlo.validate(
                backtest_result, trades=trades
            )

        if 'bootstrap' in validators:
            report.bootstrap_result = self.bootstrap.validate(
                backtest_result, trades=trades
            )

        if 'pbo' in validators:
            report.pbo_result = self.pbo.validate(
                backtest_result, trades=trades, returns_matrix=returns_matrix
            )

        if 'prop_firm' in validators and returns is not None and len(returns) >= 10:
            rules = prop_firm_rules or PropFirmRules()
            report.prop_firm_result = self.prop_firm.simulate_challenge(
                returns / 100,  # Convert to decimal
                rules
            )

        # Calculate overall verdict and recommendations
        report.overall_verdict = self._calculate_verdict(report)
        report.recommendations = self._generate_recommendations(report)

        return report

    def _extract_returns(self, trades: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract returns array from trades."""
        if not trades:
            return None

        returns = []
        for trade in trades:
            if isinstance(trade, dict):
                pnl = trade.get('pnl_pct')
            else:
                pnl = getattr(trade, 'pnl_pct', None)

            if pnl is not None:
                returns.append(pnl)

        return np.array(returns) if returns else None

    def _calculate_verdict(self, report: ValidationReport) -> str:
        """Calculate overall verdict from individual results."""
        statuses = []

        if report.permutation_result:
            statuses.append(report.permutation_result.status)
        if report.monte_carlo_result:
            statuses.append(report.monte_carlo_result.status)
        if report.bootstrap_result:
            statuses.append(report.bootstrap_result.status)
        if report.pbo_result:
            statuses.append(report.pbo_result.status)

        if not statuses:
            return "UNKNOWN"

        # Worst case verdict
        if ValidationStatus.ERROR in statuses:
            return "FAIL"
        if ValidationStatus.FAIL in statuses:
            return "FAIL"
        if ValidationStatus.WARN in statuses:
            return "WARN"
        return "PASS"

    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        # Permutation test recommendations
        if report.permutation_result:
            if report.permutation_result.status == ValidationStatus.FAIL:
                recommendations.append(
                    "Edge is not statistically significant - consider more trades or different entry rules"
                )
            elif report.permutation_result.status == ValidationStatus.WARN:
                recommendations.append(
                    "Edge is marginally significant - gather more data before live trading"
                )

        # Monte Carlo recommendations
        if report.monte_carlo_result:
            metrics = report.monte_carlo_result.metrics
            if metrics.get('risk_of_ruin', 0) > 0.10:
                recommendations.append(
                    f"High risk of ruin ({metrics['risk_of_ruin']:.1%}) - reduce position size"
                )
            if metrics.get('median_max_drawdown', 0) > 25:
                recommendations.append(
                    f"Expected drawdowns are significant ({metrics['median_max_drawdown']:.1f}%) - ensure adequate capital"
                )

        # Bootstrap recommendations
        if report.bootstrap_result:
            ci = report.bootstrap_result.metrics.get('sharpe_ci', [0, 0])
            if ci[0] < 0:
                recommendations.append(
                    "Lower bound of Sharpe CI is negative - profitability is uncertain"
                )

        # PBO recommendations
        if report.pbo_result:
            pbo = report.pbo_result.metrics.get('pbo', 0)
            if pbo > 0.50:
                recommendations.append(
                    f"High probability of overfitting ({pbo:.0%}) - simplify strategy or gather more data"
                )
            elif pbo > 0.25:
                recommendations.append(
                    f"Moderate overfitting risk ({pbo:.0%}) - validate on fresh data before live trading"
                )

        # Prop firm recommendations
        if report.prop_firm_result:
            pass_rate = report.prop_firm_result.pass_rate
            if pass_rate < 0.30:
                recommendations.append(
                    f"Low prop firm pass probability ({pass_rate:.0%}) - not suitable for funded trading"
                )
            elif pass_rate < 0.50:
                recommendations.append(
                    f"Moderate prop firm pass rate ({pass_rate:.0%}) - consider tighter risk management"
                )

            # Check failure reasons
            fail_reasons = report.prop_firm_result.fail_reasons
            if fail_reasons.get('daily_loss_limit', 0) > 0.3:
                recommendations.append(
                    "Frequent daily loss limit breaches - consider reducing position size"
                )
            if fail_reasons.get('total_loss_limit', 0) > 0.3:
                recommendations.append(
                    "Frequent total drawdown breaches - strategy may be too aggressive"
                )

        if not recommendations:
            recommendations.append("Strategy passed all validation checks - ready for forward testing")

        return recommendations


def quick_validate(
    trades: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> ValidationReport:
    """
    Quick validation function for simple use cases.

    Args:
        trades: List of trade dicts with 'pnl_pct' field
        config: Optional configuration dict

    Returns:
        ValidationReport with all results

    Example:
        trades = [{'pnl_pct': 2.5}, {'pnl_pct': -1.2}, ...]
        report = quick_validate(trades)
        print(report)
    """
    service = ValidationService(config)
    return service.validate_strategy(trades)
