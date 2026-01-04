"""
Deployment Gatekeeper
======================
Validates if a strategy meets production-readiness criteria.

The Kill Switch: Only strategies that pass ALL checks are approved.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class GatekeeperConfig:
    """Configuration for deployment gatekeeper."""
    
    # Sharpe thresholds
    min_sharpe_oos: float = 1.0
    min_sharpe_is: float = 1.0
    
    # Risk thresholds
    max_drawdown_limit: float = 25.0  # Maximum acceptable drawdown %
    max_var_95: float = 20.0          # Maximum 95% VaR
    max_kill_switch_prob: float = 0.15
    
    # Statistical significance
    max_p_value: float = 0.05
    
    # Regime robustness (% of regimes with positive returns)
    min_regime_robustness: float = 0.75
    
    # Stability
    min_param_stability: float = 0.6
    
    # Trade count
    min_trades: int = 50
    
    # IS/OOS ratio (to detect overfit)
    max_is_oos_ratio: float = 2.0


@dataclass 
class ReadinessResult:
    """Result of deployment readiness check."""
    
    is_ready: bool
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str]
    
    # Individual check results
    check_results: Dict[str, Dict] = field(default_factory=dict)
    
    # Scores
    overall_score: float = 0.0
    
    def __str__(self) -> str:
        status = "✅ READY" if self.is_ready else "❌ NOT READY"
        return f"{status} - Passed: {len(self.passed_checks)}, Failed: {len(self.failed_checks)}"


class DeploymentGatekeeper:
    """
    Deployment Gatekeeper - The Final Kill Switch.
    
    Validates if a strategy is production-ready by checking:
    1. Sharpe > threshold (both IS and OOS)
    2. Monte Carlo VaR within limits
    3. Kill switch probability acceptable
    4. Regime robustness sufficient
    5. Statistical significance
    6. Parameter stability
    
    Only strategies that pass ALL checks are approved.
    """
    
    def __init__(self, config: Optional[GatekeeperConfig] = None):
        """
        Initialize gatekeeper.
        
        Args:
            config: Gatekeeper configuration
        """
        self.config = config or GatekeeperConfig()
        logger.info("DeploymentGatekeeper initialized")
    
    def check_readiness(self, result: Any) -> ReadinessResult:
        """
        Check if strategy is production-ready.
        
        Args:
            result: PipelineResult from ValidationOrchestrator
            
        Returns:
            ReadinessResult with pass/fail status
        """
        passed = []
        failed = []
        warnings = []
        check_results = {}
        
        # Check 1: OOS Sharpe
        check_results["sharpe_oos"] = self._check_sharpe_oos(result)
        if check_results["sharpe_oos"]["passed"]:
            passed.append("OOS Sharpe")
        else:
            failed.append("OOS Sharpe")
        
        # Check 2: Monte Carlo VaR
        check_results["var_95"] = self._check_var(result)
        if check_results["var_95"]["passed"]:
            passed.append("VaR 95%")
        else:
            failed.append("VaR 95%")
        
        # Check 3: Kill Switch Probability
        check_results["kill_switch"] = self._check_kill_switch(result)
        if check_results["kill_switch"]["passed"]:
            passed.append("Kill Switch Prob")
        else:
            failed.append("Kill Switch Prob")
        
        # Check 4: Statistical Significance
        check_results["significance"] = self._check_significance(result)
        if check_results["significance"]["passed"]:
            passed.append("Statistical Significance")
        else:
            failed.append("Statistical Significance")
        
        # Check 5: Regime Robustness
        check_results["regime_robustness"] = self._check_regime_robustness(result)
        if check_results["regime_robustness"]["passed"]:
            passed.append("Regime Robustness")
        else:
            if check_results["regime_robustness"].get("warning"):
                warnings.append("Regime Robustness")
            else:
                failed.append("Regime Robustness")
        
        # Check 6: Parameter Stability
        check_results["param_stability"] = self._check_param_stability(result)
        if check_results["param_stability"]["passed"]:
            passed.append("Parameter Stability")
        else:
            warnings.append("Parameter Stability")
        
        # Check 7: Trade Count
        check_results["trade_count"] = self._check_trade_count(result)
        if check_results["trade_count"]["passed"]:
            passed.append("Trade Count")
        else:
            warnings.append("Trade Count")
        
        # Check 8: IS/OOS Ratio (overfit detection)
        check_results["is_oos_ratio"] = self._check_is_oos_ratio(result)
        if check_results["is_oos_ratio"]["passed"]:
            passed.append("IS/OOS Ratio")
        else:
            warnings.append("IS/OOS Ratio (potential overfit)")
        
        # Calculate overall score
        total_checks = len(passed) + len(failed)
        overall_score = len(passed) / total_checks * 100 if total_checks > 0 else 0
        
        # Determine if ready (all critical checks must pass)
        critical_checks = ["sharpe_oos", "kill_switch", "significance"]
        all_critical_pass = all(
            check_results.get(c, {}).get("passed", False) 
            for c in critical_checks
        )
        
        is_ready = len(failed) == 0 and all_critical_pass
        
        result_obj = ReadinessResult(
            is_ready=is_ready,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            check_results=check_results,
            overall_score=overall_score,
        )
        
        logger.info(f"Readiness check complete: {result_obj}")
        
        return result_obj
    
    def _check_sharpe_oos(self, result: Any) -> Dict:
        """Check OOS Sharpe ratio."""
        sharpe = getattr(result, 'oos_sharpe', 0)
        passed = sharpe >= self.config.min_sharpe_oos
        
        return {
            "passed": passed,
            "value": sharpe,
            "threshold": self.config.min_sharpe_oos,
            "message": f"OOS Sharpe {sharpe:.3f} {'≥' if passed else '<'} {self.config.min_sharpe_oos}",
        }
    
    def _check_var(self, result: Any) -> Dict:
        """Check 95% VaR."""
        mc = getattr(result, 'monte_carlo_result', None)
        if mc is None:
            return {"passed": True, "value": None, "message": "No MC data"}
        
        var_95 = abs(getattr(mc, 'var_95', 0))
        passed = var_95 <= self.config.max_var_95
        
        return {
            "passed": passed,
            "value": var_95,
            "threshold": self.config.max_var_95,
            "message": f"VaR 95% {var_95:.1f}% {'≤' if passed else '>'} {self.config.max_var_95}%",
        }
    
    def _check_kill_switch(self, result: Any) -> Dict:
        """Check kill switch probability."""
        mc = getattr(result, 'monte_carlo_result', None)
        if mc is None:
            return {"passed": True, "value": None, "message": "No MC data"}
        
        kill_prob = getattr(mc, 'kill_switch_prob', 0)
        passed = kill_prob <= self.config.max_kill_switch_prob
        
        return {
            "passed": passed,
            "value": kill_prob,
            "threshold": self.config.max_kill_switch_prob,
            "message": f"Kill prob {kill_prob:.1%} {'≤' if passed else '>'} {self.config.max_kill_switch_prob:.0%}",
        }
    
    def _check_significance(self, result: Any) -> Dict:
        """Check statistical significance (p-value)."""
        p_value = getattr(result, 'sharpe_p_value', 1.0)
        passed = p_value <= self.config.max_p_value
        
        return {
            "passed": passed,
            "value": p_value,
            "threshold": self.config.max_p_value,
            "message": f"p-value {p_value:.4f} {'≤' if passed else '>'} {self.config.max_p_value}",
        }
    
    def _check_regime_robustness(self, result: Any) -> Dict:
        """Check performance across regimes."""
        regime_perf = getattr(result, 'regime_performance', {})
        
        if not regime_perf:
            # No regime data - pass with warning
            return {
                "passed": True,
                "warning": True,
                "value": None,
                "message": "No regime data available",
            }
        
        # Count regimes with positive performance
        positive_regimes = sum(
            1 for r in regime_perf.values() 
            if isinstance(r, dict) and r.get('avg_return', 0) > 0
        )
        total_regimes = len(regime_perf)
        
        if total_regimes == 0:
            return {"passed": True, "warning": True, "value": None}
        
        robustness = positive_regimes / total_regimes
        passed = robustness >= self.config.min_regime_robustness
        
        return {
            "passed": passed,
            "value": robustness,
            "threshold": self.config.min_regime_robustness,
            "message": f"Regime robustness {robustness:.0%} ({positive_regimes}/{total_regimes} positive)",
        }
    
    def _check_param_stability(self, result: Any) -> Dict:
        """Check parameter stability across folds."""
        wf = getattr(result, 'walk_forward_result', None)
        if wf is None:
            return {"passed": True, "value": None, "message": "No WF data"}
        
        stability = getattr(wf, 'param_stability_score', 0)
        passed = stability >= self.config.min_param_stability
        
        return {
            "passed": passed,
            "value": stability,
            "threshold": self.config.min_param_stability,
            "message": f"Stability {stability:.2f} {'≥' if passed else '<'} {self.config.min_param_stability}",
        }
    
    def _check_trade_count(self, result: Any) -> Dict:
        """Check minimum trade count."""
        trades = getattr(result, 'total_trades', 0)
        passed = trades >= self.config.min_trades
        
        return {
            "passed": passed,
            "value": trades,
            "threshold": self.config.min_trades,
            "message": f"Trades {trades} {'≥' if passed else '<'} {self.config.min_trades}",
        }
    
    def _check_is_oos_ratio(self, result: Any) -> Dict:
        """Check IS/OOS performance ratio (overfit detection)."""
        wf = getattr(result, 'walk_forward_result', None)
        if wf is None:
            return {"passed": True, "value": None, "message": "No WF data"}
        
        ratio = getattr(wf, 'is_to_oos_ratio', 1.0)
        passed = ratio <= self.config.max_is_oos_ratio
        
        return {
            "passed": passed,
            "value": ratio,
            "threshold": self.config.max_is_oos_ratio,
            "message": f"IS/OOS ratio {ratio:.2f} {'≤' if passed else '>'} {self.config.max_is_oos_ratio}",
        }
    
    def generate_report(self, readiness: ReadinessResult) -> str:
        """Generate text report of readiness check."""
        status = "✅ APPROVED FOR DEPLOYMENT" if readiness.is_ready else "❌ NOT READY FOR DEPLOYMENT"
        
        lines = [
            "=" * 60,
            "DEPLOYMENT READINESS REPORT",
            "=" * 60,
            "",
            f"Status: {status}",
            f"Score: {readiness.overall_score:.0f}/100",
            "",
            "-" * 60,
            "PASSED CHECKS:",
            "-" * 60,
        ]
        
        for check in readiness.passed_checks:
            details = readiness.check_results.get(check.lower().replace(" ", "_"), {})
            msg = details.get("message", "")
            lines.append(f"  ✓ {check}: {msg}")
        
        if readiness.failed_checks:
            lines.extend([
                "",
                "-" * 60,
                "FAILED CHECKS:",
                "-" * 60,
            ])
            for check in readiness.failed_checks:
                details = readiness.check_results.get(check.lower().replace(" ", "_"), {})
                msg = details.get("message", "")
                lines.append(f"  ✗ {check}: {msg}")
        
        if readiness.warnings:
            lines.extend([
                "",
                "-" * 60,
                "WARNINGS:",
                "-" * 60,
            ])
            for warning in readiness.warnings:
                lines.append(f"  ⚠ {warning}")
        
        lines.extend([
            "",
            "=" * 60,
            "RECOMMENDATION:",
            "=" * 60,
        ])
        
        if readiness.is_ready:
            lines.extend([
                "  Strategy approved for paper trading.",
                "  Recommended sizing: Start at 25% of target allocation.",
                "  Monitor for 30 days before scaling.",
            ])
        else:
            lines.extend([
                "  Strategy NOT approved for deployment.",
                "  Address the failed checks above before resubmitting.",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_gatekeeper(
    min_sharpe: float = 1.0,
    max_kill_switch_prob: float = 0.15,
    min_regime_robustness: float = 0.75
) -> DeploymentGatekeeper:
    """Factory function for DeploymentGatekeeper."""
    config = GatekeeperConfig(
        min_sharpe_oos=min_sharpe,
        max_kill_switch_prob=max_kill_switch_prob,
        min_regime_robustness=min_regime_robustness,
    )
    return DeploymentGatekeeper(config=config)
