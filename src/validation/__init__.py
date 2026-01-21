"""
Validation Module
=================
VRD 2.0 Statistical Validation Suite

Provides validators for testing edge significance and risk analysis:
- PermutationTest: Statistical significance via shuffle testing
- MonteCarloSim: Risk analysis via equity path simulation
- BootstrapResample: Confidence intervals via resampling
- PBOCalculator: Probability of Backtest Overfitting
- PropFirmSimulator: Prop firm challenge pass probability
- ValidationService: Unified validation orchestrator
"""

from src.validation.base import (
    BaseValidator,
    ValidationResult,
    ValidationStatus,
    ValidationSuite
)
from src.validation.permutation import PermutationTest, PermutationConfig
from src.validation.monte_carlo import (
    MonteCarloSim,
    MonteCarloConfig,
    PropFirmRules,
    PropFirmResult,
    PropFirmSimulator,
)
from src.validation.bootstrap import BootstrapResample, BootstrapConfig
from src.validation.pbo import PBOCalculator, PBOResult
from src.validation.service import ValidationService, ValidationReport, quick_validate

# Type alias for backward compatibility
PermutationResult = ValidationResult
MonteCarloResult = ValidationResult
BootstrapResult = ValidationResult

# Class aliases for backward compatibility with tests
MonteCarloSimulator = MonteCarloSim
BlockBootstrap = BootstrapResample

__all__ = [
    # Base classes
    'BaseValidator',
    'ValidationResult',
    'ValidationStatus',
    'ValidationSuite',

    # Validators
    'PermutationTest',
    'MonteCarloSim',
    'MonteCarloSimulator',  # Alias
    'BootstrapResample',
    'BlockBootstrap',  # Alias
    'PBOCalculator',
    'PropFirmSimulator',
    'ValidationService',

    # Config classes
    'PermutationConfig',
    'MonteCarloConfig',
    'BootstrapConfig',

    # Result classes
    'PBOResult',
    'PropFirmRules',
    'PropFirmResult',
    'ValidationReport',

    # Result aliases (for backward compatibility)
    'PermutationResult',
    'MonteCarloResult',
    'BootstrapResult',

    # Convenience functions
    'quick_validate',
]


def run_validation_suite(
    backtest_result: dict,
    trades: list = None,
    validators: list = None,
    config: dict = None
) -> ValidationSuite:
    """
    Run multiple validators and return combined results.
    
    Args:
        backtest_result: Results from BacktestEngine
        trades: List of trade dictionaries
        validators: List of validator names to run (default: all)
        config: Configuration overrides per validator
    
    Returns:
        ValidationSuite with all results
    
    Example:
        suite = run_validation_suite(
            backtest_result=results,
            trades=trades_list,
            validators=['permutation_test', 'monte_carlo']
        )
        print(suite)
    """
    available = {
        'permutation_test': PermutationTest,
        'monte_carlo': MonteCarloSim,
        'bootstrap': BootstrapResample,
    }
    
    if validators is None:
        validators = list(available.keys())
    
    config = config or {}
    suite = ValidationSuite()
    
    for name in validators:
        if name not in available:
            continue
        
        validator_config = config.get(name, {})
        validator = available[name](config=validator_config)
        
        result = validator.validate(backtest_result, trades=trades)
        suite.add_result(result)
    
    return suite


# Helper function aliases for direct access
def run_permutation_test(backtest_result, trades=None, config=None):
    """Convenience function to run permutation test."""
    validator = PermutationTest(config=config or {})
    return validator.validate(backtest_result, trades=trades)


def run_monte_carlo(backtest_result, trades=None, config=None):
    """Convenience function to run Monte Carlo simulation."""
    validator = MonteCarloSim(config=config or {})
    return validator.validate(backtest_result, trades=trades)


def compute_all_confidence_intervals(backtest_result, trades=None, config=None):
    """Convenience function to compute bootstrap confidence intervals."""
    validator = BootstrapResample(config=config or {})
    return validator.validate(backtest_result, trades=trades)
