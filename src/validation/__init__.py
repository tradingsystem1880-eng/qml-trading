"""
Validation Module for QML Trading System
=========================================
Institutional-grade strategy validation framework with:
- VRD 2.0 (Versioned Research Database)
- Purged Walk-Forward Optimization
- Statistical Robustness Testing (Permutation, Monte Carlo, Bootstrap)
"""

from src.validation.database import VRDDatabase, ExperimentRecord
from src.validation.tracker import ExperimentTracker
from src.validation.walk_forward import PurgedWalkForwardEngine, WalkForwardConfig, FoldResult
from src.validation.permutation import PermutationTest, PermutationResult
from src.validation.monte_carlo import MonteCarloSimulator, MonteCarloResult
from src.validation.bootstrap import BlockBootstrap, BootstrapResult
from src.validation.validator import StrategyValidator

__all__ = [
    # Database
    "VRDDatabase",
    "ExperimentRecord",
    # Tracker
    "ExperimentTracker",
    # Walk-Forward
    "PurgedWalkForwardEngine",
    "WalkForwardConfig",
    "FoldResult",
    # Statistical Testing
    "PermutationTest",
    "PermutationResult",
    "MonteCarloSimulator",
    "MonteCarloResult",
    "BlockBootstrap",
    "BootstrapResult",
    # High-Level
    "StrategyValidator",
]
