"""
Optimization Module for Phase 7.6 and 7.7
=========================================
Bayesian optimization for pattern detection and trade management parameters.

Phase 7.6: Detection parameter optimization (15 params)
Phase 7.7: Extended optimization with trade simulation (25 params)
"""

from src.optimization.bayesian_optimizer import (
    BayesianOptimizer,
    OptimizationConfig,
    OptimizationResult,
    PARAM_SPACE,
)
from src.optimization.parallel_runner import (
    ParallelDetectionRunner,
    DetectionResult,
    AggregateResult,
)

# Phase 7.7 additions
from src.optimization.trade_simulator import (
    TradeSimulator,
    TradeManagementConfig,
    SimulatedTrade,
    SimulationResult,
    ExitReason,
)
from src.optimization.objectives import (
    ObjectiveFunction,
    ObjectiveResult,
    ObjectiveConfig,
    ObjectiveType,
    create_objective,
    get_all_objective_types,
    CountQualityObjective,
    SharpeObjective,
    ExpectancyObjective,
    ProfitFactorObjective,
    MaxDrawdownObjective,
    CompositeObjective,
)
from src.optimization.extended_runner import (
    ExtendedDetectionRunner,
    ExtendedRunnerConfig,
    WalkForwardConfig,
    WalkForwardResult,
    ClusterValidationResult,
    SYMBOL_CLUSTERS,
    ALL_CLUSTERED_SYMBOLS,
)

__all__ = [
    # Phase 7.6
    'BayesianOptimizer',
    'OptimizationConfig',
    'OptimizationResult',
    'PARAM_SPACE',
    'ParallelDetectionRunner',
    'DetectionResult',
    'AggregateResult',

    # Phase 7.7 - Trade Simulator
    'TradeSimulator',
    'TradeManagementConfig',
    'SimulatedTrade',
    'SimulationResult',
    'ExitReason',

    # Phase 7.7 - Objectives
    'ObjectiveFunction',
    'ObjectiveResult',
    'ObjectiveConfig',
    'ObjectiveType',
    'create_objective',
    'get_all_objective_types',
    'CountQualityObjective',
    'SharpeObjective',
    'ExpectancyObjective',
    'ProfitFactorObjective',
    'MaxDrawdownObjective',
    'CompositeObjective',

    # Phase 7.7 - Extended Runner
    'ExtendedDetectionRunner',
    'ExtendedRunnerConfig',
    'WalkForwardConfig',
    'WalkForwardResult',
    'ClusterValidationResult',
    'SYMBOL_CLUSTERS',
    'ALL_CLUSTERED_SYMBOLS',
]
