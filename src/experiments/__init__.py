"""
Experiment Lab Module - Phase 6
===============================
A/B testing framework for systematic parameter optimization.

Components:
- ParameterSet: Single parameter configuration
- GridSearchConfig: Defines search space
- ParameterGridManager: Deduplication and tracking
- ExperimentRunner: Automated backtesting
- Statistics: BH correction and significance testing

Usage:
    from src.experiments import (
        ParameterSet, GridSearchConfig, ParameterGridManager,
        ExperimentRunner, ExperimentResult,
        benjamini_hochberg_correction, get_significant_discoveries
    )
    from src.data.sqlite_manager import SQLiteManager

    # Setup
    db = SQLiteManager()
    manager = ParameterGridManager(db)

    # Check grid size
    config = GridSearchConfig()
    print(f"Total combinations: {config.total_combinations():,}")

    # Get untested parameters
    for params in manager.get_untested(config, limit=100):
        print(f"Testing: {params}")

    # After running experiments, analyze significance
    results = get_significant_discoveries(experiment_results)
    print(f"Significant: {len(results['significant'])}")
"""

from .parameter_grid import (
    ParameterSet,
    GridSearchConfig,
    ParameterGridManager,
)

from .runner import (
    ExperimentResult,
    ExperimentRunner,
)

from .statistics import (
    SignificanceResult,
    benjamini_hochberg_correction,
    analyze_experiment_significance,
    rank_experiments,
    calculate_experiment_p_value,
    add_p_values_to_experiments,
    get_significant_discoveries,
)

__all__ = [
    # Parameter Grid
    'ParameterSet',
    'GridSearchConfig',
    'ParameterGridManager',

    # Runner
    'ExperimentResult',
    'ExperimentRunner',

    # Statistics
    'SignificanceResult',
    'benjamini_hochberg_correction',
    'analyze_experiment_significance',
    'rank_experiments',
    'calculate_experiment_p_value',
    'add_p_values_to_experiments',
    'get_significant_discoveries',
]
