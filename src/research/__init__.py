"""
Research Module
===============
Experimental concepts and analysis tools.

These are for RESEARCH ONLY - not production ready.

Phase 9.7 additions:
- ResearchJournal: Track experiments, prevent re-testing failed ideas
- FeatureValidator: Rigorous validation pipeline for new features
"""

from .smart_money_concepts import (
    OrderBlock,
    OrderBlockType,
    FairValueGap,
    FVGType,
    OrderBlockDetector,
    FVGDetector,
    SMCAnalyzer,
    get_smc_features,
)

from .research_journal import (
    ResearchJournal,
    ExperimentResult,
)

from .feature_validator import (
    FeatureValidator,
    FeatureValidatorConfig,
    FeatureValidationResult,
    ValidationStepResult,
)

__all__ = [
    # Smart Money Concepts
    'OrderBlock',
    'OrderBlockType',
    'FairValueGap',
    'FVGType',
    'OrderBlockDetector',
    'FVGDetector',
    'SMCAnalyzer',
    'get_smc_features',
    # Research Infrastructure (Phase 9.7)
    'ResearchJournal',
    'ExperimentResult',
    'FeatureValidator',
    'FeatureValidatorConfig',
    'FeatureValidationResult',
    'ValidationStepResult',
]
