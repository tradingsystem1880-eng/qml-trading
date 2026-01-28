"""
Research Module
===============
Experimental concepts and analysis tools.

These are for RESEARCH ONLY - not production ready.
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

__all__ = [
    'OrderBlock',
    'OrderBlockType',
    'FairValueGap',
    'FVGType',
    'OrderBlockDetector',
    'FVGDetector',
    'SMCAnalyzer',
    'get_smc_features',
]
