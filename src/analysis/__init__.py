"""
Analysis Module for QML Trading System
=======================================
Advanced analytics including regime detection and sensitivity analysis.
"""

from src.analysis.regimes import RegimeClassifier, RegimeConfig, RegimeResult
from src.analysis.sensitivity import ParameterScanner, SensitivityConfig, SensitivityResult
from src.analysis.diagnostics import (
    AdvancedDiagnostics,
    DiagnosticsResult,
    VolatilityExpansionResult,
    DrawdownDecomposition,
    CorrelationAnalysis,
)

__all__ = [
    "RegimeClassifier",
    "RegimeConfig",
    "RegimeResult",
    "ParameterScanner",
    "SensitivityConfig",
    "SensitivityResult",
    "AdvancedDiagnostics",
    "DiagnosticsResult",
    "VolatilityExpansionResult",
    "DrawdownDecomposition",
    "CorrelationAnalysis",
]
