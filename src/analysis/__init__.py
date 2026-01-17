"""
Analysis Module for QML Trading System
=======================================
Advanced analytics including regime detection and sensitivity analysis.
"""

from src.analysis.sensitivity import SensitivityVisualizer

# Lazy imports to avoid circular dependencies
def get_regime_classifier():
    from src.analysis.regimes import RegimeClassifier
    return RegimeClassifier

def get_diagnostics():
    from src.analysis.diagnostics import AdvancedDiagnostics
    return AdvancedDiagnostics

__all__ = [
    "SensitivityVisualizer",
    "get_regime_classifier",
    "get_diagnostics",
]

