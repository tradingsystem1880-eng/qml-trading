"""
Reporting Module
================
Tools for logging experiments and generating reports.

Components:
- ExperimentLogger: SQLite-based experiment storage
- DossierGenerator: HTML report generation
"""

from src.reporting.storage import ExperimentLogger
from src.reporting.dossier import DossierGenerator

__all__ = ['ExperimentLogger', 'DossierGenerator']
