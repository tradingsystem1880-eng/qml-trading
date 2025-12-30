"""
QML Pattern Detection Module
=============================
Core detection engine for identifying Quasimodo (QML) patterns.

Submodules:
    - swing: Swing point identification
    - structure: Market structure analysis
    - choch: Change of Character detection
    - bos: Break of Structure detection
    - detector: Main QML pattern detector
"""

from src.detection.swing import SwingDetector
from src.detection.structure import StructureAnalyzer
from src.detection.choch import CHoCHDetector
from src.detection.bos import BoSDetector
from src.detection.detector import QMLDetector

__all__ = [
    "SwingDetector",
    "StructureAnalyzer", 
    "CHoCHDetector",
    "BoSDetector",
    "QMLDetector"
]

