"""
Adaptive Function Approximation Engine (AFAE)
Core Neuro-Symbolic Series Expansion Package
"""

from .symbolic_engine import SymbolicExpansionEngine
from .neural_optimizer import HybridSeriesLayer, NeuralCoefficientOptimizer
from .router import AdaptiveBasisRouter

__version__ = "0.2.0"
__author__ = "Rishit Gupta"

__all__ = [
    "SymbolicExpansionEngine",
    "HybridSeriesLayer",
    "NeuralCoefficientOptimizer",
    "AdaptiveBasisRouter",
]
