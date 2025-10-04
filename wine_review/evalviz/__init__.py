# wine_review/evalviz/__init__.py
"""
Evaluation & Visualization utilities for the Wine Review project.

This subpackage provides:
- Metrics helpers for model evaluation
- Visualization utilities (confusion matrices, PR curves, threshold sweeps, etc.)
"""

from .plots import plot_cm   # confusion matrix heatmap

# you can add more once you implement them
# from .metrics import *       # macro/micro metrics, AUPRC helpers, etc.

__all__ = [
    "plot_cm",
    # "plot_threshold_sweep",
    # "plot_f1_per_label",
]
