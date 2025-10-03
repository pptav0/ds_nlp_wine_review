# wine_review/modeling/__init__.py
"""
Modeling subpackage for the Wine Review project.

Provides:
- Weighted BCE Trainer for multilabel tasks
- Flavor multi-label dataset & inference
- Sentiment single-label dataset, metrics & predictor
- (Optional) taxonomy classifiers for variety/country
"""

# utils
from .utils import WeightedBCETrainer

# flavor (multi-label)
from .flavor import MultiLabelTextDS, infer_probs

# sentiment (single-label)
from .sentiment import (
    points_to_sentiment,
    SingleLabelTextDS,
    sent_metrics,
    predict_sentiment,
)

# taxonomy (to be added in taxonomy.py)
try:
    from .taxonomy import train_one_classifier, load_classifier, predict_cls

except ImportError:
    # taxonomy helpers may not be defined yet; ignore if missing
    pass

__all__ = [
    # utils
    "WeightedBCETrainer",
    # flavor
    "MultiLabelTextDS",
    "infer_probs",
    # sentiment
    "points_to_sentiment",
    "SingleLabelTextDS",
    "sent_metrics",
    "predict_sentiment",
    # taxonomy (optional)
    "train_one_classifier",
    "load_classifier",
    "predict_cls",
]
