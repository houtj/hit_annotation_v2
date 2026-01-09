"""Task-specific handler functions"""

from .segmentation import extract_points_from_prediction
from .classification import get_classification_prediction

__all__ = ['extract_points_from_prediction', 'get_classification_prediction']
