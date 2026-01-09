"""Loss functions for different tasks"""

from .segmentation import create_segmentation_loss, calculate_segmentation_class_weights
from .classification import create_classification_loss

__all__ = [
    'create_segmentation_loss',
    'calculate_segmentation_class_weights',
    'create_classification_loss'
]
