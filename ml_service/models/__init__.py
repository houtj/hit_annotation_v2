"""Model heads for different tasks"""

from .segmentation_head import BinarySegmentationHead
from .classification_head import MultiClassHead

__all__ = ['BinarySegmentationHead', 'MultiClassHead']
