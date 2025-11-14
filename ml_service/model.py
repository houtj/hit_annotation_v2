"""Binary segmentation model architecture"""

import torch
import torch.nn as nn


class BinarySegmentationHead(nn.Module):
    """
    1x1 convolutional head for binary segmentation (foreground/background)
    
    Architecture:
        - Input: (B, in_channels, H, W) - DINOv3 features
        - Conv2d 1x1: in_channels -> 1
        - Sigmoid activation
        - Output: (B, 1, H, W) - probability map [0, 1]
    
    Loss:
        - Binary Cross Entropy (BCE) loss on point locations
        - Optional dense supervision on full feature maps
    """
    
    def __init__(self, in_channels: int = 384):
        """
        Initialize binary segmentation head
        
        Args:
            in_channels: Number of input feature channels (384 for DINOv3-small)
        """
        super().__init__()
        self.in_channels = in_channels
        self.classifier = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features of shape (B, in_channels, H, W)
        
        Returns:
            Probability map of shape (B, 1, H, W) with values in [0, 1]
        """
        logits = self.classifier(x)
        return self.sigmoid(logits)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits without sigmoid (useful for some loss functions)
        
        Args:
            x: Input features of shape (B, in_channels, H, W)
        
        Returns:
            Raw logits of shape (B, 1, H, W)
        """
        return self.classifier(x)

