"""Binary segmentation model architecture"""

import torch
import torch.nn as nn


class BinarySegmentationHead(nn.Module):
    """
    Lightweight CNN head for binary segmentation (foreground/background)
    
    Architecture:
        - Input: (B, in_channels, H, W) - DINOv3 features (384, 96, 96)
        - Conv 3x3: in_channels → 128, ReLU, BatchNorm
        - Conv 3x3: 128 → 64, ReLU, BatchNorm
        - Conv 1x1: 64 → 1, Sigmoid
        - Output: (B, 1, H, W) - probability map [0, 1]
    
    Receptive Field: 5×5 patches (80×80 pixels in original image)
    
    This provides spatial context so the model can understand that neighboring
    patches form continuous structures (e.g., thin lines), rather than treating
    each patch independently.
    
    Parameters: ~50k (vs ~400 for simple 1x1 conv)
    """
    
    def __init__(self, in_channels: int = 384, hidden_dim1: int = 128, hidden_dim2: int = 64):
        """
        Initialize CNN segmentation head
        
        Args:
            in_channels: Number of input feature channels (384 for DINOv3-small)
            hidden_dim1: Hidden dimension for first conv layer
            hidden_dim2: Hidden dimension for second conv layer
        """
        super().__init__()
        self.in_channels = in_channels
        
        # First conv block: 3x3 conv with spatial context
        self.conv1 = nn.Conv2d(in_channels, hidden_dim1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second conv block: further refine spatial features
        self.conv2 = nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Final classifier: 1x1 conv to output
        self.classifier = nn.Conv2d(hidden_dim2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features of shape (B, in_channels, H, W)
        
        Returns:
            Probability map of shape (B, 1, H, W) with values in [0, 1]
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Classifier
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
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Classifier (no sigmoid)
        return self.classifier(x)
