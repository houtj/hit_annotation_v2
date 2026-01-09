"""Multi-class classification model architecture"""

import torch
import torch.nn as nn


class MultiClassHead(nn.Module):
    """
    Classification head for multi-class image classification
    
    Architecture:
        - Input: (B, in_channels, H, W) - DINOv3 features (384, 96, 96)
        - Global Average Pooling: (B, in_channels, H, W) → (B, in_channels)
        - FC Layer 1: in_channels → hidden_dim, ReLU, Dropout
        - FC Layer 2: hidden_dim → num_classes
        - Output: (B, num_classes) - class logits
    
    The global average pooling aggregates spatial information across the entire
    feature map, making the model robust to different input sizes and focusing
    on global image-level features rather than local patterns.
    
    Parameters: ~50k for 4 classes (vs ~50k for segmentation head)
    """
    
    def __init__(self, in_channels: int = 384, num_classes: int = 4, hidden_dim: int = 128, dropout: float = 0.3):
        """
        Initialize classification head
        
        Args:
            in_channels: Number of input feature channels (384 for DINOv3-small)
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for FC layer
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Global average pooling - aggregates spatial features
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Two-layer MLP classifier
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features of shape (B, in_channels, H, W)
        
        Returns:
            Class logits of shape (B, num_classes)
        """
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.gap(x)
        
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1)
        
        # MLP classifier
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicted class and confidence
        
        Args:
            x: Input features of shape (B, in_channels, H, W)
        
        Returns:
            Tuple of (predicted_class_indices, confidence_scores)
            - predicted_class_indices: (B,) - index of predicted class
            - confidence_scores: (B,) - confidence score [0, 1]
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        return predicted, confidence
