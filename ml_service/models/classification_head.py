"""Multi-class classification model architecture"""

import torch
import torch.nn as nn


class MultiClassHead(nn.Module):
    """
    Lightweight CNN head for multi-class image classification
    
    Architecture:
        - Input: (B, in_channels, H, W) - DINOv3 features (384, 96, 96)
        - Conv 3x3: in_channels → 128, ReLU, BatchNorm
        - Conv 3x3: 128 → 64, ReLU, BatchNorm
        - Global Average Pooling: (B, 64, H, W) → (B, 64)
        - Linear: 64 → num_classes
        - Output: (B, num_classes) - class logits (apply softmax externally)
    
    Uses global average pooling to aggregate spatial features into a single
    classification prediction per image.
    
    Parameters: ~50k (similar to segmentation head)
    """
    
    def __init__(
        self, 
        num_classes: int, 
        in_channels: int = 384, 
        hidden_dim1: int = 128, 
        hidden_dim2: int = 64
    ):
        """
        Initialize CNN classification head
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input feature channels (384 for DINOv3-small)
            hidden_dim1: Hidden dimension for first conv layer
            hidden_dim2: Hidden dimension for second conv layer
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # First conv block: 3x3 conv with spatial context
        self.conv1 = nn.Conv2d(in_channels, hidden_dim1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second conv block: further refine spatial features
        self.conv2 = nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final classifier: linear layer to class logits
        self.classifier = nn.Linear(hidden_dim2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features of shape (B, in_channels, H, W)
        
        Returns:
            Class logits of shape (B, num_classes)
            Note: Does NOT apply softmax - use with CrossEntropyLoss which expects raw logits
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        logits = self.classifier(x)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (softmax applied)
        
        Args:
            x: Input features of shape (B, in_channels, H, W)
        
        Returns:
            Class probabilities of shape (B, num_classes) summing to 1
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicted class and confidence
        
        Args:
            x: Input features of shape (B, in_channels, H, W)
        
        Returns:
            Tuple of (predicted_class_indices, confidence_scores)
            - predicted_class_indices: (B,) tensor of class indices
            - confidence_scores: (B,) tensor of confidence scores [0, 1]
        """
        proba = self.predict_proba(x)
        confidence, predicted = torch.max(proba, dim=1)
        return predicted, confidence
