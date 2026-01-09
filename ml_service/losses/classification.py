"""Classification loss functions (CrossEntropy-based)"""

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """
    Cross Entropy loss for multi-class classification with class weighting
    """
    
    def __init__(self, class_weights: torch.Tensor | None = None):
        """
        Args:
            class_weights: Optional tensor of shape (num_classes,) with per-class weights
        """
        super().__init__()
        self.class_weights = class_weights
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss
        
        Args:
            logits: (B, num_classes) raw logits from model
            targets: (B,) class indices
        
        Returns:
            Scalar loss value
        """
        return self.loss_fn(logits, targets)


def create_classification_loss(
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None
) -> ClassificationLoss:
    """
    Create a classification loss function
    
    Args:
        class_weights: Optional tensor of shape (num_classes,) with per-class weights
        device: Device to place the loss function on
    
    Returns:
        ClassificationLoss instance
    """
    loss = ClassificationLoss(class_weights)
    if device is not None:
        loss = loss.to(device)
    return loss
