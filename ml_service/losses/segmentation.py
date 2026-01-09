"""Segmentation loss functions (BCE-based)"""

from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_segmentation_class_weights(
    train_data: List[Dict],
    device: torch.device
) -> torch.Tensor:
    """
    Calculate class weights based on the frequency of each class (inverse frequency weighting)
    
    Args:
        train_data: List of training samples (each with 'points' key)
        device: torch device
    
    Returns:
        Tensor of shape (2,) with weights for [background, foreground]
    """
    class_counts = torch.zeros(2, dtype=torch.float32)
    
    for sample in train_data:
        points = sample['points']
        for _, _, class_idx in points:
            class_counts[class_idx] += 1
    
    if class_counts.sum() == 0:
        return torch.ones(2, device=device)
    
    # Inverse frequency weighting: weight = total / (num_classes * count)
    total_points = class_counts.sum()
    weights = total_points / (2.0 * class_counts)
    
    # Handle zero counts (set weight to 1.0)
    weights[class_counts == 0] = 1.0
    
    # Normalize weights so they sum to num_classes (for stability)
    weights = weights / weights.mean()
    
    print(f"Class distribution: Background={int(class_counts[0])}, Foreground={int(class_counts[1])}")
    print(f"Class weights: Background={weights[0]:.3f}, Foreground={weights[1]:.3f}")
    
    return weights.to(device)


class SegmentationLoss(nn.Module):
    """
    Binary Cross Entropy loss for segmentation with per-point weighting
    
    This loss function:
    1. Only computes loss at labeled pixel locations (sparse supervision)
    2. Applies class weights for handling imbalanced classes
    """
    
    def __init__(self, class_weights: torch.Tensor | None = None):
        """
        Args:
            class_weights: Tensor of shape (2,) with weights for [background, foreground]
        """
        super().__init__()
        self.class_weights = class_weights
    
    def forward(
        self,
        predictions: torch.Tensor,
        label_mask: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss only at labeled locations
        
        Args:
            predictions: (B, 1, H, W) probability predictions
            label_mask: (B, H, W) bool mask of labeled pixels
            target_mask: (B, H, W) target values at labeled pixels
        
        Returns:
            Scalar loss value
        """
        # Squeeze channel dimension from predictions
        predictions = predictions.squeeze(1)  # (B, H, W)
        
        # Get predictions and targets at labeled locations
        pred_at_labels = predictions[label_mask]
        target_at_labels = target_mask[label_mask]
        
        if pred_at_labels.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Compute per-sample BCE loss
        bce = F.binary_cross_entropy(pred_at_labels, target_at_labels, reduction='none')
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = torch.where(
                target_at_labels > 0.5,
                self.class_weights[1],  # foreground weight
                self.class_weights[0]   # background weight
            )
            bce = bce * weights
        
        return bce.mean()


def create_segmentation_loss(class_weights: torch.Tensor | None = None) -> SegmentationLoss:
    """
    Create a segmentation loss function
    
    Args:
        class_weights: Optional tensor of shape (2,) with weights for [background, foreground]
    
    Returns:
        SegmentationLoss instance
    """
    return SegmentationLoss(class_weights)
