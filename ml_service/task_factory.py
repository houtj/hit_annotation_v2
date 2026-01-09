"""Task factory for creating task-specific components"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Callable, Tuple

from model import BinarySegmentationHead
from classification_head import MultiClassHead


def create_model_head(task_type: str, config: dict, device: torch.device) -> nn.Module:
    """
    Create task-specific model head
    
    Args:
        task_type: Either "segmentation" or "classification"
        config: Configuration dictionary with task-specific parameters
        device: torch device
    
    Returns:
        Model head (BinarySegmentationHead or MultiClassHead)
    """
    if task_type == "segmentation":
        head = BinarySegmentationHead(in_channels=384)
    elif task_type == "classification":
        # Get number of classes from config
        num_classes = len(config.get("classes", []))
        if num_classes == 0:
            raise ValueError("No classes defined in config for classification task")
        head = MultiClassHead(in_channels=384, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    head = head.to(device)
    return head


def create_loss_fn(task_type: str) -> Callable:
    """
    Create task-specific loss function
    
    Args:
        task_type: Either "segmentation" or "classification"
    
    Returns:
        Loss function
    """
    if task_type == "segmentation":
        # Binary cross-entropy loss for segmentation
        return nn.BCELoss()
    elif task_type == "classification":
        # Cross-entropy loss for multi-class classification
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def load_training_data_for_task(
    task_type: str,
    file_ids: list,
    session_dir: Path,
    config: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load training data based on task type
    
    Args:
        task_type: Either "segmentation" or "classification"
        file_ids: List of file IDs with human labels
        session_dir: Path to session directory
        config: Configuration dictionary
    
    Returns:
        Tuple of (features, labels)
        - For segmentation: features (N, 384), labels (N,) point-level
        - For classification: features (B, 384, H, W), labels (B,) image-level
    """
    if task_type == "segmentation":
        from data_loader import load_training_data
        return load_training_data(file_ids, session_dir)
    elif task_type == "classification":
        from data_loader import load_classification_data
        return load_classification_data(file_ids, session_dir, config)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_checkpoint_prefix(task_type: str) -> str:
    """
    Get checkpoint filename prefix based on task type
    
    Args:
        task_type: Either "segmentation" or "classification"
    
    Returns:
        Checkpoint filename prefix
    """
    if task_type == "segmentation":
        return "binary_seg_head"
    elif task_type == "classification":
        return "classification_head"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
