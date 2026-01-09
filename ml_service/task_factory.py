"""
Task Factory - Central dispatcher for task-specific components

This module provides factory methods to create task-specific model heads,
data loaders, and loss functions based on the task type configuration.
"""

from pathlib import Path
from typing import List, Dict, Callable
import torch
import torch.nn as nn

from models import BinarySegmentationHead, MultiClassHead
from data_loaders import (
    load_segmentation_training_data,
    load_classification_training_data,
    load_classes
)
from data_loaders.classification import calculate_class_weights_classification
from losses import (
    create_segmentation_loss,
    calculate_segmentation_class_weights,
    create_classification_loss
)


def create_model_head(
    task_type: str,
    num_classes: int = 2,
    in_channels: int = 384
) -> nn.Module:
    """
    Create a model head based on task type
    
    Args:
        task_type: "segmentation" or "classification"
        num_classes: Number of classes (only used for classification)
        in_channels: Number of input feature channels (384 for DINOv3-small)
    
    Returns:
        Appropriate model head (BinarySegmentationHead or MultiClassHead)
    """
    if task_type == "segmentation":
        return BinarySegmentationHead(in_channels=in_channels)
    elif task_type == "classification":
        return MultiClassHead(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def load_training_data(
    task_type: str,
    file_ids: List[int],
    session_dir: Path,
    resize: int = 1536,
    class_names: List[str] | None = None
) -> List[Dict]:
    """
    Load training data based on task type
    
    Args:
        task_type: "segmentation" or "classification"
        file_ids: List of file IDs to load
        session_dir: Path to session directory
        resize: Target resize dimension (used for segmentation)
        class_names: List of class names (used for classification)
    
    Returns:
        List of training sample dictionaries (format depends on task)
    """
    if task_type == "segmentation":
        return load_segmentation_training_data(file_ids, session_dir, resize)
    elif task_type == "classification":
        return load_classification_training_data(file_ids, session_dir, class_names)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def create_loss_fn(
    task_type: str,
    training_data: List[Dict],
    num_classes: int,
    device: torch.device
) -> tuple[nn.Module, torch.Tensor]:
    """
    Create loss function and class weights based on task type
    
    Args:
        task_type: "segmentation" or "classification"
        training_data: Training data samples (for computing class weights)
        num_classes: Number of classes
        device: Target device
    
    Returns:
        Tuple of (loss_function, class_weights)
    """
    if task_type == "segmentation":
        class_weights = calculate_segmentation_class_weights(training_data, device)
        loss_fn = create_segmentation_loss(class_weights)
        return loss_fn, class_weights
    elif task_type == "classification":
        class_weights = calculate_class_weights_classification(
            training_data, num_classes, device
        )
        loss_fn = create_classification_loss(class_weights, device)
        return loss_fn, class_weights
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_class_names(session_dir: Path) -> List[str]:
    """
    Get list of class names from database
    
    Args:
        session_dir: Path to session directory
    
    Returns:
        List of class names in sorted order
    """
    db_path = session_dir / "annotations.db"
    classes = load_classes(db_path)
    return [c['classname'] for c in classes]


def get_num_classes(session_dir: Path) -> int:
    """
    Get number of classes from database
    
    Args:
        session_dir: Path to session directory
    
    Returns:
        Number of classes
    """
    return len(get_class_names(session_dir))


def decode_segmentation_prediction(
    output: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Decode segmentation model output to binary mask
    
    Args:
        output: Model output of shape (B, 1, H, W) with probabilities
        threshold: Threshold for binarization
    
    Returns:
        Binary mask of shape (B, 1, H, W)
    """
    return (output > threshold).float()


def decode_classification_prediction(
    output: torch.Tensor,
    class_names: List[str]
) -> List[Dict]:
    """
    Decode classification model output to prediction dictionaries
    
    Args:
        output: Model output of shape (B, num_classes) with logits
        class_names: List of class names
    
    Returns:
        List of prediction dictionaries with keys:
        - class: predicted class name
        - confidence: confidence score
        - probabilities: dict of class -> probability
    """
    proba = torch.softmax(output, dim=1)  # (B, num_classes)
    confidence, predicted = torch.max(proba, dim=1)  # (B,), (B,)
    
    results = []
    for i in range(output.shape[0]):
        pred_idx = predicted[i].item()
        conf = confidence[i].item()
        prob_dict = {
            class_names[j]: proba[i, j].item()
            for j in range(len(class_names))
        }
        
        results.append({
            "type": "class",
            "class": class_names[pred_idx],
            "confidence": conf,
            "probabilities": prob_dict
        })
    
    return results


def get_checkpoint_prefix(task_type: str) -> str:
    """
    Get checkpoint filename prefix based on task type
    
    Args:
        task_type: "segmentation" or "classification"
    
    Returns:
        Prefix string for checkpoint files
    """
    if task_type == "segmentation":
        return "binary_seg_head"
    elif task_type == "classification":
        return "classification_head"
    else:
        return "model_head"


def create_model_head_from_config(
    task_type: str,
    config: Dict,
    device: torch.device
) -> nn.Module:
    """
    Create a model head based on task type and config
    
    Args:
        task_type: "segmentation" or "classification"
        config: Configuration dictionary with 'classes' key
        device: Target device
    
    Returns:
        Model head on the specified device
    """
    num_classes = len(config.get('classes', []))
    head = create_model_head(task_type, num_classes=num_classes)
    return head.to(device)
