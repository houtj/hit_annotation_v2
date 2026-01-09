"""Classification-specific data loading (image-based)"""

from pathlib import Path
from typing import List, Dict
import torch

from .base import load_features, load_labels, load_classes


def get_class_label(labels: List[Dict]) -> str | None:
    """
    Extract class label from label data (for classification task)
    
    Args:
        labels: List of label dictionaries
    
    Returns:
        Class name string if found, None otherwise
    """
    for label in labels:
        if label.get('type') == 'class':
            return label.get('classname')
    return None


def load_classification_training_data(
    file_ids: List[int],
    session_dir: Path,
    class_names: List[str] | None = None
) -> List[Dict]:
    """
    Load training data for classification (image-level labeling)
    
    Args:
        file_ids: List of file IDs to load
        session_dir: Path to session directory
        class_names: Optional list of class names (for consistent ordering)
                    If None, will load from database
    
    Returns:
        List of training sample dictionaries with keys:
        - file_id: int
        - features: torch.Tensor (384, H, W) - full feature map
        - class_idx: int - class index (0 to num_classes-1)
        - class_name: str - class name
        - orig_width: int
        - orig_height: int
    """
    db_path = session_dir / "annotations.db"
    
    # Load class names if not provided
    if class_names is None:
        classes = load_classes(db_path)
        class_names = [c['classname'] for c in classes]
    
    # Create class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    training_data = []
    
    for file_id in file_ids:
        try:
            # Load features
            features, orig_width, orig_height = load_features(file_id, session_dir)
            
            # Load labels
            labels = load_labels(file_id, db_path)
            
            # Get class label
            class_name = get_class_label(labels)
            
            if class_name is None:
                # Skip files without class labels
                continue
            
            if class_name not in class_to_idx:
                print(f"Warning: Unknown class '{class_name}' for file {file_id}, skipping")
                continue
            
            class_idx = class_to_idx[class_name]
            
            training_data.append({
                'file_id': file_id,
                'features': features,
                'class_idx': class_idx,
                'class_name': class_name,
                'orig_width': orig_width,
                'orig_height': orig_height
            })
        
        except Exception as e:
            print(f"Warning: Failed to load file {file_id}: {e}")
            continue
    
    return training_data


def create_classification_batch(
    samples: List[Dict],
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a batch from classification samples
    
    Args:
        samples: List of sample dictionaries from load_classification_training_data
        device: Target device (cuda/cpu)
    
    Returns:
        Tuple of (features_batch, labels_batch):
        - features_batch: (B, 384, H, W) tensor
        - labels_batch: (B,) tensor of class indices
    """
    features_list = [s['features'] for s in samples]
    labels_list = [s['class_idx'] for s in samples]
    
    # Stack features (assumes all have same spatial dimensions)
    features_batch = torch.stack(features_list, dim=0).to(device)
    labels_batch = torch.tensor(labels_list, dtype=torch.long, device=device)
    
    return features_batch, labels_batch


def calculate_class_weights_classification(
    training_data: List[Dict],
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Calculate class weights based on frequency (inverse frequency weighting)
    
    Args:
        training_data: List of training samples
        num_classes: Total number of classes
        device: Target device
    
    Returns:
        Tensor of shape (num_classes,) with weights
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    
    for sample in training_data:
        class_counts[sample['class_idx']] += 1
    
    if class_counts.sum() == 0:
        return torch.ones(num_classes, device=device)
    
    # Inverse frequency weighting
    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts)
    
    # Handle zero counts (set weight to 1.0)
    weights[class_counts == 0] = 1.0
    
    # Normalize weights
    weights = weights / weights.mean()
    
    print(f"Class distribution: {dict(enumerate(class_counts.tolist()))}")
    print(f"Class weights: {dict(enumerate(weights.tolist()))}")
    
    return weights.to(device)
