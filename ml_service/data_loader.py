"""Data loading utilities for training"""

import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import random


def load_human_labeled_files(db_path: Path) -> List[int]:
    """
    Load list of file IDs that have human labels
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        List of file IDs with human labels
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query files that have non-empty labels not created by "auto*"
    query = """
        SELECT DISTINCT l.file_id
        FROM labels l
        WHERE l.label_data != '[]'
        AND l.created_by NOT LIKE 'auto%'
    """
    
    cursor.execute(query)
    file_ids = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return file_ids


def load_features(file_id: int, session_dir: Path) -> Tuple[torch.Tensor, int, int]:
    """
    Load DINOv3 features for a file
    
    Args:
        file_id: File ID in database
        session_dir: Path to session directory
    
    Returns:
        Tuple of (features, original_width, original_height)
        - features: torch.Tensor of shape (384, H, W) in fp16
        - original_width: Original image width
        - original_height: Original image height
    """
    db_path = session_dir / "annotations.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get file info
    cursor.execute(
        "SELECT feature_path, width, height FROM files WHERE id = ?",
        (file_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"File ID {file_id} not found in database")
    
    feature_path, width, height = row
    
    # Load features from .npy file
    feature_full_path = session_dir / feature_path
    if not feature_full_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_full_path}")
    
    features = np.load(feature_full_path)
    features_tensor = torch.from_numpy(features).float()  # Convert fp16 to fp32
    
    return features_tensor, width, height


def load_labels(file_id: int, db_path: Path) -> List[Dict]:
    """
    Load label data for a file
    
    Args:
        file_id: File ID in database
        db_path: Path to SQLite database
    
    Returns:
        List of label dictionaries (parsed from JSON)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT label_data FROM labels WHERE file_id = ?",
        (file_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row or not row[0]:
        return []
    
    import json
    return json.loads(row[0])


def map_points_to_feature_coords(
    points: List[Dict],
    orig_width: int,
    orig_height: int,
    resize: int = 1536
) -> List[Tuple[int, int, int]]:
    """
    Map point annotations to feature map coordinates
    
    Args:
        points: List of point label dicts with keys: x, y, classname (in pixel coordinates)
        orig_width: Original image width in pixels
        orig_height: Original image height in pixels
        resize: Target resize dimension (default 1536)
    
    Returns:
        List of (fy, fx, class_idx) tuples where:
        - fy, fx: feature map coordinates
        - class_idx: 1 for foreground, 0 for background
    
    Coordinate mapping (matching reference dinov3_training.py):
        1. Original (x, y) in pixels
        2. Normalize to [0, 1]: col = x / (orig_width - 1), row = y / (orig_height - 1)
        3. Scale to resized: x_pad = col * (new_w - 1), y_pad = row * (new_h - 1)
        4. Feature map: fx = floor(x_pad / 16), fy = floor(y_pad / 16)
    """
    # Calculate scale factor
    scale = resize / max(orig_width, orig_height)
    
    # Calculate resized dimensions (rounded to multiples of 16)
    new_w = (int(orig_width * scale) // 16) * 16
    new_h = (int(orig_height * scale) // 16) * 16
    
    mapped_points = []
    for point in points:
        if point.get('type') != 'point':
            continue
        
        x = point.get('x')
        y = point.get('y')
        classname = point.get('classname', 'foreground')
        
        if x is None or y is None:
            continue
        
        # Normalize pixel coordinates to [0, 1] (matching reference)
        col = float(x) / max(1, orig_width - 1)
        row = float(y) / max(1, orig_height - 1)
        
        # Map to padded/resized image coordinates (matching reference)
        x_pad = col * (new_w - 1)
        y_pad = row * (new_h - 1)
        
        # Map to feature coordinates (16x downsampling, using floor)
        fx = int(np.floor(x_pad / 16.0))
        fy = int(np.floor(y_pad / 16.0))
        
        # Calculate feature map dimensions
        feat_w = new_w // 16
        feat_h = new_h // 16
        
        # Clamp to valid range
        fx = min(max(fx, 0), feat_w - 1)
        fy = min(max(fy, 0), feat_h - 1)
        
        # Map class name to index (1=foreground, 0=background)
        class_idx = 1 if classname.lower() == 'foreground' else 0
        
        mapped_points.append((fy, fx, class_idx))
    
    return mapped_points


def split_train_test(
    file_ids: List[int],
    test_ratio: float = 0.2,
    min_train: int = 1,
    min_test: int = 1
) -> Tuple[List[int], List[int]]:
    """
    Split file IDs into train and test sets
    
    Args:
        file_ids: List of file IDs to split
        test_ratio: Ratio of files for test set (default 0.2 = 20%)
        min_train: Minimum number of training files
        min_test: Minimum number of test files
    
    Returns:
        Tuple of (train_ids, test_ids)
    """
    if len(file_ids) < (min_train + min_test):
        raise ValueError(
            f"Need at least {min_train + min_test} files, got {len(file_ids)}"
        )
    
    # Shuffle file IDs
    file_ids_shuffled = file_ids.copy()
    random.shuffle(file_ids_shuffled)
    
    # Calculate split point
    n_test = max(min_test, int(len(file_ids) * test_ratio))
    n_test = min(n_test, len(file_ids) - min_train)  # Ensure enough for train
    
    test_ids = file_ids_shuffled[:n_test]
    train_ids = file_ids_shuffled[n_test:]
    
    return train_ids, test_ids


def create_label_masks(
    points: List[Tuple[int, int, int]],
    feat_height: int,
    feat_width: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create dense masks from sparse point labels for efficient vectorized training
    
    Args:
        points: List of (fy, fx, class_idx) tuples
        feat_height: Feature map height
        feat_width: Feature map width
    
    Returns:
        Tuple of (label_mask, target_mask, weight_mask):
        - label_mask: (H, W) bool tensor, True where labels exist
        - target_mask: (H, W) float tensor, target values (0 or 1) at labeled locations
        - weight_mask: (H, W) float tensor, weights at labeled locations (for class balancing)
    """
    # Initialize masks (all zeros/False)
    label_mask = torch.zeros(feat_height, feat_width, dtype=torch.bool)
    target_mask = torch.zeros(feat_height, feat_width, dtype=torch.float32)
    weight_mask = torch.ones(feat_height, feat_width, dtype=torch.float32)  # Default weight 1.0
    
    # Fill in labeled locations
    for fy, fx, class_idx in points:
        label_mask[fy, fx] = True
        target_mask[fy, fx] = float(class_idx)
        # Weight will be set during training based on class weights
    
    return label_mask, target_mask, weight_mask


def load_training_data(
    file_ids: List[int],
    session_dir: Path,
    resize: int = 1536
) -> List[Dict]:
    """
    Load training data for a list of files with pre-computed masks for efficient training
    
    Args:
        file_ids: List of file IDs to load
        session_dir: Path to session directory
        resize: Target resize dimension
    
    Returns:
        List of training sample dictionaries with keys:
        - file_id: int
        - features: torch.Tensor (384, H, W)
        - points: List[(fy, fx, class_idx)] (kept for compatibility)
        - label_mask: torch.Tensor (H, W) bool - where labels exist
        - target_mask: torch.Tensor (H, W) float - target values at labeled locations
        - orig_width: int
        - orig_height: int
    """
    db_path = session_dir / "annotations.db"
    training_data = []
    
    for file_id in file_ids:
        try:
            # Load features
            features, orig_width, orig_height = load_features(file_id, session_dir)
            feat_height, feat_width = features.shape[1], features.shape[2]
            
            # Load labels
            labels = load_labels(file_id, db_path)
            
            # Map points to feature coordinates
            points = map_points_to_feature_coords(
                labels, orig_width, orig_height, resize
            )
            
            if not points:
                # Skip files with no valid point annotations
                continue
            
            # Create dense masks from sparse points
            label_mask, target_mask, weight_mask = create_label_masks(
                points, feat_height, feat_width
            )
            
            training_data.append({
                'file_id': file_id,
                'features': features,
                'points': points,  # Keep for deduplication logic
                'label_mask': label_mask,
                'target_mask': target_mask,
                'weight_mask': weight_mask,
                'orig_width': orig_width,
                'orig_height': orig_height
            })
        
        except Exception as e:
            print(f"Warning: Failed to load file {file_id}: {e}")
            continue
    
    return training_data

