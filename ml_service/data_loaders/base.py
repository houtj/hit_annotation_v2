"""Base data loading utilities shared across tasks"""

import sqlite3
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch


def load_human_labeled_files(db_path: Path, task: str = "segmentation") -> List[int]:
    """
    Load list of file IDs that have human labels
    
    Args:
        db_path: Path to SQLite database
        task: Task type ("segmentation" or "classification")
    
    Returns:
        List of file IDs with human labels
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if task == "segmentation":
        # For segmentation, look for point-type labels with human origin
        query = """
            SELECT DISTINCT l.file_id
            FROM labels l
            WHERE l.label_data != '[]'
            AND l.created_by NOT LIKE 'auto%'
        """
    else:
        # For classification, look for class-type labels with human origin
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
        - features: torch.Tensor of shape (384, H, W) in fp32
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
    
    return json.loads(row[0])


def load_classes(db_path: Path) -> List[Dict]:
    """
    Load class definitions from database
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        List of class dictionaries with 'classname' and 'color'
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT classname, color FROM classes ORDER BY classname")
    rows = cursor.fetchall()
    conn.close()
    
    return [{"classname": row[0], "color": row[1]} for row in rows]


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
