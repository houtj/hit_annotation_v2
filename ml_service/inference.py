"""Inference utilities for generating predictions"""

import sqlite3
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import BinarySegmentationHead
from data_loader import load_features


def predict_full_image(
    head: BinarySegmentationHead,
    file_id: int,
    session_dir: Path,
    version: str,
    device: torch.device,
    threshold: float = 0.5  # Not used anymore, kept for API compatibility
) -> Path:
    """
    Generate full-resolution probability map for an image
    
    Args:
        head: Trained segmentation head
        file_id: File ID in database
        session_dir: Path to session directory
        version: Model version string (e.g., "1.2")
        device: torch device (cuda or cpu)
        threshold: Not used (kept for API compatibility)
    
    Returns:
        Path to saved prediction mask PNG file (grayscale, 0-255 representing probabilities 0-1)
    
    Steps:
        1. Load pre-extracted features (384, 96, 96) from .npy
        2. Pass through head -> (1, 96, 96) probability map
        3. Upsample to original image resolution using bilinear interpolation
        4. Convert probabilities (0-1) to grayscale (0-255) for smooth visualization
        5. Delete old prediction if exists
        6. Save to session/storage/predictions/file_{file_id}.png
        7. Update Prediction table in database
    """
    # Load features and original dimensions
    features, orig_width, orig_height = load_features(file_id, session_dir)
    
    # Move to device and add batch dimension
    features = features.unsqueeze(0).to(device)  # (1, 384, H, W)
    
    # Generate prediction
    head.eval()
    with torch.no_grad():
        prob_map = head(features)  # (1, 1, H, W)
    
    # Remove batch dimension: (1, H, W)
    prob_map = prob_map.squeeze(0).squeeze(0)  # Remove batch and channel dims: (H, W)
    
    # Calculate resize parameters (matching init_session.py and reference)
    resize = 1536
    scale = resize / max(orig_width, orig_height)
    new_w = (int(orig_width * scale) // 16) * 16
    new_h = (int(orig_height * scale) // 16) * 16
    
    # Crop prob_map to valid region (excluding padding)
    # Reference: lines 802-807 in dinov3_training.py
    Fh, Fw = prob_map.shape  # Feature map dimensions (96, 96)
    image_rows_01 = new_h / max(new_w, new_h)
    image_cols_01 = new_w / max(new_w, new_h)
    
    if image_rows_01 < 1.0:
        # Image is wider than tall, crop height
        prob_map = prob_map[: int(Fh * image_rows_01), :]
    elif image_cols_01 < 1.0:
        # Image is taller than wide, crop width
        prob_map = prob_map[:, : int(Fw * image_cols_01)]
    
    # Now upsample the cropped prob_map to original image resolution
    prob_map_upsampled = F.interpolate(
        prob_map.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims: (1, 1, H', W')
        size=(orig_height, orig_width),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # Remove batch and channel dims: (orig_height, orig_width)
    
    # Convert probability map (0-1) to grayscale values (0-255) for visualization
    # Keep the soft probabilities instead of hard thresholding for smoother edges
    prob_map_numpy = prob_map_upsampled.cpu().numpy()
    prob_map_vis = (prob_map_numpy * 255).astype(np.uint8)
    
    # Create predictions directory
    predictions_dir = session_dir / "storage" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Check database for existing prediction to delete old file
    db_path = session_dir / "annotations.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT path FROM predictions WHERE file_id = ?",
        (file_id,)
    )
    existing = cursor.fetchone()
    
    if existing:
        # Delete old prediction file
        old_path = session_dir / existing[0]
        if old_path.exists():
            try:
                old_path.unlink()
                print(f"  Deleted old prediction: {old_path.name}")
            except Exception as e:
                print(f"  Warning: Could not delete old prediction {old_path.name}: {e}")
    
    # Use simple filename without version (one prediction per file)
    mask_filename = f"file_{file_id}.png"
    mask_path = predictions_dir / mask_filename
    
    # Save as grayscale image with soft probabilities
    mask_img = Image.fromarray(prob_map_vis, mode='L')
    mask_img.save(mask_path)
    
    # Store relative path
    relative_path = f"storage/predictions/{mask_filename}"
    
    if existing:
        # Update existing prediction
        cursor.execute(
            "UPDATE predictions SET path = ? WHERE file_id = ?",
            (relative_path, file_id)
        )
    else:
        # Insert new prediction
        cursor.execute(
            "INSERT INTO predictions (file_id, path) VALUES (?, ?)",
            (file_id, relative_path)
        )
    
    conn.commit()
    conn.close()
    
    return mask_path


def predict_batch(
    head: BinarySegmentationHead,
    file_ids: list[int],
    session_dir: Path,
    version: str,
    device: torch.device,
    threshold: float = 0.5
) -> list[Path]:
    """
    Generate predictions for a batch of files
    
    Args:
        head: Trained segmentation head
        file_ids: List of file IDs to predict
        session_dir: Path to session directory
        version: Model version string
        device: torch device
        threshold: Probability threshold for binary classification
    
    Returns:
        List of paths to saved prediction masks
    """
    mask_paths = []
    
    for file_id in file_ids:
        try:
            mask_path = predict_full_image(
                head, file_id, session_dir, version, device, threshold
            )
            mask_paths.append(mask_path)
        except Exception as e:
            print(f"Error predicting file {file_id}: {e}")
            continue
    
    return mask_paths

