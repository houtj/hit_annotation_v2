"""Inference utilities for generating predictions - Supports segmentation and classification"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from data_loaders import load_features


def predict_full_image(
    head: nn.Module,
    file_id: int,
    session_dir: Path,
    version: str,
    device: torch.device,
    threshold: float = 0.5  # Not used anymore, kept for API compatibility
) -> Path:
    """
    Generate full-resolution probability map for an image (segmentation task)
    
    Args:
        head: Trained segmentation head
        file_id: File ID in database
        session_dir: Path to session directory
        version: Model version string (e.g., "1.2")
        device: torch device (cuda or cpu)
        threshold: Not used (kept for API compatibility)
    
    Returns:
        Path to saved prediction mask PNG file (grayscale, 0-255 representing probabilities 0-1)
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
        "SELECT prediction_data FROM predictions WHERE file_id = ?",
        (file_id,)
    )
    existing = cursor.fetchone()
    
    if existing:
        # Delete old prediction file if it exists
        try:
            existing_data = json.loads(existing[0]) if existing[0] else {}
            if existing_data.get('type') == 'mask' and 'path' in existing_data:
                old_path = session_dir / existing_data['path']
                if old_path.exists():
                    old_path.unlink()
                    print(f"  Deleted old prediction: {old_path.name}")
        except Exception as e:
            print(f"  Warning: Could not delete old prediction: {e}")
    
    # Use simple filename without version (one prediction per file)
    mask_filename = f"file_{file_id}.png"
    mask_path = predictions_dir / mask_filename
    
    # Save as grayscale image with soft probabilities
    mask_img = Image.fromarray(prob_map_vis, mode='L')
    mask_img.save(mask_path)
    
    # Store relative path in prediction_data JSON
    relative_path = f"storage/predictions/{mask_filename}"
    prediction_data = json.dumps({
        "type": "mask",
        "path": relative_path
    })
    
    if existing:
        # Update existing prediction
        cursor.execute(
            "UPDATE predictions SET prediction_data = ? WHERE file_id = ?",
            (prediction_data, file_id)
        )
    else:
        # Insert new prediction
        cursor.execute(
            "INSERT INTO predictions (file_id, prediction_data) VALUES (?, ?)",
            (file_id, prediction_data)
        )
    
    conn.commit()
    conn.close()
    
    return mask_path


def predict_classification(
    head: nn.Module,
    file_id: int,
    session_dir: Path,
    version: str,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """
    Generate classification prediction for an image
    
    Args:
        head: Trained classification head
        file_id: File ID in database
        session_dir: Path to session directory
        version: Model version string (e.g., "1.2")
        device: torch device (cuda or cpu)
        class_names: List of class names
    
    Returns:
        Dictionary with prediction result:
        {
            "type": "class",
            "class": str,           # Predicted class name
            "confidence": float,    # Confidence score 0-1
            "probabilities": {      # Per-class probabilities
                "class1": float,
                "class2": float,
                ...
            }
        }
    """
    # Load features
    features, orig_width, orig_height = load_features(file_id, session_dir)
    
    # Move to device and add batch dimension
    features = features.unsqueeze(0).to(device)  # (1, 384, H, W)
    
    # Generate prediction
    head.eval()
    with torch.no_grad():
        logits = head(features)  # (1, num_classes)
        proba = torch.softmax(logits, dim=1)  # (1, num_classes)
    
    # Get predicted class and confidence
    confidence, predicted_idx = torch.max(proba, dim=1)
    predicted_idx = predicted_idx.item()
    confidence = confidence.item()
    
    # Build probabilities dictionary
    probabilities = {
        class_names[i]: proba[0, i].item()
        for i in range(len(class_names))
    }
    
    # Create prediction result
    prediction_result = {
        "type": "class",
        "class": class_names[predicted_idx],
        "confidence": confidence,
        "probabilities": probabilities
    }
    
    # Store in database
    db_path = session_dir / "annotations.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id FROM predictions WHERE file_id = ?",
        (file_id,)
    )
    existing = cursor.fetchone()
    
    prediction_data = json.dumps(prediction_result)
    
    if existing:
        # Update existing prediction
        cursor.execute(
            "UPDATE predictions SET prediction_data = ? WHERE file_id = ?",
            (prediction_data, file_id)
        )
    else:
        # Insert new prediction
        cursor.execute(
            "INSERT INTO predictions (file_id, prediction_data) VALUES (?, ?)",
            (file_id, prediction_data)
        )
    
    conn.commit()
    conn.close()
    
    return prediction_result


def predict_batch(
    head: nn.Module,
    file_ids: List[int],
    session_dir: Path,
    version: str,
    device: torch.device,
    task_type: str = "segmentation",
    class_names: List[str] = None
) -> List:
    """
    Generate predictions for a batch of files
    
    Args:
        head: Trained model head
        file_ids: List of file IDs to predict
        session_dir: Path to session directory
        version: Model version string
        device: torch device
        task_type: "segmentation" or "classification"
        class_names: List of class names (required for classification)
    
    Returns:
        List of prediction results (Paths for segmentation, Dicts for classification)
    """
    results = []
    
    for file_id in file_ids:
        try:
            if task_type == "segmentation":
                result = predict_full_image(
                    head, file_id, session_dir, version, device
                )
            else:
                result = predict_classification(
                    head, file_id, session_dir, version, device, class_names
                )
            results.append(result)
        except Exception as e:
            print(f"Error predicting file {file_id}: {e}")
            continue
    
    return results
