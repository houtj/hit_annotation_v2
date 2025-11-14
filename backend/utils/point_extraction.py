"""
Utility functions for extracting points from prediction masks
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.ndimage import distance_transform_edt, gaussian_filter
import cv2


def extract_points_from_mask(
    mask_path: Path,
    max_points: int = 500,
    confidence_threshold: float = 0.15,
    min_distance: float = 3.0,
    gradient_weight: float = 2.0,
    foreground_color: str = '#ff0000',
    background_color: str = '#0000ff'
) -> List[Dict]:
    """
    Extract important points from a prediction mask using gradient-based stratified sampling
    
    Args:
        mask_path: Path to the prediction mask (grayscale PNG, 0-255)
        max_points: Maximum number of points to extract
        confidence_threshold: Only consider pixels with prob < threshold or > (1 - threshold)
        min_distance: Minimum distance between sampled points (in pixels)
        gradient_weight: Weight for gradient-based importance (higher = prefer edges)
        foreground_color: Hex color for foreground points (from Class table)
        background_color: Hex color for background points (from Class table)
    
    Returns:
        List of point dicts with keys: type, x, y, classname, color, origin
    """
    # Load mask
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    
    height, width = mask.shape
    
    # Step 1: Filter high-confidence pixels
    foreground_mask = mask > (1.0 - confidence_threshold)
    background_mask = mask < confidence_threshold
    
    # Step 2: Compute gradients
    gradient_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize gradient
    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    # Step 3: Compute importance maps
    # Foreground importance: combine confidence and gradient
    fg_confidence = mask.copy()
    fg_confidence[~foreground_mask] = 0
    fg_importance = fg_confidence + gradient_weight * gradient_magnitude * foreground_mask
    
    # Background importance: combine confidence and gradient
    bg_confidence = 1.0 - mask
    bg_confidence[~background_mask] = 0
    bg_importance = bg_confidence + gradient_weight * gradient_magnitude * background_mask
    
    # Step 4: Poisson disk sampling with importance weighting
    fg_points = poisson_disk_sampling_weighted(
        fg_importance,
        foreground_mask,
        min_distance=min_distance,
        max_points=max_points // 2
    )
    
    bg_points = poisson_disk_sampling_weighted(
        bg_importance,
        background_mask,
        min_distance=min_distance,
        max_points=max_points // 2
    )
    
    # Step 5: Convert to point format
    extracted_points = []
    
    for y, x in fg_points:
        extracted_points.append({
            'type': 'point',
            'x': int(x),
            'y': int(y),
            'classname': 'foreground',
            'color': foreground_color,  # Use color from Class table
            'origin': 'pred'  # Mark as extracted from prediction
        })
    
    for y, x in bg_points:
        extracted_points.append({
            'type': 'point',
            'x': int(x),
            'y': int(y),
            'classname': 'background',
            'color': background_color,  # Use color from Class table
            'origin': 'pred'  # Mark as extracted from prediction
        })
    
    return extracted_points


def poisson_disk_sampling_weighted(
    importance_map: np.ndarray,
    valid_mask: np.ndarray,
    min_distance: float,
    max_points: int
) -> List[Tuple[int, int]]:
    """
    Perform weighted Poisson disk sampling
    
    Args:
        importance_map: 2D array with importance weights
        valid_mask: Boolean mask indicating valid sampling locations
        min_distance: Minimum distance between samples
        max_points: Maximum number of points to sample
    
    Returns:
        List of (y, x) coordinates
    """
    height, width = importance_map.shape
    
    # Get candidate pixels
    candidate_coords = np.argwhere(valid_mask)
    
    if len(candidate_coords) == 0:
        return []
    
    # Get importance values for candidates
    candidate_importance = importance_map[valid_mask]
    
    # Normalize to probabilities
    if candidate_importance.sum() > 0:
        probabilities = candidate_importance / candidate_importance.sum()
    else:
        probabilities = np.ones(len(candidate_coords)) / len(candidate_coords)
    
    # Initialize sampled points
    sampled_points = []
    
    # Build spatial index for fast distance queries
    # We'll use a simple grid-based approach
    grid_size = int(min_distance)
    if grid_size < 1:
        grid_size = 1
    
    occupied_grid = set()
    
    # Sample points iteratively
    attempts = 0
    max_attempts = max_points * 50  # Prevent infinite loop
    
    while len(sampled_points) < max_points and attempts < max_attempts:
        attempts += 1
        
        # Sample a candidate based on importance
        idx = np.random.choice(len(candidate_coords), p=probabilities)
        y, x = candidate_coords[idx]
        
        # Check if too close to existing points
        grid_y = y // grid_size
        grid_x = x // grid_size
        
        # Check neighborhood in grid
        too_close = False
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if (grid_y + dy, grid_x + dx) in occupied_grid:
                    # Check actual distance
                    for sy, sx in sampled_points:
                        dist = np.sqrt((y - sy)**2 + (x - sx)**2)
                        if dist < min_distance:
                            too_close = True
                            break
                if too_close:
                    break
            if too_close:
                break
        
        if not too_close:
            sampled_points.append((y, x))
            occupied_grid.add((grid_y, grid_x))
            
            # Reduce probability of sampling nearby points
            # Create a mask for nearby candidates
            distances = np.sqrt((candidate_coords[:, 0] - y)**2 + (candidate_coords[:, 1] - x)**2)
            nearby_mask = distances < min_distance * 2
            probabilities[nearby_mask] *= 0.1
            
            # Renormalize
            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()
            else:
                break
    
    return sampled_points


def merge_with_human_labels(
    extracted_points: List[Dict],
    human_labels: List[Dict]
) -> List[Dict]:
    """
    Merge extracted points with existing human labels
    
    If an extracted point has the same rounded coordinate as a human label,
    keep the human label instead. Only points with origin="human" are considered
    human labels. Points without an origin field are treated as human (backward compatibility).
    
    Args:
        extracted_points: Points extracted from prediction mask (all have origin="pred")
        human_labels: All existing labels (may have mixed origins)
    
    Returns:
        Merged list of points
    """
    # Separate human labels from auto-extracted labels
    true_human_labels = []
    human_coords = set()
    
    for label in human_labels:
        if label.get('type') == 'point':
            origin = label.get('origin')
            # Treat as human if origin is "human" or missing (backward compatibility)
            is_human = (origin == 'human' or origin is None)
            
            if is_human:
                true_human_labels.append(label)
                x = label.get('x')
                y = label.get('y')
                if x is not None and y is not None:
                    human_coords.add((round(x), round(y)))
    
    # Start with true human labels
    merged_points = list(true_human_labels)
    
    # Add extracted points that don't overlap with human labels
    for point in extracted_points:
        x = point.get('x')
        y = point.get('y')
        if x is not None and y is not None:
            if (round(x), round(y)) not in human_coords:
                merged_points.append(point)
    
    return merged_points

