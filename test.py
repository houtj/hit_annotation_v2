"""
Test script for extracting important points from prediction masks
Uses gradient-based stratified sampling
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def extract_important_points(
    mask_path: str,
    max_points: int = 500,
    confidence_low: float = 0.15,
    confidence_high: float = 0.85,
    edge_ratio: float = 0.75
) -> tuple:
    """
    Extract important points from a prediction mask
    
    Args:
        mask_path: Path to the prediction mask (grayscale, 0-255)
        max_points: Maximum number of points to extract
        confidence_low: Low confidence threshold (0-1)
        confidence_high: High confidence threshold (0-1)
        edge_ratio: Proportion of points to allocate to edges (0-1)
    
    Returns:
        Tuple of (foreground_points, background_points, mask_array, gradient_map)
    """
    # Load mask and convert to probability (0-1)
    mask_img = Image.open(mask_path).convert('L')
    mask = np.array(mask_img, dtype=np.float32) / 255.0
    
    print(f"Image size: {mask.shape}")
    print(f"Probability range: [{mask.min():.3f}, {mask.max():.3f}]")
    
    # 1. Confidence filtering: only high-confidence pixels
    confident_bg = mask < confidence_low  # Strong background
    confident_fg = mask > confidence_high  # Strong foreground
    confident_mask = confident_bg | confident_fg
    
    print(f"Confident pixels: {confident_mask.sum()} / {mask.size} ({confident_mask.sum()/mask.size*100:.1f}%)")
    
    # 2. Compute gradient magnitude (edge detection)
    # Use Sobel operator
    grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient to [0, 1]
    if gradient.max() > 0:
        gradient = gradient / gradient.max()
    
    print(f"Gradient range: [{gradient.min():.3f}, {gradient.max():.3f}]")
    
    # 3. Separate edge and interior regions
    # Define edge as high gradient + confident
    gradient_threshold = 0.1  # Pixels with gradient > threshold are edges
    is_edge = (gradient > gradient_threshold) & confident_mask
    is_interior = (gradient <= gradient_threshold) & confident_mask
    
    print(f"Edge pixels: {is_edge.sum()}")
    print(f"Interior pixels: {is_interior.sum()}")
    
    # 4. Allocate points between edge and interior
    n_edge_points = int(max_points * edge_ratio)
    n_interior_points = max_points - n_edge_points
    
    # 5. Sample points
    fg_points = []
    bg_points = []
    
    # Sample edge points (foreground)
    edge_fg = is_edge & confident_fg
    if edge_fg.sum() > 0:
        edge_fg_coords = np.argwhere(edge_fg)
        n_sample = min(n_edge_points // 2, len(edge_fg_coords))
        if n_sample > 0:
            # Weight by gradient (higher gradient = more important)
            weights = gradient[edge_fg_coords[:, 0], edge_fg_coords[:, 1]]
            weights = weights / weights.sum()
            indices = np.random.choice(len(edge_fg_coords), size=n_sample, replace=False, p=weights)
            sampled = edge_fg_coords[indices]
            fg_points.extend([(int(x), int(y)) for y, x in sampled])
    
    # Sample edge points (background)
    edge_bg = is_edge & confident_bg
    if edge_bg.sum() > 0:
        edge_bg_coords = np.argwhere(edge_bg)
        n_sample = min(n_edge_points // 2, len(edge_bg_coords))
        if n_sample > 0:
            weights = gradient[edge_bg_coords[:, 0], edge_bg_coords[:, 1]]
            weights = weights / weights.sum()
            indices = np.random.choice(len(edge_bg_coords), size=n_sample, replace=False, p=weights)
            sampled = edge_bg_coords[indices]
            bg_points.extend([(int(x), int(y)) for y, x in sampled])
    
    # Sample interior points (foreground)
    interior_fg = is_interior & confident_fg
    if interior_fg.sum() > 0:
        interior_fg_coords = np.argwhere(interior_fg)
        n_sample = min(n_interior_points // 2, len(interior_fg_coords))
        if n_sample > 0:
            indices = np.random.choice(len(interior_fg_coords), size=n_sample, replace=False)
            sampled = interior_fg_coords[indices]
            fg_points.extend([(int(x), int(y)) for y, x in sampled])
    
    # Sample interior points (background)
    interior_bg = is_interior & confident_bg
    if interior_bg.sum() > 0:
        interior_bg_coords = np.argwhere(interior_bg)
        n_sample = min(n_interior_points // 2, len(interior_bg_coords))
        if n_sample > 0:
            indices = np.random.choice(len(interior_bg_coords), size=n_sample, replace=False)
            sampled = interior_bg_coords[indices]
            bg_points.extend([(int(x), int(y)) for y, x in sampled])
    
    print(f"\nExtracted points:")
    print(f"  Foreground: {len(fg_points)}")
    print(f"  Background: {len(bg_points)}")
    print(f"  Total: {len(fg_points) + len(bg_points)}")
    
    return fg_points, bg_points, mask, gradient


def visualize_results(mask, gradient, fg_points, bg_points, output_path):
    """Create visualization of the extracted points"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original mask
    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].set_title('Original Prediction Mask')
    axes[0, 0].axis('off')
    
    # Gradient map
    axes[0, 1].imshow(gradient, cmap='hot')
    axes[0, 1].set_title('Gradient Map (Edge Detection)')
    axes[0, 1].axis('off')
    
    # Points on mask
    axes[1, 0].imshow(mask, cmap='gray')
    if fg_points:
        fg_x, fg_y = zip(*fg_points)
        axes[1, 0].scatter(fg_x, fg_y, c='green', s=20, alpha=0.8, label='Foreground', edgecolors='white', linewidths=0.5)
    if bg_points:
        bg_x, bg_y = zip(*bg_points)
        axes[1, 0].scatter(bg_x, bg_y, c='red', s=20, alpha=0.8, label='Background', edgecolors='white', linewidths=0.5)
    axes[1, 0].set_title(f'Extracted Points (Total: {len(fg_points) + len(bg_points)})')
    axes[1, 0].legend()
    axes[1, 0].axis('off')
    
    # Points density heatmap
    point_map = np.zeros_like(mask)
    for x, y in fg_points:
        point_map[y, x] = 1.0
    for x, y in bg_points:
        point_map[y, x] = 0.5
    
    # Apply Gaussian blur to show density
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(point_map, sigma=5)
    
    axes[1, 1].imshow(density, cmap='viridis')
    axes[1, 1].set_title('Point Density Map')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Input and output paths
    input_path = "/Users/houtj/projects/active_annotation/test_smooth_output_method1.png"
    output_path = "/Users/houtj/projects/active_annotation/test_output_dip_smoothed.png"
    
    print("="*60)
    print("Point Extraction from Prediction Mask")
    print("="*60)
    
    # Extract points
    fg_points, bg_points, mask, gradient = extract_important_points(
        input_path,
        max_points=4000,
        confidence_low=0.4,
        confidence_high=0.6,
        edge_ratio=0.75
    )
    
    # Visualize
    visualize_results(mask, gradient, fg_points, bg_points, output_path)
    
    print("\nDone!")

