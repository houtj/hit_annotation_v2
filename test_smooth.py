"""
Test script for smoothing binary masks
Implements two methods: Distance Transform and Dilation+Blur
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter


def method1_distance_transform(binary_mask, sigma=5.0):
    """
    Method 1: Distance Transform + Exponential Decay
    
    Creates smooth probability map using distance from 1-valued pixels
    
    Args:
        binary_mask: Binary mask (0 or 1)
        sigma: Controls falloff rate (larger = slower decay)
    
    Returns:
        Smoothed probability map (0-1)
    """
    # Compute distance from each pixel to nearest 1-valued pixel
    distance = distance_transform_edt(1 - binary_mask)
    
    # Apply exponential decay: prob = exp(-distance / sigma)
    prob_map = np.exp(-distance / sigma)
    
    # Ensure original 1s stay at 1.0
    prob_map[binary_mask == 1] = 1.0
    
    return prob_map


def method2_dilate_blur(binary_mask, dilation_size=5, blur_sigma=3.0):
    """
    Method 2: Morphological Dilation + Gaussian Blur
    
    Dilates the mask then applies Gaussian blur for smooth transitions
    
    Args:
        binary_mask: Binary mask (0 or 1)
        dilation_size: Number of pixels to dilate
        blur_sigma: Sigma for Gaussian blur
    
    Returns:
        Smoothed probability map (0-1)
    """
    # Create structuring element for dilation
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    
    # Dilate the binary mask
    dilated = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
    
    # Convert to float and apply Gaussian blur
    prob_map = dilated.astype(np.float32)
    prob_map = gaussian_filter(prob_map, sigma=blur_sigma)
    
    # Normalize to [0, 1]
    if prob_map.max() > 0:
        prob_map = prob_map / prob_map.max()
    
    # Ensure original 1s stay at 1.0
    prob_map[binary_mask == 1] = 1.0
    
    return prob_map


def load_binary_mask(mask_path):
    """Load and convert mask to binary (0 or 1)"""
    mask_img = Image.open(mask_path)
    
    # Convert to numpy array
    if mask_img.mode == 'L':
        # Grayscale
        mask = np.array(mask_img, dtype=np.float32) / 255.0
    elif mask_img.mode in ['RGB', 'RGBA']:
        # Color image - convert to grayscale
        mask = np.array(mask_img.convert('L'), dtype=np.float32) / 255.0
    else:
        mask = np.array(mask_img, dtype=np.float32)
    
    # Threshold to binary (anything > 0.5 becomes 1)
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    return binary_mask


def visualize_smoothing_results(binary_mask, smooth1, smooth2, output_path):
    """Create comprehensive visualization of smoothing results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original and Method 1
    axes[0, 0].imshow(binary_mask, cmap='gray')
    axes[0, 0].set_title('Original Binary Mask')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(smooth1, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Method 1: Distance Transform\n(Exponential Decay)')
    axes[0, 1].axis('off')
    
    # Show probability profile for Method 1
    # Take a horizontal slice through the middle
    mid_row = smooth1.shape[0] // 2
    profile1 = smooth1[mid_row, :]
    axes[0, 2].plot(profile1, 'r-', linewidth=2)
    axes[0, 2].set_ylim([0, 1.1])
    axes[0, 2].set_title('Probability Profile (Middle Row)\nMethod 1')
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Probability')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Method 2
    axes[1, 0].imshow(smooth2, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Method 2: Dilation + Blur')
    axes[1, 0].axis('off')
    
    # Difference map
    diff = np.abs(smooth1 - smooth2)
    im = axes[1, 1].imshow(diff, cmap='viridis', vmin=0, vmax=0.5)
    axes[1, 1].set_title('Absolute Difference\n(Method 1 - Method 2)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # Show probability profile for Method 2
    profile2 = smooth2[mid_row, :]
    axes[1, 2].plot(profile2, 'b-', linewidth=2)
    axes[1, 2].set_ylim([0, 1.1])
    axes[1, 2].set_title('Probability Profile (Middle Row)\nMethod 2')
    axes[1, 2].set_xlabel('Column')
    axes[1, 2].set_ylabel('Probability')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    return fig


def compute_statistics(binary_mask, smooth_mask):
    """Compute useful statistics about the smoothing"""
    
    # Number of original 1s
    n_ones = binary_mask.sum()
    
    # Area with probability > 0.5
    n_high_prob = (smooth_mask > 0.5).sum()
    
    # Area with probability > 0.1
    n_low_prob = (smooth_mask > 0.1).sum()
    
    # Check that original 1s are preserved
    original_ones_preserved = np.allclose(smooth_mask[binary_mask == 1], 1.0)
    
    return {
        'original_ones': int(n_ones),
        'high_prob_pixels': int(n_high_prob),
        'low_prob_pixels': int(n_low_prob),
        'ones_preserved': original_ones_preserved,
        'expansion_ratio_50': n_high_prob / max(n_ones, 1),
        'expansion_ratio_10': n_low_prob / max(n_ones, 1)
    }


if __name__ == "__main__":
    # Paths
    input_path = "/Users/houtj/projects/active_annotation/data/hitl_data/condabri_north_349/seg_labels/az_170_dip_75/mask_dip_75_az_170_1.png"
    output_path = "/Users/houtj/projects/active_annotation/test_smooth_output.png"
    
    print("="*70)
    print("Binary Mask Smoothing Test")
    print("="*70)
    
    # Load binary mask
    print(f"\nLoading mask from: {input_path}")
    binary_mask = load_binary_mask(input_path)
    print(f"Mask shape: {binary_mask.shape}")
    print(f"Number of 1-valued pixels: {binary_mask.sum()}")
    print(f"Percentage of 1s: {binary_mask.sum() / binary_mask.size * 100:.2f}%")
    
    # Method 1: Distance Transform
    print("\n" + "-"*70)
    print("Method 1: Distance Transform + Exponential Decay")
    print("-"*70)
    smooth1 = method1_distance_transform(binary_mask, sigma=5.0)
    stats1 = compute_statistics(binary_mask, smooth1)
    print(f"  Original 1s preserved: {stats1['ones_preserved']}")
    print(f"  Pixels with prob > 0.5: {stats1['high_prob_pixels']} (expansion: {stats1['expansion_ratio_50']:.2f}x)")
    print(f"  Pixels with prob > 0.1: {stats1['low_prob_pixels']} (expansion: {stats1['expansion_ratio_10']:.2f}x)")
    
    # Method 2: Dilation + Blur
    print("\n" + "-"*70)
    print("Method 2: Morphological Dilation + Gaussian Blur")
    print("-"*70)
    smooth2 = method2_dilate_blur(binary_mask, dilation_size=5, blur_sigma=3.0)
    stats2 = compute_statistics(binary_mask, smooth2)
    print(f"  Original 1s preserved: {stats2['ones_preserved']}")
    print(f"  Pixels with prob > 0.5: {stats2['high_prob_pixels']} (expansion: {stats2['expansion_ratio_50']:.2f}x)")
    print(f"  Pixels with prob > 0.1: {stats2['low_prob_pixels']} (expansion: {stats2['expansion_ratio_10']:.2f}x)")
    
    # Visualize results
    print("\n" + "-"*70)
    print("Generating visualization...")
    print("-"*70)
    visualize_smoothing_results(binary_mask, smooth1, smooth2, output_path)
    
    # Save the smooth masks
    smooth1_path = output_path.replace('.png', '_method1.png')
    smooth2_path = output_path.replace('.png', '_method2.png')
    
    Image.fromarray((smooth1 * 255).astype(np.uint8)).save(smooth1_path)
    Image.fromarray((smooth2 * 255).astype(np.uint8)).save(smooth2_path)
    
    print(f"  Method 1 mask saved to: {smooth1_path}")
    print(f"  Method 2 mask saved to: {smooth2_path}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)

