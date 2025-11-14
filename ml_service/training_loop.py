"""Training loop with suspension for periodic inference"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BinarySegmentationHead
from inference import predict_full_image
from utils import notify_prediction_ready, send_training_metric


def calculate_class_weights(train_data: List[Dict], device: torch.device) -> torch.Tensor:
    """
    Calculate class weights based on the frequency of each class (inverse frequency weighting)
    
    Args:
        train_data: List of training samples
        device: torch device
    
    Returns:
        Tensor of shape (2,) with weights for [background, foreground]
    """
    class_counts = torch.zeros(2, dtype=torch.float32)
    
    for sample in train_data:
        points = sample['points']
        for _, _, class_idx in points:
            class_counts[class_idx] += 1
    
    if class_counts.sum() == 0:
        return torch.ones(2, device=device)
    
    # Inverse frequency weighting: weight = total / (num_classes * count)
    total_points = class_counts.sum()
    weights = total_points / (2.0 * class_counts)
    
    # Handle zero counts (set weight to 1.0)
    weights[class_counts == 0] = 1.0
    
    # Normalize weights so they sum to num_classes (for stability)
    weights = weights / weights.mean()
    
    print(f"Class distribution: Background={int(class_counts[0])}, Foreground={int(class_counts[1])}")
    print(f"Class weights: Background={weights[0]:.3f}, Foreground={weights[1]:.3f}")
    
    return weights.to(device)


def check_stop_signal(db_path: Path) -> bool:
    """
    Check if training should stop
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        True if training_trigger is set to 2 (stop signal)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT value FROM config WHERE key = 'training_trigger'"
    )
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0] == '2':
        return True
    return False


def train_epoch(
    head: BinarySegmentationHead,
    train_data: List[Dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor = None
) -> float:
    """
    Train for one epoch with optional class balancing
    
    Args:
        head: Segmentation head model
        train_data: List of training samples
        optimizer: Optimizer
        device: torch device
        class_weights: Optional tensor of shape (2,) with weights for [background, foreground]
    
    Returns:
        Average training loss for the epoch
    """
    head.train()
    total_loss = 0.0
    total_points = 0
    
    for sample in train_data:
        features = sample['features'].unsqueeze(0).to(device)  # (1, 384, H, W)
        points = sample['points']  # List of (fy, fx, class_idx)
        
        if not points:
            continue
        
        # Forward pass
        prob_map = head(features)  # (1, 1, H, W)
        prob_map = prob_map.squeeze(0).squeeze(0)  # (H, W)
        
        # Extract probabilities at point locations
        point_probs = []
        point_targets = []
        point_weights = []
        
        for fy, fx, class_idx in points:
            point_probs.append(prob_map[fy, fx])
            point_targets.append(float(class_idx))
            # Assign weight based on class
            if class_weights is not None:
                point_weights.append(class_weights[class_idx].item())
            else:
                point_weights.append(1.0)
        
        if not point_probs:
            continue
        
        # Convert to tensors
        point_probs_tensor = torch.stack(point_probs)
        point_targets_tensor = torch.tensor(point_targets, device=device)
        point_weights_tensor = torch.tensor(point_weights, device=device)
        
        # Weighted binary cross-entropy loss
        loss = F.binary_cross_entropy(point_probs_tensor, point_targets_tensor, weight=point_weights_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(points)
        total_points += len(points)
    
    return total_loss / total_points if total_points > 0 else 0.0


def validate(
    head: BinarySegmentationHead,
    test_data: List[Dict],
    device: torch.device
) -> float:
    """
    Validate model on test set
    
    Args:
        head: Segmentation head model
        test_data: List of test samples
        device: torch device
    
    Returns:
        Average test loss
    """
    head.eval()
    total_loss = 0.0
    total_points = 0
    
    with torch.no_grad():
        for sample in test_data:
            features = sample['features'].unsqueeze(0).to(device)
            points = sample['points']
            
            if not points:
                continue
            
            prob_map = head(features).squeeze(0).squeeze(0)
            
            point_probs = []
            point_targets = []
            
            for fy, fx, class_idx in points:
                point_probs.append(prob_map[fy, fx])
                point_targets.append(float(class_idx))
            
            if not point_probs:
                continue
            
            point_probs_tensor = torch.stack(point_probs)
            point_targets_tensor = torch.tensor(point_targets, device=device)
            
            loss = F.binary_cross_entropy(point_probs_tensor, point_targets_tensor)
            
            total_loss += loss.item() * len(points)
            total_points += len(points)
    
    return total_loss / total_points if total_points > 0 else 0.0


def train_with_suspension(
    head: BinarySegmentationHead,
    train_data: List[Dict],
    test_data: List[Dict],
    session_dir: Path,
    current_file_id: int,
    major_version: int,
    device: torch.device,
    prediction_interval: int = 20,
    early_stop_patience: int = 5,
    early_stop_threshold: float = 0.001,
    max_epochs: int = 1000,
    learning_rate: float = 1e-3
) -> Tuple[bool, int, float]:
    """
    Train with periodic suspension for inference
    
    Args:
        head: Segmentation head model
        train_data: List of training samples
        test_data: List of test samples
        session_dir: Path to session directory
        current_file_id: File ID for prediction during suspension
        major_version: Major version number
        device: torch device
        prediction_interval: Suspend every N epochs for inference
        early_stop_patience: Stop if test loss doesn't improve for N checks
        early_stop_threshold: Minimum improvement threshold
        max_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        Tuple of (should_stop, final_minor_version, best_test_loss)
        - should_stop: True if early stop triggered or user requested stop
        - final_minor_version: Final minor version number
        - best_test_loss: Best test loss achieved
    """
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    db_path = session_dir / "annotations.db"
    
    best_test_loss = float('inf')
    patience_counter = 0
    minor_version = 0
    
    print(f"Starting training for version {major_version}.{minor_version}")
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Prediction interval: {prediction_interval} epochs")
    print(f"Early stop patience: {early_stop_patience}, threshold: {early_stop_threshold}")
    
    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(train_data, device)
    
    for epoch in range(max_epochs):
        # Check for stop signal
        if check_stop_signal(db_path):
            print(f"\n[Epoch {epoch}] Stop signal received. Terminating training.")
            
            # Generate final prediction before stopping
            print(f"\nGenerating final prediction before stop...")
            try:
                minor_version += 1
                version_str = f"{major_version}.{minor_version}"
                
                mask_path = predict_full_image(
                    head, current_file_id, session_dir, version_str, device
                )
                print(f"  Generated final prediction: {mask_path.name}")
                
                # Notify frontend via HTTP
                notify_prediction_ready(current_file_id, version_str)
            except Exception as e:
                print(f"  Error generating final prediction: {e}")
            
            return True, minor_version, best_test_loss
        
        # Suspension for inference
        if epoch % prediction_interval == 0 and epoch > 0:
            print(f"\n[Epoch {epoch}] Suspending for inference...")
            
            # Increment minor version
            minor_version += 1
            version_str = f"{major_version}.{minor_version}"
            
            # Save checkpoint
            checkpoint_dir = session_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"binary_seg_head_v{version_str.replace('.', '_')}.pth"
            torch.save(head.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path.name}")
            
            # Run inference on current file
            try:
                mask_path = predict_full_image(
                    head, current_file_id, session_dir, version_str, device
                )
                print(f"  Generated prediction: {mask_path.name}")
                
                # Notify frontend via HTTP
                notify_prediction_ready(current_file_id, version_str)
            
            except Exception as e:
                print(f"  Error during inference: {e}")
            
            # Check for stop signal again after inference
            if check_stop_signal(db_path):
                print(f"  Stop signal received after inference. Terminating training.")
                return True, minor_version, best_test_loss
            
            print(f"  Resuming training...\n")
        
        # Train one epoch with class balancing
        train_loss = train_epoch(head, train_data, optimizer, device, class_weights)
        
        # Validate
        test_loss = validate(head, test_data, device)
        
        # Send metrics to backend for storage and WebSocket broadcast
        version_str = f"{major_version}.{minor_version}"
        send_training_metric(version_str, epoch, train_loss, test_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # Early stopping check
        if test_loss < best_test_loss - early_stop_threshold:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"\n[Epoch {epoch}] Early stopping triggered. Test loss did not improve for {early_stop_patience} checks.")
            print(f"Best test loss: {best_test_loss:.4f}")
            
            # Generate final prediction before returning
            print(f"\nGenerating final prediction...")
            try:
                minor_version += 1
                version_str = f"{major_version}.{minor_version}"
                
                mask_path = predict_full_image(
                    head, current_file_id, session_dir, version_str, device
                )
                print(f"  Generated final prediction: {mask_path.name}")
                
                # Notify frontend via HTTP
                notify_prediction_ready(current_file_id, version_str)
            except Exception as e:
                print(f"  Error generating final prediction: {e}")
            
            return True, minor_version, best_test_loss
    
    print(f"\n[Epoch {max_epochs}] Reached maximum epochs.")
    print(f"Best test loss: {best_test_loss:.4f}")
    
    # Generate final prediction before returning
    print(f"\nGenerating final prediction...")
    try:
        minor_version += 1
        version_str = f"{major_version}.{minor_version}"
        
        mask_path = predict_full_image(
            head, current_file_id, session_dir, version_str, device
        )
        print(f"  Generated final prediction: {mask_path.name}")
        
        # Notify frontend via HTTP
        notify_prediction_ready(current_file_id, version_str)
    except Exception as e:
        print(f"  Error generating final prediction: {e}")
    
    return False, minor_version, best_test_loss

