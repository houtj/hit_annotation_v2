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


def deduplicate_masks(
    label_mask: torch.Tensor,
    target_mask: torch.Tensor,
    points: List[Tuple[int, int, int]],
    class_weights: torch.Tensor = None,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Deduplicate points and update masks to handle conflicts
    
    Handles conflicts when multiple pixel coordinates map to the same feature cell.
    Resolution strategy:
    - If all labels agree: use that label with combined weight
    - If labels conflict: use soft label (average of classes) with combined weight
    
    Args:
        label_mask: (H, W) bool tensor indicating labeled locations
        target_mask: (H, W) float tensor with target values
        points: List of (fy, fx, class_idx) tuples (for conflict detection)
        class_weights: Optional tensor of shape (2,) with weights for [background, foreground]
        verbose: Whether to print deduplication statistics
    
    Returns:
        Tuple of (label_mask, target_mask, weight_mask, stats):
        - label_mask: (H, W) bool tensor (potentially updated)
        - target_mask: (H, W) float tensor (potentially updated with soft labels)
        - weight_mask: (H, W) float tensor with class-based weights
        - stats: Dict with 'original', 'deduplicated', 'conflicts' counts
    """
    from collections import defaultdict
    
    # Group points by (fy, fx) coordinate to detect conflicts
    coord_to_labels = defaultdict(list)
    for fy, fx, class_idx in points:
        coord_to_labels[(fy, fx)].append(class_idx)
    
    # Create weight mask
    H, W = label_mask.shape
    weight_mask = torch.ones(H, W, dtype=torch.float32, device=label_mask.device)
    
    num_conflicts = 0
    
    # Update masks with deduplication and class weights
    for (fy, fx), labels in coord_to_labels.items():
        # Check for conflicts
        if len(set(labels)) > 1:
            num_conflicts += 1
            if verbose:
                print(f"  Warning: Conflicting labels at ({fy}, {fx}): {labels} -> using soft label {sum(labels)/len(labels):.2f}")
            # Update target to soft label
            target_mask[fy, fx] = sum(labels) / len(labels)
        
        # Set class-based weight
        if class_weights is not None:
            # For soft labels, use weighted average of class weights
            weight = sum(class_weights[int(label)].item() for label in labels) / len(labels)
            weight_mask[fy, fx] = weight
    
    stats = {
        'original': len(points),
        'deduplicated': len(coord_to_labels),
        'conflicts': num_conflicts
    }
    
    return label_mask, target_mask, weight_mask, stats


def train_epoch(
    head: BinarySegmentationHead,
    train_data: List[Dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor = None,
    batch_size: int = 32,
    accumulation_steps: int = 1
) -> float:
    """
    Train for one epoch with vectorized mask-based operations (NO Python loops over points!)
    
    Args:
        head: Segmentation head model
        train_data: List of training samples (with pre-computed masks)
        optimizer: Optimizer
        device: torch device
        class_weights: Optional tensor of shape (2,) with weights for [background, foreground]
        batch_size: Number of samples to process in parallel
        accumulation_steps: Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)
    
    Returns:
        Average training loss for the epoch
    """
    head.train()
    total_loss = 0.0
    total_points = 0
    
    num_samples = len(train_data)
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch = train_data[batch_start:batch_end]
        
        # Collect batch data
        batch_features = []
        batch_label_masks = []
        batch_target_masks = []
        batch_weight_masks = []
        
        for sample in batch:
            if not sample['points']:
                continue
            
            # Deduplicate and get updated masks
            label_mask, target_mask, weight_mask, _ = deduplicate_masks(
                sample['label_mask'].clone(),
                sample['target_mask'].clone(),
                sample['points'],
                class_weights
            )
            
            batch_features.append(sample['features'])
            batch_label_masks.append(label_mask)
            batch_target_masks.append(target_mask)
            batch_weight_masks.append(weight_mask)
        
        if not batch_features:
            continue
        
        # Stack into batch tensors
        if batch_features[0].device != device:
            features_batch = torch.stack(batch_features).to(device)
        else:
            features_batch = torch.stack(batch_features)
        
        label_masks_batch = torch.stack(batch_label_masks).to(device)  # (B, H, W) bool
        target_masks_batch = torch.stack(batch_target_masks).to(device)  # (B, H, W) float
        weight_masks_batch = torch.stack(batch_weight_masks).to(device)  # (B, H, W) float
        
        # Forward pass (batched)
        prob_maps = head(features_batch)  # (B, 1, H, W)
        prob_maps = prob_maps.squeeze(1)  # (B, H, W)
        
        # Vectorized loss computation over entire batch (NO LOOPS!)
        # Extract only labeled locations using boolean indexing
        probs_labeled = prob_maps[label_masks_batch]  # (N_total,) where N_total = sum of all labeled points
        targets_labeled = target_masks_batch[label_masks_batch]  # (N_total,)
        weights_labeled = weight_masks_batch[label_masks_batch]  # (N_total,)
        
        if len(probs_labeled) > 0:
            # Compute weighted BCE loss over all labeled points in batch
            loss = F.binary_cross_entropy(probs_labeled, targets_labeled, weight=weights_labeled)
            
            # Backward pass with gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_start // batch_size + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * len(probs_labeled) * accumulation_steps
            total_points += len(probs_labeled)
    
    # Final optimizer step if there are remaining gradients
    if (num_samples // batch_size) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / total_points if total_points > 0 else 0.0


def validate(
    head: BinarySegmentationHead,
    test_data: List[Dict],
    device: torch.device,
    batch_size: int = 64
) -> float:
    """
    Validate model on test set with batching
    
    Args:
        head: Segmentation head model
        test_data: List of test samples
        device: torch device
        batch_size: Batch size for validation (can be larger than training)
    
    Returns:
        Average test loss
    """
    head.eval()
    total_loss = 0.0
    total_points = 0
    
    num_samples = len(test_data)
    
    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch = test_data[batch_start:batch_end]
            
            # Collect batch data
            batch_features = []
            batch_label_masks = []
            batch_target_masks = []
            
            for sample in batch:
                if not sample['points']:
                    continue
                
                # Deduplicate (no class weights for validation)
                label_mask, target_mask, _, _ = deduplicate_masks(
                    sample['label_mask'].clone(),
                    sample['target_mask'].clone(),
                    sample['points'],
                    class_weights=None
                )
                
                batch_features.append(sample['features'])
                batch_label_masks.append(label_mask)
                batch_target_masks.append(target_mask)
            
            if not batch_features:
                continue
            
            # Stack into batch tensors
            if batch_features[0].device != device:
                features_batch = torch.stack(batch_features).to(device)
            else:
                features_batch = torch.stack(batch_features)
            
            label_masks_batch = torch.stack(batch_label_masks).to(device)
            target_masks_batch = torch.stack(batch_target_masks).to(device)
            
            # Forward pass (batched)
            prob_maps = head(features_batch)  # (B, 1, H, W)
            prob_maps = prob_maps.squeeze(1)  # (B, H, W)
            
            # Vectorized loss computation (NO LOOPS!)
            probs_labeled = prob_maps[label_masks_batch]
            targets_labeled = target_masks_batch[label_masks_batch]
            
            if len(probs_labeled) > 0:
                loss = F.binary_cross_entropy(probs_labeled, targets_labeled)
                
                total_loss += loss.item() * len(probs_labeled)
                total_points += len(probs_labeled)
    
    return total_loss / total_points if total_points > 0 else 0.0


def train_classification_epoch(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_size: int = 16
) -> float:
    """Train one epoch for classification task"""
    head.train()
    total_loss = 0.0
    num_samples = features.size(0)
    
    # Shuffle indices
    indices = torch.randperm(num_samples)
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_features = features[batch_indices]
        batch_labels = labels[batch_indices]
        
        optimizer.zero_grad()
        logits = head(batch_features)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_indices.size(0)
    
    return total_loss / num_samples


def validate_classification(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    batch_size: int = 32
) -> float:
    """Validate for classification task"""
    head.eval()
    total_loss = 0.0
    num_samples = features.size(0)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_features = features[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            logits = head(batch_features)
            loss = loss_fn(logits, batch_labels)
            
            total_loss += loss.item() * batch_features.size(0)
    
    return total_loss / num_samples


def train_with_suspension(
    head: nn.Module,
    train_data: List[Dict] | None,
    test_data: List[Dict] | None,
    session_dir: Path,
    current_file_id: int,
    major_version: int,
    device: torch.device,
    prediction_interval: int = 20,
    early_stop_patience: int = 5,
    early_stop_threshold: float = 0.001,
    max_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = None,
    task_type: str = "segmentation",
    train_ids: List[int] = None,
    test_ids: List[int] = None,
    config: dict = None
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
        batch_size: Batch size for training (auto-detected if None)
    
    Returns:
        Tuple of (should_stop, final_minor_version, best_test_loss)
        - should_stop: True if early stop triggered or user requested stop
        - final_minor_version: Final minor version number
        - best_test_loss: Best test loss achieved
    """
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    db_path = session_dir / "annotations.db"
    
    # Load data for classification if needed
    if task_type == "classification" and train_data is None:
        from data_loader import load_classification_data
        print("Loading classification data...")
        train_features, train_labels = load_classification_data(train_ids, session_dir, config)
        test_features, test_labels = load_classification_data(test_ids, session_dir, config)
        # Move to device
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)
    
    # Auto-detect batch size based on device
    if batch_size is None:
        if device.type == 'cuda':
            # For CUDA, adjust based on GPU memory
            try:
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_mem_gb >= 80:  # H100 or A100 80GB
                    batch_size = 128
                elif gpu_mem_gb >= 40:  # A100 40GB
                    batch_size = 64
                elif gpu_mem_gb >= 16:  # V100, RTX 3090
                    batch_size = 32
                else:
                    batch_size = 16
            except:
                batch_size = 32
        elif device.type == 'mps':
            batch_size = 16  # Apple Silicon - conservative
        else:
            batch_size = 8  # CPU - small batch
    
    val_batch_size = min(batch_size * 2, 128)  # Validation can use larger batches
    
    best_test_loss = float('inf')
    patience_counter = 0
    minor_version = 0
    
    print(f"Starting training for version {major_version}.{minor_version}")
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Batch size: {batch_size} (train), {val_batch_size} (validation)")
    print(f"Prediction interval: {prediction_interval} epochs")
    print(f"Early stop patience: {early_stop_patience}, threshold: {early_stop_threshold}")
    
    # Preload features to GPU if possible (for high-end GPUs)
    if device.type == 'cuda':
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem_gb >= 40:  # A100/H100 - preload for speed
                print(f"\nPreloading features to GPU for faster training...")
                for sample in train_data + test_data:
                    sample['features'] = sample['features'].to(device)
                print(f"  Features preloaded to GPU memory")
        except:
            pass  # Keep features on CPU if preloading fails
    
    # Segmentation-specific setup
    if task_type == "segmentation":
        # Calculate class weights for balanced training
        class_weights = calculate_class_weights(train_data, device)
        
        # Diagnostic: Check for point collisions across all training data
        total_original = 0
        total_deduplicated = 0
        total_conflicts = 0
        for sample in train_data:
            _, _, _, stats = deduplicate_masks(
                sample['label_mask'],
                sample['target_mask'],
                sample['points'],
                class_weights
            )
            total_original += stats['original']
            total_deduplicated += stats['deduplicated']
            total_conflicts += stats['conflicts']
        
        if total_original > total_deduplicated:
            print(f"\n⚠️  Point Collision Warning:")
            print(f"  Original points: {total_original}")
            print(f"  After deduplication: {total_deduplicated} ({total_original - total_deduplicated} duplicates removed)")
            print(f"  Conflicting labels: {total_conflicts} locations")
            if total_conflicts > 0:
                print(f"  (Conflicts resolved using soft labels: average of conflicting classes)")
            print()
    else:
        class_weights = None
    
    for epoch in range(max_epochs):
        # Check for stop signal
        if check_stop_signal(db_path):
            print(f"\n[Epoch {epoch}] Stop signal received. Terminating training.")
            
            # Generate final prediction before stopping (segmentation only)
            if task_type == "segmentation":
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
            from task_factory import get_checkpoint_prefix
            checkpoint_dir = session_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_prefix = get_checkpoint_prefix(task_type)
            checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_v{version_str.replace('.', '_')}.pth"
            torch.save(head.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path.name}")
            
            # Run inference on current file (segmentation only)
            if task_type == "segmentation":
                try:
                    mask_path = predict_full_image(
                        head, current_file_id, session_dir, version_str, device
                    )
                    print(f"  Generated prediction: {mask_path.name}")
                    
                    # Notify frontend via HTTP
                    notify_prediction_ready(current_file_id, version_str)
                
                except Exception as e:
                    print(f"  Error during inference: {e}")
            else:
                print(f"  Skipping inference for classification task")
            
            # Check for stop signal again after inference
            if check_stop_signal(db_path):
                print(f"  Stop signal received after inference. Terminating training.")
                return True, minor_version, best_test_loss
            
            print(f"  Resuming training...\n")
        
        # Train one epoch
        if task_type == "segmentation":
            train_loss = train_epoch(head, train_data, optimizer, device, class_weights, batch_size)
            test_loss = validate(head, test_data, device, val_batch_size)
        elif task_type == "classification":
            from task_factory import create_loss_fn
            loss_fn = create_loss_fn(task_type)
            train_loss = train_classification_epoch(head, train_features, train_labels, optimizer, loss_fn, batch_size)
            test_loss = validate_classification(head, test_features, test_labels, loss_fn, val_batch_size)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
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
            
            # Generate final prediction before returning (segmentation only)
            if task_type == "segmentation":
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
    
    # Generate final prediction before returning (segmentation only)
    if task_type == "segmentation":
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

