"""Training loop with suspension for periodic inference - Supports both segmentation and classification"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import notify_prediction_ready, send_training_metric
from data_loaders import load_segmentation_training_data, load_classification_training_data
from data_loaders.classification import calculate_class_weights_classification
from losses import calculate_segmentation_class_weights


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


# ==============================================================================
# Segmentation-specific functions
# ==============================================================================

def deduplicate_masks(
    label_mask: torch.Tensor,
    target_mask: torch.Tensor,
    points: List[Tuple[int, int, int]],
    class_weights: torch.Tensor = None,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Deduplicate points and update masks to handle conflicts
    """
    from collections import defaultdict
    
    coord_to_labels = defaultdict(list)
    for fy, fx, class_idx in points:
        coord_to_labels[(fy, fx)].append(class_idx)
    
    H, W = label_mask.shape
    weight_mask = torch.ones(H, W, dtype=torch.float32, device=label_mask.device)
    
    num_conflicts = 0
    
    for (fy, fx), labels in coord_to_labels.items():
        if len(set(labels)) > 1:
            num_conflicts += 1
            target_mask[fy, fx] = sum(labels) / len(labels)
        
        if class_weights is not None:
            weight = sum(class_weights[int(label)].item() for label in labels) / len(labels)
            weight_mask[fy, fx] = weight
    
    stats = {
        'original': len(points),
        'deduplicated': len(coord_to_labels),
        'conflicts': num_conflicts
    }
    
    return label_mask, target_mask, weight_mask, stats


def train_epoch_segmentation(
    head: nn.Module,
    train_data: List[Dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor = None,
    batch_size: int = 32,
    accumulation_steps: int = 1
) -> float:
    """
    Train for one epoch for segmentation with vectorized mask-based operations
    """
    head.train()
    total_loss = 0.0
    total_points = 0
    
    num_samples = len(train_data)
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch = train_data[batch_start:batch_end]
        
        batch_features = []
        batch_label_masks = []
        batch_target_masks = []
        batch_weight_masks = []
        
        for sample in batch:
            if not sample['points']:
                continue
            
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
        
        features_batch = torch.stack(batch_features).to(device)
        label_masks_batch = torch.stack(batch_label_masks).to(device)
        target_masks_batch = torch.stack(batch_target_masks).to(device)
        weight_masks_batch = torch.stack(batch_weight_masks).to(device)
        
        prob_maps = head(features_batch)  # (B, 1, H, W)
        prob_maps = prob_maps.squeeze(1)  # (B, H, W)
        
        probs_labeled = prob_maps[label_masks_batch]
        targets_labeled = target_masks_batch[label_masks_batch]
        weights_labeled = weight_masks_batch[label_masks_batch]
        
        if len(probs_labeled) > 0:
            loss = F.binary_cross_entropy(probs_labeled, targets_labeled, weight=weights_labeled)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_start // batch_size + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * len(probs_labeled) * accumulation_steps
            total_points += len(probs_labeled)
    
    if (num_samples // batch_size) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / total_points if total_points > 0 else 0.0


def validate_segmentation(
    head: nn.Module,
    test_data: List[Dict],
    device: torch.device,
    batch_size: int = 64
) -> float:
    """
    Validate model on test set for segmentation
    """
    head.eval()
    total_loss = 0.0
    total_points = 0
    
    num_samples = len(test_data)
    
    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch = test_data[batch_start:batch_end]
            
            batch_features = []
            batch_label_masks = []
            batch_target_masks = []
            
            for sample in batch:
                if not sample['points']:
                    continue
                
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
            
            features_batch = torch.stack(batch_features).to(device)
            label_masks_batch = torch.stack(batch_label_masks).to(device)
            target_masks_batch = torch.stack(batch_target_masks).to(device)
            
            prob_maps = head(features_batch)
            prob_maps = prob_maps.squeeze(1)
            
            probs_labeled = prob_maps[label_masks_batch]
            targets_labeled = target_masks_batch[label_masks_batch]
            
            if len(probs_labeled) > 0:
                loss = F.binary_cross_entropy(probs_labeled, targets_labeled)
                total_loss += loss.item() * len(probs_labeled)
                total_points += len(probs_labeled)
    
    return total_loss / total_points if total_points > 0 else 0.0


# ==============================================================================
# Classification-specific functions
# ==============================================================================

def train_epoch_classification(
    head: nn.Module,
    train_data: List[Dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor = None,
    batch_size: int = 32
) -> float:
    """
    Train for one epoch for classification (image-based batching)
    """
    head.train()
    total_loss = 0.0
    total_samples = 0
    
    num_samples = len(train_data)
    
    # Create loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch = train_data[batch_start:batch_end]
        
        # Stack features and labels
        features_list = [s['features'] for s in batch]
        labels_list = [s['class_idx'] for s in batch]
        
        features_batch = torch.stack(features_list).to(device)
        labels_batch = torch.tensor(labels_list, dtype=torch.long, device=device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = head(features_batch)  # (B, num_classes)
        
        # Compute loss
        loss = loss_fn(logits, labels_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch)
        total_samples += len(batch)
    
    return total_loss / total_samples if total_samples > 0 else 0.0


def validate_classification(
    head: nn.Module,
    test_data: List[Dict],
    device: torch.device,
    batch_size: int = 64
) -> float:
    """
    Validate model on test set for classification
    """
    head.eval()
    total_loss = 0.0
    total_samples = 0
    
    num_samples = len(test_data)
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch = test_data[batch_start:batch_end]
            
            features_list = [s['features'] for s in batch]
            labels_list = [s['class_idx'] for s in batch]
            
            features_batch = torch.stack(features_list).to(device)
            labels_batch = torch.tensor(labels_list, dtype=torch.long, device=device)
            
            logits = head(features_batch)
            loss = loss_fn(logits, labels_batch)
            
            total_loss += loss.item() * len(batch)
            total_samples += len(batch)
    
    return total_loss / total_samples if total_samples > 0 else 0.0


# ==============================================================================
# Main training loop (task-agnostic with suspension)
# ==============================================================================

def train_with_suspension(
    head: nn.Module,
    train_ids: List[int],
    test_ids: List[int],
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
    class_names: List[str] = None
) -> Tuple[bool, int, float]:
    """
    Train with periodic suspension for inference - supports both segmentation and classification
    
    Args:
        head: Model head (BinarySegmentationHead or MultiClassHead)
        train_ids: List of file IDs for training
        test_ids: List of file IDs for testing
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
        task_type: "segmentation" or "classification"
        class_names: List of class names (required for classification)
    
    Returns:
        Tuple of (should_stop, final_minor_version, best_test_loss)
    """
    from inference import predict_full_image, predict_classification
    from task_factory import get_checkpoint_prefix
    
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    db_path = session_dir / "annotations.db"
    
    # Auto-detect batch size based on device
    if batch_size is None:
        if device.type == 'cuda':
            try:
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_mem_gb >= 80:
                    batch_size = 128
                elif gpu_mem_gb >= 40:
                    batch_size = 64
                elif gpu_mem_gb >= 16:
                    batch_size = 32
                else:
                    batch_size = 16
            except:
                batch_size = 32
        elif device.type == 'mps':
            batch_size = 16
        else:
            batch_size = 8
    
    val_batch_size = min(batch_size * 2, 128)
    
    # Load training data based on task type
    print(f"\nLoading {task_type} training data...")
    if task_type == "segmentation":
        train_data = load_segmentation_training_data(train_ids, session_dir)
        test_data = load_segmentation_training_data(test_ids, session_dir)
        print(f"  Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            print("  Error: No valid training or test data")
            return True, 0, float('inf')
        
        # Calculate class weights
        class_weights = calculate_segmentation_class_weights(train_data, device)
        
    elif task_type == "classification":
        train_data = load_classification_training_data(train_ids, session_dir, class_names)
        test_data = load_classification_training_data(test_ids, session_dir, class_names)
        print(f"  Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            print("  Error: No valid training or test data")
            return True, 0, float('inf')
        
        # Calculate class weights
        num_classes = len(class_names) if class_names else head.num_classes
        class_weights = calculate_class_weights_classification(train_data, num_classes, device)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    best_test_loss = float('inf')
    patience_counter = 0
    minor_version = 0
    
    print(f"\nStarting training for version {major_version}.{minor_version}")
    print(f"Task type: {task_type}")
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Batch size: {batch_size} (train), {val_batch_size} (validation)")
    print(f"Prediction interval: {prediction_interval} epochs")
    print(f"Early stop patience: {early_stop_patience}, threshold: {early_stop_threshold}")
    
    # Preload features to GPU if possible
    if device.type == 'cuda':
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem_gb >= 40:
                print(f"\nPreloading features to GPU for faster training...")
                for sample in train_data + test_data:
                    sample['features'] = sample['features'].to(device)
                print(f"  Features preloaded to GPU memory")
        except:
            pass
    
    checkpoint_prefix = get_checkpoint_prefix(task_type)
    
    for epoch in range(max_epochs):
        # Check for stop signal
        if check_stop_signal(db_path):
            print(f"\n[Epoch {epoch}] Stop signal received. Terminating training.")
            
            # Generate final prediction before stopping
            print(f"\nGenerating final prediction before stop...")
            try:
                minor_version += 1
                version_str = f"{major_version}.{minor_version}"
                
                if task_type == "segmentation":
                    mask_path = predict_full_image(
                        head, current_file_id, session_dir, version_str, device
                    )
                    print(f"  Generated final prediction: {mask_path.name}")
                else:
                    predict_classification(
                        head, current_file_id, session_dir, version_str, device, class_names
                    )
                    print(f"  Generated classification prediction")
                
                notify_prediction_ready(current_file_id, version_str)
            except Exception as e:
                print(f"  Error generating final prediction: {e}")
            
            return True, minor_version, best_test_loss
        
        # Suspension for inference
        if epoch % prediction_interval == 0 and epoch > 0:
            print(f"\n[Epoch {epoch}] Suspending for inference...")
            
            minor_version += 1
            version_str = f"{major_version}.{minor_version}"
            
            # Save checkpoint
            checkpoint_dir = session_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_v{version_str.replace('.', '_')}.pth"
            torch.save(head.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path.name}")
            
            # Run inference on current file
            try:
                if task_type == "segmentation":
                    mask_path = predict_full_image(
                        head, current_file_id, session_dir, version_str, device
                    )
                    print(f"  Generated prediction: {mask_path.name}")
                else:
                    predict_classification(
                        head, current_file_id, session_dir, version_str, device, class_names
                    )
                    print(f"  Generated classification prediction")
                
                notify_prediction_ready(current_file_id, version_str)
            except Exception as e:
                print(f"  Error during inference: {e}")
            
            if check_stop_signal(db_path):
                print(f"  Stop signal received after inference. Terminating training.")
                return True, minor_version, best_test_loss
            
            print(f"  Resuming training...\n")
        
        # Train one epoch
        if task_type == "segmentation":
            train_loss = train_epoch_segmentation(head, train_data, optimizer, device, class_weights, batch_size)
            test_loss = validate_segmentation(head, test_data, device, val_batch_size)
        else:
            train_loss = train_epoch_classification(head, train_data, optimizer, device, class_weights, batch_size)
            test_loss = validate_classification(head, test_data, device, val_batch_size)
        
        # Send metrics to backend
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
            print(f"\n[Epoch {epoch}] Early stopping triggered.")
            print(f"Best test loss: {best_test_loss:.4f}")
            
            # Generate final prediction
            print(f"\nGenerating final prediction...")
            try:
                minor_version += 1
                version_str = f"{major_version}.{minor_version}"
                
                if task_type == "segmentation":
                    mask_path = predict_full_image(
                        head, current_file_id, session_dir, version_str, device
                    )
                    print(f"  Generated final prediction: {mask_path.name}")
                else:
                    predict_classification(
                        head, current_file_id, session_dir, version_str, device, class_names
                    )
                    print(f"  Generated classification prediction")
                
                notify_prediction_ready(current_file_id, version_str)
            except Exception as e:
                print(f"  Error generating final prediction: {e}")
            
            return True, minor_version, best_test_loss
    
    print(f"\n[Epoch {max_epochs}] Reached maximum epochs.")
    print(f"Best test loss: {best_test_loss:.4f}")
    
    # Generate final prediction
    print(f"\nGenerating final prediction...")
    try:
        minor_version += 1
        version_str = f"{major_version}.{minor_version}"
        
        if task_type == "segmentation":
            mask_path = predict_full_image(
                head, current_file_id, session_dir, version_str, device
            )
            print(f"  Generated final prediction: {mask_path.name}")
        else:
            predict_classification(
                head, current_file_id, session_dir, version_str, device, class_names
            )
            print(f"  Generated classification prediction")
        
        notify_prediction_ready(current_file_id, version_str)
    except Exception as e:
        print(f"  Error generating final prediction: {e}")
    
    return False, minor_version, best_test_loss