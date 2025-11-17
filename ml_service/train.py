#!/usr/bin/env python3
"""Main training script for ML service"""

import sqlite3
import time
from pathlib import Path
from datetime import datetime
import torch

from model import BinarySegmentationHead
from data_loader import (
    load_human_labeled_files,
    load_training_data,
    split_train_test
)
from training_loop import train_with_suspension


def load_config(db_path: Path) -> dict:
    """
    Load configuration from database
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        Dictionary of config key-value pairs
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT key, value FROM config")
    config = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    return config


def update_config(db_path: Path, key: str, value: str):
    """
    Update a single config value
    
    Args:
        db_path: Path to SQLite database
        key: Config key
        value: New value
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE config SET value = ? WHERE key = ?",
        (value, key)
    )
    
    conn.commit()
    conn.close()


def create_model_version_record(
    db_path: Path,
    version: str,
    status: str = "training"
):
    """
    Create or update ModelVersion record
    
    Args:
        db_path: Path to SQLite database
        version: Version string (e.g., "1.0")
        status: Status string ("training", "completed", "failed")
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    # Check if version exists
    cursor.execute(
        "SELECT version FROM model_versions WHERE version = ?",
        (version,)
    )
    existing = cursor.fetchone()
    
    if existing:
        # Update existing
        if status == "completed":
            cursor.execute(
                "UPDATE model_versions SET status = ?, training_end_at = ? WHERE version = ?",
                (status, now, version)
            )
        else:
            cursor.execute(
                "UPDATE model_versions SET status = ? WHERE version = ?",
                (status, version)
            )
    else:
        # Insert new
        cursor.execute(
            "INSERT INTO model_versions (version, training_start_at, status) VALUES (?, ?, ?)",
            (version, now, status)
        )
    
    conn.commit()
    conn.close()


def update_model_version_record(
    db_path: Path,
    version: str,
    status: str,
    metrics: dict = None,
    checkpoint_path: str = None
):
    """
    Update ModelVersion record with completion info
    
    Args:
        db_path: Path to SQLite database
        version: Version string
        status: Status string ("completed", "failed")
        metrics: Optional metrics dictionary
        checkpoint_path: Optional path to checkpoint
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    if metrics:
        import json
        metrics_json = json.dumps(metrics)
        cursor.execute(
            "UPDATE model_versions SET status = ?, training_end_at = ?, metrics = ?, path = ? WHERE version = ?",
            (status, now, metrics_json, checkpoint_path, version)
        )
    else:
        cursor.execute(
            "UPDATE model_versions SET status = ?, training_end_at = ? WHERE version = ?",
            (status, now, version)
        )
    
    conn.commit()
    conn.close()


def main():
    """Main training loop"""
    # Paths
    session_dir = Path(__file__).parent.parent / "session"
    db_path = session_dir / "annotations.db"
    checkpoint_dir = session_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ML SERVICE - BINARY SEGMENTATION TRAINING")
    print("=" * 70)
    print(f"Session directory: {session_dir.absolute()}")
    print(f"Database: {db_path.absolute()}")
    print(f"Checkpoints: {checkpoint_dir.absolute()}")
    print("=" * 70)
    
    # Check device with better support for different platforms
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nDevice: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nDevice: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"\nDevice: CPU")
        print(f"  Warning: Training on CPU will be very slow!")
    
    # Main polling loop
    print("\nWaiting for training trigger...")
    
    while True:
        # Poll database for training trigger
        config = load_config(db_path)
        
        training_trigger = config.get('training_trigger', '0')
        
        if training_trigger == '0':
            # Idle - wait and continue
            time.sleep(1)
            continue
        
        if training_trigger == '2':
            # Stop requested - reset trigger and continue polling
            print("\nStop signal received (no active training).")
            update_config(db_path, 'training_trigger', '0')
            continue
        
        # training_trigger == '1': Start new training session
        print("\n" + "=" * 70)
        print("NEW TRAINING SESSION STARTED")
        print("=" * 70)
        
        try:
            # Parse config
            prediction_interval = int(config.get('prediction_interval', '20'))
            early_stop_patience = int(config.get('early_stop_patience', '5'))
            early_stop_threshold = float(config.get('early_stop_threshold', '0.001'))
            current_file_id_str = config.get('current_file_id', '')
            
            if not current_file_id_str or current_file_id_str == '':
                print("Error: No current_file_id set. Skipping training.")
                update_config(db_path, 'training_trigger', '0')
                continue
            
            current_file_id = int(current_file_id_str)
            
            # Get current version
            model_version = config.get('model_version', '0.0')
            try:
                major_str, minor_str = model_version.split('.')
                major = int(major_str)
            except (ValueError, AttributeError):
                major = 0
            
            version_str = f"{major}.0"
            
            print(f"\nConfiguration:")
            print(f"  Version: {version_str}")
            print(f"  Current file ID: {current_file_id}")
            print(f"  Prediction interval: {prediction_interval} epochs")
            print(f"  Early stop patience: {early_stop_patience}")
            print(f"  Early stop threshold: {early_stop_threshold}")
            
            # Create ModelVersion record
            create_model_version_record(db_path, version_str, "training")
            
            # Load human-labeled files
            print(f"\nLoading training data...")
            labeled_files = load_human_labeled_files(db_path)
            print(f"  Found {len(labeled_files)} human-labeled files")
            
            if len(labeled_files) < 2:
                print("  Error: Need at least 2 labeled files for train/test split")
                update_model_version_record(db_path, version_str, "failed")
                update_config(db_path, 'training_trigger', '0')
                continue
            
            # Split train/test
            train_ids, test_ids = split_train_test(labeled_files, test_ratio=0.2)
            print(f"  Train files: {len(train_ids)}")
            print(f"  Test files: {len(test_ids)}")
            
            # Load data
            print(f"\n Loading features and labels...")
            train_data = load_training_data(train_ids, session_dir)
            test_data = load_training_data(test_ids, session_dir)
            print(f"  Train samples: {len(train_data)}")
            print(f"  Test samples: {len(test_data)}")
            
            if len(train_data) == 0 or len(test_data) == 0:
                print("  Error: No valid training or test data")
                update_model_version_record(db_path, version_str, "failed")
                update_config(db_path, 'training_trigger', '0')
                continue
            
            # Initialize or load model
            print(f"\nInitializing model...")
            head = BinarySegmentationHead(in_channels=384)
            
            # Count parameters
            total_params = sum(p.numel() for p in head.parameters())
            trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            # Try to load latest checkpoint for warm start
            latest_checkpoint = checkpoint_dir / "binary_seg_head_latest.pth"
            if latest_checkpoint.exists():
                try:
                    head.load_state_dict(torch.load(latest_checkpoint, map_location='cpu'), strict=True)
                    print(f"  Loaded checkpoint: {latest_checkpoint.name}")
                except Exception as e:
                    print(f"  Warning: Could not load checkpoint (architecture mismatch?): {e}")
                    print(f"  Starting from scratch with new architecture")
            else:
                print(f"  Starting from scratch (no checkpoint found)")
            
            head = head.to(device)
            print(f"  Model initialized on {device}")
            
            # Training loop with suspension
            print(f"\nStarting training...")
            should_stop, final_minor, best_test_loss = train_with_suspension(
                head=head,
                train_data=train_data,
                test_data=test_data,
                session_dir=session_dir,
                current_file_id=current_file_id,
                major_version=major,
                device=device,
                prediction_interval=prediction_interval,
                early_stop_patience=early_stop_patience,
                early_stop_threshold=early_stop_threshold,
                max_epochs=1000,
                learning_rate=1e-3
            )
            
            # Save final checkpoint
            final_version = f"{major}.{final_minor}"
            final_checkpoint = checkpoint_dir / "binary_seg_head_latest.pth"
            versioned_checkpoint = checkpoint_dir / f"binary_seg_head_v{final_version.replace('.', '_')}.pth"
            
            torch.save(head.state_dict(), final_checkpoint)
            torch.save(head.state_dict(), versioned_checkpoint)
            print(f"\nSaved final checkpoint: {final_checkpoint.name}")
            print(f"Saved versioned checkpoint: {versioned_checkpoint.name}")
            
            # Update ModelVersion record
            metrics = {
                "best_test_loss": best_test_loss,
                "final_minor_version": final_minor
            }
            update_model_version_record(
                db_path,
                version_str,
                "completed",
                metrics=metrics,
                checkpoint_path=str(versioned_checkpoint.relative_to(session_dir))
            )
            
            # Update config
            update_config(db_path, 'model_version', final_version)
            print(f"\nUpdated model version to: {final_version}")
        
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            
            # Mark version as failed
            try:
                version_str = config.get('model_version', '0.0')
                update_model_version_record(db_path, version_str, "failed")
            except:
                pass
        
        finally:
            # Reset training trigger
            update_config(db_path, 'training_trigger', '0')
            print("\n" + "=" * 70)
            print("TRAINING SESSION ENDED")
            print("=" * 70)
            print("\nWaiting for next training trigger...\n")


if __name__ == "__main__":
    main()

