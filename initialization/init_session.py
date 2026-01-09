#!/usr/bin/env python3
"""
Initialize a new annotation session.

Creates session folder structure and populates database with images.

USAGE:
======
    uv run python init_session.py --config dip_picking_config.yaml

Configuration File:
===================
All initialization parameters must be specified in a YAML configuration file.
See dip_picking_config.yaml for a fully documented example for dip picking tasks.

Key configuration sections:
  - Data Paths: image_dir, formats, labels_dir, max_files
  - Feature Extraction: resize (DINOv3 target size)
  - ML Training: prediction_interval, early_stop_patience, early_stop_threshold
  - Point Extraction: max_points, confidence_threshold, min_distance, gradient_weight
  - Classes: name and color for each segmentation class

Configuration Parameters (stored in Config table):
===================================================
All parameters from the config file are stored in the database's Config table
and can be accessed by the backend and ML service during annotation and training.

Feature Extraction:
  - resize: 1536                    # Target size for DINOv3 feature extraction

ML Training:
  - prediction_interval: 20         # Epochs between predictions
  - early_stop_patience: 5          # Epochs to wait for improvement
  - early_stop_threshold: 0.001     # Minimum improvement threshold
  - training_trigger: 0             # Training control (0=idle, 1=start, 2=stop)
  - current_file_id: ""             # Currently annotated file ID
  - model_version: "0.0"            # Current model version (X.X format)

Point Extraction:
  - max_points: 500                 # Maximum points to extract from prediction
  - confidence_threshold: 0.15      # Confidence threshold (pixels <0.15 or >0.85)
  - min_distance: 3.0               # Minimum distance between points (pixels)
  - gradient_weight: 2.0            # Gradient importance weight (higher=more edges)

Classes (example for dip picking):
  - foreground: #00FF00 (green)     # Dip reflections/events
  - background: #FF0000 (red)       # Non-dip areas
"""

import argparse
import asyncio
import os
import sys
import shutil
import json
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from db.models import Base, File, Class, Config, Label

# Default paths (can be overridden by config file)
DEFAULT_SESSION_DIR = Path(__file__).parent.parent / "session"
DEFAULT_DINOV3_REPO = Path(__file__).parent.parent / "models" / "dinov3"
DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_RESIZE_VALUE = 1536

# Global variables (set by load_config or defaults)
SESSION_DIR = DEFAULT_SESSION_DIR
STORAGE_DIR = DEFAULT_SESSION_DIR / "storage" / "input"
FEATURES_DIR = DEFAULT_SESSION_DIR / "storage" / "features"
DB_PATH = DEFAULT_SESSION_DIR / "annotations.db"
DINOV3_REPO = DEFAULT_DINOV3_REPO
WEIGHTS_PATH = DEFAULT_WEIGHTS_PATH
RESIZE_VALUE = DEFAULT_RESIZE_VALUE

# Add DINOv3 to path
sys.path.insert(0, str(DINOV3_REPO))


async def init_database():
    """Initialize database tables"""
    db_url = f"sqlite+aiosqlite:///{DB_PATH.absolute()}"
    engine = create_async_engine(db_url, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    return engine


async def add_file_to_db(session, filename: str, filepath: str, width: int, height: int, feature_path: str):
    """Add file record to database"""
    file_record = File(
        filename=filename,
        filepath=filepath,
        width=width,
        height=height,
        feature_path=feature_path,
    )
    session.add(file_record)
    await session.flush()  # Flush to get the file_id
    return file_record


async def add_label_to_db(session, file_id: int, label_data: list, created_by: str = "imported"):
    """Add label record to database from imported JSON"""
    label_record = Label(
        file_id=file_id,
        created_by=created_by,
        label_data=label_data,
    )
    session.add(label_record)


def find_matching_label_file(image_path: Path, labels_dir: Path) -> Path | None:
    """
    Find matching label JSON file for an image
    
    Searches for a JSON file with the same stem (filename without extension)
    in the labels directory, preserving the folder structure.
    
    Handles naming variations:
    - Exact match: image.png -> image.json
    - Prefix variation: generated_*.png -> mask_*.json
    
    Args:
        image_path: Path to the image file
        labels_dir: Root directory containing label JSON files
    
    Returns:
        Path to matching label file, or None if not found
    """
    if not labels_dir or not labels_dir.exists():
        return None
    
    # Get the stem (filename without extension)
    image_stem = image_path.stem
    
    # Try exact stem match first
    possible_paths = list(labels_dir.rglob(f"{image_stem}.json"))
    
    if possible_paths:
        return possible_paths[0]
    
    # Try common naming pattern variations
    # Pattern: generated_* -> mask_*
    if image_stem.startswith("generated_"):
        alternative_stem = image_stem.replace("generated_", "mask_", 1)
        possible_paths = list(labels_dir.rglob(f"{alternative_stem}.json"))
        if possible_paths:
            return possible_paths[0]
    
    # Pattern: mask_* -> generated_*
    if image_stem.startswith("mask_"):
        alternative_stem = image_stem.replace("mask_", "generated_", 1)
        possible_paths = list(labels_dir.rglob(f"{alternative_stem}.json"))
        if possible_paths:
            return possible_paths[0]
    
    return None


def load_label_json(label_path: Path) -> list:
    """Load label data from JSON file"""
    try:
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        # Validate it's a list
        if not isinstance(label_data, list):
            print(f"  ⚠ Invalid label format in {label_path.name}: expected list")
            return []
        
        return label_data
    except Exception as e:
        print(f"  ✗ Error loading label {label_path.name}: {e}")
        return []


def load_dinov3_model():
    """Load DINOv3 model using torch.hub with local source"""
    try:
        global DINOV3_REPO, WEIGHTS_PATH
        
        if not DINOV3_REPO.exists():
            raise FileNotFoundError(f"DINOv3 repo not found at: {DINOV3_REPO}")
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Weights not found at: {WEIGHTS_PATH}")
        
        # Load model using torch.hub with local source
        model = torch.hub.load(
            repo_or_dir=str(DINOV3_REPO),
            model='dinov3_vits16',
            source='local',
            weights=str(WEIGHTS_PATH)
        )
        
        # Set to eval mode and freeze parameters
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"DINOv3 model loaded on {device}")
        return model, device
    except Exception as e:
        print(f"Error loading DINOv3 model: {e}")
        print(f"Please ensure DINOv3 repo structure is:")
        print(f"  models/dinov3/dinov3/  (code)")
        print(f"  models/weights/  (model weights)")
        import traceback
        traceback.print_exc()
        return None, None


def resize_pad(img: Image.Image, target_size: int = 1536) -> tuple[Image.Image, int, int]:
    """Resize image with largest side to target_size and zero-pad to square.
    
    Returns (padded_image, new_w, new_h), where new_w/new_h are the resized image
    content dimensions inside the padded image.
    """
    from PIL import ImageOps
    
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Round to multiples of 16 for patch size
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Canvas size must also be multiple of 16
    canvas_size = (target_size // 16) * 16
    # Pad with zeros (black), aligned to top-left
    padded = ImageOps.pad(resized, (canvas_size, canvas_size), color=(0, 0, 0), centering=(0, 0))
    return padded, new_w, new_h


def normalize_image(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor for DINOv3.
    
    Returns tensor of shape (1, 3, H, W) with ImageNet normalization.
    """
    x = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).unsqueeze(0)
    x = x.float() / 255.0
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def extract_features(model, device, img_array: np.ndarray, resize: int) -> np.ndarray:
    """Extract DINOv3 features for an image.
    
    Returns features as numpy array of shape (F, H, W) where:
    - F is feature dimension (384 for vits16)
    - H, W are feature map dimensions (e.g., 96x96 for 1536x1536 input)
    """
    try:
        # Convert to PIL Image
        img = Image.fromarray(img_array.astype('uint8'))
        
        # Resize and pad to target size
        img_padded, _, _ = resize_pad(img, target_size=resize)
        
        # Normalize
        img_tensor = normalize_image(img_padded).to(device)
        
        # Extract features using forward_features
        with torch.no_grad():
            feats_dict = model.forward_features(img_tensor)
            # Get patch tokens (excludes CLS token)
            feats = feats_dict["x_norm_patchtokens"]
            
            # Reshape from (1, N_patches, F) to (F, H, W)
            Ph = int(img_tensor.shape[2] / 16)  # Patch height
            Pw = int(img_tensor.shape[3] / 16)  # Patch width
            F = feats.shape[-1]  # Feature dimension
            feats = feats.permute(0, 2, 1).reshape(1, F, Ph, Pw)
        
        # Convert to numpy (fp16 for space efficiency) and remove batch dimension
        features_np = feats[0].cpu().to(torch.float16).numpy()
        return features_np
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


def convert_image_to_npy(image_path: Path, output_path: Path) -> tuple[bool, int, int]:
    """Convert image to numpy array and save as .npy. Returns (success, width, height)"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        np.save(output_path, img_array)
        return True, img.width, img.height
    except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return False, 0, 0


async def scan_and_process_images(
    image_dir: Path, 
    formats: list[str], 
    labels_dir: Path | None = None, 
    max_files: int | None = None,
    classes: list[dict] | None = None,
    ml_config: dict | None = None,
    point_config: dict | None = None
):
    """
    Scan directory for images, convert and add to database
    
    Args:
        image_dir: Directory containing images
        formats: List of image file extensions to process
        labels_dir: Optional directory containing label JSON files
        max_files: Optional maximum number of files to process (for testing/limiting)
    """
    # Check if session exists
    session_exists = SESSION_DIR.exists()
    reinit = False
    
    if session_exists:
        print(f"\nSession directory already exists: {SESSION_DIR.absolute()}")
        response = input("Do you want to re-initialize (remove existing data)? [y/N]: ").strip().lower()
        reinit = response in ['y', 'yes']
        
        if reinit:
            print(f"Removing existing session directory...")
            shutil.rmtree(SESSION_DIR)
            session_exists = False
        else:
            print("Adding to existing session...")
    
    # Create session structure if needed
    if not session_exists:
        SESSION_DIR.mkdir(parents=True)
        STORAGE_DIR.mkdir(parents=True)
        FEATURES_DIR.mkdir(parents=True)
        print(f"Created session directory: {SESSION_DIR.absolute()}")
    
    # Initialize or connect to database
    engine = await init_database()
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    
    # Find all image files
    image_files = []
    for fmt in formats:
        image_files.extend(image_dir.rglob(f"*.{fmt}"))
        image_files.extend(image_dir.rglob(f"*.{fmt.upper()}"))
    
    # Sort files for consistent ordering
    image_files = sorted(image_files)
    
    # Apply max_files limit if specified
    total_found = len(image_files)
    if max_files and max_files < total_found:
        image_files = image_files[:max_files]
        print(f"\nFound {total_found} images (limiting to {max_files})")
    else:
        print(f"\nFound {len(image_files)} images")
    
    # Add default classes and config (only if new session)
    if not session_exists or reinit:
        async with AsyncSessionLocal() as db_session:
            # ============================================================
            # Classes (from config or defaults)
            # ============================================================
            if classes is None:
                classes = [
                    {"name": "foreground", "color": "#00FF00"},
                    {"name": "background", "color": "#FF0000"}
                ]
            
            for cls in classes:
                class_record = Class(classname=cls["name"], color=cls["color"])
                db_session.add(class_record)
            
            # ============================================================
            # Configuration Parameters (from config or defaults)
            # ============================================================
            # All config values are stored as strings in the database
            
            # --- Feature Extraction ---
            resize_config = Config(key="resize", value=str(RESIZE_VALUE))
            db_session.add(resize_config)
            
            # --- ML Training (with defaults) ---
            if ml_config is None:
                ml_config = {
                    "prediction_interval": 20,
                    "early_stop_patience": 5,
                    "early_stop_threshold": 0.001
                }
            
            prediction_interval_config = Config(key="prediction_interval", value=str(ml_config["prediction_interval"]))
            db_session.add(prediction_interval_config)
            
            early_stop_patience_config = Config(key="early_stop_patience", value=str(ml_config["early_stop_patience"]))
            db_session.add(early_stop_patience_config)
            
            early_stop_threshold_config = Config(key="early_stop_threshold", value=str(ml_config["early_stop_threshold"]))
            db_session.add(early_stop_threshold_config)
            
            # Training control parameters (always default)
            training_trigger_config = Config(key="training_trigger", value="0")
            db_session.add(training_trigger_config)
            
            current_file_id_config = Config(key="current_file_id", value="")
            db_session.add(current_file_id_config)
            
            model_version_config = Config(key="model_version", value="0.0")
            db_session.add(model_version_config)
            
            # --- Point Extraction (with defaults) ---
            if point_config is None:
                point_config = {
                    "max_points": 500,
                    "confidence_threshold": 0.15,
                    "min_distance": 3.0,
                    "gradient_weight": 2.0
                }
            
            max_points_config = Config(key="max_points", value=str(point_config["max_points"]))
            db_session.add(max_points_config)
            
            confidence_threshold_config = Config(key="confidence_threshold", value=str(point_config["confidence_threshold"]))
            db_session.add(confidence_threshold_config)
            
            min_distance_config = Config(key="min_distance", value=str(point_config["min_distance"]))
            db_session.add(min_distance_config)
            
            gradient_weight_config = Config(key="gradient_weight", value=str(point_config["gradient_weight"]))
            db_session.add(gradient_weight_config)
            
            await db_session.commit()
        
        class_names = [cls["name"] for cls in classes]
        print(f"Added classes: {', '.join(class_names)}")
        print(f"Added config: resize={RESIZE_VALUE}, ML training params, point extraction params")
    
    # Load DINOv3 model
    print("\nLoading DINOv3 model...")
    model, device = load_dinov3_model()
    if model is None:
        print("Failed to load DINOv3 model. Exiting.")
        return
    
    # Process images
    processed = 0
    skipped = 0
    labels_imported = 0
    async with AsyncSessionLocal() as db_session:
        for img_path in image_files:
            # Create .npy filename (preserve relative structure in filename)
            relative_path = img_path.relative_to(image_dir)
            # Flatten the path structure into filename
            npy_filename = str(relative_path).replace(os.sep, '_').replace('.', '_') + '.npy'
            npy_path = STORAGE_DIR / npy_filename
            feature_filename = npy_filename.replace('.npy', '_features.npy')
            feature_path = FEATURES_DIR / feature_filename
            
            # Skip if file already exists (when adding to existing session)
            if npy_path.exists() and feature_path.exists() and session_exists and not reinit:
                skipped += 1
                continue
            
            # Convert to .npy
            success, width, height = convert_image_to_npy(img_path, npy_path)
            if success:
                # Extract features
                img_array = np.load(npy_path)
                features = extract_features(model, device, img_array, RESIZE_VALUE)
                
                if features.size > 0:
                    # Save features
                    np.save(feature_path, features)
                    
                    # Store paths to database
                    storage_relative = f"storage/input/{npy_filename}"
                    feature_relative = f"storage/features/{feature_filename}"
                    file_record = await add_file_to_db(db_session, img_path.name, storage_relative, width, height, feature_relative)
                    processed += 1
                    
                    # Check for matching label file and import if exists
                    if labels_dir:
                        label_path = find_matching_label_file(img_path, labels_dir)
                        if label_path:
                            label_data = load_label_json(label_path)
                            if label_data:
                                await add_label_to_db(db_session, file_record.id, label_data, created_by="imported")
                                labels_imported += 1
                    
                    if processed % 10 == 0:
                        print(f"Processed {processed}/{len(image_files)} images...")
        
        await db_session.commit()
    
    await engine.dispose()
    print(f"\n✓ Successfully processed {processed} images")
    if labels_imported > 0:
        print(f"✓ Imported {labels_imported} labels")
    if skipped > 0:
        print(f"✓ Skipped {skipped} existing images")
    print(f"✓ Database: {DB_PATH.absolute()}")
    print(f"✓ Storage: {STORAGE_DIR.absolute()}")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path string relative to base directory"""
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Initialize annotation session with images and optional labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with config file:
  uv run python init_session.py --config dip_picking_config.yaml
  
  # Override max_files for testing:
  uv run python init_session.py --config dip_picking_config.yaml --max-files 10
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        required=False,
        help="Maximum number of files to process (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file does not exist: {config_path}")
        return
    
    print(f"Loading configuration from: {config_path.absolute()}")
    config = load_config(config_path)
    config_dir = config_path.parent
    
    # Set global paths from config or defaults
    global SESSION_DIR, STORAGE_DIR, FEATURES_DIR, DB_PATH, DINOV3_REPO, WEIGHTS_PATH, RESIZE_VALUE
    
    if "session_dir" in config:
        SESSION_DIR = resolve_path(config["session_dir"], config_dir)
    
    if "dinov3_repo" in config:
        DINOV3_REPO = resolve_path(config["dinov3_repo"], config_dir)
        # Update sys.path
        sys.path.insert(0, str(DINOV3_REPO))
    
    if "weights_path" in config:
        WEIGHTS_PATH = resolve_path(config["weights_path"], config_dir)
    
    if "resize" in config:
        RESIZE_VALUE = config["resize"]
    
    # Update derived paths
    STORAGE_DIR = SESSION_DIR / "storage" / "input"
    FEATURES_DIR = SESSION_DIR / "storage" / "features"
    DB_PATH = SESSION_DIR / "annotations.db"
    
    # Parse image directory (required from config)
    if "image_dir" not in config or not config["image_dir"]:
        print("Error: 'image_dir' must be specified in config file")
        return
    
    image_dir = resolve_path(config["image_dir"], config_dir)
    if not image_dir.exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        return
    
    # Parse formats (required from config)
    if "formats" not in config or not config["formats"]:
        print("Error: 'formats' must be specified in config file")
        return
    
    formats = [fmt.strip() for fmt in config["formats"].split(',')]
    
    # Parse optional labels directory (from config)
    labels_dir = None
    if "labels_dir" in config and config["labels_dir"]:
        labels_dir = resolve_path(config["labels_dir"], config_dir)
        if labels_dir and not labels_dir.exists():
            print(f"Warning: Labels directory does not exist: {labels_dir}")
            labels_dir = None
    
    # Parse optional max_files (command-line arg overrides config)
    max_files = args.max_files
    if max_files is None and "max_files" in config:
        max_files = config["max_files"]
    
    # Extract classes, ML config, and point extraction config
    classes = config.get("classes", None)
    ml_config = {
        "prediction_interval": config.get("prediction_interval", 20),
        "early_stop_patience": config.get("early_stop_patience", 5),
        "early_stop_threshold": config.get("early_stop_threshold", 0.001)
    }
    point_config = {
        "max_points": config.get("max_points", 500),
        "confidence_threshold": config.get("confidence_threshold", 0.15),
        "min_distance": config.get("min_distance", 3.0),
        "gradient_weight": config.get("gradient_weight", 2.0)
    }
    
    # Print configuration summary
    print("=" * 60)
    print("INITIALIZING ANNOTATION SESSION")
    print("=" * 60)
    print(f"Image directory: {image_dir.absolute()}")
    print(f"Image formats: {', '.join(formats)}")
    if labels_dir:
        print(f"Labels directory: {labels_dir.absolute()}")
    if max_files:
        print(f"Max files limit: {max_files}")
    print(f"Session directory: {SESSION_DIR.absolute()}")
    print(f"DINOv3 repo: {DINOV3_REPO.absolute()}")
    print(f"Weights: {WEIGHTS_PATH.absolute()}")
    print(f"Resize: {RESIZE_VALUE}")
    print("=" * 60)
    
    # Run async processing
    asyncio.run(scan_and_process_images(
        image_dir, 
        formats, 
        labels_dir, 
        max_files,
        classes,
        ml_config,
        point_config
    ))
    
    print("\n✓ Session initialized successfully!")


if __name__ == "__main__":
    main()

