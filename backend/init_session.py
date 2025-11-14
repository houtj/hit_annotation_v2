#!/usr/bin/env python3
"""
Initialize a new annotation session.

Creates session folder structure and populates database with images.
"""

import argparse
import asyncio
import os
import sys
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from db.models import Base, File, Class, Config

# Add DINOv3 to path
DINOV3_REPO = Path(__file__).parent / "feature_extraction_model" / "dinov3"
sys.path.insert(0, str(DINOV3_REPO))

SESSION_DIR = Path("../session")
STORAGE_DIR = SESSION_DIR / "storage" / "input"
FEATURES_DIR = SESSION_DIR / "storage" / "features"
DB_PATH = SESSION_DIR / "annotations.db"
RESIZE_VALUE = 1536


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


def load_dinov3_model():
    """Load DINOv3 model using torch.hub with local source"""
    try:
        # Path to DINOv3 repo and weights
        dinov3_repo = Path(__file__).parent / "feature_extraction_model" / "dinov3"
        weights_path = Path(__file__).parent / "feature_extraction_model" / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        
        if not dinov3_repo.exists():
            raise FileNotFoundError(f"DINOv3 repo not found at: {dinov3_repo}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at: {weights_path}")
        
        # Load model using torch.hub with local source
        model = torch.hub.load(
            repo_or_dir=str(dinov3_repo),
            model='dinov3_vits16',
            source='local',
            weights=str(weights_path)
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
        print(f"  backend/feature_extraction_model/dinov3/dinov3/  (code)")
        print(f"  backend/feature_extraction_model/weights/  (model weights)")
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


async def scan_and_process_images(image_dir: Path, formats: list[str]):
    """Scan directory for images, convert and add to database"""
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
    
    print(f"\nFound {len(image_files)} images")
    
    # Add default classes and config (only if new session)
    if not session_exists or reinit:
        async with AsyncSessionLocal() as db_session:
            foreground_class = Class(classname="foreground", color="#00FF00")
            background_class = Class(classname="background", color="#FF0000")
            db_session.add(foreground_class)
            db_session.add(background_class)
            
            # Add config
            resize_config = Config(key="resize", value=str(RESIZE_VALUE))
            db_session.add(resize_config)
            
            # ML training config
            prediction_interval_config = Config(key="prediction_interval", value="20")
            early_stop_patience_config = Config(key="early_stop_patience", value="5")
            early_stop_threshold_config = Config(key="early_stop_threshold", value="0.001")
            training_trigger_config = Config(key="training_trigger", value="0")
            current_file_id_config = Config(key="current_file_id", value="")
            model_version_config = Config(key="model_version", value="0.0")
            
            db_session.add(prediction_interval_config)
            db_session.add(early_stop_patience_config)
            db_session.add(early_stop_threshold_config)
            db_session.add(training_trigger_config)
            db_session.add(current_file_id_config)
            db_session.add(model_version_config)
            
            await db_session.commit()
        
        print("Added default classes: foreground, background")
        print(f"Added config: resize={RESIZE_VALUE}")
    
    # Load DINOv3 model
    print("\nLoading DINOv3 model...")
    model, device = load_dinov3_model()
    if model is None:
        print("Failed to load DINOv3 model. Exiting.")
        return
    
    # Process images
    processed = 0
    skipped = 0
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
                    await add_file_to_db(db_session, img_path.name, storage_relative, width, height, feature_relative)
                    processed += 1
                    if processed % 10 == 0:
                        print(f"Processed {processed}/{len(image_files)} images...")
        
        await db_session.commit()
    
    await engine.dispose()
    print(f"\n✓ Successfully processed {processed} images")
    if skipped > 0:
        print(f"✓ Skipped {skipped} existing images")
    print(f"✓ Database: {DB_PATH.absolute()}")
    print(f"✓ Storage: {STORAGE_DIR.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize annotation session with images"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--formats",
        type=str,
        required=True,
        help="Comma-separated list of image formats (e.g., jpg,png,jpeg)"
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        return
    
    formats = [fmt.strip() for fmt in args.formats.split(',')]
    
    print("=" * 60)
    print("INITIALIZING ANNOTATION SESSION")
    print("=" * 60)
    print(f"Image directory: {image_dir.absolute()}")
    print(f"Image formats: {', '.join(formats)}")
    print(f"Session directory: {SESSION_DIR.absolute()}")
    print("=" * 60)
    
    # Run async processing
    asyncio.run(scan_and_process_images(image_dir, formats))
    
    print("\n✓ Session initialized successfully!")


if __name__ == "__main__":
    main()

