#!/usr/bin/env python3
"""
Initialize a new annotation session.

Creates session folder structure and populates database with images.
"""

import argparse
import asyncio
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from db.models import Base, File, Class


SESSION_DIR = Path("../session")
STORAGE_DIR = SESSION_DIR / "storage" / "input"
DB_PATH = SESSION_DIR / "annotations.db"


async def init_database():
    """Initialize database tables"""
    db_url = f"sqlite+aiosqlite:///{DB_PATH.absolute()}"
    engine = create_async_engine(db_url, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    return engine


async def add_file_to_db(session, filename: str, filepath: str, width: int, height: int):
    """Add file record to database"""
    file_record = File(
        filename=filename,
        filepath=filepath,
        width=width,
        height=height,
    )
    session.add(file_record)


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
    
    # Add default classes (only if new session)
    if not session_exists or reinit:
        async with AsyncSessionLocal() as db_session:
            foreground_class = Class(classname="foreground", color="#00FF00")
            background_class = Class(classname="background", color="#FF0000")
            db_session.add(foreground_class)
            db_session.add(background_class)
            await db_session.commit()
        
        print("Added default classes: foreground, background")
    
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
            
            # Skip if file already exists (when adding to existing session)
            if npy_path.exists() and session_exists and not reinit:
                skipped += 1
                continue
            
            # Convert to .npy
            success, width, height = convert_image_to_npy(img_path, npy_path)
            if success:
                # Store relative path to storage
                storage_relative = f"storage/input/{npy_filename}"
                await add_file_to_db(db_session, img_path.name, storage_relative, width, height)
                processed += 1
                if processed % 100 == 0:
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

