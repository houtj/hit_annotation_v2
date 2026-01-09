"""Label management API endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import List
from pathlib import Path
import numpy as np
from PIL import Image
import io

from db.database import get_db, SESSION_DIR
from db.models import File, Label, Class
from utils.point_extraction import extract_points_from_mask, merge_with_human_labels

router = APIRouter(prefix="/api", tags=["labels"])


class LabelDataItem(BaseModel):
    type: str
    classname: str
    color: str
    x: float | None = None
    y: float | None = None
    path: str | None = None
    origin: str | None = None  # 'human' or 'pred'


class CreateLabelRequest(BaseModel):
    label_data: List[LabelDataItem]
    created_by: str


@router.get("/files/{file_id}")
async def get_file(file_id: int, db: AsyncSession = Depends(get_db)):
    """Get single file details"""
    query = select(File).where(File.id == file_id)
    result = await db.execute(query)
    file = result.scalar_one_or_none()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get label if exists
    label_query = select(Label).where(Label.file_id == file_id)
    label_result = await db.execute(label_query)
    label = label_result.scalar_one_or_none()
    
    return {
        "id": file.id,
        "filename": file.filename,
        "filepath": file.filepath,
        "width": file.width,
        "height": file.height,
        "label": {
            "id": label.id,
            "label_data": label.label_data,
            "created_by": label.created_by,
            "updated_at": label.updated_at.isoformat(),
        } if label else None,
    }


@router.get("/classes")
async def get_classes(db: AsyncSession = Depends(get_db)):
    """Get all annotation classes"""
    query = select(Class).order_by(Class.classname)
    result = await db.execute(query)
    classes = result.scalars().all()
    
    return [
        {
            "classname": cls.classname,
            "color": cls.color,
        }
        for cls in classes
    ]


@router.get("/files/{file_id}/image")
async def get_file_image(file_id: int, db: AsyncSession = Depends(get_db)):
    """Get image data for a file"""
    query = select(File).where(File.id == file_id)
    result = await db.execute(query)
    file = result.scalar_one_or_none()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Load .npy file
    npy_path = SESSION_DIR / file.filepath
    if not npy_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    # Load numpy array and convert to image
    img_array = np.load(npy_path)
    img = Image.fromarray(img_array.astype('uint8'))
    
    # Convert to PNG bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")


@router.post("/files/{file_id}/labels")
async def create_or_update_label(
    file_id: int,
    request: CreateLabelRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create or update label for a file"""
    # Check if file exists
    file_query = select(File).where(File.id == file_id)
    file_result = await db.execute(file_query)
    file = file_result.scalar_one_or_none()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if label exists
    label_query = select(Label).where(Label.file_id == file_id)
    label_result = await db.execute(label_query)
    label = label_result.scalar_one_or_none()
    
    # Convert label_data to dict list
    label_data_list = [item.model_dump() for item in request.label_data]
    
    if label:
        # Update existing label
        label.label_data = label_data_list
        label.created_by = request.created_by
    else:
        # Create new label
        label = Label(
            file_id=file_id,
            created_by=request.created_by,
            label_data=label_data_list,
        )
        db.add(label)
    
    await db.commit()
    await db.refresh(label)
    
    return {
        "id": label.id,
        "file_id": label.file_id,
        "label_data": label.label_data,
        "created_by": label.created_by,
        "updated_at": label.updated_at.isoformat(),
    }


@router.get("/files/{file_id}/prediction")
async def get_prediction(file_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get prediction for a file
    
    For segmentation: Returns the prediction mask as a PNG image
    For classification: Returns JSON with class and confidence
    """
    from db.models import Prediction
    
    # Get prediction for this file
    pred_query = select(Prediction).where(Prediction.file_id == file_id)
    pred_result = await db.execute(pred_query)
    prediction = pred_result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No prediction available for this file")
    
    pred_data = prediction.prediction_data
    
    if pred_data.get("type") == "mask":
        # Segmentation: return mask image
        mask_path = SESSION_DIR / pred_data["path"]
        if not mask_path.exists():
            raise HTTPException(status_code=404, detail="Prediction file not found on disk")
        
        try:
            mask_img = Image.open(mask_path)
            
            # Convert to PNG bytes
            img_byte_arr = io.BytesIO()
            mask_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading prediction mask: {str(e)}")
    
    elif pred_data.get("type") == "class":
        # Classification: return JSON
        return {
            "type": "class",
            "class": pred_data.get("class"),
            "confidence": pred_data.get("confidence")
        }
    
    else:
        raise HTTPException(status_code=500, detail="Unknown prediction type")


@router.post("/files/{file_id}/extract-points")
async def extract_points(
    file_id: int,
    created_by: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract points from prediction mask and save to database
    
    This endpoint:
    1. Removes all previously extracted points (origin="pred")
    2. Extracts fresh points from the current prediction mask
    3. Merges new extracted points with existing human labels (origin="human")
    4. Preserves human labels when coordinates overlap
    
    Args:
        file_id: ID of the file
        created_by: Username (passed as query parameter for tracking)
    
    Returns:
        Updated label with merged points, including counts of:
        - extracted_count: Number of new points extracted
        - human_count: Number of human labels preserved
        - removed_count: Number of old extracted points removed
        - total_count: Total points after merge
    """
    from db.models import Prediction
    
    # Check if file exists
    file_query = select(File).where(File.id == file_id)
    file_result = await db.execute(file_query)
    file = file_result.scalar_one_or_none()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if prediction exists
    pred_query = select(Prediction).where(Prediction.file_id == file_id)
    pred_result = await db.execute(pred_query)
    prediction = pred_result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No prediction available for this file")
    
    # Get prediction mask path
    mask_path = SESSION_DIR / prediction.path
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found on disk")
    
    # Get class colors from database
    classes_query = select(Class)
    classes_result = await db.execute(classes_query)
    classes = classes_result.scalars().all()
    
    # Build color map
    color_map = {cls.classname: cls.color for cls in classes}
    foreground_color = color_map.get('foreground', '#ff0000')
    background_color = color_map.get('background', '#0000ff')
    
    # Get point extraction parameters from config
    from db.models import Config
    config_query = select(Config).where(Config.key.in_([
        'max_points', 'confidence_threshold', 'min_distance', 'gradient_weight'
    ]))
    config_result = await db.execute(config_query)
    configs = {cfg.key: cfg.value for cfg in config_result.scalars().all()}
    
    # Parse config values with defaults
    max_points = int(configs.get('max_points', '500'))
    confidence_threshold = float(configs.get('confidence_threshold', '0.15'))
    min_distance = float(configs.get('min_distance', '3.0'))
    gradient_weight = float(configs.get('gradient_weight', '2.0'))
    
    # Extract points from mask
    try:
        extracted_points = extract_points_from_mask(
            mask_path,
            max_points=max_points,
            confidence_threshold=confidence_threshold,
            min_distance=min_distance,
            gradient_weight=gradient_weight,
            foreground_color=foreground_color,
            background_color=background_color
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting points: {str(e)}")
    
    # Get existing labels
    label_query = select(Label).where(Label.file_id == file_id)
    label_result = await db.execute(label_query)
    existing_label = label_result.scalar_one_or_none()
    
    # Filter only human-created points (remove all previously extracted points)
    human_labels = []
    removed_pred_count = 0
    if existing_label and existing_label.label_data:
        for label in existing_label.label_data:
            origin = label.get('origin')
            # Keep only human points (origin="human" or None for backward compatibility)
            if origin == 'human' or origin is None:
                human_labels.append(label)
            elif origin == 'pred':
                removed_pred_count += 1
    
    # Merge fresh extracted points with human labels
    merged_points = merge_with_human_labels(extracted_points, human_labels)
    
    # Update or create label
    if existing_label:
        # Update existing label
        existing_label.label_data = merged_points
        # Mark as hybrid if there are human points, otherwise pure auto
        if human_labels:
            existing_label.created_by = f"{created_by} + auto: extracted"
        else:
            existing_label.created_by = "auto: extracted"
    else:
        # Create new label
        existing_label = Label(
            file_id=file_id,
            created_by="auto: extracted",
            label_data=merged_points,
        )
        db.add(existing_label)
    
    await db.commit()
    await db.refresh(existing_label)
    
    return {
        "id": existing_label.id,
        "file_id": existing_label.file_id,
        "label_data": existing_label.label_data,
        "created_by": existing_label.created_by,
        "updated_at": existing_label.updated_at.isoformat(),
        "extracted_count": len(extracted_points),
        "human_count": len(human_labels),
        "removed_count": removed_pred_count,
        "total_count": len(merged_points),
    }

