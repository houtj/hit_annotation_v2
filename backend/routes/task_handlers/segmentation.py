"""Segmentation-specific task handlers"""

from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.models import File, Label, Prediction, Class, Config
from db.database import SESSION_DIR
from utils.point_extraction import extract_points_from_mask, merge_with_human_labels


async def extract_points_from_prediction(
    file_id: int,
    created_by: str,
    db: AsyncSession
) -> dict:
    """
    Extract points from prediction mask and save to database
    
    This function:
    1. Removes all previously extracted points (origin="pred")
    2. Extracts fresh points from the current prediction mask
    3. Merges new extracted points with existing human labels (origin="human")
    4. Preserves human labels when coordinates overlap
    
    Args:
        file_id: ID of the file
        created_by: Username for tracking
        db: Database session
    
    Returns:
        Dictionary with updated label data and counts
    """
    import json
    
    # Check if file exists
    file_query = select(File).where(File.id == file_id)
    file_result = await db.execute(file_query)
    file = file_result.scalar_one_or_none()
    
    if not file:
        raise ValueError("File not found")
    
    # Check if prediction exists
    pred_query = select(Prediction).where(Prediction.file_id == file_id)
    pred_result = await db.execute(pred_query)
    prediction = pred_result.scalar_one_or_none()
    
    if not prediction:
        raise ValueError("No prediction available for this file")
    
    # Get prediction path from prediction_data
    try:
        pred_data = json.loads(prediction.prediction_data) if isinstance(prediction.prediction_data, str) else prediction.prediction_data
        if pred_data.get('type') != 'mask':
            raise ValueError("Prediction is not a mask type")
        mask_relative_path = pred_data.get('path')
    except (json.JSONDecodeError, KeyError):
        raise ValueError("Invalid prediction data format")
    
    mask_path = SESSION_DIR / mask_relative_path
    if not mask_path.exists():
        raise ValueError("Prediction file not found on disk")
    
    # Get class colors from database
    classes_query = select(Class)
    classes_result = await db.execute(classes_query)
    classes = classes_result.scalars().all()
    
    color_map = {cls.classname: cls.color for cls in classes}
    foreground_color = color_map.get('foreground', '#ff0000')
    background_color = color_map.get('background', '#0000ff')
    
    # Get point extraction parameters from config
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
    extracted_points = extract_points_from_mask(
        mask_path,
        max_points=max_points,
        confidence_threshold=confidence_threshold,
        min_distance=min_distance,
        gradient_weight=gradient_weight,
        foreground_color=foreground_color,
        background_color=background_color
    )
    
    # Get existing labels
    label_query = select(Label).where(Label.file_id == file_id)
    label_result = await db.execute(label_query)
    existing_label = label_result.scalar_one_or_none()
    
    # Filter only human-created points
    human_labels = []
    removed_pred_count = 0
    if existing_label and existing_label.label_data:
        for label in existing_label.label_data:
            origin = label.get('origin')
            if origin == 'human' or origin is None:
                human_labels.append(label)
            elif origin == 'pred':
                removed_pred_count += 1
    
    # Merge fresh extracted points with human labels
    merged_points = merge_with_human_labels(extracted_points, human_labels)
    
    # Update or create label
    if existing_label:
        existing_label.label_data = merged_points
        if human_labels:
            existing_label.created_by = f"{created_by} + auto: extracted"
        else:
            existing_label.created_by = "auto: extracted"
    else:
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
