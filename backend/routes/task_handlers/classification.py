"""Classification-specific task handlers"""

import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.models import Prediction


async def get_classification_prediction(
    file_id: int,
    db: AsyncSession
) -> dict | None:
    """
    Get classification prediction for a file
    
    Args:
        file_id: ID of the file
        db: Database session
    
    Returns:
        Dictionary with prediction data or None if no prediction exists
        Format:
        {
            "type": "class",
            "class": str,           # Predicted class name
            "confidence": float,    # Confidence score 0-1
            "probabilities": {...}  # Per-class probabilities
        }
    """
    pred_query = select(Prediction).where(Prediction.file_id == file_id)
    pred_result = await db.execute(pred_query)
    prediction = pred_result.scalar_one_or_none()
    
    if not prediction:
        return None
    
    try:
        pred_data = prediction.prediction_data
        if isinstance(pred_data, str):
            pred_data = json.loads(pred_data)
        
        if pred_data.get('type') != 'class':
            return None
        
        return pred_data
    except (json.JSONDecodeError, AttributeError):
        return None


async def save_classification_label(
    file_id: int,
    class_name: str,
    created_by: str,
    db: AsyncSession
) -> dict:
    """
    Save a classification label for a file
    
    Args:
        file_id: ID of the file
        class_name: Name of the class to assign
        created_by: Username who created the label
        db: Database session
    
    Returns:
        Dictionary with saved label data
    """
    from db.models import File, Label
    
    # Check if file exists
    file_query = select(File).where(File.id == file_id)
    file_result = await db.execute(file_query)
    file = file_result.scalar_one_or_none()
    
    if not file:
        raise ValueError("File not found")
    
    # Create label data
    label_data = [{
        "type": "class",
        "classname": class_name,
        "origin": "human"
    }]
    
    # Check if label exists
    label_query = select(Label).where(Label.file_id == file_id)
    label_result = await db.execute(label_query)
    existing_label = label_result.scalar_one_or_none()
    
    if existing_label:
        existing_label.label_data = label_data
        existing_label.created_by = created_by
    else:
        existing_label = Label(
            file_id=file_id,
            created_by=created_by,
            label_data=label_data,
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
    }
