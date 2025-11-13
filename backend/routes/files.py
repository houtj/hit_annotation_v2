"""File management API endpoints"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List

from db.database import get_db
from db.models import File, Label, Prediction, ModelVersion

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/")
async def get_files(db: AsyncSession = Depends(get_db)):
    """
    Get all files with their annotation status
    
    Returns:
        List of files with labeled status and labeler information
    """
    # Query files with their labels
    query = select(File).order_by(File.filename)
    result = await db.execute(query)
    files = result.scalars().all()
    
    # Build response with labeled status
    files_data = []
    for file in files:
        # Get label for this file
        label_query = select(Label).where(Label.file_id == file.id)
        label_result = await db.execute(label_query)
        label = label_result.scalar_one_or_none()
        
        # Get prediction for this file
        pred_query = select(Prediction).where(Prediction.file_id == file.id)
        pred_result = await db.execute(pred_query)
        prediction = pred_result.scalar_one_or_none()
        
        # Determine labeled status
        if label and len(label.label_data) > 0:
            labeled = "manual"
            labeler = label.created_by
        elif prediction:
            labeled = "auto"
            # Get model version from prediction path or latest version
            version_query = select(ModelVersion).where(
                ModelVersion.status == "completed"
            ).order_by(ModelVersion.training_end_at.desc())
            version_result = await db.execute(version_query)
            latest_version = version_result.scalar_one_or_none()
            labeler = f"auto: {latest_version.version}" if latest_version else "auto"
        else:
            labeled = "no"
            labeler = "no"
        
        files_data.append({
            "id": file.id,
            "filename": file.filename,
            "filepath": file.filepath,
            "labeled": labeled,
            "labeler": labeler,
        })
    
    return files_data


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """
    Get folder statistics
    
    Returns:
        - total_files: Total number of files
        - manual_labeled: Number of manually labeled files
        - model_version: Latest completed model version
    """
    # Total files
    total_query = select(func.count(File.id))
    total_result = await db.execute(total_query)
    total_files = total_result.scalar()
    
    # Manual labeled (files with labels that have label_data)
    manual_query = select(func.count(Label.id.distinct())).where(
        Label.label_data != '[]'
    )
    manual_result = await db.execute(manual_query)
    manual_labeled = manual_result.scalar() or 0
    
    # Latest model version
    version_query = select(ModelVersion).where(
        ModelVersion.status == "completed"
    ).order_by(ModelVersion.training_end_at.desc())
    version_result = await db.execute(version_query)
    latest_version = version_result.scalar_one_or_none()
    
    return {
        "total_files": total_files,
        "manual_labeled": manual_labeled,
        "model_version": latest_version.version if latest_version else "None",
    }

