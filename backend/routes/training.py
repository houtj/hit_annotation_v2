"""Training control API endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel

from db.database import get_db
from db.models import Config

router = APIRouter(prefix="/api/training", tags=["training"])


class StartTrainingResponse(BaseModel):
    version: str
    message: str


@router.post("/start", response_model=StartTrainingResponse)
async def start_training(file_id: int, db: AsyncSession = Depends(get_db)):
    """
    Trigger training and set current file for prediction
    
    Args:
        file_id: ID of the file currently being annotated
        
    Returns:
        New version number and status message
    """
    # Set current_file_id
    await db.execute(
        update(Config)
        .where(Config.key == "current_file_id")
        .values(value=str(file_id))
    )
    
    # Get current version and increment major version
    version_query = select(Config).where(Config.key == "model_version")
    version_result = await db.execute(version_query)
    version_config = version_result.scalar_one_or_none()
    
    if version_config:
        current_version = version_config.value
        try:
            major, minor = current_version.split('.')
            new_major = int(major) + 1
            new_version = f"{new_major}.0"
        except (ValueError, AttributeError):
            new_version = "1.0"
    else:
        new_version = "1.0"
    
    # Update version (will be used when training starts)
    await db.execute(
        update(Config)
        .where(Config.key == "model_version")
        .values(value=new_version)
    )
    
    # Set training trigger to 1 (start training)
    await db.execute(
        update(Config)
        .where(Config.key == "training_trigger")
        .values(value="1")
    )
    
    await db.commit()
    
    return StartTrainingResponse(
        version=new_version,
        message=f"Training started for version {new_version}"
    )


@router.post("/stop")
async def stop_training(db: AsyncSession = Depends(get_db)):
    """
    Stop ongoing training
    
    Sets the training_trigger flag to 2, which signals the ML service
    to stop training at the next checkpoint.
    """
    # Set training trigger to 2 (stop training)
    await db.execute(
        update(Config)
        .where(Config.key == "training_trigger")
        .values(value="2")
    )
    
    await db.commit()
    
    return {"message": "Training stop signal sent"}

