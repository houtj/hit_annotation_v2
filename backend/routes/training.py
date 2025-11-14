"""Training control API endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel

from db.database import get_db
from db.models import Config, ModelVersion
from routes.websocket import notify_training_progress
import json

router = APIRouter(prefix="/api/training", tags=["training"])


class StartTrainingResponse(BaseModel):
    version: str
    message: str


class MetricData(BaseModel):
    version: str
    epoch: int
    train_loss: float
    test_loss: float


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


@router.get("/metrics/{major_version}")
async def get_metrics(major_version: int, db: AsyncSession = Depends(get_db)):
    """
    Get all training metrics for a major version
    
    Args:
        major_version: Major version number (e.g., 3 returns metrics from 3.0, 3.1, 3.2, etc.)
        
    Returns:
        Array of metrics sorted by epoch: [{"epoch": 0, "train_loss": 0.5, "test_loss": 0.6}, ...]
    """
    # Query all ModelVersion rows where version starts with "{major_version}."
    query = select(ModelVersion).where(
        ModelVersion.version.like(f"{major_version}.%")
    ).order_by(ModelVersion.version)
    
    result = await db.execute(query)
    versions = result.scalars().all()
    
    # Aggregate metrics from all minor versions
    all_metrics = []
    for version in versions:
        if version.metrics:
            # Metrics is stored as JSON array
            try:
                metrics_array = json.loads(version.metrics) if isinstance(version.metrics, str) else version.metrics
                if isinstance(metrics_array, list):
                    all_metrics.extend(metrics_array)
            except (json.JSONDecodeError, TypeError):
                continue
    
    # Sort by epoch
    all_metrics.sort(key=lambda m: m.get("epoch", 0))
    
    return all_metrics


@router.post("/metrics")
async def append_metric(metric_data: MetricData, db: AsyncSession = Depends(get_db)):
    """
    Append a new metric entry to the current model version and broadcast via WebSocket
    
    Args:
        metric_data: Contains version, epoch, train_loss, test_loss
        
    Returns:
        Success message
    """
    # Find the corresponding ModelVersion row
    query = select(ModelVersion).where(ModelVersion.version == metric_data.version)
    result = await db.execute(query)
    model_version = result.scalar_one_or_none()
    
    if not model_version:
        # Create new ModelVersion entry if it doesn't exist
        from datetime import datetime
        model_version = ModelVersion(
            version=metric_data.version,
            training_start_at=datetime.utcnow(),
            status="training",
            metrics=json.dumps([])
        )
        db.add(model_version)
        await db.flush()
    
    # Load existing metrics array
    try:
        metrics_array = json.loads(model_version.metrics) if isinstance(model_version.metrics, str) else (model_version.metrics or [])
        if not isinstance(metrics_array, list):
            metrics_array = []
    except (json.JSONDecodeError, TypeError):
        metrics_array = []
    
    # Append new metric entry
    new_metric = {
        "epoch": metric_data.epoch,
        "train_loss": metric_data.train_loss,
        "test_loss": metric_data.test_loss
    }
    metrics_array.append(new_metric)
    
    # Save to database
    model_version.metrics = json.dumps(metrics_array)
    await db.commit()
    
    # Broadcast via WebSocket to all connected clients
    await notify_training_progress(
        metric_data.version,
        metric_data.epoch,
        metric_data.train_loss,
        metric_data.test_loss
    )
    
    return {"message": "Metric appended successfully"}

