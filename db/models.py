"""
SQLAlchemy Data Models for Active Learning Annotation Tool

This module defines the database schema for the application.
"""

from datetime import datetime
from typing import Any
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all models"""
    pass


class File(Base):
    """
    Files table - stores information about uploaded images
    
    Attributes:
        id: Primary key
        filename: Original filename
        filepath: Path to the file on disk (relative to data directory)
        width: Image width in pixels
        height: Image height in pixels
        feature_path: Path to extracted DINOv3 features (.npy file)
    """
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False, unique=True)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    feature_path = Column(String(512), nullable=False)

    # Relationships
    labels = relationship("Label", back_populates="file", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="file", cascade="all, delete-orphan")


class Label(Base):
    """
    Labels table - stores annotation data for each file
    
    Attributes:
        id: Primary key
        file_id: Foreign key to files table
        created_by: User who created/updated the label (or "auto: extracted" for ML-generated labels)
        updated_at: Timestamp of last update
        label_data: JSON array of label objects. Each object can be one of three types:
        
            Point label format (for segmentation tasks):
            {
                "type": "point",
                "classname": str,      # e.g., "tree", "building"
                "color": str,          # Hex color code, e.g., "#FF5733"
                "x": float,            # X coordinate in original image space (0.0 to image width)
                "y": float,            # Y coordinate in original image space (0.0 to image height)
                "origin": str          # Either "human" (user-created) or "pred" (extracted from ML prediction)
            }
            
            Class label format (for classification tasks):
            {
                "type": "class",
                "classname": str,      # e.g., "animal", "car", "human"
                "origin": str          # Either "human" (user-created) or "pred" (predicted by model)
            }
            
            Examples: 
                Segmentation: [
                    {"type": "point", "classname": "tree", "color": "#00FF00", "x": 125.5, "y": 200.3, "origin": "human"}
                ]
                Classification: [
                    {"type": "class", "classname": "animal", "origin": "human"}
                ]
    """
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(String(100), nullable=False)
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    label_data = Column(JSON, nullable=False, default=list)

    # Relationships
    file = relationship("File", back_populates="labels")


class Prediction(Base):
    """
    Predictions table - stores ML model predictions for each file
    
    Attributes:
        id: Primary key
        file_id: Foreign key to files table
        prediction_data: JSON object containing prediction data. Format depends on task type:
        
            Segmentation prediction format:
            {
                "type": "mask",
                "path": str  # Path to the prediction mask file (2D array saved as PNG)
            }
            
            Classification prediction format:
            {
                "type": "class",
                "class": str,          # Predicted class name (e.g., "animal", "car")
                "confidence": float    # Confidence score [0.0, 1.0]
            }
            
            Examples:
                Segmentation: {"type": "mask", "path": "storage/predictions/file_1.png"}
                Classification: {"type": "class", "class": "animal", "confidence": 0.95}
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    prediction_data = Column(JSON, nullable=False)

    # Relationships
    file = relationship("File", back_populates="predictions")


class ModelVersion(Base):
    """
    Model versions table - tracks ML model training history
    
    Attributes:
        version: Version string (primary key, e.g., "v1.0", "v1.1", "v2.5")
                 Format: "{major_version}.{epoch}" where major version increments with each training run
        training_start_at: Timestamp when training started
        training_end_at: Timestamp when training completed (null if still training)
        status: Training status - one of: "training", "completed", "failed"
        metrics: JSON array of training metrics, one entry per epoch.
        
            Format: Array of metric objects
            [
                {
                    "epoch": int,        # Epoch number (0-indexed)
                    "train_loss": float, # Training loss for this epoch
                    "test_loss": float   # Validation/test loss for this epoch
                },
                ...
            ]
            
            Example: [
                {"epoch": 0, "train_loss": 0.523, "test_loss": 0.612},
                {"epoch": 1, "train_loss": 0.345, "test_loss": 0.401},
                {"epoch": 2, "train_loss": 0.234, "test_loss": 0.298}
            ]
            
            Note: Metrics are appended incrementally as training progresses
        path: Path to the saved model checkpoint (relative to session directory)
    """
    __tablename__ = "model_versions"

    version = Column(String(50), primary_key=True)
    training_start_at = Column(DateTime, nullable=False, default=func.now())
    training_end_at = Column(DateTime, nullable=True)
    status = Column(String(50), nullable=False, default="training")
    metrics = Column(JSON, nullable=True)
    path = Column(String(512), nullable=True)


class Class(Base):
    """
    Classes table - defines available annotation classes
    
    Attributes:
        classname: Class name (primary key, e.g., "tree", "building")
        color: Hex color code for visualization (e.g., "#FF5733")
    """
    __tablename__ = "classes"

    classname = Column(String(100), primary_key=True)
    color = Column(String(7), nullable=False)  # Hex color: #RRGGBB


class Config(Base):
    """
    Config table - stores global application configuration
    
    Attributes:
        key: Configuration key (primary key)
        value: Configuration value as string
        
    Common configuration keys:
        task: Task type - either "segmentation" or "classification"
        resize: Target size for DINOv3 feature extraction (e.g., "1536")
        prediction_interval: Epochs between predictions (e.g., "20")
        early_stop_patience: Epochs to wait for improvement (e.g., "5")
        early_stop_threshold: Minimum improvement threshold (e.g., "0.001")
        max_points: Maximum points to extract from prediction (segmentation only)
        confidence_threshold: Confidence threshold for point extraction (segmentation only)
        min_distance: Minimum distance between extracted points (segmentation only)
        gradient_weight: Gradient importance weight for point extraction (segmentation only)
    """
    __tablename__ = "config"

    key = Column(String(100), primary_key=True)
    value = Column(String(255), nullable=False)



