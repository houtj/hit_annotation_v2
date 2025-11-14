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
        created_by: User who created/updated the label
        updated_at: Timestamp of last update
        label_data: JSON array of label objects
            Point label: {"type": "point", "classname": str, "color": str, "x": float, "y": float}
            Mask label: {"type": "mask", "classname": str, "color": str, "path": str}
    """
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(String(100), nullable=False)
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    label_data = Column(JSON, nullable=False, default=list)  # List of label dicts

    # Relationships
    file = relationship("File", back_populates="labels")


class Prediction(Base):
    """
    Predictions table - stores ML model predictions for each file
    
    Attributes:
        id: Primary key
        file_id: Foreign key to files table
        path: Path to the prediction mask file (2D array saved as image/npy)
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    path = Column(String(512), nullable=False)

    # Relationships
    file = relationship("File", back_populates="predictions")


class ModelVersion(Base):
    """
    Model versions table - tracks ML model training history
    
    Attributes:
        version: Version string (primary key, e.g., "v1.0", "v1.1")
        training_start_at: Timestamp when training started
        training_end_at: Timestamp when training completed (null if still training)
        status: Training status (e.g., "training", "completed", "failed")
        metrics: JSON object with training metrics (loss, accuracy, etc.)
        path: Path to the saved model checkpoint
    """
    __tablename__ = "model_versions"

    version = Column(String(50), primary_key=True)
    training_start_at = Column(DateTime, nullable=False, default=func.now())
    training_end_at = Column(DateTime, nullable=True)
    status = Column(String(50), nullable=False, default="training")  # training, completed, failed
    metrics = Column(JSON, nullable=True)  # {"loss": 0.123, "accuracy": 0.95, ...}
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
    """
    __tablename__ = "config"

    key = Column(String(100), primary_key=True)
    value = Column(String(255), nullable=False)



