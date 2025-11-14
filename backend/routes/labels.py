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

router = APIRouter(prefix="/api", tags=["labels"])


class LabelDataItem(BaseModel):
    type: str
    classname: str
    color: str
    x: float | None = None
    y: float | None = None
    path: str | None = None


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

