"""Configuration API endpoints"""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
async def get_config(request: Request):
    """
    Get cached application configuration
    
    Returns configuration loaded at startup including:
    - task: Task type ("segmentation" or "classification")
    """
    return {
        "task": request.app.state.app.task_type
    }
