"""WebSocket endpoint for real-time updates"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import json

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Mark connection as dead
                dead_connections.append(connection)
        
        # Remove dead connections
        for connection in dead_connections:
            self.disconnect(connection)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "status",
            "message": "Connected to server"
        })
        
        # Keep connection alive and receive messages
        while True:
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def notify_prediction_ready(file_id: int, version: str = ""):
    """
    Notify frontend that prediction is ready for a file
    
    Args:
        file_id: ID of the file with new prediction
        version: Model version that generated the prediction
    """
    await manager.broadcast({
        "type": "prediction_ready",
        "file_id": file_id,
        "version": version
    })


async def notify_training_progress(version: str, epoch: int, train_loss: float, test_loss: float):
    """
    Notify frontend about training progress
    
    Args:
        version: Current model version
        epoch: Current training epoch
        train_loss: Training loss
        test_loss: Test/validation loss
    """
    message = {
        "type": "training_progress",
        "version": version,
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss
    }
    
    await manager.broadcast(message)


# HTTP endpoint for ML service to trigger notifications
from fastapi import APIRouter
from pydantic import BaseModel

notification_router = APIRouter(prefix="/api/notifications", tags=["notifications"])


class PredictionNotification(BaseModel):
    file_id: int
    version: str


@notification_router.post("/prediction-ready")
async def trigger_prediction_notification(notification: PredictionNotification):
    """
    Endpoint for ML service to trigger WebSocket notification
    
    Called by the ML service when a prediction is ready
    """
    await notify_prediction_ready(notification.file_id, notification.version)
    return {"status": "notified"}

