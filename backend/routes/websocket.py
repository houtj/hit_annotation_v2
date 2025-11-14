"""WebSocket endpoint for real-time updates"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List

router = APIRouter()

# Store active connections
active_connections: List[WebSocket] = []


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
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
        active_connections.remove(websocket)


async def broadcast_message(message: dict):
    """Broadcast message to all connected clients"""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            # Remove dead connections
            active_connections.remove(connection)

