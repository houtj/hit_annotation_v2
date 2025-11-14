"""Utility functions for ML service"""

import requests


def notify_prediction_ready(file_id: int, version: str, backend_url: str = "http://localhost:8000"):
    """
    Notify backend that a prediction is ready via HTTP
    
    Args:
        file_id: ID of the file with new prediction
        version: Model version that generated the prediction
        backend_url: URL of the FastAPI backend
    """
    try:
        response = requests.post(
            f"{backend_url}/api/notifications/prediction-ready",
            json={"file_id": file_id, "version": version},
            timeout=2
        )
        if response.ok:
            print(f"  ✓ Notification sent to frontend via WebSocket")
        else:
            print(f"  ✗ Failed to notify frontend: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Failed to notify frontend: {e}")

