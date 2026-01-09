"""FastAPI main application"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import select

from db.database import init_db, get_db, AsyncSessionLocal
from db.models import Config, Class
from routes import files, labels, training
from routes.websocket import router as websocket_router, notification_router


class AppState:
    """Application-level state cached at startup"""
    def __init__(self):
        self.task_type: str = "segmentation"  # Default
        self.classes: list = []
        self.config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load config on startup"""
    await init_db()
    
    # Load and cache task type and classes
    async with AsyncSessionLocal() as db:
        # Load config
        config_query = select(Config)
        config_result = await db.execute(config_query)
        configs = config_result.scalars().all()
        app.state.app_config = {cfg.key: cfg.value for cfg in configs}
        
        # Set task type
        app.state.task_type = app.state.app_config.get('task', 'segmentation')
        
        # Load classes
        classes_query = select(Class).order_by(Class.classname)
        classes_result = await db.execute(classes_query)
        classes = classes_result.scalars().all()
        app.state.classes = [
            {"classname": cls.classname, "color": cls.color}
            for cls in classes
        ]
    
    print(f"[Startup] Task type: {app.state.task_type}")
    print(f"[Startup] Classes: {[c['classname'] for c in app.state.classes]}")
    
    yield


# Create FastAPI app
app = FastAPI(
    title="Active Learning Annotation API",
    description="API for image annotation with active learning",
    version="0.1.0",
    lifespan=lifespan,
)

# Initialize app state
app.state.task_type = "segmentation"
app.state.classes = []
app.state.app_config = {}


# No-cache middleware to prevent browser caching
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Add no-cache headers for API endpoints
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

# Add no-cache middleware first
app.add_middleware(NoCacheMiddleware)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files.router)
app.include_router(labels.router)
app.include_router(websocket_router)
app.include_router(notification_router)
app.include_router(training.router)

# Serve static files (for production)
# app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/api/config")
async def get_config(request: Request):
    """
    Get application configuration
    
    Returns cached config including task type and classes.
    This endpoint does NOT query the database - it returns cached values.
    """
    return {
        "task": request.app.state.task_type,
        "classes": request.app.state.classes,
    }
