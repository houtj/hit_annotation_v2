"""FastAPI main application"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import select

from db.database import init_db, get_db
from db.models import Config, Class
from routes import files, labels, training, config
from routes.websocket import router as websocket_router, notification_router


class AppState:
    """Application state for caching configuration"""
    def __init__(self):
        self.task_type: str | None = None
        self.classes: list[dict] | None = None
        self.config: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load configuration on startup"""
    await init_db()
    
    # Load configuration into app state
    app.state.app = AppState()
    
    async for db in get_db():
        # Load task type and config
        config_query = select(Config)
        config_result = await db.execute(config_query)
        configs = config_result.scalars().all()
        app.state.app.config = {cfg.key: cfg.value for cfg in configs}
        app.state.app.task_type = app.state.app.config.get('task', 'segmentation')
        
        # Load classes
        classes_query = select(Class)
        classes_result = await db.execute(classes_query)
        classes = classes_result.scalars().all()
        app.state.app.classes = [{"classname": cls.classname, "color": cls.color} for cls in classes]
        
        print(f"Loaded configuration: task_type={app.state.app.task_type}, classes={len(app.state.app.classes)}")
        break
    
    yield


# Create FastAPI app
app = FastAPI(
    title="Active Learning Annotation API",
    description="API for image annotation with active learning",
    version="0.1.0",
    lifespan=lifespan,
)

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
app.include_router(config.router)
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
