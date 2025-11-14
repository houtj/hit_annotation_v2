"""FastAPI main application"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware

from db.database import init_db
from routes import files, labels, training
from routes.websocket import router as websocket_router, notification_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    await init_db()
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
