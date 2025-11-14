"""FastAPI main application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from db.database import init_db
from routes import files, websocket, labels


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
app.include_router(websocket.router)

# Serve static files (for production)
# app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}
