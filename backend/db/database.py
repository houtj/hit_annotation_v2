"""Database connection and session management"""

import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import event
from db.models import Base

# Use session directory for database
SESSION_DIR = Path(__file__).parent.parent.parent / "session"
DATABASE_URL = f"sqlite+aiosqlite:///{SESSION_DIR.absolute()}/annotations.db"

# Export SESSION_DIR for other modules
__all__ = ['SESSION_DIR', 'get_db', 'init_db']

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

