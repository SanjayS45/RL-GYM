"""
Main FastAPI Application

Entry point for the RL-GYM API server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .routes import training, environments, agents, datasets
from .websocket import manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RL-GYM API Server...")
    yield
    # Shutdown
    logger.info("Shutting down RL-GYM API Server...")
    await manager.disconnect_all()


# Create FastAPI app
app = FastAPI(
    title="RL-GYM API",
    description="Interactive Reinforcement Learning Training Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Note: Routers already have their own prefixes defined
app.include_router(training.router)
app.include_router(environments.router)
app.include_router(agents.router)
app.include_router(datasets.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RL-GYM API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

