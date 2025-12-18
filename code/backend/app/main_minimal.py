"""
Minimal FastAPI application for Fluxion backend - Simplified version for testing
"""

import logging
import time
from contextlib import asynccontextmanager

from config.settings import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.app.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Application startup time
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Fluxion backend application (minimal mode)...")
    yield
    logger.info("Shutting down Fluxion backend application...")


# Create FastAPI application
app = FastAPI(
    title=settings.app.APP_NAME,
    description="Fluxion DeFi Supply Chain Platform API - Minimal Mode",
    version=settings.app.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.app.VERSION,
        "uptime": time.time() - app_start_time,
        "mode": "minimal",
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Fluxion DeFi Supply Chain Platform API - Minimal Mode",
        "version": settings.app.VERSION,
        "environment": settings.app.ENVIRONMENT,
        "docs_url": "/docs",
        "health_url": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main_minimal:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
