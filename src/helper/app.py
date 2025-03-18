"""
Main entry point for the application.
"""
import os
import sys
from pathlib import Path

# Add the project root to sys.path for correct imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from src.helper.config import settings
from src.helper.utils.logger import get_logger

# Initialize logger
logger = get_logger("app")

# Initialize application
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import router with absolute import from project root
from src.helper.api.routes import router as memory_router

# Register routers
app.include_router(memory_router, prefix="/api")

# Add global exception handler for logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

@app.get("/")
async def hello_world():
    """Return a Hello World message."""
    logger.info("Hello World endpoint called")
    return {"message": "Hello World from Memory-enabled Helper!"}

# Add custom startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Server running at http://{settings.HOST}:{settings.PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info("Application shutting down")

def main():
    """Entry point for launching via poetry."""
    try:
        uvicorn.run(
            "src.helper.app:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()