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
from fastapi import FastAPI
from src.helper.config import settings

# Initialize application
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Import router with absolute import from project root
from src.helper.api.routes import router as memory_router

# Register routers
app.include_router(memory_router, prefix="/api")

@app.get("/")
async def hello_world():
    """Return a Hello World message."""
    return {"message": "Hello World from Memory-enabled Helper!"}

def main():
    """Entry point for launching via poetry."""
    uvicorn.run(
        "src.helper.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

if __name__ == "__main__":
    main()