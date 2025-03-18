"""
Main entry point for the application.
"""
import sys
import os
from pathlib import Path

# Добавляем директорию src в sys.path для корректных импортов
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # переходим на уровень выше src
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI

# Initialize application
app = FastAPI(title="Helper App with Memory", version="0.1.0")

# Используем абсолютный импорт от корня проекта
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
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

if __name__ == "__main__":
    main()