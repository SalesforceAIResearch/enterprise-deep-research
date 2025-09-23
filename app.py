
import os
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

# Application version
VERSION = "v0.6.5"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Deep Research API",
    description="An API for performing deep research on topics using LLMs and web search",
    version=VERSION,
)

# Load environment variables from .env and Secrets
load_dotenv(override=True)

# Set default values for critical configuration
llm_provider = os.environ.get('LLM_PROVIDER', 'openai')
llm_model = os.environ.get('LLM_MODEL', 'o3-mini')
max_loops = os.environ.get('MAX_WEB_RESEARCH_LOOPS', '20')

# Debug information
# print("\n=== Environment Variables Debug ===")
# print(f"REPL_ENVIRONMENT: {os.environ.get('REPL_ENVIRONMENT')}")
# 
# print(f"All environment variables: {dict(os.environ)}")

llm_provider = os.environ.get('LLM_PROVIDER')
llm_model = os.environ.get('LLM_MODEL')
max_loops = os.environ.get('MAX_WEB_RESEARCH_LOOPS')

print("\n=== LLM Configuration ===")
print(f"LLM_PROVIDER: {llm_provider}")
print(f"LLM_MODEL: {llm_model}")
print(f"MAX_WEB_RESEARCH_LOOPS: {max_loops}")
print("==============================\n")

# Add error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Try to import our routers
try:
    from routers.research import router as research_router
    from routers.file_analysis import router as file_analysis_router
    logger.info("Successfully imported routers")
except ImportError as e:
    logger.error(f"Error importing routers: {e}")
    raise

# Configure CORS and cache control
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

app.add_middleware(CacheControlMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(research_router)
app.include_router(file_analysis_router)

# Mount the React build directory
app.mount("/static", StaticFiles(directory="ai-research-assistant/build/static"), name="static")
app.mount("/", StaticFiles(directory="ai-research-assistant/build", html=True), name="root")

@app.get("/{path:path}")
async def serve_react(path: str):
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    return FileResponse("ai-research-assistant/build/index.html")

@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "message": "Deep Research API is running",
        "version": VERSION,
        "endpoints": {
            "POST /deep-research": "Perform deep research on a topic",
            "POST /api/files/upload": "Upload and analyze files",
            "GET /api/files/{file_id}/analysis": "Get file analysis results",
        },
        "documentation": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    import logging
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info", workers=1)
