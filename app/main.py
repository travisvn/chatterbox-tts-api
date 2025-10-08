"""
Main FastAPI application
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import mimetypes
import logging
from app.core.tts_model import initialize_model
from app.core.voice_library import get_voice_library
from app.core.background_tasks import start_background_processor, stop_background_processor
from app.api.router import api_router
from app.config import Config
from app.core.version import get_version
from app.core.tts_model import unload_model, _idle_monitor   # add import
ascii_art = r"""


  /$$$$$$  /$$                   /$$     /$$                                 /$$$$$$$                     
 /$$__  $$| $$                  | $$    | $$                                | $$__  $$                    
| $$  \__/| $$$$$$$   /$$$$$$  /$$$$$$ /$$$$$$    /$$$$$$   /$$$$$$         | $$  \ $$  /$$$$$$  /$$   /$$
| $$      | $$__  $$ |____  $$|_  $$_/|_  $$_/   /$$__  $$ /$$__  $$ /$$$$$$| $$$$$$$  /$$__  $$|  $$ /$$/
| $$      | $$  \ $$  /$$$$$$$  | $$    | $$    | $$$$$$$$| $$  \__/|______/| $$__  $$| $$  \ $$ \  $$$$/ 
| $$    $$| $$  | $$ /$$__  $$  | $$ /$$| $$ /$$| $$_____/| $$              | $$  \ $$| $$  | $$  >$$  $$ 
|  $$$$$$/| $$  | $$|  $$$$$$$  |  $$$$/|  $$$$/|  $$$$$$$| $$              | $$$$$$$/|  $$$$$$/ /$$/\  $$
 \______/ |__/  |__/ \_______/   \___/   \___/   \_______/|__/              |_______/  \______/ |__/  \__/                                                                                                        
                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
"""


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(ascii_art)
    
    # Start model initialization in the background
    # This allows the server to respond to health checks immediately
    # while the model loads asynchronously

    import asyncio

    # ❌ remove this (no eager load)
    # model_init_task = asyncio.create_task(initialize_model())

    # ✅ start idle monitor
    asyncio.create_task(_idle_monitor())

    # Initialize voice library to restore default voice settings
    print("Initializing voice library...")
    voice_lib = get_voice_library()
    default_voice = voice_lib.get_default_voice()
    if default_voice:
        print(f"Restored default voice: {default_voice}")
    else:
        print("Using system default voice")

    # Start background processor for long text TTS jobs
    print("Starting long text background processor...")
    await start_background_processor()
    print("Long text background processor started")

    # Note: We don't await the model initialization here
    # The server will start immediately and health checks will show initialization status
    
    yield
    
    # Shutdown (cleanup if needed)
    # Stop background processor
    print("Stopping long text background processor...")
    await stop_background_processor()
    print("Long text background processor stopped")

    # Cancel model initialization if it's still running
    # ✅ force unload model
    await unload_model()

    # ❌ remove cancel logic since no startup task
    # if not model_init_task.done():
    #     model_init_task.cancel()
    #     ...


# Create FastAPI app
app = FastAPI(
    title="Chatterbox TTS API",
    description="REST API for Chatterbox TTS with OpenAI-compatible endpoints",
    version=get_version(),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
cors_origins = Config.CORS_ORIGINS
if cors_origins == "*":
    allowed_origins = ["*"]
else:
    # Split comma-separated origins and strip whitespace
    allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main router
app.include_router(api_router)

# --- ✅ Serve frontend build if present ---
FRONTEND_BUILD_DIR = os.path.join("frontend", "dist")

if os.path.exists(FRONTEND_BUILD_DIR):
    mimetypes.add_type("text/javascript", ".js")  # fix .js MIME type
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_BUILD_DIR, html=True),
        name="spa-static-files",
    )
    print(f"Serving frontend from: {FRONTEND_BUILD_DIR}")
else:
    print(
        f"Frontend build directory not found at '{FRONTEND_BUILD_DIR}'. Serving API only."
    )



# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "internal_error"
            }
        }
    ) 