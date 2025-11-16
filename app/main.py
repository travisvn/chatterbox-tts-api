"""
Main FastAPI application
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.tts_model import initialize_model
from app.core.voice_library import get_voice_library
from app.core.background_tasks import start_background_processor, stop_background_processor
from app.api.router import api_router
from app.config import Config
from app.core.version import get_version


ascii_art = r"""
  ____ _           _   _            _               
 / ___| |__   __ _| |_| |_ ___ _ __| |__   _____  __
| |   | '_ \ / _` | __| __/ _ \ '__| '_ \ / _ \ \/ /
| |___| | | | (_| | |_| ||  __/ |  | |_) | (_) >  < 
 \____|_| |_|\__,_|\__|\__\___|_|  |_.__/ \___/_/\_\
                                                    
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
    model_init_task = asyncio.create_task(initialize_model())
    
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
    if model_init_task and not model_init_task.done():
        model_init_task.cancel()
        try:
            await model_init_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Catch any other exceptions that might have occurred during initialization
            print(f"Model initialization raised an exception: {e}")


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


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    # Ensure consistent error response format
    # If detail is already a dict with error key, use it as is
    # Otherwise, wrap it in the standard error format
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        content = exc.detail
    elif isinstance(exc.detail, dict):
        # If it's a dict but not in error format, wrap it
        content = {"error": exc.detail}
    else:
        # If it's a string, create standard error format
        content = {
            "error": {
                "message": str(exc.detail),
                "type": "http_error"
            }
        }
    return JSONResponse(
        status_code=exc.status_code,
        content=content
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