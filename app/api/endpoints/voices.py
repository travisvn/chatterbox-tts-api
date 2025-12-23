"""
Voice library management endpoints
"""

import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

from app.models import ErrorResponse, SupportedLanguagesResponse, SupportedLanguageItem
from app.core.voice_library import get_voice_library, SUPPORTED_VOICE_FORMATS
from app.core.aliases import add_route_aliases
from app.core.tts_model import is_multilingual, get_supported_languages
from app.config import Config

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)

@router.get(
    "/languages",
    response_model=SupportedLanguagesResponse,
    responses={
        200: {"description": "List of supported languages"},
        500: {"model": ErrorResponse}
    },
    summary="Get supported languages",
    description="Get all languages supported by the multilingual TTS model"
)
async def get_supported_languages_endpoint():
    """Get supported languages for multilingual TTS"""
    try:
        if not is_multilingual():
            # If not using multilingual model, only English is supported
            languages = [SupportedLanguageItem(code="en", name="English")]
        else:
            # Get supported languages from multilingual model
            supported_langs = get_supported_languages()
            languages = [
                SupportedLanguageItem(code=code, name=name) 
                for code, name in supported_langs.items()
            ]
        # Build full response including required fields
        return SupportedLanguagesResponse(
            languages=languages,
            count=len(languages),
            model_type="multilingual" if is_multilingual() else "standard",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to get supported languages: {str(e)}", "type": "language_error"}}
        )

@router.get(
    "/audio/voices",
    responses={
        200: {"description": "List of voices in library"},
        500: {"model": ErrorResponse}
    },
    summary="List all voices in library",
    description="Get a list of all uploaded voices in the voice library"
)
async def list_voices():
    """List all voices in the voice library"""
    try:
        voice_lib = get_voice_library()
        voices = voice_lib.list_voices()
        
        return {
            "voices": voices,
            "count": len(voices)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to list voices: {str(e)}", "type": "voice_library_error"}}
        )

@router.get(
    "/voices/default",
    responses={
        200: {"description": "Current default voice information"},
        500: {"model": ErrorResponse}
    },
    summary="Get current default voice",
    description="Get information about the currently configured default voice"
)
async def get_default_voice():
    """Get the current default voice"""
    try:
        voice_lib = get_voice_library()
        
        # Get the persistent default voice setting
        default_voice_name = voice_lib.get_default_voice()
        
        if default_voice_name:
            # Default voice is from the voice library
            voice_info = voice_lib.get_voice_info(default_voice_name)
            if voice_info:
                return {
                    "default_voice": default_voice_name,
                    "source": "voice_library",
                    "voice_info": voice_info
                }
            else:
                # Voice was deleted, clear the setting
                voice_lib.clear_default_voice()
        
        # Default voice is not from voice library (file system voice)
        return {
            "default_voice": None,
            "source": "file_system",
            "path": Config.VOICE_SAMPLE_PATH
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to get default voice: {str(e)}", "type": "default_voice_error"}}
        )

@router.post(
    "/voices/default",
    responses={
        200: {"description": "Default voice updated successfully"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Set default voice",
    description="Set a voice from the library as the default voice for TTS generation"
)
async def set_default_voice(
    voice_name: str = Form(..., description="Name of the voice to set as default", min_length=1, max_length=100)
):
    """Set a voice from the library as the default voice"""
    try:
        voice_lib = get_voice_library()
        
        # Use the persistent default voice setting method
        if not voice_lib.set_default_voice(voice_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Voice '{voice_name}' not found in voice library", "type": "voice_not_found_error"}}
            )
        
        voice_info = voice_lib.get_voice_info(voice_name)
        
        return {
            "message": "Default voice updated successfully",
            "default_voice": voice_name,
            "voice_info": voice_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to set default voice: {str(e)}", "type": "default_voice_error"}}
        )

@router.delete(
    "/voices/default",
    responses={
        200: {"description": "Default voice reset to system default"},
        500: {"model": ErrorResponse}
    },
    summary="Reset default voice",
    description="Reset default voice to system default (file system voice)"
)
async def reset_default_voice():
    """Reset default voice to system default"""
    try:
        voice_lib = get_voice_library()
        
        # Use the persistent default voice clearing method
        voice_lib.clear_default_voice()
        
        return {
            "message": "Default voice reset to system default",
            "default_voice": None,
            "source": "file_system",
            "path": Config.VOICE_SAMPLE_PATH
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to reset default voice: {str(e)}", "type": "default_voice_error"}}
        )

@router.post(
    "/voices",
    responses={
        201: {"description": "Voice uploaded successfully"},
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Upload a new voice to the library",
    description="Upload a voice file to the voice library with a custom name"
)
async def upload_voice(
    voice_name: str = Form(..., description="Name for the voice (used in API calls)", min_length=1, max_length=100),
    voice_file: UploadFile = File(..., description="Voice audio file"),
    language: str = Form("en", description="Language code for the voice (e.g., 'en', 'fr', 'es')", min_length=2, max_length=5)
):
    """Upload a new voice to the library"""
    
    # Validate voice file
    if not voice_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": "No filename provided", "type": "invalid_request_error"}}
        )
    
    # Check file extension
    file_ext = os.path.splitext(voice_file.filename.lower())[1]
    if file_ext not in SUPPORTED_VOICE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(SUPPORTED_VOICE_FORMATS)}",
                    "type": "invalid_request_error"
                }
            }
        )
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if hasattr(voice_file, 'size') and voice_file.size and voice_file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"File too large. Maximum size: {max_size // (1024*1024)}MB",
                    "type": "invalid_request_error"
                }
            }
        )
    
    # Validate language if multilingual model is available
    if is_multilingual():
        supported_langs = get_supported_languages()
        if language not in supported_langs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": f"Unsupported language: {language}. Supported languages: {', '.join(supported_langs.keys())}",
                        "type": "invalid_request_error"
                    }
                }
            )
    elif language != "en":
        # If not using multilingual model, only English is supported
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Only English (en) is supported when not using multilingual model",
                    "type": "invalid_request_error"
                }
            }
        )
    
    try:
        # Read file content
        file_content = await voice_file.read()
        
        # Add voice to library with language
        voice_lib = get_voice_library()
        metadata = voice_lib.add_voice(voice_name, file_content, voice_file.filename, language)
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Voice uploaded successfully",
                "voice": {
                    "name": metadata["name"],
                    "filename": metadata["filename"],
                    "file_size": metadata["file_size"],
                    "upload_date": metadata["upload_date"],
                    "language": metadata.get("language", "en")
                }
            }
        )
        
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": {"message": str(e), "type": "voice_exists_error"}}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": str(e), "type": "validation_error"}}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to upload voice: {str(e)}", "type": "voice_library_error"}}
        )


@router.get(
    "/voices/{voice_name}",
    responses={
        200: {"description": "Voice information"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get voice information",
    description="Get detailed information about a specific voice"
)
async def get_voice_info(voice_name: str):
    """Get information about a specific voice"""
    try:
        voice_lib = get_voice_library()
        voice_info = voice_lib.get_voice_info(voice_name)
        
        if voice_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Voice '{voice_name}' not found", "type": "voice_not_found_error"}}
            )
        
        return {"voice": voice_info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to get voice info: {str(e)}", "type": "voice_library_error"}}
        )


@router.get(
    "/voices/{voice_name}/download",
    response_class=FileResponse,
    responses={
        200: {"content": {"audio/*": {}}},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Download a voice file",
    description="Download the original voice file"
)
async def download_voice(voice_name: str):
    """Download a voice file"""
    try:
        voice_lib = get_voice_library()
        voice_path = voice_lib.get_voice_path(voice_name)
        
        if voice_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Voice '{voice_name}' not found", "type": "voice_not_found_error"}}
            )
        
        voice_info = voice_lib.get_voice_info(voice_name)
        filename = voice_info["filename"] if voice_info else f"{voice_name}.wav"
        
        return FileResponse(
            voice_path,
            filename=filename,
            media_type="audio/wav"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to download voice: {str(e)}", "type": "voice_library_error"}}
        )


@router.put(
    "/voices/{voice_name}",
    responses={
        200: {"description": "Voice renamed successfully"},
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Rename a voice",
    description="Rename an existing voice in the library"
)
async def rename_voice(
    voice_name: str,
    new_name: str = Form(..., description="New name for the voice", min_length=1, max_length=100)
):
    """Rename a voice in the library"""
    try:
        voice_lib = get_voice_library()
        success = voice_lib.rename_voice(voice_name, new_name)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Voice '{voice_name}' not found", "type": "voice_not_found_error"}}
            )
        
        return {
            "message": "Voice renamed successfully",
            "old_name": voice_name,
            "new_name": new_name
        }
        
    except HTTPException:
        raise
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": {"message": str(e), "type": "voice_exists_error"}}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": str(e), "type": "validation_error"}}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to rename voice: {str(e)}", "type": "voice_library_error"}}
        )


@router.delete(
    "/voices/{voice_name}",
    responses={
        200: {"description": "Voice deleted successfully"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Delete a voice",
    description="Delete a voice from the library"
)
async def delete_voice(voice_name: str):
    """Delete a voice from the library"""
    try:
        voice_lib = get_voice_library()
        success = voice_lib.delete_voice(voice_name)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Voice '{voice_name}' not found", "type": "voice_not_found_error"}}
            )
        
        return {
            "message": "Voice deleted successfully",
            "voice_name": voice_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to delete voice: {str(e)}", "type": "voice_library_error"}}
        )


@router.post(
    "/voices/{voice_name}/aliases",
    responses={
        201: {"description": "Alias added successfully"},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Add an alias to a voice",
    description="Add an alternative name (alias) for a voice"
)
async def add_voice_alias(
    voice_name: str,
    alias: str = Form(..., description="The alias to add", min_length=1, max_length=100)
):
    """Add an alias to a voice"""
    try:
        voice_lib = get_voice_library()
        success = voice_lib.add_alias(voice_name, alias)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Voice '{voice_name}' not found", "type": "voice_not_found_error"}}
            )
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Alias added successfully",
                "voice_name": voice_name,
                "alias": alias,
                "aliases": voice_lib.list_aliases(voice_name)
            }
        )
        
    except HTTPException:
        raise
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": {"message": str(e), "type": "alias_exists_error"}}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": str(e), "type": "validation_error"}}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to add alias: {str(e)}", "type": "voice_library_error"}}
        )


@router.delete(
    "/voices/{voice_name}/aliases/{alias}",
    responses={
        200: {"description": "Alias removed successfully"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Remove an alias from a voice",
    description="Remove an alternative name (alias) from a voice"
)
async def remove_voice_alias(voice_name: str, alias: str):
    """Remove an alias from a voice"""
    try:
        voice_lib = get_voice_library()
        success = voice_lib.remove_alias(voice_name, alias)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Voice '{voice_name}' or alias '{alias}' not found", "type": "voice_not_found_error"}}
            )
        
        return {
            "message": "Alias removed successfully",
            "voice_name": voice_name,
            "alias": alias,
            "aliases": voice_lib.list_aliases(voice_name)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to remove alias: {str(e)}", "type": "voice_library_error"}}
        )


@router.get(
    "/voices/{voice_name}/aliases",
    responses={
        200: {"description": "List of voice aliases"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="List aliases for a voice",
    description="Get all alternative names (aliases) for a voice"
)
async def list_voice_aliases(voice_name: str):
    """List all aliases for a voice"""
    try:
        voice_lib = get_voice_library()
        
        # Check if voice exists
        if voice_lib.get_voice_info(voice_name) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Voice '{voice_name}' not found", "type": "voice_not_found_error"}}
            )
        
        aliases = voice_lib.list_aliases(voice_name)
        
        return {
            "voice_name": voice_name,
            "aliases": aliases,
            "count": len(aliases)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to list aliases: {str(e)}", "type": "voice_library_error"}}
        )


@router.get(
    "/voices/all-names",
    responses={
        200: {"description": "List of all voice names and aliases"},
        500: {"model": ErrorResponse}
    },
    summary="List all voice names and aliases",
    description="Get all available voice names (including aliases) in the library"
)
async def list_all_voice_names():
    """List all voice names and aliases"""
    try:
        voice_lib = get_voice_library()
        all_names = voice_lib.get_all_voice_names()
        
        return {
            "voice_names": all_names,
            "count": len(all_names)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to list voice names: {str(e)}", "type": "voice_library_error"}}
        )


@router.post(
    "/voices/cleanup",
    responses={
        200: {"description": "Cleanup completed"},
        500: {"model": ErrorResponse}
    },
    summary="Clean up missing voice files",
    description="Remove metadata entries for voice files that no longer exist"
)
async def cleanup_voices():
    """Clean up missing voice files from metadata"""
    try:
        voice_lib = get_voice_library()
        removed_voices = voice_lib.cleanup_missing_files()
        
        return {
            "message": "Cleanup completed",
            "removed_voices": removed_voices,
            "count": len(removed_voices)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Failed to cleanup voices: {str(e)}", "type": "voice_library_error"}}
        )

# Export the base router for the main app to use
__all__ = ["base_router"] 
