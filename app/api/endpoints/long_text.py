"""
Long text TTS endpoints for processing texts that exceed the configured minimum length
"""

import asyncio
import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.models.long_text import (
    LongTextRequest,
    LongTextJobResponse,
    LongTextJobCreateResponse,
    LongTextJobAction,
    LongTextJobActionType,
    LongTextJobList,
    LongTextSSEEvent,
    LongTextJobStatus,
    LongTextJobUpdateRequest,
    LongTextJobRetryRequest,
    LongTextJobDetails,
    LongTextHistoryStats,
    BulkJobAction,
    BulkJobActionResponse,
    LongTextHistorySort
)
from app.config import Config
from app.core.long_text_jobs import get_job_manager
from app.core.background_tasks import get_processor
from app.core.quality_presets import get_quality_preset
from app.core.text_processing import validate_long_text_input, estimate_processing_time
from app.core import add_route_aliases

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.post("/audio/speech/long", response_model=LongTextJobCreateResponse)
async def create_long_text_job(request: LongTextRequest):
    """
    Submit a long text TTS job for background processing.

    Text must exceed the configured minimum length to use this endpoint.
    For shorter texts, use /audio/speech instead.
    """
    try:
        # Validate the input text
        is_valid, error_message = validate_long_text_input(request.input)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": error_message,
                        "type": "invalid_request_error"
                    }
                }
            )

        # Resolve quality and chunking configuration
        preset_name = request.get_quality_preset()
        preset_config = get_quality_preset(preset_name)

        cfg_weight = request.cfg_weight if request.cfg_weight is not None else preset_config["cfg_weight"]
        temperature = request.temperature if request.temperature is not None else preset_config["temperature"]
        chunk_size = request.get_chunk_size(preset_config)
        silence_padding = request.get_silence_padding()
        chunking_strategy = request.get_chunking_strategy()
        pause_settings = request.resolve_pause_settings()

        # Get job manager and processor
        job_manager = get_job_manager()
        processor = get_processor()

        # Create the job
        job_id, estimated_chunks = job_manager.create_job(
            text=request.input,
            voice=request.voice,
            output_format=request.response_format or "mp3",
            exaggeration=request.exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            session_id=request.session_id,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            silence_padding=silence_padding,
            quality_preset=preset_name,
            enable_pauses=pause_settings["enable"],
            custom_pauses=pause_settings["custom"],
        )

        # Submit for background processing
        await processor.submit_job(job_id)

        # Estimate processing time
        estimated_time = estimate_processing_time(len(request.input), chunk_size=chunk_size)

        return LongTextJobCreateResponse(
            job_id=job_id,
            status=LongTextJobStatus.PENDING,
            estimated_processing_time_seconds=estimated_time,
            total_chunks=estimated_chunks,
            message="Job submitted for processing",
            status_url=f"/audio/speech/long/{job_id}",
            sse_url=f"/audio/speech/long/{job_id}/sse"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error"
                }
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to create job: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.get("/audio/speech/long/{job_id}", response_model=LongTextJobResponse)
async def get_job_status(job_id: str):
    """
    Get the status and progress of a long text TTS job.
    """
    try:
        job_manager = get_job_manager()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        # Get job metadata and progress
        metadata = job_manager._load_job_metadata(job_id)
        progress = job_manager.get_progress(job_id)

        if not metadata or not progress:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to load job data",
                        "type": "api_error"
                    }
                }
            )

        # Determine download URL if completed
        download_url = None
        if metadata.status == LongTextJobStatus.COMPLETED and metadata.output_path:
            download_url = f"/audio/speech/long/{job_id}/download"

        # Determine action capabilities
        can_pause = metadata.status == LongTextJobStatus.PROCESSING
        can_resume = metadata.status == LongTextJobStatus.PAUSED

        return LongTextJobResponse(
            job_id=job_id,
            status=metadata.status,
            progress=progress,
            metadata=metadata,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            download_url=download_url,
            can_pause=can_pause,
            can_resume=can_resume
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get job status: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.get("/audio/speech/long/{job_id}/download")
async def download_job_audio(job_id: str):
    """
    Download the completed audio file for a long text TTS job.
    """
    try:
        job_manager = get_job_manager()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        # Get job metadata
        metadata = job_manager._load_job_metadata(job_id)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to load job metadata",
                        "type": "api_error"
                    }
                }
            )

        # Check if job is completed
        if metadata.status != LongTextJobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "message": f"Job is not completed (status: {metadata.status})",
                        "type": "invalid_request_error"
                    }
                }
            )

        # Check if output file exists
        if not metadata.output_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Job completed but no output file available",
                        "type": "api_error"
                    }
                }
            )

        # Construct full path
        job_dir = Path(Config.LONG_TEXT_DATA_DIR) / job_id
        output_file = job_dir / metadata.output_path

        if not output_file.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Output file not found on disk",
                        "type": "api_error"
                    }
                }
            )

        # Determine media type based on format
        SUPPORTED_FORMATS = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg"
        }
        media_type = SUPPORTED_FORMATS.get(metadata.output_format, "audio/wav")
        if metadata.output_format not in SUPPORTED_FORMATS:
            logger.warning(f"Unknown format {metadata.output_format}, defaulting to wav")

        # Return file response
        return FileResponse(
            path=str(output_file),
            media_type=media_type,
            filename=f"long_text_{job_id}.{metadata.output_format}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to download audio: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.put("/audio/speech/long/{job_id}/pause")
async def pause_job(job_id: str):
    """
    Pause a currently processing long text TTS job.
    """
    try:
        job_manager = get_job_manager()
        processor = get_processor()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        # Check job status
        metadata = job_manager._load_job_metadata(job_id)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to load job metadata",
                        "type": "api_error"
                    }
                }
            )

        if metadata.status != LongTextJobStatus.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "message": f"Cannot pause job in {metadata.status} state",
                        "type": "invalid_request_error"
                    }
                }
            )

        # Attempt to pause the job
        success = await processor.pause_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to pause job",
                        "type": "api_error"
                    }
                }
            )

        return {"message": f"Job {job_id} paused successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to pause job: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.put("/audio/speech/long/{job_id}/resume")
async def resume_job(job_id: str):
    """
    Resume a paused long text TTS job.
    """
    try:
        job_manager = get_job_manager()
        processor = get_processor()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        # Check job status
        metadata = job_manager._load_job_metadata(job_id)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to load job metadata",
                        "type": "api_error"
                    }
                }
            )

        if metadata.status != LongTextJobStatus.PAUSED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "message": f"Cannot resume job in {metadata.status} state",
                        "type": "invalid_request_error"
                    }
                }
            )

        # Resume the job by re-submitting it
        await processor.submit_job(job_id)

        return {"message": f"Job {job_id} resumed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to resume job: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.delete("/audio/speech/long/{job_id}")
async def cancel_job(job_id: str, action: LongTextJobActionType = Query(LongTextJobActionType.CANCEL, description="Action to perform: cancel or delete")):
    """
    Cancel or delete a long text TTS job.
    """
    try:
        job_manager = get_job_manager()
        processor = get_processor()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        if action == LongTextJobActionType.CANCEL:
            # Cancel the job (if running) and mark as cancelled
            await processor.pause_job(job_id)  # This cancels active processing
            job_manager.cancel_job(job_id)
            return {"message": f"Job {job_id} cancelled successfully"}

        elif action == LongTextJobActionType.DELETE:
            # Delete the job completely
            await processor.pause_job(job_id)  # Cancel if running
            job_manager.delete_job(job_id)
            return {"message": f"Job {job_id} deleted successfully"}

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": f"Invalid action: {action}",
                        "type": "invalid_request_error"
                    }
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to {action} job: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.get("/audio/speech/long", response_model=LongTextJobList)
async def list_jobs(
    session_id: Optional[str] = None,
    job_status: Optional[LongTextJobStatus] = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    List long text TTS jobs, optionally filtered by status.
    Note: session_id is accepted but not used for filtering (shows all jobs for better UX).
    """
    try:
        job_manager = get_job_manager()

        # Get filtered jobs - session_id filtering removed for better UX
        job_list = job_manager.list_jobs(session_id=session_id, limit=limit)

        # Apply additional status filtering if requested
        if job_status is not None:
            filtered_jobs = [job for job in job_list.jobs if job.status == job_status]
            job_list.jobs = filtered_jobs

        return job_list

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to list jobs: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.get("/audio/speech/long/{job_id}/sse")
async def job_progress_sse(job_id: str):
    """
    Server-Sent Events stream for real-time job progress updates.
    """
    try:
        job_manager = get_job_manager()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        async def event_generator():
            """Generate SSE events for job progress"""
            last_status = None
            last_progress = None

            while True:
                try:
                    # Get current job status and progress
                    metadata = job_manager._load_job_metadata(job_id)
                    progress = job_manager.get_progress(job_id)

                    if not metadata or not progress:
                        break

                    # Check if we should send an update
                    send_update = (
                        metadata.status != last_status or
                        (progress.overall_progress != last_progress and
                         progress.overall_progress % 5 == 0)  # Send every 5% progress
                    )

                    if send_update:
                        event = LongTextSSEEvent(
                            job_id=job_id,
                            event_type="progress",
                            data={
                                "status": metadata.status,
                                "progress": progress.overall_progress,
                                "current_chunk": progress.current_chunk.index if progress.current_chunk else None,
                                "total_chunks": metadata.total_chunks,
                                "estimated_remaining_seconds": progress.estimated_remaining_seconds
                            }
                        )

                        yield {
                            "event": event.event_type,
                            "data": json.dumps(event.data)
                        }

                        last_status = metadata.status
                        last_progress = progress.overall_progress

                    # If job is completed, failed, or cancelled, send final event and exit
                    if metadata.status in [LongTextJobStatus.COMPLETED, LongTextJobStatus.FAILED, LongTextJobStatus.CANCELLED]:
                        final_event = LongTextSSEEvent(
                            job_id=job_id,
                            event_type="completed" if metadata.status == LongTextJobStatus.COMPLETED else "error",
                            data={
                                "status": metadata.status,
                                "message": "Job completed successfully" if metadata.status == LongTextJobStatus.COMPLETED else metadata.error
                            }
                        )

                        yield {
                            "event": final_event.event_type,
                            "data": json.dumps(final_event.data)
                        }
                        break

                    # Wait before next check
                    await asyncio.sleep(2)

                except Exception as e:
                    # Send error event and exit
                    error_event = LongTextSSEEvent(
                        job_id=job_id,
                        event_type="error",
                        data={
                            "message": f"Error monitoring job: {str(e)}"
                        }
                    )

                    yield {
                        "event": error_event.event_type,
                        "data": json.dumps(error_event.data)
                    }
                    break

        return EventSourceResponse(event_generator())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to start SSE stream: {str(e)}",
                    "type": "api_error"
                }
            }
        )


# History-specific endpoints
@router.get("/audio/speech/long-history", response_model=LongTextJobList)
async def list_history_jobs(
    session_id: Optional[str] = None,
    status: Optional[LongTextJobStatus] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    is_archived: Optional[bool] = None,
    sort: LongTextHistorySort = LongTextHistorySort.COMPLETED_DESC,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    List long text TTS jobs for history view with advanced filtering and sorting.
    """
    try:
        from datetime import datetime
        job_manager = get_job_manager()

        # Parse date strings
        start_datetime = None
        end_datetime = None
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"message": "Invalid start_date format", "type": "invalid_request_error"}}
                )

        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"message": "Invalid end_date format", "type": "invalid_request_error"}}
                )

        # Get filtered jobs
        job_list = job_manager.list_history_jobs(
            session_id=session_id,
            status_filter=status,
            start_date=start_datetime,
            end_date=end_datetime,
            search_text=search,
            is_archived=is_archived,
            sort_by=sort.value,
            limit=limit,
            offset=offset
        )

        return job_list

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to list history jobs: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.get("/audio/speech/long-history/stats", response_model=LongTextHistoryStats)
async def get_history_stats(session_id: Optional[str] = None):
    """
    Get statistics for long text TTS history.
    """
    try:
        job_manager = get_job_manager()
        stats_data = job_manager.get_history_stats(session_id=session_id)
        return LongTextHistoryStats(**stats_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get history stats: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.get("/audio/speech/long/{job_id}/details", response_model=LongTextJobDetails)
async def get_job_details(job_id: str):
    """
    Get detailed information about a specific job including chunk details.
    """
    try:
        job_manager = get_job_manager()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        # Load metadata and chunks
        metadata = job_manager._load_job_metadata(job_id)
        chunks = job_manager._load_chunks_data(job_id)
        input_text = job_manager._load_input_text(job_id) or ""

        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to load job metadata",
                        "type": "api_error"
                    }
                }
            )

        # Track access
        job_manager.track_job_access(job_id)

        return LongTextJobDetails(
            metadata=metadata,
            chunks=chunks,
            input_text=input_text,
            error_log=[metadata.error] if metadata.error else [],
            performance_metrics={
                "total_processing_time_ms": metadata.total_processing_time_ms,
                "avg_chunk_time_ms": metadata.total_processing_time_ms / len(chunks) if chunks else 0,
                "success_rate": len([c for c in chunks if c.audio_file]) / len(chunks) if chunks else 0
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get job details: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.patch("/audio/speech/long/{job_id}", response_model=LongTextJobAction)
async def update_job_metadata(job_id: str, update_request: LongTextJobUpdateRequest):
    """
    Update job metadata (name, tags, archive status).
    """
    try:
        job_manager = get_job_manager()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        # Update metadata
        success = job_manager.update_job_metadata(
            job_id=job_id,
            display_name=update_request.display_name,
            tags=update_request.tags,
            is_archived=update_request.is_archived
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to update job metadata",
                        "type": "api_error"
                    }
                }
            )

        # Get updated metadata
        metadata = job_manager._load_job_metadata(job_id)

        return LongTextJobAction(
            success=True,
            message="Job metadata updated successfully",
            status=metadata.status if metadata else LongTextJobStatus.FAILED
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to update job: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.post("/audio/speech/long/{job_id}/retry", response_model=LongTextJobCreateResponse)
async def retry_job(job_id: str, retry_request: LongTextJobRetryRequest):
    """
    Retry a failed job, optionally with new parameters.
    """
    try:
        job_manager = get_job_manager()
        processor = get_processor()

        # Check if job exists
        if not job_manager.job_exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job {job_id} not found",
                        "type": "not_found_error"
                    }
                }
            )

        # Retry the job
        new_job_id = job_manager.retry_job(
            job_id=job_id,
            preserve_chunks=retry_request.preserve_chunks,
            new_parameters=retry_request.new_parameters
        )

        if not new_job_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Cannot retry job in current state",
                        "type": "invalid_request_error"
                    }
                }
            )

        # Submit new job for processing
        await processor.submit_job(new_job_id)

        # Get new job metadata for response
        new_metadata = job_manager._load_job_metadata(new_job_id)
        if not new_metadata:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to load new job metadata",
                        "type": "api_error"
                    }
                }
            )

        return LongTextJobCreateResponse(
            job_id=new_job_id,
            status=new_metadata.status,
            message=f"Retry job created successfully (retry #{new_metadata.retry_count})",
            total_chunks=new_metadata.total_chunks,
            status_url=f"/audio/speech/long/{new_job_id}",
            sse_url=f"/audio/speech/long/{new_job_id}/sse"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to retry job: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.delete("/audio/speech/long/history")
async def clear_history(
    session_id: Optional[str] = None,
    confirm: bool = Query(False, description="Confirmation that user wants to clear history")
):
    """
    Clear job history. Requires confirmation.
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "History clearing requires confirmation (confirm=true)",
                        "type": "invalid_request_error"
                    }
                }
            )

        job_manager = get_job_manager()

        # Get all jobs to clear
        jobs_to_clear = job_manager.list_history_jobs(session_id=session_id, limit=1000)

        cleared_count = 0
        failed_count = 0

        for job in jobs_to_clear.jobs:
            if job.status in [LongTextJobStatus.COMPLETED, LongTextJobStatus.FAILED, LongTextJobStatus.CANCELLED]:
                if job_manager.delete_job(job.job_id):
                    cleared_count += 1
                else:
                    failed_count += 1

        return {
            "message": f"Cleared {cleared_count} jobs from history",
            "cleared_count": cleared_count,
            "failed_count": failed_count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to clear history: {str(e)}",
                    "type": "api_error"
                }
            }
        )


@router.post("/audio/speech/long/bulk", response_model=BulkJobActionResponse)
async def bulk_job_action(bulk_request: BulkJobAction):
    """
    Perform bulk operations on multiple jobs.
    """
    try:
        if not bulk_request.confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Bulk operations require confirmation",
                        "type": "invalid_request_error"
                    }
                }
            )

        job_manager = get_job_manager()
        processor = get_processor()

        success_count = 0
        failed_count = 0
        failed_jobs = []

        for job_id in bulk_request.job_ids:
            try:
                if bulk_request.action == "delete":
                    await processor.pause_job(job_id)  # Cancel if running
                    if job_manager.delete_job(job_id):
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_jobs.append(job_id)

                elif bulk_request.action == "archive":
                    if job_manager.archive_job(job_id):
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_jobs.append(job_id)

                elif bulk_request.action == "unarchive":
                    if job_manager.unarchive_job(job_id):
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_jobs.append(job_id)

                elif bulk_request.action == "retry":
                    new_job_id = job_manager.retry_job(job_id)
                    if new_job_id:
                        await processor.submit_job(new_job_id)
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_jobs.append(job_id)

                else:
                    failed_count += 1
                    failed_jobs.append(job_id)

            except Exception as e:
                failed_count += 1
                failed_jobs.append(job_id)

        total_count = len(bulk_request.job_ids)

        return BulkJobActionResponse(
            success_count=success_count,
            failed_count=failed_count,
            total_count=total_count,
            failed_jobs=failed_jobs,
            message=f"Bulk {bulk_request.action}: {success_count}/{total_count} successful"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to perform bulk action: {str(e)}",
                    "type": "api_error"
                }
            }
        )