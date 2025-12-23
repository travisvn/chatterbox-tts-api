"""
TTS request status tracking and management
"""

import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass, asdict


class TTSStatus(Enum):
    """TTS request status enumeration"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING_TEXT = "processing_text"
    CHUNKING = "chunking"
    GENERATING_AUDIO = "generating_audio"
    CONCATENATING = "concatenating"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TTSProgressInfo:
    """Progress information for TTS generation"""
    current_chunk: int = 0
    total_chunks: int = 0
    current_step: str = ""
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_chunks == 0:
            return 0.0
        return (self.current_chunk / self.total_chunks) * 100.0


@dataclass 
class TTSRequestInfo:
    """Complete TTS request information"""
    request_id: str
    status: TTSStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    text_length: int = 0
    text_preview: str = ""
    voice_source: str = "default"
    parameters: Dict[str, Any] = None
    progress: TTSProgressInfo = None
    error_message: Optional[str] = None
    memory_usage: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.parameters is None:
            self.parameters = {}
        if self.progress is None:
            self.progress = TTSProgressInfo()
        if self.memory_usage is None:
            self.memory_usage = {}
    
    @property
    def duration_seconds(self) -> float:
        """Calculate request duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    @property
    def is_active(self) -> bool:
        """Check if request is currently active"""
        return self.status not in [TTSStatus.COMPLETED, TTSStatus.ERROR, TTSStatus.IDLE]


class TTSStatusManager:
    """Thread-safe TTS status manager"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._current_request: Optional[TTSRequestInfo] = None
        self._request_history: List[TTSRequestInfo] = []
        self._max_history = 10  # Keep last 10 requests
        self._total_requests = 0
    
    def start_request(
        self,
        text: str,
        voice_source: str = "default",
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start tracking a new TTS request"""
        with self._lock:
            request_id = str(uuid.uuid4())[:8]  # Short ID for logging
            
            self._current_request = TTSRequestInfo(
                request_id=request_id,
                status=TTSStatus.INITIALIZING,
                start_time=datetime.now(timezone.utc),
                text_length=len(text),
                text_preview=text[:100] + "..." if len(text) > 100 else text,
                voice_source=voice_source,
                parameters=parameters or {}
            )
            
            self._total_requests += 1
            return request_id
    
    def update_status(
        self,
        request_id: str,
        status: TTSStatus,
        current_step: str = "",
        current_chunk: int = None,
        total_chunks: int = None,
        memory_usage: Optional[Dict[str, float]] = None,
        error_message: Optional[str] = None
    ):
        """Update request status and progress"""
        with self._lock:
            if not self._current_request or self._current_request.request_id != request_id:
                return
            
            self._current_request.status = status
            
            if current_step:
                self._current_request.progress.current_step = current_step
            
            if current_chunk is not None:
                self._current_request.progress.current_chunk = current_chunk
            
            if total_chunks is not None:
                self._current_request.progress.total_chunks = total_chunks
                # Estimate completion time based on progress
                if current_chunk and current_chunk > 0:
                    elapsed = self._current_request.duration_seconds
                    estimated_total = (elapsed / current_chunk) * total_chunks
                    remaining = max(0, estimated_total - elapsed)
                    self._current_request.progress.estimated_completion = datetime.fromtimestamp(
                        datetime.now(timezone.utc).timestamp() + remaining,
                        tz=timezone.utc
                    )
            
            if memory_usage:
                self._current_request.memory_usage.update(memory_usage)
            
            if error_message:
                self._current_request.error_message = error_message
            
            # If completed or error, finalize request
            if status in [TTSStatus.COMPLETED, TTSStatus.ERROR]:
                self._current_request.end_time = datetime.now(timezone.utc)
                self._finalize_request()
    
    def _finalize_request(self):
        """Move current request to history"""
        if self._current_request:
            # Add to history
            self._request_history.append(self._current_request)
            
            # Trim history if too long
            if len(self._request_history) > self._max_history:
                self._request_history = self._request_history[-self._max_history:]
            
            # Clear current request 
            self._current_request = None
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        with self._lock:
            if not self._current_request:
                return {
                    "status": TTSStatus.IDLE.value,
                    "is_processing": False,
                    "total_requests": self._total_requests,
                    "message": "No active requests"
                }
            
            request_dict = asdict(self._current_request)
            
            # Convert datetime objects to timestamps
            request_dict['start_time'] = self._current_request.start_time.timestamp()
            if self._current_request.end_time:
                request_dict['end_time'] = self._current_request.end_time.timestamp()
            
            # Add computed properties
            request_dict['duration_seconds'] = self._current_request.duration_seconds
            request_dict['is_active'] = self._current_request.is_active
            request_dict['is_processing'] = True
            request_dict['total_requests'] = self._total_requests
            
            # Format progress info and ensure all required fields are present
            if self._current_request.progress.estimated_completion:
                request_dict['progress']['estimated_completion'] = (
                    self._current_request.progress.estimated_completion
                )
            
            # Ensure progress_percentage is always present
            request_dict['progress']['progress_percentage'] = self._current_request.progress.progress_percentage
            
            return request_dict
    
    def get_request_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent request history"""
        with self._lock:
            history = []
            for request in self._request_history[-limit:]:
                request_dict = asdict(request)
                request_dict['start_time'] = request.start_time.timestamp()
                if request.end_time:
                    request_dict['end_time'] = request.end_time.timestamp()
                request_dict['duration_seconds'] = request.duration_seconds
                history.append(request_dict)
            
            return list(reversed(history))  # Most recent first
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self._lock:
            completed_requests = [r for r in self._request_history if r.status == TTSStatus.COMPLETED]
            error_requests = [r for r in self._request_history if r.status == TTSStatus.ERROR]
            
            if completed_requests:
                avg_duration = sum(r.duration_seconds for r in completed_requests) / len(completed_requests)
                avg_text_length = sum(r.text_length for r in completed_requests) / len(completed_requests)
            else:
                avg_duration = 0
                avg_text_length = 0
            
            return {
                "total_requests": self._total_requests,
                "completed_requests": len(completed_requests),
                "error_requests": len(error_requests),
                "success_rate": (
                    len(completed_requests) / max(1, len(completed_requests) + len(error_requests))
                ) * 100,
                "average_duration_seconds": avg_duration,
                "average_text_length": avg_text_length,
                "is_processing": self._current_request is not None
            }
    
    def clear_history(self):
        """Clear request history (keep current request)"""
        with self._lock:
            self._request_history.clear()


# Global status manager instance
_status_manager = TTSStatusManager()


# Public API functions
def start_tts_request(
    text: str,
    voice_source: str = "default",
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """Start tracking a new TTS request"""
    return _status_manager.start_request(text, voice_source, parameters)


def update_tts_status(
    request_id: str,
    status: TTSStatus,
    current_step: str = "",
    current_chunk: int = None,
    total_chunks: int = None,
    memory_usage: Optional[Dict[str, float]] = None,
    error_message: Optional[str] = None
):
    """Update TTS request status"""
    _status_manager.update_status(
        request_id, status, current_step, current_chunk, 
        total_chunks, memory_usage, error_message
    )


def get_tts_status() -> Dict[str, Any]:
    """Get current TTS processing status"""
    return _status_manager.get_current_status()


def get_tts_history(limit: int = 5) -> List[Dict[str, Any]]:
    """Get TTS request history"""
    return _status_manager.get_request_history(limit)


def get_tts_statistics() -> Dict[str, Any]:
    """Get TTS processing statistics"""
    return _status_manager.get_statistics()


def clear_tts_history():
    """Clear TTS request history"""
    _status_manager.clear_history() 