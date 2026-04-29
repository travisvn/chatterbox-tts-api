"""
Memory management endpoints
"""

import torch
from typing import Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException, status

from app.core import get_memory_info, cleanup_memory, add_route_aliases
from app.config import Config

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)

# Global request counter for memory tracking
REQUEST_COUNTER = 0

# Memory alert thresholds (configurable)
MEMORY_ALERT_THRESHOLDS = {
    "cpu_memory_percent": 85.0,  # Alert when CPU memory usage exceeds 85%
    "gpu_memory_mb": 8192,       # Alert when GPU memory exceeds 8GB
}


def get_memory_alerts(memory_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check for memory alerts based on current usage"""
    alerts = []
    
    # CPU memory alert
    if memory_info.get("cpu_memory_percent", 0) > MEMORY_ALERT_THRESHOLDS["cpu_memory_percent"]:
        alerts.append({
            "type": "cpu_memory_high",
            "level": "warning",
            "message": f"CPU memory usage is high: {memory_info['cpu_memory_percent']:.1f}%",
            "threshold": MEMORY_ALERT_THRESHOLDS["cpu_memory_percent"],
            "current": memory_info["cpu_memory_percent"]
        })
    
    # GPU memory alert
    gpu_memory = memory_info.get("gpu_memory_allocated_mb", 0)
    if gpu_memory > MEMORY_ALERT_THRESHOLDS["gpu_memory_mb"]:
        alerts.append({
            "type": "gpu_memory_high",
            "level": "warning", 
            "message": f"GPU memory usage is high: {gpu_memory:.1f}MB",
            "threshold": MEMORY_ALERT_THRESHOLDS["gpu_memory_mb"],
            "current": gpu_memory
        })
    
    # Memory leak detection (simplified)
    if REQUEST_COUNTER > 10:  # Only after some requests
        if memory_info.get("cpu_memory_percent", 0) > 90:
            alerts.append({
                "type": "potential_memory_leak",
                "level": "error",
                "message": "Potential memory leak detected - consider restarting",
                "suggestion": "High memory usage after multiple requests"
            })
    
    return {
        "has_alerts": len(alerts) > 0,
        "alert_count": len(alerts),
        "alerts": alerts
    }


@router.get(
    "/memory",
    summary="Memory information and management",
    description="Get current memory usage and perform cleanup operations"
)
async def memory_management(
    cleanup: bool = Query(False, description="Perform memory cleanup"),
    force_cuda_clear: bool = Query(False, description="Force CUDA cache clearing"),
    include_alerts: bool = Query(True, description="Include memory alerts")
):
    """Memory management endpoint for monitoring and cleanup"""
    global REQUEST_COUNTER
    
    memory_info = get_memory_info()
    
    result = {
        "memory_info": memory_info,
        "request_counter": REQUEST_COUNTER,
        "cleanup_performed": False,
        "cuda_cache_cleared": False,
        "mps_cache_cleared": False
    }
    
    # Add memory alerts if requested
    if include_alerts:
        result["alerts"] = get_memory_alerts(memory_info)
    
    if cleanup:
        try:
            # Perform cleanup
            collected_objects = cleanup_memory(force_cuda_clear)
            result["cleanup_performed"] = True
            result["collected_objects"] = collected_objects
            
            # Clear CUDA cache if requested or if using GPU
            if torch.cuda.is_available() and force_cuda_clear:
                result["cuda_cache_cleared"] = True
                
            # Clear MPS cache if requested or if using Apple Silicon GPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and force_cuda_clear:
                result["mps_cache_cleared"] = True
            
            # Get updated memory info after cleanup
            result["memory_info_after_cleanup"] = get_memory_info()
            
            print(f"🧹 Manual cleanup performed: {collected_objects} objects collected")
            
        except Exception as e:
            result["cleanup_error"] = str(e)
    
    return result


@router.post(
    "/memory/reset",
    summary="Reset memory tracking",
    description="Reset request counter and clear all caches (use with caution)"
)
async def reset_memory_tracking(confirm: bool = Query(False, description="Confirm the reset operation")):
    """Reset memory tracking and perform aggressive cleanup"""
    global REQUEST_COUNTER
    
    if not confirm:
        return {
            "message": "Memory reset requires confirmation. Set confirm=true to proceed.",
            "warning": "This will reset request counter and clear all caches."
        }
    
    try:
        # Reset counter
        old_counter = REQUEST_COUNTER
        REQUEST_COUNTER = 0
        
        # Aggressive cleanup
        collected = cleanup_memory(force_cuda_clear=True)
        
        if torch.cuda.is_available():
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        
        memory_info = get_memory_info()
        
        print(f"🔄 Memory tracking reset: counter {old_counter} → 0, collected {collected} objects")
        
        return {
            "status": "success",
            "previous_request_counter": old_counter,
            "collected_objects": collected,
            "memory_info": memory_info,
            "cuda_stats_reset": torch.cuda.is_available()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "memory_info": get_memory_info()
        }


@router.get(
    "/memory/config",
    summary="Get memory management configuration",
    description="Get current memory management settings and thresholds"
)
async def get_memory_config():
    """Get memory management configuration"""
    return {
        "config": {
            "memory_cleanup_interval": Config.MEMORY_CLEANUP_INTERVAL,
            "cuda_cache_clear_interval": Config.CUDA_CACHE_CLEAR_INTERVAL,
            "enable_memory_monitoring": Config.ENABLE_MEMORY_MONITORING,
            "max_chunk_length": Config.MAX_CHUNK_LENGTH,
            "max_total_length": Config.MAX_TOTAL_LENGTH
        },
        "alert_thresholds": MEMORY_ALERT_THRESHOLDS,
        "device_info": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
    }


@router.post(
    "/memory/config",
    summary="Update memory alert thresholds",
    description="Update memory alert thresholds (temporary, not persisted)"
)
async def update_memory_config(
    cpu_memory_percent: Optional[float] = Query(None, description="CPU memory alert threshold (percentage)", ge=50, le=95),
    gpu_memory_mb: Optional[float] = Query(None, description="GPU memory alert threshold (MB)", ge=1024, le=32768)
):
    """Update memory alert thresholds"""
    global MEMORY_ALERT_THRESHOLDS
    
    old_thresholds = MEMORY_ALERT_THRESHOLDS.copy()
    
    if cpu_memory_percent is not None:
        MEMORY_ALERT_THRESHOLDS["cpu_memory_percent"] = cpu_memory_percent
    
    if gpu_memory_mb is not None:
        MEMORY_ALERT_THRESHOLDS["gpu_memory_mb"] = gpu_memory_mb
    
    return {
        "status": "success",
        "message": "Memory alert thresholds updated",
        "old_thresholds": old_thresholds,
        "new_thresholds": MEMORY_ALERT_THRESHOLDS,
        "note": "These changes are temporary and will reset on server restart"
    }


@router.get(
    "/memory/recommendations",
    summary="Get memory optimization recommendations",
    description="Get personalized recommendations for memory optimization"
)
async def get_memory_recommendations():
    """Get memory optimization recommendations based on current usage"""
    memory_info = get_memory_info()
    recommendations = []
    
    # Analyze current memory usage
    cpu_percent = memory_info.get("cpu_memory_percent", 0)
    gpu_memory = memory_info.get("gpu_memory_allocated_mb", 0)
    
    # CPU memory recommendations
    if cpu_percent > 80:
        recommendations.append({
            "type": "cpu_memory",
            "priority": "high",
            "title": "High CPU Memory Usage",
            "description": f"CPU memory usage is at {cpu_percent:.1f}%",
            "actions": [
                "Consider reducing MAX_CHUNK_LENGTH in configuration",
                "Increase MEMORY_CLEANUP_INTERVAL frequency",
                "Monitor for memory leaks in long-running processes"
            ]
        })
    elif cpu_percent > 60:
        recommendations.append({
            "type": "cpu_memory", 
            "priority": "medium",
            "title": "Moderate CPU Memory Usage",
            "description": f"CPU memory usage is at {cpu_percent:.1f}%",
            "actions": [
                "Monitor memory trends during peak usage",
                "Consider manual cleanup during heavy loads"
            ]
        })
    
    # GPU memory recommendations
    if torch.cuda.is_available() and gpu_memory > 6000:
        recommendations.append({
            "type": "gpu_memory",
            "priority": "high", 
            "title": "High GPU Memory Usage",
            "description": f"GPU memory usage is at {gpu_memory:.1f}MB",
            "actions": [
                "Enable more frequent CUDA cache clearing",
                "Reduce batch sizes or chunk lengths",
                "Consider model precision optimizations"
            ]
        })
    
    # Request counter recommendations
    if REQUEST_COUNTER > 50:
        recommendations.append({
            "type": "maintenance",
            "priority": "medium",
            "title": "High Request Count",
            "description": f"Processed {REQUEST_COUNTER} requests since last reset",
            "actions": [
                "Consider periodic memory resets",
                "Monitor for gradual memory increases",
                "Schedule maintenance windows for resets"
            ]
        })
    
    # Configuration recommendations
    if Config.MEMORY_CLEANUP_INTERVAL > 5:
        recommendations.append({
            "type": "configuration",
            "priority": "low",
            "title": "Cleanup Interval Optimization",
            "description": "Memory cleanup interval could be more frequent",
            "actions": [
                f"Consider reducing MEMORY_CLEANUP_INTERVAL from {Config.MEMORY_CLEANUP_INTERVAL} to 3-5",
                "Monitor impact on performance vs memory usage"
            ]
        })
    
    return {
        "memory_info": memory_info,
        "request_counter": REQUEST_COUNTER,
        "recommendations": recommendations,
        "recommendation_count": len(recommendations),
        "has_critical_issues": any(r["priority"] == "high" for r in recommendations)
    }


# Export the base router for the main app to use
__all__ = ["base_router"] 