"""
Endpoint alias management utilities
"""

from typing import Dict, List, Any, Callable
from functools import wraps


# Centralized endpoint aliases configuration
# Format: "primary_endpoint": ["alias1", "alias2", ...]
ENDPOINT_ALIASES: Dict[str, List[str]] = {
    "/audio/speech": ["/v1/audio/speech", "/tts"],
    "/audio/speech/upload": ["/v1/audio/speech/upload", "/tts/upload"],
    "/audio/speech/stream": ["/v1/audio/speech/stream", "/tts/stream"],
    "/audio/speech/stream/upload": ["/v1/audio/speech/stream/upload", "/tts/stream/upload"],
    "/languages": ["/v1/languages"],
    "/voices": ["/v1/voices", "/voice-library", "/voice_library"],
    "/voices/default": ["/v1/voices/default", "/default-voice"],
    "/voices/{voice_name}": ["/v1/voices/{voice_name}"],
    "/voices/{voice_name}/download": ["/v1/voices/{voice_name}/download"],
    "/voices/{voice_name}/aliases": ["/v1/voices/{voice_name}/aliases"],
    "/voices/{voice_name}/aliases/{alias}": ["/v1/voices/{voice_name}/aliases/{alias}"],
    "/voices/all-names": ["/v1/voices/all-names"],
    "/voices/cleanup": ["/v1/voices/cleanup"],
    "/health": ["/v1/health", "/status"],
    "/ping": ["/v1/ping"],
    "/models": ["/v1/models"],
    "/config": ["/v1/config"],
    "/endpoints": ["/v1/endpoints", "/routes"],
    "/memory": ["/v1/memory"],
    "/memory/config": ["/v1/memory/config"],
    "/memory/reset": ["/v1/memory/reset"],
    "/memory/recommendations": ["/v1/memory/recommendations"],
    "/status": ["/v1/status", "/processing", "/processing/status"],
    "/status/progress": ["/v1/status/progress", "/progress"],
    "/status/history": ["/v1/status/history", "/history"],
    "/status/statistics": ["/v1/status/statistics", "/stats"],
    "/status/history/clear": ["/v1/status/history/clear"],
    "/info": ["/v1/info", "/api/info"],
    "/audio/speech/long": ["/v1/audio/speech/long"],
    "/audio/speech/long/jobs": ["/v1/audio/speech/long/jobs"],
    "/audio/speech/long/jobs/{job_id}": ["/v1/audio/speech/long/jobs/{job_id}"],
    "/audio/speech/long/{job_id}/download": ["/v1/audio/speech/long/{job_id}/download"],
    "/audio/speech/long/{job_id}/pause": ["/v1/audio/speech/long/{job_id}/pause"],
    "/audio/speech/long/{job_id}/resume": ["/v1/audio/speech/long/{job_id}/resume"],
    "/audio/speech/long/{job_id}/sse": ["/v1/audio/speech/long/{job_id}/sse"],
    "/audio/speech/long/{job_id}": ["/v1/audio/speech/long/{job_id}"],
    "/audio/speech/long-history": ["/v1/audio/speech/long-history"],
    "/audio/speech/long-history/stats": ["/v1/audio/speech/long-history/stats"],
    "/audio/speech/long/{job_id}/details": ["/v1/audio/speech/long/{job_id}/details"],
    "/audio/speech/long/{job_id}/retry": ["/v1/audio/speech/long/{job_id}/retry"],
    "/audio/speech/long/bulk": ["/v1/audio/speech/long/bulk"],
}


def alias_route(primary_path: str, alias_paths: List[str] = None, **kwargs):
    """
    Decorator to create endpoint aliases.
    
    Args:
        primary_path: The main endpoint path (included in schema)
        alias_paths: List of alias paths (excluded from schema). If None, uses ENDPOINT_ALIASES lookup.
        **kwargs: Additional FastAPI route parameters
    
    Example:
        @alias_route("/audio/speech")
        async def text_to_speech():
            pass
            
        # Creates /audio/speech (primary) and /v1/audio/speech (alias)
    """
    def decorator(router_method: Callable):
        def route_decorator(*args, **route_kwargs):
            # Merge kwargs with route_kwargs, giving precedence to route_kwargs
            final_kwargs = {**kwargs, **route_kwargs}
            
            def endpoint_decorator(func):
                # Add the primary endpoint
                router_method(primary_path, **final_kwargs)(func)
                
                # Determine alias paths
                target_aliases = alias_paths or ENDPOINT_ALIASES.get(primary_path, [])
                
                # Add each alias endpoint
                for alias_path in target_aliases:
                    # Create alias kwargs (exclude from schema)
                    alias_kwargs = final_kwargs.copy()
                    alias_kwargs['include_in_schema'] = False
                    
                    # Add the alias endpoint
                    router_method(alias_path, **alias_kwargs)(func)
                
                return func
            return endpoint_decorator
        return route_decorator
    return decorator


def add_route_aliases(router):
    """
    Add aliased versions of router methods.
    
    Returns a router with aliased post, get, put, delete, patch methods.
    """
    class AliasedRouter:
        def __init__(self, original_router):
            self._router = original_router
            
        def __getattr__(self, name):
            # Pass through all other router attributes
            return getattr(self._router, name)
        
        def _create_aliased_method(self, method_name: str, path: str, **kwargs):
            """Create a method that adds both primary and alias endpoints"""
            original_method = getattr(self._router, method_name)
            
            def decorator(func):
                # Add the primary endpoint
                original_method(path, **kwargs)(func)
                
                # Get alias paths for this primary path
                alias_paths = ENDPOINT_ALIASES.get(path, [])
                
                # Add each alias endpoint
                for alias_path in alias_paths:
                    # Create alias kwargs (exclude from schema)
                    alias_kwargs = kwargs.copy()
                    alias_kwargs['include_in_schema'] = False
                    
                    # Add the alias endpoint
                    original_method(alias_path, **alias_kwargs)(func)
                
                return func
            return decorator
        
        def post(self, path: str, **kwargs):
            return self._create_aliased_method('post', path, **kwargs)
            
        def get(self, path: str, **kwargs):
            return self._create_aliased_method('get', path, **kwargs)
            
        def put(self, path: str, **kwargs):
            return self._create_aliased_method('put', path, **kwargs)
            
        def delete(self, path: str, **kwargs):
            return self._create_aliased_method('delete', path, **kwargs)
            
        def patch(self, path: str, **kwargs):
            return self._create_aliased_method('patch', path, **kwargs)
    
    return AliasedRouter(router)


def get_all_aliases() -> Dict[str, List[str]]:
    """Return all configured endpoint aliases"""
    return ENDPOINT_ALIASES.copy()


def add_custom_alias(primary_path: str, alias_path: str) -> None:
    """Add a custom endpoint alias at runtime"""
    if primary_path not in ENDPOINT_ALIASES:
        ENDPOINT_ALIASES[primary_path] = []
    if alias_path not in ENDPOINT_ALIASES[primary_path]:
        ENDPOINT_ALIASES[primary_path].append(alias_path)


def add_multiple_aliases(primary_path: str, alias_paths: List[str]) -> None:
    """Add multiple aliases to an endpoint at runtime"""
    if primary_path not in ENDPOINT_ALIASES:
        ENDPOINT_ALIASES[primary_path] = []
    for alias_path in alias_paths:
        if alias_path not in ENDPOINT_ALIASES[primary_path]:
            ENDPOINT_ALIASES[primary_path].append(alias_path)


def remove_alias(primary_path: str, alias_path: str = None) -> None:
    """Remove a specific alias or all aliases for an endpoint"""
    if alias_path:
        # Remove specific alias
        if primary_path in ENDPOINT_ALIASES and alias_path in ENDPOINT_ALIASES[primary_path]:
            ENDPOINT_ALIASES[primary_path].remove(alias_path)
    else:
        # Remove all aliases for this endpoint
        ENDPOINT_ALIASES.pop(primary_path, None)


def get_endpoint_info() -> Dict[str, Any]:
    """Get comprehensive endpoint information including all mappings"""
    info = {
        "total_endpoints": len(ENDPOINT_ALIASES),
        "total_aliases": sum(len(aliases) for aliases in ENDPOINT_ALIASES.values()),
        "mappings": {}
    }
    
    for primary, aliases in ENDPOINT_ALIASES.items():
        info["mappings"][primary] = {
            "primary": primary,
            "aliases": aliases,
            "total_paths": 1 + len(aliases)
        }
    
    return info 