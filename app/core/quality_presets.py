"""Quality presets for TTS generation configurable via environment variables."""

from app.config import Config


def get_quality_preset(preset_name: str | None = None) -> dict:
    """Return the quality preset configuration.

    Args:
        preset_name: Name of the preset to retrieve. Defaults to the configured
            LONG_TEXT_QUALITY_PRESET when ``None`` is provided.

    Returns:
        A dictionary with ``chunk_size``, ``cfg_weight`` and ``temperature``
        settings. Falls back to the ``balanced`` preset when the requested name
        is not defined.
    """

    if preset_name is None:
        preset_name = Config.LONG_TEXT_QUALITY_PRESET

    return Config.QUALITY_PRESETS.get(preset_name, Config.QUALITY_PRESETS["balanced"])


def get_chunk_size_for_preset(preset_name: str | None = None) -> int:
    """Return the chunk size associated with a preset.

    Args:
        preset_name: Optional preset name. When omitted the configured default
            preset is used.

    Returns:
        Chunk size for the preset, falling back to ``Config.LONG_TEXT_CHUNK_SIZE``
        if the preset is not found or does not define a chunk size.
    """

    preset = get_quality_preset(preset_name) if preset_name else None

    if preset is not None:
        chunk_size = preset.get("chunk_size")
        if isinstance(chunk_size, int) and chunk_size > 0:
            return chunk_size

    return Config.LONG_TEXT_CHUNK_SIZE
