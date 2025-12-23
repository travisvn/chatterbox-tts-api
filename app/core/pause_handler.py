"""Utility classes for punctuation-based pause handling."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Representation of a processed text segment."""

    text: str
    pause_after_ms: int
    original_separator: Optional[str] = None


class PauseHandler:
    """Split text around punctuation and expose pause metadata."""

    DEFAULT_PAUSES: Dict[str, int] = {
        r"\.\.\.": Config.ELLIPSIS_PAUSE_MS,
        r"—": Config.EM_DASH_PAUSE_MS,
        r"–": Config.EN_DASH_PAUSE_MS,
        r"\.": Config.PERIOD_PAUSE_MS,
        r"\n\n": Config.PARAGRAPH_PAUSE_MS,
        r"\n": Config.LINE_BREAK_PAUSE_MS,
    }

    def __init__(
        self,
        enable_pauses: bool = True,
        custom_pauses: Optional[Dict[str, int]] = None,
        min_pause_ms: int = Config.MIN_PAUSE_MS,
        max_pause_ms: int = Config.MAX_PAUSE_MS,
    ) -> None:
        self.enable_pauses = enable_pauses
        self.min_pause_ms = min_pause_ms
        self.max_pause_ms = max_pause_ms

        self.pause_patterns: Dict[str, int] = {}
        for pattern, duration in self.DEFAULT_PAUSES.items():
            self.pause_patterns[self._normalize_pattern(pattern)] = int(duration)

        if custom_pauses:
            for raw_pattern, duration in custom_pauses.items():
                try:
                    normalized = self._normalize_pattern(raw_pattern)
                    self.pause_patterns[normalized] = int(duration)
                except (TypeError, ValueError) as exc:
                    logger.debug("Ignoring invalid custom pause %r: %s", raw_pattern, exc)

        logger.debug("PauseHandler initialised with %d patterns", len(self.pause_patterns))

    def process(self, text: str) -> List[TextChunk]:
        """Split text and annotate pauses."""

        cleaned = text.strip()
        if not self.enable_pauses or not cleaned:
            return [TextChunk(text=cleaned, pause_after_ms=0)] if cleaned else []

        matches: List[Dict[str, object]] = []
        for pattern, duration in self.pause_patterns.items():
            compiled = re.compile(pattern)
            for match in compiled.finditer(text):
                matches.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "separator": match.group(0),
                        "pause_ms": int(duration),
                    }
                )

        matches.sort(key=lambda m: (int(m["start"]), -(int(m["end"]) - int(m["start"]))))

        chunks: List[TextChunk] = []
        position = 0
        for match in matches:
            start = int(match["start"])
            end = int(match["end"])
            if start < position:
                continue

            chunk_text = text[position:start].strip()
            if chunk_text:
                pause_ms = self._clamp_pause(int(match["pause_ms"]))
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        pause_after_ms=pause_ms,
                        original_separator=str(match["separator"]),
                    )
                )

            position = end

        remaining = text[position:].strip()
        if remaining:
            chunks.append(TextChunk(text=remaining, pause_after_ms=0, original_separator=None))

        logger.debug("Split text into %d pause-aware chunks", len(chunks))
        return chunks

    def estimate_total_pause_time(self, text: str) -> int:
        """Estimate cumulative pause duration for ``text``."""

        return sum(chunk.pause_after_ms for chunk in self.process(text))

    def get_pause_summary(self, text: str) -> Dict[str, object]:
        """Return statistics about pauses for ``text``."""

        chunks = self.process(text)
        pause_types: Dict[str, int] = {}
        for chunk in chunks:
            if chunk.pause_after_ms > 0:
                separator = chunk.original_separator or "other"
                pause_types[separator] = pause_types.get(separator, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_pause_ms": sum(chunk.pause_after_ms for chunk in chunks),
            "pause_types": pause_types,
            "chunks_with_pauses": sum(1 for chunk in chunks if chunk.pause_after_ms > 0),
        }

    def _clamp_pause(self, pause_ms: int) -> int:
        return max(self.min_pause_ms, min(pause_ms, self.max_pause_ms))

    @staticmethod
    def _normalize_pattern(pattern: str) -> str:
        if pattern is None:
            raise ValueError("Pause pattern cannot be None")
        if "\\" in pattern:
            return pattern
        return re.escape(pattern)


def split_text_with_pauses(
    text: str,
    enable_pauses: bool = True,
    custom_pauses: Optional[Dict[str, int]] = None,
    min_pause_ms: int = Config.MIN_PAUSE_MS,
    max_pause_ms: int = Config.MAX_PAUSE_MS,
) -> List[TextChunk]:
    """Convenience wrapper around :class:`PauseHandler`."""

    pause_mapping: Dict[str, int] = {
        r"\.\.\.": Config.ELLIPSIS_PAUSE_MS,
        r"—": Config.EM_DASH_PAUSE_MS,
        r"–": Config.EN_DASH_PAUSE_MS,
        r"\.": Config.PERIOD_PAUSE_MS,
        r"\n\n": Config.PARAGRAPH_PAUSE_MS,
        r"\n": Config.LINE_BREAK_PAUSE_MS,
    }

    if custom_pauses:
        pause_mapping.update(custom_pauses)

    handler = PauseHandler(
        enable_pauses=enable_pauses,
        custom_pauses=pause_mapping,
        min_pause_ms=min_pause_ms,
        max_pause_ms=max_pause_ms,
    )
    return handler.process(text)
