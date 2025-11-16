"""
Text processing utilities for TTS
"""

import gc
import torch
import re
from typing import List, Optional, Tuple
from app.config import Config
from app.models.long_text import LongTextChunk


def split_text_into_chunks(text: str, max_length: int = None) -> list:
    """Split text into manageable chunks for TTS processing"""
    if max_length is None:
        max_length = Config.MAX_CHUNK_LENGTH
    
    if len(text) <= max_length:
        return [text]
    
    # Try to split at sentence boundaries first
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    chunks = []
    current_chunk = ""
    
    # Split into sentences
    sentences = []
    temp_text = text
    
    while temp_text:
        best_split = len(temp_text)
        best_ending = ""
        
        for ending in sentence_endings:
            pos = temp_text.find(ending)
            if pos != -1 and pos < best_split:
                best_split = pos + len(ending)
                best_ending = ending
        
        if best_split == len(temp_text):
            # No sentence ending found, take the rest
            sentences.append(temp_text)
            break
        else:
            sentences.append(temp_text[:best_split])
            temp_text = temp_text[best_split:]
    
    # Group sentences into chunks
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If single sentence is too long, split it further
            if len(sentence) > max_length:
                # Split at commas, semicolons, etc.
                sub_delimiters = [', ', '; ', ' - ', ' — ']
                sub_chunks = [sentence]
                
                for delimiter in sub_delimiters:
                    new_sub_chunks = []
                    for chunk in sub_chunks:
                        if len(chunk) <= max_length:
                            new_sub_chunks.append(chunk)
                        else:
                            parts = chunk.split(delimiter)
                            current_part = ""
                            for part in parts:
                                if len(current_part) + len(delimiter) + len(part) <= max_length:
                                    current_part += (delimiter if current_part else "") + part
                                else:
                                    if current_part:
                                        new_sub_chunks.append(current_part)
                                    current_part = part
                            if current_part:
                                new_sub_chunks.append(current_part)
                    sub_chunks = new_sub_chunks
                
                # Add sub-chunks
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) <= max_length:
                        chunks.append(sub_chunk.strip())
                    else:
                        # Last resort: split by words
                        words = sub_chunk.split()
                        current_word_chunk = ""
                        for word in words:
                            if len(current_word_chunk) + len(word) + 1 <= max_length:
                                current_word_chunk += (" " if current_word_chunk else "") + word
                            else:
                                if current_word_chunk:
                                    chunks.append(current_word_chunk)
                                current_word_chunk = word
                        if current_word_chunk:
                            chunks.append(current_word_chunk)
                current_chunk = ""
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks


def split_text_for_streaming(
    text: str,
    chunk_size: Optional[int] = None,
    strategy: Optional[str] = None,
    quality: Optional[str] = None
) -> List[str]:
    """
    Split text into chunks optimized for streaming with different strategies.
    
    Args:
        text: Input text to split
        chunk_size: Target chunk size (characters)
        strategy: Splitting strategy ('sentence', 'paragraph', 'fixed', 'word')
        quality: Quality preset ('fast', 'balanced', 'high')
    
    Returns:
        List of text chunks optimized for streaming
    """
    # Apply quality presets
    if quality:
        if quality == "fast":
            chunk_size = chunk_size or 100
            strategy = strategy or "word"
        elif quality == "balanced":
            chunk_size = chunk_size or 200
            strategy = strategy or "sentence"
        elif quality == "high":
            chunk_size = chunk_size or 300
            strategy = strategy or "paragraph"
    
    # Set defaults
    chunk_size = chunk_size or 200
    strategy = strategy or "sentence"
    
    # Apply strategy-specific splitting
    if strategy == "paragraph":
        return _split_by_paragraphs(text, chunk_size)
    elif strategy == "sentence":
        return _split_by_sentences(text, chunk_size)
    elif strategy == "word":
        return _split_by_words(text, chunk_size)
    elif strategy == "fixed":
        return _split_by_fixed_size(text, chunk_size)
    else:
        # Default to sentence splitting
        return _split_by_sentences(text, chunk_size)


def _split_by_paragraphs(text: str, max_length: int) -> List[str]:
    """Split text by paragraph breaks, respecting max length"""
    # Split by double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If paragraph fits with current chunk
        if len(current_chunk) + len(paragraph) + 2 <= max_length:  # +2 for paragraph break
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph is too long, split it by sentences
            if len(paragraph) > max_length:
                sentence_chunks = _split_by_sentences(paragraph, max_length)
                chunks.extend(sentence_chunks)
                current_chunk = ""
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]


def _split_by_sentences(text: str, max_length: int) -> List[str]:
    """Split text by sentence boundaries, respecting max length"""
    # Enhanced sentence splitting regex
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If sentence fits with current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_length:  # +1 for space
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If sentence is too long, split it further
            if len(sentence) > max_length:
                sub_chunks = _split_long_sentence(sentence, max_length)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]


def _split_by_words(text: str, max_length: int) -> List[str]:
    """Split text by word boundaries, respecting max length"""
    words = text.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        # If word fits with current chunk
        if len(current_chunk) + len(word) + 1 <= max_length:  # +1 for space
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If single word is too long, force it into its own chunk
            if len(word) > max_length:
                # Split very long words at character boundaries
                for i in range(0, len(word), max_length):
                    chunks.append(word[i:i + max_length])
                current_chunk = ""
            else:
                current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]


def _split_by_fixed_size(text: str, chunk_size: int) -> List[str]:
    """Split text into fixed-size chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    
    return chunks


def _split_long_sentence(sentence: str, max_length: int) -> List[str]:
    """Split a long sentence at natural break points"""
    # Try to split at commas, semicolons, etc.
    delimiters = [', ', '; ', ' - ', ' — ', ': ', ' and ', ' or ', ' but ']
    
    chunks = [sentence]
    
    for delimiter in delimiters:
        new_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                new_chunks.append(chunk)
            else:
                parts = chunk.split(delimiter)
                current_part = ""
                for part in parts:
                    if len(current_part) + len(delimiter) + len(part) <= max_length:
                        current_part += (delimiter if current_part else "") + part
                    else:
                        if current_part:
                            new_chunks.append(current_part)
                        current_part = part
                if current_part:
                    new_chunks.append(current_part)
        chunks = new_chunks
    
    # Final fallback: split by words
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            word_chunks = _split_by_words(chunk, max_length)
            final_chunks.extend(word_chunks)
    
    return [chunk.strip() for chunk in final_chunks if chunk.strip()]


def get_streaming_settings(
    streaming_chunk_size: Optional[int],
    streaming_strategy: Optional[str],
    streaming_quality: Optional[str]
) -> dict:
    """
    Get optimized streaming settings based on parameters.
    
    Returns a dictionary with optimized settings for streaming.
    """
    settings = {
        "chunk_size": streaming_chunk_size or 200,
        "strategy": streaming_strategy or "sentence",
        "quality": streaming_quality or "balanced"
    }
    
    # Apply quality presets if not explicitly overridden
    if streaming_quality and not streaming_chunk_size:
        if streaming_quality == "fast":
            settings["chunk_size"] = 100
        elif streaming_quality == "high":
            settings["chunk_size"] = 300
    
    if streaming_quality and not streaming_strategy:
        if streaming_quality == "fast":
            settings["strategy"] = "word"
        elif streaming_quality == "high":
            settings["strategy"] = "paragraph"
    
    return settings


def concatenate_audio_chunks(audio_chunks: list, sample_rate: int) -> torch.Tensor:
    """Concatenate multiple audio tensors with proper memory management"""
    if not audio_chunks:
        raise ValueError("Cannot concatenate empty list of audio chunks")

    if len(audio_chunks) == 1:
        return audio_chunks[0]

    # Add small silence between chunks (0.1 seconds)
    silence_samples = int(0.1 * sample_rate)

    # Create silence tensor on the same device as audio chunks
    device = audio_chunks[0].device if hasattr(audio_chunks[0], 'device') else 'cpu'
    silence = torch.zeros(1, silence_samples, device=device)
    
    # Use torch.no_grad() to prevent gradient tracking
    with torch.no_grad():
        concatenated = audio_chunks[0]
        
        for i, chunk in enumerate(audio_chunks[1:], 1):
            # Concatenate current result with silence and next chunk
            concatenated = torch.cat([concatenated, silence, chunk], dim=1)
            
            # Optional: cleanup intermediate tensors for very long sequences
            if i % 10 == 0:  # Every 10 chunks
                gc.collect()
    
    # Clean up silence tensor
    del silence
    
    return concatenated


def chunk_text(text: str, strategy: str = "sentence", max_length: Optional[int] = None) -> List[str]:
    """Split text according to the requested strategy."""

    if max_length is None or max_length <= 0:
        max_length = Config.LONG_TEXT_CHUNK_SIZE

    # Respect the standard TTS hard limit with a small buffer to avoid boundary errors
    effective_max = min(max_length, Config.MAX_TOTAL_LENGTH - 100)
    if effective_max <= 0:
        effective_max = max_length

    cleaned = text.strip()
    if not cleaned:
        return []

    normalized_strategy = (strategy or "sentence").lower()

    if normalized_strategy == "fixed":
        return [
            chunk.strip()
            for chunk in (cleaned[i : i + effective_max] for i in range(0, len(cleaned), effective_max))
            if chunk.strip()
        ]

    if normalized_strategy == "paragraph":
        return _chunk_by_paragraphs(cleaned, effective_max)

    if normalized_strategy == "word":
        return _chunk_by_words(cleaned, effective_max)

    # Default strategy combines hierarchical paragraph/sentence/word splitting
    return _chunk_hierarchical(cleaned, effective_max)


def _chunk_by_paragraphs(text: str, max_length: int) -> List[str]:
    """Chunk text prioritising paragraph boundaries."""

    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
    if not paragraphs:
        return _chunk_hierarchical(text, max_length)

    chunks: List[str] = []
    current: Optional[str] = None

    for paragraph in paragraphs:
        if current is None:
            current = paragraph
            continue

        candidate = f"{current}\n\n{paragraph}"
        if len(candidate) <= max_length:
            current = candidate
        else:
            chunks.extend(_chunk_hierarchical(current, max_length))
            current = paragraph

    if current:
        chunks.extend(_chunk_hierarchical(current, max_length))

    return chunks


def _chunk_by_words(text: str, max_length: int) -> List[str]:
    """Chunk text using word boundaries."""

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    current_words: List[str] = []
    current_length = 0

    for word in words:
        # Include a space when the current chunk already has words
        additional_length = len(word) if not current_words else len(word) + 1

        if current_words and current_length + additional_length > max_length:
            chunk = " ".join(current_words).strip()
            if chunk:
                chunks.append(chunk)
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += additional_length

    if current_words:
        chunk = " ".join(current_words).strip()
        if chunk:
            chunks.append(chunk)

    refined_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) > max_length:
            refined_chunks.extend(_chunk_hierarchical(chunk, max_length))
        else:
            refined_chunks.append(chunk)

    return refined_chunks


def _chunk_hierarchical(text: str, max_length: int) -> List[str]:
    """Hierarchical chunking that mirrors the legacy behaviour."""

    chunks: List[str] = []
    remaining_text = text.strip()

    while remaining_text:
        if len(remaining_text) <= max_length:
            chunks.append(remaining_text)
            break

        chunk_text, remaining = _find_best_split_point(remaining_text, max_length, 0)

        if not chunk_text:
            chunk_text = remaining_text[:max_length].strip()
            remaining = remaining_text[max_length:].strip()

        chunks.append(chunk_text)
        remaining_text = remaining

    return [chunk for chunk in chunks if chunk]


def split_text_for_long_generation(
    text: str,
    max_chunk_size: Optional[int] = None,
    strategy: Optional[str] = None,
) -> List[LongTextChunk]:
    """Split long text into structured chunks ready for generation."""

    if max_chunk_size is None or max_chunk_size <= 0:
        max_chunk_size = Config.LONG_TEXT_CHUNK_SIZE

    resolved_strategy = strategy or Config.LONG_TEXT_CHUNKING_STRATEGY

    chunk_strings = chunk_text(text, strategy=resolved_strategy, max_length=max_chunk_size)
    if not chunk_strings:
        return []

    chunks: List[LongTextChunk] = []
    for index, chunk_body in enumerate(chunk_strings):
        preview = chunk_body[:50] + ("..." if len(chunk_body) > 50 else "")
        chunks.append(
            LongTextChunk(
                index=index,
                text=chunk_body,
                text_preview=preview,
                character_count=len(chunk_body),
            )
        )

    return chunks


def _find_best_split_point(text: str, max_length: int, overlap_chars: int = 0) -> Tuple[str, str]:
    """
    Find the best point to split text while preserving semantic boundaries.

    Returns:
        Tuple of (chunk_text, remaining_text)
    """
    if len(text) <= max_length:
        return text, ""

    # Strategy 1: Split at paragraph boundaries
    split_result = _try_split_at_paragraphs(text, max_length, overlap_chars)
    if split_result:
        return split_result

    # Strategy 2: Split at sentence boundaries
    split_result = _try_split_at_sentences(text, max_length, overlap_chars)
    if split_result:
        return split_result

    # Strategy 3: Split at clause boundaries
    split_result = _try_split_at_clauses(text, max_length, overlap_chars)
    if split_result:
        return split_result

    # Strategy 4: Split at word boundaries (last resort)
    return _split_at_words(text, max_length, overlap_chars)


def _try_split_at_paragraphs(text: str, max_length: int, overlap_chars: int) -> Optional[Tuple[str, str]]:
    """Try to split at paragraph boundaries (double newlines)"""
    # Find all paragraph breaks
    paragraph_pattern = r'\n\s*\n'
    matches = list(re.finditer(paragraph_pattern, text))

    if not matches:
        return None

    # Find the best paragraph break within our limit
    best_split = None
    for match in matches:
        split_pos = match.end()
        if split_pos <= max_length:
            best_split = split_pos
        else:
            break

    if best_split and best_split > max_length * 0.5:  # Don't take chunks that are too small
        chunk_text = text[:best_split].strip()
        remaining_text = text[max(0, best_split - overlap_chars):].strip()
        return chunk_text, remaining_text

    return None


def _try_split_at_sentences(text: str, max_length: int, overlap_chars: int) -> Optional[Tuple[str, str]]:
    """Try to split at sentence boundaries"""
    # Enhanced sentence boundary detection
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n', '."', '!"', '?"', ".'", "!'", "?'"]

    best_split = None
    for ending in sentence_endings:
        pos = 0
        while pos < len(text):
            found = text.find(ending, pos)
            if found == -1:
                break

            split_pos = found + len(ending)
            if split_pos <= max_length:
                best_split = split_pos
                pos = found + 1
            else:
                break

    if best_split and best_split > max_length * 0.4:  # Don't take chunks that are too small
        chunk_text = text[:best_split].strip()
        remaining_text = text[max(0, best_split - overlap_chars):].strip()
        return chunk_text, remaining_text

    return None


def _try_split_at_clauses(text: str, max_length: int, overlap_chars: int) -> Optional[Tuple[str, str]]:
    """Try to split at clause boundaries (commas, semicolons, etc.)"""
    clause_delimiters = [', ', '; ', ': ', ' - ', ' — ', ' and ', ' or ', ' but ', ' while ', ' when ']

    best_split = None
    for delimiter in clause_delimiters:
        pos = 0
        while pos < len(text):
            found = text.find(delimiter, pos)
            if found == -1:
                break

            split_pos = found + len(delimiter)
            if split_pos <= max_length:
                best_split = split_pos
                pos = found + 1
            else:
                break

    if best_split and best_split > max_length * 0.3:  # Don't take chunks that are too small
        chunk_text = text[:best_split].strip()
        remaining_text = text[max(0, best_split - overlap_chars):].strip()
        return chunk_text, remaining_text

    return None


def _split_at_words(text: str, max_length: int, overlap_chars: int) -> Tuple[str, str]:
    """Split at word boundaries as last resort"""
    if len(text) <= max_length:
        return text, ""

    # Find the last space before our limit
    split_pos = text.rfind(' ', 0, max_length)

    if split_pos == -1:  # No space found, force split
        split_pos = max_length

    chunk_text = text[:split_pos].strip()
    remaining_text = text[max(0, split_pos - overlap_chars):].strip()

    return chunk_text, remaining_text


def estimate_processing_time(
    text_length: int,
    avg_chars_per_second: float = 25.0,
    chunk_size: Optional[int] = None,
) -> int:
    """
    Estimate processing time for long text TTS generation.

    Args:
        text_length: Total characters in text
        avg_chars_per_second: Average processing rate (characters per second)
        chunk_size: Optional chunk size override used for estimation

    Returns:
        Estimated processing time in seconds
    """
    # Base estimate + overhead for chunking and concatenation
    base_time = text_length / avg_chars_per_second

    effective_chunk_size = chunk_size or Config.LONG_TEXT_CHUNK_SIZE
    if effective_chunk_size <= 0:
        effective_chunk_size = Config.LONG_TEXT_CHUNK_SIZE

    # Add overhead: 5 seconds for setup + 2 seconds per chunk + 10 seconds for concatenation
    num_chunks = max(1, (text_length + effective_chunk_size - 1) // effective_chunk_size)
    overhead = 5 + (num_chunks * 2) + 10

    return int(base_time + overhead)


def validate_long_text_input(text: str) -> Tuple[bool, str]:
    """
    Validate text for long text TTS generation.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Input text cannot be empty"

    text_length = len(text.strip())
    min_length = Config.get_long_text_min_length()
    max_length = Config.get_long_text_max_length()

    if text_length < min_length:
        return False, (
            "Text must be at least {} characters for long-text processing (received {} characters)".format(
                min_length,
                text_length,
            )
        )

    if text_length > max_length:
        return False, f"Text is too long ({text_length} characters). Maximum allowed: {max_length}"

    # Check for excessive repetition (potential spam/abuse)
    words = text.split()
    if len(set(words)) < len(words) * 0.1:  # Less than 10% unique words
        return False, "Text appears to be excessively repetitive"

    return True, "" 