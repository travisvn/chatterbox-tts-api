"""Unit tests for configurable long text limits"""

from pathlib import Path
from typing import Iterator
import sys

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config
from app.core.text_processing import validate_long_text_input
from app.models.long_text import LongTextRequest


@pytest.fixture()
def reset_long_text_limits() -> Iterator[None]:
    """Restore Config long text limits after each test"""
    original_min = Config.LONG_TEXT_MIN_LENGTH
    original_max = Config.LONG_TEXT_MAX_LENGTH
    yield
    Config.LONG_TEXT_MIN_LENGTH = original_min
    Config.LONG_TEXT_MAX_LENGTH = original_max


def test_long_text_request_accepts_env_configured_min(monkeypatch: pytest.MonkeyPatch, reset_long_text_limits: None) -> None:
    """Long text requests should accept inputs that meet the configured minimum"""
    monkeypatch.setenv("LONG_TEXT_MIN_LENGTH", "100")
    monkeypatch.setenv("LONG_TEXT_MAX_LENGTH", "1000")

    sample_text = "x" * 150
    request = LongTextRequest(input=sample_text)

    assert request.input == sample_text


def test_long_text_request_rejects_below_min(monkeypatch: pytest.MonkeyPatch, reset_long_text_limits: None) -> None:
    """Validation should fail when text length is below the configured minimum"""
    monkeypatch.setenv("LONG_TEXT_MIN_LENGTH", "200")
    monkeypatch.setenv("LONG_TEXT_MAX_LENGTH", "1000")

    with pytest.raises(ValidationError) as exc_info:
        LongTextRequest(input="y" * 150)

    assert "200" in str(exc_info.value)


def test_validate_long_text_input_uses_configured_limits(monkeypatch: pytest.MonkeyPatch, reset_long_text_limits: None) -> None:
    """Core validation should leverage runtime configuration for min and max lengths"""
    monkeypatch.setenv("LONG_TEXT_MIN_LENGTH", "120")
    monkeypatch.setenv("LONG_TEXT_MAX_LENGTH", "400")

    is_valid, _ = validate_long_text_input("z" * 200)
    assert is_valid

    too_short_valid, too_short_message = validate_long_text_input("z" * 100)
    assert not too_short_valid
    assert "120" in too_short_message

    too_long_valid, too_long_message = validate_long_text_input("z" * 450)
    assert not too_long_valid
    assert "400" in too_long_message
