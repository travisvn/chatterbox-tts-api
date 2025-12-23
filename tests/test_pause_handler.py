import pytest

from app.core.pause_handler import PauseHandler, split_text_with_pauses, TextChunk


class TestPauseHandler:
    def test_basic_ellipsis_pause(self):
        handler = PauseHandler()
        chunks = handler.process("Hello... world")

        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[0].pause_after_ms == 600
        assert chunks[1].text == "world"
        assert chunks[1].pause_after_ms == 0

    def test_em_dash_pause(self):
        handler = PauseHandler()
        chunks = handler.process("Hello—world")

        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[0].pause_after_ms == 400
        assert chunks[1].text == "world"

    def test_en_dash_pause(self):
        handler = PauseHandler()
        chunks = handler.process("Numbers 1–2")

        assert len(chunks) == 2
        assert chunks[0].text == "Numbers 1"
        assert chunks[0].pause_after_ms == 350
        assert chunks[1].text == "2"

    def test_multiple_pauses(self):
        handler = PauseHandler()
        text = "Hello... I was thinking—maybe tomorrow?"
        chunks = handler.process(text)

        assert len(chunks) == 3
        assert chunks[0].text == "Hello"
        assert chunks[0].pause_after_ms == 600
        assert chunks[1].text == "I was thinking"
        assert chunks[1].pause_after_ms == 400
        assert chunks[2].text == "maybe tomorrow?"
        assert chunks[2].pause_after_ms == 0

    def test_no_pauses_when_disabled(self):
        handler = PauseHandler(enable_pauses=False)
        chunks = handler.process("Hello... world—test")

        assert len(chunks) == 1
        assert chunks[0].text == "Hello... world—test"
        assert chunks[0].pause_after_ms == 0

    def test_line_break_pause(self):
        handler = PauseHandler()
        chunks = handler.process("Line one\nLine two")

        assert len(chunks) == 2
        assert chunks[0].text == "Line one"
        assert chunks[0].pause_after_ms == 250
        assert chunks[1].text == "Line two"

    def test_paragraph_break_pause(self):
        handler = PauseHandler()
        chunks = handler.process("Paragraph one\n\nParagraph two")

        assert len(chunks) == 2
        assert chunks[0].text == "Paragraph one"
        assert chunks[0].pause_after_ms == 500
        assert chunks[1].text == "Paragraph two"

    def test_custom_pause_durations(self):
        custom = {r"\.\.\.": 1000}
        handler = PauseHandler(custom_pauses=custom)
        chunks = handler.process("Hello... world")

        assert chunks[0].pause_after_ms == 1000

    def test_pause_clamping(self):
        handler = PauseHandler(min_pause_ms=200, max_pause_ms=500)
        custom = {r"\.\.\.": 2000}
        handler.pause_patterns.update(custom)

        chunks = handler.process("Hello... world")
        assert chunks[0].pause_after_ms == 500

    def test_empty_text(self):
        handler = PauseHandler()
        chunks = handler.process("")

        assert len(chunks) == 0

    def test_no_pause_punctuation(self):
        handler = PauseHandler()
        chunks = handler.process("Hello world, how are you?")

        assert len(chunks) == 1
        assert chunks[0].text == "Hello world, how are you?"
        assert chunks[0].pause_after_ms == 0

    def test_estimate_total_pause_time(self):
        handler = PauseHandler()
        text = "Hello... world—test"
        total_pause = handler.estimate_total_pause_time(text)

        assert total_pause == 1000

    def test_pause_summary(self):
        handler = PauseHandler()
        text = "Hello... world—test... again"
        summary = handler.get_pause_summary(text)

        assert summary['total_chunks'] == 4
        assert summary['chunks_with_pauses'] == 3
        assert summary['pause_types']['...'] == 2
        assert summary['pause_types']['—'] == 1

    def test_convenience_function(self):
        chunks = split_text_with_pauses("Hello... world")

        assert len(chunks) == 2
        assert isinstance(chunks[0], TextChunk)
        assert chunks[0].pause_after_ms == 600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
