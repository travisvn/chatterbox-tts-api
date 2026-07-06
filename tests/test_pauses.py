import unittest
import re
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock dependencies for unit testing without environment
sys.modules['torch'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['pydantic'] = MagicMock()
sys.modules['chatterbox'] = MagicMock()
sys.modules['chatterbox.tts'] = MagicMock()
sys.modules['chatterbox.mtl_tts'] = MagicMock()
sys.modules['app.models.long_text'] = MagicMock()
sys.modules['app.core.mtl'] = MagicMock()
sys.modules['app.core.status'] = MagicMock()
sys.modules['app.core.voice_library'] = MagicMock()
sys.modules['app.core.memory'] = MagicMock()

# Import the module to test
import app.core.text_processing as tp

class TestPauses(unittest.TestCase):
    def test_split_text_with_pauses(self):
        """Test that [pause:ms] tags are correctly converted to markers"""
        text = "Hello world. [pause:500ms] This is a test."
        chunks = tp.split_text_into_chunks(text)
        
        self.assertIn("__PAUSE_MS:500__", chunks)
        self.assertEqual(chunks[0], "Hello world.")
        self.assertEqual(chunks[2], "This is a test.")

    def test_split_text_with_default_pause(self):
        """Test that standard [pause] tags default to 200ms"""
        text = "Break [pause] here."
        chunks = tp.split_text_into_chunks(text)
        
        self.assertIn("__PAUSE_MS:200__", chunks)

    def test_split_text_with_newlines(self):
        """Test that newlines cause text to split into chunks (preserving pauses)"""
        text = "Line 1\nLine 2"
        chunks = tp.split_text_into_chunks(text)
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "Line 1")
        self.assertEqual(chunks[1], "Line 2")

    def test_complex_pause_tags(self):
        """Test various formats of the pause tag"""
        text = "[pause:100] [pause:100ms] [pause:1000ms]"
        chunks = tp.split_text_into_chunks(text)
        
        self.assertEqual(chunks[0], "__PAUSE_MS:100__")
        self.assertEqual(chunks[1], "__PAUSE_MS:100__")
        self.assertEqual(chunks[2], "__PAUSE_MS:1000__")

if __name__ == '__main__':
    unittest.main()
