"""
Voice library management for storing and retrieving user-uploaded voices
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from app.config import Config

# Supported audio formats for voice uploads
SUPPORTED_VOICE_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}

class VoiceLibrary:
    """Manages a library of voice samples for TTS generation"""
    
    def __init__(self, library_dir: str = None):
        self.library_dir = Path(library_dir or Config.VOICE_LIBRARY_DIR)
        self.metadata_file = self.library_dir / "voices.json"
        self.config_file = self.library_dir / "config.json"
        self._ensure_library_dir()
        self._metadata = self._load_metadata()
        self._config = self._load_config()
    
    def _ensure_library_dir(self):
        """Ensure the voice library directory exists"""
        self.library_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """Load voice metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {"voices": {}, "version": "2.0"}
    
    def _save_metadata(self):
        """Save voice metadata to JSON file"""
        try:
            # Write to temp file first for atomic operation
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)
            # Atomic rename
            temp_file.replace(self.metadata_file)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save voice metadata: {e}")
            raise RuntimeError(f"Failed to save voice library metadata: {e}")
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {
            "default_voice": None,
            "default_voice_path": None,
            "version": "1.0",
            "last_updated": None
        }
    
    def _save_config(self):
        """Save configuration to JSON file"""
        try:
            self._config["last_updated"] = datetime.now().isoformat()
            # Write to temp file first for atomic operation
            temp_file = self.config_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            # Atomic rename
            temp_file.replace(self.config_file)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save voice library config: {e}")
            raise RuntimeError(f"Failed to save voice library configuration: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for the voice file for deduplication"""
        if not file_path.exists():
            raise FileNotFoundError(f"Voice file not found: {file_path}")

        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except (IOError, OSError) as e:
            raise RuntimeError(f"Failed to hash voice file {file_path}: {e}")
    
    def add_voice(self, voice_name: str, file_content: bytes, original_filename: str, language: str = "en") -> Dict:
        """
        Add a voice to the library
        
        Args:
            voice_name: The name to store the voice as (will be used in API calls)
            file_content: The binary content of the voice file
            original_filename: The original filename (used to determine extension)
            language: Language code for this voice (default: "en")
        
        Returns:
            Dict with voice metadata
        
        Raises:
            ValueError: If voice name is invalid or file format unsupported
            FileExistsError: If voice name already exists
        """
        # Validate voice name
        if not voice_name or not voice_name.strip():
            raise ValueError("Voice name cannot be empty")
        
        voice_name = voice_name.strip()
        
        # Validate language code
        if not language or not language.strip():
            raise ValueError("Language code cannot be empty")
        
        language = language.strip().lower()
        
        # Check for invalid characters in voice name
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in voice_name for char in invalid_chars):
            raise ValueError(f"Voice name contains invalid characters: {invalid_chars}")
        
        # Get file extension
        file_ext = Path(original_filename).suffix.lower()
        if file_ext not in SUPPORTED_VOICE_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: {', '.join(SUPPORTED_VOICE_FORMATS)}")
        
        # Check if voice name already exists (including aliases)
        if voice_name in self._metadata["voices"]:
            raise FileExistsError(f"Voice '{voice_name}' already exists")
        
        if self._get_voice_by_alias(voice_name) is not None:
            raise FileExistsError(f"Name '{voice_name}' is already used as an alias")
        
        # Save the voice file with the original extension
        voice_filename = f"{voice_name}{file_ext}"
        voice_path = self.library_dir / voice_filename
        
        with open(voice_path, 'wb') as f:
            f.write(file_content)
        
        # Generate file hash for deduplication tracking
        file_hash = self._get_file_hash(voice_path)
        
        # Create metadata entry
        metadata = {
            "name": voice_name,
            "filename": voice_filename,
            "original_filename": original_filename,
            "file_extension": file_ext,
            "file_size": len(file_content),
            "file_hash": file_hash,
            "upload_date": datetime.now().isoformat(),
            "path": str(voice_path),
            "language": language,
            "aliases": []  # Initialize empty aliases list
        }
        
        # Save metadata
        self._metadata["voices"][voice_name] = metadata
        self._save_metadata()
        
        return metadata
    
    def get_voice_path(self, voice_name: str) -> Optional[str]:
        """
        Get the file path for a voice by name or alias
        
        Args:
            voice_name: The name or alias of the voice
            
        Returns:
            Path to the voice file, or None if not found
        """
        # First try direct name lookup
        if voice_name in self._metadata["voices"]:
            metadata = self._metadata["voices"][voice_name]
            voice_path = Path(metadata["path"])
            
            if not voice_path.exists():
                # File is missing, remove from metadata
                del self._metadata["voices"][voice_name]
                self._save_metadata()
                return None
            
            return str(voice_path)
        
        # Then try alias lookup
        actual_name = self._get_voice_by_alias(voice_name)
        if actual_name:
            return self.get_voice_path(actual_name)
        
        return None
    
    def list_voices(self) -> List[Dict]:
        """
        List all voices in the library
        
        Returns:
            List of voice metadata dictionaries
        """
        voices = []
        voices_to_remove = []
        
        for voice_name, metadata in self._metadata["voices"].items():
            voice_path = Path(metadata["path"])
            if voice_path.exists():
                # Ensure aliases and language fields exist for backward compatibility
                voice_data = {
                    **metadata,
                    "exists": True,
                    "aliases": metadata.get("aliases", []),
                    "language": metadata.get("language", "en")  # Default to English for old voices
                }
                voices.append(voice_data)
            else:
                # Mark for removal if file is missing
                voices_to_remove.append(voice_name)
        
        # Clean up missing files from metadata
        for voice_name in voices_to_remove:
            del self._metadata["voices"][voice_name]
        
        if voices_to_remove:
            self._save_metadata()
        
        # Sort by upload date (newest first)
        voices.sort(key=lambda x: x["upload_date"], reverse=True)
        return voices
    
    def delete_voice(self, voice_name: str) -> bool:
        """
        Delete a voice from the library
        
        Args:
            voice_name: The name of the voice to delete
            
        Returns:
            True if voice was deleted, False if not found
        """
        if voice_name not in self._metadata["voices"]:
            return False
        
        metadata = self._metadata["voices"][voice_name]
        voice_path = Path(metadata["path"])
        
        # Remove file if it exists
        if voice_path.exists():
            try:
                voice_path.unlink()
            except OSError as e:
                # Don't delete from metadata if file couldn't be removed
                logger.error(f"Failed to delete voice file {voice_path}: {e}")
                raise RuntimeError(f"Voice file is in use or cannot be deleted: {e}")

        # Remove from metadata
        del self._metadata["voices"][voice_name]
        self._save_metadata()

        return True
    
    def rename_voice(self, old_name: str, new_name: str) -> bool:
        """
        Rename a voice
        
        Args:
            old_name: Current voice name
            new_name: New voice name
            
        Returns:
            True if renamed successfully, False if old voice not found
            
        Raises:
            ValueError: If new name is invalid
            FileExistsError: If new name already exists
        """
        if old_name not in self._metadata["voices"]:
            return False
        
        # Validate new name
        if not new_name or not new_name.strip():
            raise ValueError("Voice name cannot be empty")
        
        new_name = new_name.strip()
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in new_name for char in invalid_chars):
            raise ValueError(f"Voice name contains invalid characters: {invalid_chars}")
        
        # Check if new name already exists
        if new_name in self._metadata["voices"]:
            raise FileExistsError(f"Voice '{new_name}' already exists")
        
        # Get existing metadata
        metadata = self._metadata["voices"][old_name].copy()
        old_path = Path(metadata["path"])
        
        # Create new filename and path
        file_ext = metadata["file_extension"]
        new_filename = f"{new_name}{file_ext}"
        new_path = self.library_dir / new_filename
        
        # Rename the file if it exists
        if old_path.exists():
            try:
                old_path.rename(new_path)
            except OSError as e:
                raise ValueError(f"Failed to rename voice file: {e}")
        
        # Update metadata
        metadata["name"] = new_name
        metadata["filename"] = new_filename
        metadata["path"] = str(new_path)
        
        # Save under new name and remove old
        self._metadata["voices"][new_name] = metadata
        del self._metadata["voices"][old_name]
        self._save_metadata()
        
        return True
    
    def get_voice_info(self, voice_name: str) -> Optional[Dict]:
        """
        Get detailed information about a voice by name or alias
        
        Args:
            voice_name: The name or alias of the voice
            
        Returns:
            Voice metadata dictionary, or None if not found
        """
        # Resolve alias to actual name
        actual_name = self.resolve_voice_name(voice_name)
        if actual_name is None:
            return None
        
        metadata = self._metadata["voices"][actual_name]
        voice_path = Path(metadata["path"])
        
        if not voice_path.exists():
            # File is missing, remove from metadata
            del self._metadata["voices"][actual_name]
            self._save_metadata()
            return None
        
        return {
            **metadata,
            "exists": True,
            "aliases": metadata.get("aliases", [])
        }
    
    def cleanup_missing_files(self) -> List[str]:
        """
        Remove metadata entries for missing voice files
        
        Returns:
            List of removed voice names
        """
        removed_voices = []
        
        for voice_name, metadata in list(self._metadata["voices"].items()):
            voice_path = Path(metadata["path"])
            if not voice_path.exists():
                del self._metadata["voices"][voice_name]
                removed_voices.append(voice_name)
        
        if removed_voices:
            self._save_metadata()
        
        return removed_voices
    
    def set_default_voice(self, voice_name: str) -> bool:
        """
        Set a voice from the library as the default voice
        
        Args:
            voice_name: The name of the voice to set as default
            
        Returns:
            True if successful, False if voice not found
        """
        voice_path = self.get_voice_path(voice_name)
        if voice_path is None:
            return False
        
        self._config["default_voice"] = voice_name
        self._config["default_voice_path"] = voice_path
        self._save_config()
        
        # Also update the runtime configuration
        Config.VOICE_SAMPLE_PATH = voice_path
        
        return True
    
    def clear_default_voice(self):
        """
        Clear the default voice setting (revert to system default)
        """
        original_path = os.getenv('VOICE_SAMPLE_PATH', './voice-sample.mp3')
        
        self._config["default_voice"] = None
        self._config["default_voice_path"] = None
        self._save_config()
        
        # Also update the runtime configuration
        Config.VOICE_SAMPLE_PATH = original_path
    
    def get_default_voice(self) -> Optional[str]:
        """
        Get the current default voice name
        
        Returns:
            Default voice name or None if using system default
        """
        return self._config.get("default_voice")
    
    def get_default_voice_path(self) -> Optional[str]:
        """
        Get the current default voice path
        
        Returns:
            Default voice path or None if using system default
        """
        # Check if we have a persistent default voice set
        if self._config.get("default_voice_path") and Path(self._config["default_voice_path"]).exists():
            return self._config["default_voice_path"]
        return None
    
    def initialize_default_voice(self):
        """
        Initialize the default voice from persistent configuration on startup
        """
        # If we have a persistent default voice setting, apply it
        persistent_path = self.get_default_voice_path()
        if persistent_path:
            Config.VOICE_SAMPLE_PATH = persistent_path
    
    def _get_voice_by_alias(self, alias: str) -> Optional[str]:
        """
        Find the actual voice name by alias
        
        Args:
            alias: The alias to search for
            
        Returns:
            The actual voice name, or None if alias not found
        """
        for voice_name, metadata in self._metadata["voices"].items():
            aliases = metadata.get("aliases", [])
            if alias in aliases:
                return voice_name
        return None
    
    def add_alias(self, voice_name: str, alias: str) -> bool:
        """
        Add an alias to a voice
        
        Args:
            voice_name: The actual voice name
            alias: The alias to add
            
        Returns:
            True if alias was added, False if voice not found
            
        Raises:
            ValueError: If alias is invalid
            FileExistsError: If alias already exists
        """
        if voice_name not in self._metadata["voices"]:
            return False
        
        # Validate alias
        if not alias or not alias.strip():
            raise ValueError("Alias cannot be empty")
        
        alias = alias.strip()
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in alias for char in invalid_chars):
            raise ValueError(f"Alias contains invalid characters: {invalid_chars}")
        
        # Check if alias already exists as a voice name
        if alias in self._metadata["voices"]:
            raise FileExistsError(f"Alias '{alias}' conflicts with existing voice name")
        
        # Check if alias already exists as another alias
        existing_voice = self._get_voice_by_alias(alias)
        if existing_voice is not None:
            if existing_voice == voice_name:
                # Alias already exists for this voice
                return True
            else:
                raise FileExistsError(f"Alias '{alias}' already exists for voice '{existing_voice}'")
        
        # Add the alias
        aliases = self._metadata["voices"][voice_name].get("aliases", [])
        if alias not in aliases:
            aliases.append(alias)
            self._metadata["voices"][voice_name]["aliases"] = aliases
            self._save_metadata()
        
        return True
    
    def remove_alias(self, voice_name: str, alias: str) -> bool:
        """
        Remove an alias from a voice
        
        Args:
            voice_name: The actual voice name
            alias: The alias to remove
            
        Returns:
            True if alias was removed, False if voice or alias not found
        """
        if voice_name not in self._metadata["voices"]:
            return False
        
        aliases = self._metadata["voices"][voice_name].get("aliases", [])
        if alias in aliases:
            aliases.remove(alias)
            self._metadata["voices"][voice_name]["aliases"] = aliases
            self._save_metadata()
            return True
        
        return False
    
    def list_aliases(self, voice_name: str) -> List[str]:
        """
        Get all aliases for a voice
        
        Args:
            voice_name: The actual voice name
            
        Returns:
            List of aliases
        """
        if voice_name not in self._metadata["voices"]:
            return []
        
        return self._metadata["voices"][voice_name].get("aliases", [])
    
    def get_all_voice_names(self) -> List[str]:
        """
        Get all voice names and aliases
        
        Returns:
            List of all available voice names (actual names + aliases)
        """
        names = list(self._metadata["voices"].keys())
        
        # Add all aliases
        for metadata in self._metadata["voices"].values():
            aliases = metadata.get("aliases", [])
            names.extend(aliases)
        
        return names
    
    def resolve_voice_name(self, name_or_alias: str) -> Optional[str]:
        """
        Resolve a name or alias to the actual voice name
        
        Args:
            name_or_alias: Voice name or alias
            
        Returns:
            The actual voice name, or None if not found
        """
        # First check if it's an actual voice name
        if name_or_alias in self._metadata["voices"]:
            return name_or_alias
        
        # Then check if it's an alias
        return self._get_voice_by_alias(name_or_alias)
    
    def get_voice_language(self, voice_name: str) -> Optional[str]:
        """
        Get the language code for a voice by name or alias
        
        Args:
            voice_name: The name or alias of the voice
            
        Returns:
            Language code, or None if voice not found
        """
        # Resolve alias to actual name
        actual_name = self.resolve_voice_name(voice_name)
        if actual_name is None:
            return None
        
        metadata = self._metadata["voices"][actual_name]
        return metadata.get("language", "en")  # Default to English for backward compatibility


# Global voice library instance
_voice_library = None

def get_voice_library() -> VoiceLibrary:
    """Get the global voice library instance"""
    global _voice_library
    if _voice_library is None:
        _voice_library = VoiceLibrary()
        # Initialize the default voice from persistent configuration
        _voice_library.initialize_default_voice()
    return _voice_library 