"""Cache manager for expensive pipeline operations.

Uses SHA-256 hashes of input files to store and retrieve parsed transcripts,
slide parses, and notes parses, avoiding redundant processing on reruns.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import Config


class CacheManager:
    """Manages disk-based caching keyed by input content hashes."""

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        """Compute SHA-256 hash of a single file."""
        hasher = hashlib.sha256()
        path = Path(file_path)
        if not path.is_file():
            return ""
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_multi_file_hash(file_paths: List[str | Path]) -> str:
        """Compute combined SHA-256 hash for multiple files."""
        hasher = hashlib.sha256()
        for fp in sorted(str(p) for p in file_paths):
            p = Path(fp)
            if p.is_file():
                hasher.update(p.name.encode("utf-8"))
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def get(cls, cache_key: str, namespace: str) -> Optional[Any]:
        """Retrieve cached JSON data if it exists.

        Args:
            cache_key: Unique hash or key
            namespace: Stage folder (e.g., 'transcribe', 'slides', 'notes')
        """
        cache_dir = Config.CACHE_DIR / namespace
        cache_file = cache_dir / f"{cache_key}.json"
        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @classmethod
    def set(cls, cache_key: str, namespace: str, data: Any) -> None:
        """Save data to disk cache as JSON.

        Args:
            cache_key: Unique hash or key
            namespace: Stage folder (e.g., 'transcribe', 'slides', 'notes')
            data: JSON-serializable structure
        """
        cache_dir = Config.CACHE_DIR / namespace
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
