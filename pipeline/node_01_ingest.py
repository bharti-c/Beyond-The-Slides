"""Node 1: Ingest.

Validates input files, assigns a unique lecture_id, provisions project storage
under ./data/{lecture_id}/, extracts video duration, and initializes pipeline state.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

from config import Config
from cache_manager import CacheManager
from models import PipelineGraphState

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str | Path) -> float:
    """Extract duration of a video file in seconds using ffmpeg."""
    ffmpeg_bin = Config.get_ffmpeg_binary()
    cmd = [ffmpeg_bin, "-i", str(video_path)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Look for "Duration: 00:00:00.00" in stderr
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+|\d+)", proc.stderr)
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except Exception as e:
        logger.warning(f"Could not extract video duration via ffmpeg: {e}")

    # Fallback default duration
    return 180.0


def ingest_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Ingest node entrypoint for LangGraph.

    Validates inputs and sets up storage.
    """
    logger.info("Executing Node 1: Ingest")
    video_path = Path(state["video_path"]).resolve()
    slides_path = Path(state["slides_pdf_path"]).resolve()
    notes_paths = [Path(p).resolve() for p in state.get("notes_paths", [])]

    errors = list(state.get("errors", []))

    if not video_path.is_file():
        errors.append(f"Video file not found: {video_path}")
    if not slides_path.is_file():
        errors.append(f"Slides PDF not found: {slides_path}")
    for np in notes_paths:
        if not np.is_file():
            errors.append(f"Notes file not found: {np}")

    # Compute hash to create or reuse lecture_id
    video_hash = CacheManager.compute_file_hash(video_path) if video_path.is_file() else "unknown"
    lecture_id = state.get("lecture_id") or f"lec_{video_hash[:8]}_{int(time.time())}"

    # Target storage directory under ./data/{lecture_id}/
    target_dir = Config.DATA_DIR / lecture_id
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "slides").mkdir(parents=True, exist_ok=True)
    (target_dir / "notes").mkdir(parents=True, exist_ok=True)
    (target_dir / "audio").mkdir(parents=True, exist_ok=True)
    (target_dir / "chunks").mkdir(parents=True, exist_ok=True)

    # Copy files into project directory for persistent provenance
    stored_video_path = target_dir / video_path.name
    if video_path.is_file() and not stored_video_path.is_file():
        try:
            shutil.copy2(video_path, stored_video_path)
        except Exception as e:
            logger.warning(f"Could not copy video to data dir ({e}), referencing original path.")
            stored_video_path = video_path

    stored_slides_path = target_dir / slides_path.name
    if slides_path.is_file() and not stored_slides_path.is_file():
        try:
            shutil.copy2(slides_path, stored_slides_path)
        except Exception:
            stored_slides_path = slides_path

    stored_notes_paths = []
    for idx, np in enumerate(notes_paths):
        dest = target_dir / "notes" / f"note_{idx:02d}_{np.name}"
        if np.is_file() and not dest.is_file():
            try:
                shutil.copy2(np, dest)
                stored_notes_paths.append(str(dest))
            except Exception:
                stored_notes_paths.append(str(np))
        elif dest.is_file():
            stored_notes_paths.append(str(dest))

    duration = get_video_duration(stored_video_path)
    logger.info(f"Ingested lecture {lecture_id}. Video duration: {duration:.1f}s")

    return {
        "lecture_id": lecture_id,
        "video_path": str(stored_video_path),
        "slides_pdf_path": str(stored_slides_path),
        "notes_paths": stored_notes_paths,
        "video_duration": duration,
        "output_dir": str(target_dir),
        "current_stage": "ingest_complete",
        "errors": errors,
    }
