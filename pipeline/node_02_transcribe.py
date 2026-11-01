"""Node 2: Transcribe.

Extracts audio from lecture video using ffmpeg, chunks audio if needed to respect
Groq's upload limits, calls Groq Whisper with verbose_json, and aligns chunk timestamps
to absolute video time coordinates.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from config import Config
from cache_manager import CacheManager
from groq_client import groq_client
from models import PipelineGraphState, TranscriptSegment
from pipeline.node_01_ingest import get_video_duration

logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path: Path, output_audio_path: Path) -> Path:
    """Extract mono audio at 16kHz from video using ffmpeg."""
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = Config.get_ffmpeg_binary()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        str(output_audio_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_audio_path


def chunk_audio_file(
    audio_path: Path,
    output_chunks_dir: Path,
    chunk_duration_sec: int = 600,
) -> List[tuple[Path, float]]:
    """Split audio into segments of chunk_duration_sec seconds.

    Returns list of (chunk_path, start_offset_seconds).
    """
    output_chunks_dir.mkdir(parents=True, exist_ok=True)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    # If file is smaller than chunk limit, return directly with 0 offset
    if file_size_mb < Config.AUDIO_CHUNK_SIZE_MB:
        return [(audio_path, 0.0)]

    ffmpeg_bin = Config.get_ffmpeg_binary()
    chunk_template = output_chunks_dir / "chunk_%03d.mp3"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration_sec),
        "-c", "copy",
        str(chunk_template),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    chunks = sorted(output_chunks_dir.glob("chunk_*.mp3"))
    results = []
    current_offset = 0.0

    for ch in chunks:
        ch_duration = get_video_duration(ch)
        results.append((ch, current_offset))
        current_offset += ch_duration

    return results


def transcribe_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Transcribe node entrypoint for LangGraph."""
    logger.info("Executing Node 2: Transcribe")
    video_path = Path(state["video_path"])
    output_dir = Path(state["output_dir"])
    audio_path = output_dir / "audio" / "extracted_lecture.mp3"
    chunks_dir = output_dir / "chunks"

    errors = list(state.get("errors", []))

    if not video_path.is_file():
        errors.append(f"Cannot transcribe: video file not found at {video_path}")
        return {
            "transcript": [],
            "current_stage": "transcribe_failed",
            "errors": errors,
        }

    # Extract audio if not present
    if not audio_path.is_file():
        try:
            extract_audio_from_video(video_path, audio_path)
        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            errors.append(f"Audio extraction failed: {e}")
            return {
                "transcript": [],
                "current_stage": "transcribe_failed",
                "errors": errors,
            }

    # Check cache for existing transcript
    audio_hash = CacheManager.compute_file_hash(audio_path)
    cached_transcript = CacheManager.get(audio_hash, "transcribe")
    if cached_transcript:
        logger.info(f"Using cached transcript for {audio_hash[:8]}")
        segments = [TranscriptSegment(**seg) for seg in cached_transcript]
        return {
            "transcript": segments,
            "current_stage": "transcribe_complete",
            "errors": errors,
        }

    # Split into chunks if needed
    try:
        chunks_with_offsets = chunk_audio_file(audio_path, chunks_dir)
    except Exception as e:
        logger.warning(f"Audio chunking failed: {e}. Attempting direct file transcription.")
        chunks_with_offsets = [(audio_path, 0.0)]

    all_segments: List[TranscriptSegment] = []

    if not Config.validate_groq_key():
        # Fallback placeholder transcript when API key is not yet provided
        logger.warning("GROQ_API_KEY is not set. Generating sample transcript segments for development.")
        duration = state.get("video_duration", 180.0)
        sample_step = 15.0
        cur_t = 0.0
        while cur_t < duration:
            end_t = min(cur_t + sample_step, duration)
            all_segments.append(
                TranscriptSegment(
                    start=round(cur_t, 2),
                    end=round(end_t, 2),
                    text=f"Spoken lecture content covering core concepts between {cur_t:.0f}s and {end_t:.0f}s.",
                )
            )
            cur_t = end_t
    else:
        for chunk_path, offset in chunks_with_offsets:
            logger.info(f"Transcribing audio chunk {chunk_path.name} with offset {offset:.1f}s...")
            try:
                raw_segs = groq_client.transcribe_audio_chunk(chunk_path)
                for seg in raw_segs:
                    start_abs = round(offset + seg["start"], 2)
                    end_abs = round(offset + seg["end"], 2)
                    text = seg["text"].strip()
                    if text:
                        all_segments.append(
                            TranscriptSegment(start=start_abs, end=end_abs, text=text)
                        )
            except Exception as e:
                logger.error(f"Transcription error on chunk {chunk_path.name}: {e}")
                errors.append(f"Transcription error on {chunk_path.name}: {e}")

    # Cache successful transcript
    if all_segments:
        CacheManager.set(
            audio_hash,
            "transcribe",
            [seg.model_dump() for seg in all_segments],
        )

    logger.info(f"Transcription complete: {len(all_segments)} segments.")
    return {
        "transcript": all_segments,
        "current_stage": "transcribe_complete",
        "errors": errors,
    }
