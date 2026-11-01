"""Pluggable Text-To-Speech (TTS) layer for Lecture-to-Living-Course.

Provides an abstract BaseTTS interface and a high-fidelity default EdgeTTS implementation
using Microsoft Edge's neural voices with distinct personas for Host and Curious Student.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
from config import Config

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    """Abstract interface for pluggable TTS providers."""

    @abstractmethod
    async def synthesize(self, text: str, speaker: str, output_path: str | Path) -> str:
        """Synthesize text into an audio file.

        Args:
            text: Text to speak
            speaker: Speaker identifier ('Host' or 'Curious Student')
            output_path: Path where output audio file should be saved

        Returns:
            Absolute path to generated audio file
        """
        pass

    @abstractmethod
    async def synthesize_dialogue(
        self,
        script: List[Dict[str, str]],
        output_dir: str | Path,
        combined_output_path: str | Path,
    ) -> str:
        """Synthesize multiple dialogue turns and combine them into a single podcast audio file."""
        pass


class EdgeTTS(BaseTTS):
    """Default high-quality neural TTS implementation using edge-tts."""

    # Default neural voices
    VOICE_MAP: Dict[str, str] = {
        "Host": "en-US-GuyNeural",               # Natural authoritative male voice
        "Curious Student": "en-US-JennyNeural",   # Expressive, inquisitive female voice
    }

    def __init__(self, voice_map: Optional[Dict[str, str]] = None) -> None:
        self.voice_map = voice_map or self.VOICE_MAP

    async def synthesize(self, text: str, speaker: str, output_path: str | Path) -> str:
        """Synthesize a single speech line to MP3 using edge-tts."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        voice = self.voice_map.get(speaker, "en-US-GuyNeural")

        try:
            import edge_tts
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(str(out))
            return str(out)
        except Exception as e:
            logger.warning(f"EdgeTTS synthesis failed for speaker {speaker}: {e}. Creating fallback audio.")
            # Create a minimal empty/fallback audio file so the pipeline continues
            self._create_fallback_audio(out)
            return str(out)

    async def synthesize_dialogue(
        self,
        script: List[Dict[str, str]],
        output_dir: str | Path,
        combined_output_path: str | Path,
    ) -> str:
        """Synthesize turns in parallel batches and concatenate them using ffmpeg."""
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        combined_out = Path(combined_output_path).resolve()
        combined_out.parent.mkdir(parents=True, exist_ok=True)

        clip_paths: List[str] = []
        tasks = []

        for idx, line in enumerate(script):
            clip_file = out_dir / f"turn_{idx:03d}_{line.get('speaker', 'host').replace(' ', '_')}.mp3"
            clip_paths.append(str(clip_file))
            tasks.append(self.synthesize(text=line.get("text", ""), speaker=line.get("speaker", "Host"), output_path=clip_file))

        await asyncio.gather(*tasks)

        # Concatenate audio clips using ffmpeg concat demuxer
        self._concatenate_clips(clip_paths, combined_out)
        return str(combined_out)

    def _concatenate_clips(self, clip_paths: List[str], output_path: Path) -> None:
        """Concatenate MP3 files using ffmpeg."""
        valid_clips = [p for p in clip_paths if Path(p).is_file() and Path(p).stat().st_size > 0]
        if not valid_clips:
            self._create_fallback_audio(output_path)
            return

        import subprocess
        concat_list = output_path.parent / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in valid_clips:
                # ffmpeg requires escaped paths or forward slashes
                normalized = Path(p).as_posix()
                f.write(f"file '{normalized}'\n")

        ffmpeg_bin = Config.get_ffmpeg_binary()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            logger.warning(f"ffmpeg concat failed ({e}); attempting re-encoding concat fallback.")
            cmd_reencode = [
                ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-c:a", "libmp3lame",
                "-b:a", "128k",
                str(output_path),
            ]
            try:
                subprocess.run(cmd_reencode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e2:
                logger.error(f"Failed to concatenate audio clips: {e2}")
                if valid_clips:
                    import shutil
                    shutil.copy(valid_clips[0], output_path)
        finally:
            if concat_list.exists():
                try:
                    concat_list.unlink()
                except Exception:
                    pass

    def _create_fallback_audio(self, output_path: Path) -> None:
        """Generate a short silent MP3 placeholder using ffmpeg or raw bytes."""
        ffmpeg_bin = Config.get_ffmpeg_binary()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=24000:cl=mono",
            "-t", "1",
            "-q:a", "9",
            str(output_path),
        ]
        try:
            import subprocess
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            # Minimal MP3 frame header as fallback
            with open(output_path, "wb") as f:
                f.write(b"\xff\xfb\x90\x64" + b"\x00" * 500)


# Default TTS provider instance
tts_service: BaseTTS = EdgeTTS()
