"""Utility script to generate sample lecture materials for end-to-end testing.

Generates:
1. sample_data/sample_slides.pdf (2-page lecture presentation)
2. sample_data/sample_notes.png (student notes photo)
3. sample_data/sample_video.mp4 (short lecture clip with synthesized audio)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont
from config import Config
from tts import tts_service


def create_sample_slides_pdf(output_path: Path) -> Path:
    """Create a 2-page lecture slides PDF using PyMuPDF."""
    import fitz  # PyMuPDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    # Slide 1: Introduction
    page1 = doc.new_page(width=960, height=540)
    rect1 = fitz.Rect(50, 50, 910, 490)
    text1 = (
        "Sequence Alignment Fundamentals\n\n"
        "• Formulation: Finding optimal correspondence between two time sequences.\n"
        "• Common Domains: Computational biology, speech processing, and timeseries.\n"
        "• Key Objective: Minimize cumulative temporal distortion distance."
    )
    page1.draw_rect(rect1, color=(0.1, 0.3, 0.7), fill=(0.95, 0.97, 1.0))
    page1.insert_text((80, 100), text1, fontsize=20, color=(0.1, 0.15, 0.35))

    # Slide 2: Dynamic Programming Recurrence
    page2 = doc.new_page(width=960, height=540)
    rect2 = fitz.Rect(50, 50, 910, 490)
    text2 = (
        "Dynamic Programming Recurrence\n\n"
        "• Global Alignment Grid: D[i, j] = dist(i, j) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])\n"
        "• Time Complexity: O(N * M)\n"
        "• Space Complexity: O(N * M) or O(min(N, M)) with memory optimization"
    )
    page2.draw_rect(rect2, color=(0.1, 0.3, 0.7), fill=(0.95, 0.97, 1.0))
    page2.insert_text((80, 100), text2, fontsize=20, color=(0.1, 0.15, 0.35))

    doc.save(str(output_path))
    doc.close()
    return output_path


def create_sample_notes_image(output_path: Path) -> Path:
    """Create a student handwritten notes photo using Pillow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (800, 1000), color=(252, 250, 242))
    draw = ImageDraw.Draw(img)

    # Draw notebook lines
    for y in range(80, 980, 36):
        draw.line([(60, y), (740, y)], fill=(210, 220, 235), width=1)
    draw.line([(120, 40), (120, 980)], fill=(240, 180, 180), width=2)

    # Add handwritten text notes
    draw.text((140, 70), "Lecture 4 Notes: Alignment Algorithms", fill=(20, 30, 80))
    draw.text((140, 140), "- Goal: align video, slides & speech", fill=(30, 30, 70))
    draw.text((140, 210), "- DP matrix fills from top-left (0,0)", fill=(30, 30, 70))
    draw.text((140, 280), "- Prof caveat: step constraint MUST be monotonic!", fill=(180, 40, 40))
    draw.text((140, 350), "- Midterm hint: banded search cuts time by 70%", fill=(150, 30, 120))

    # Draw simple alignment diagram sketch
    draw.rectangle([200, 450, 550, 750], outline=(40, 50, 100), width=2)
    for i in range(1, 4):
        draw.line([(200 + i * 87, 450), (200 + i * 87, 750)], fill=(120, 130, 160), width=1)
        draw.line([(200, 450 + i * 75), (550, 450 + i * 75)], fill=(120, 130, 160), width=1)
    draw.line([(200, 450), (287, 525), (374, 600), (461, 675), (550, 750)], fill=(220, 40, 40), width=3)
    draw.text((220, 770), "Figure: Monotonic optimal path diagonal", fill=(60, 60, 80))

    img.save(str(output_path))
    return output_path


async def create_sample_lecture_video(output_path: Path) -> Path:
    """Synthesize a short lecture audio and combine it with test video frames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_audio = output_path.parent / "temp_lecture_voice.mp3"

    speech_text = (
        "Welcome to today's lecture. In slide one, we introduce sequence alignment and distance metrics. "
        "Here is an important exam hint: dynamic time warping requires a strictly monotonic step constraint to prevent temporal loops. "
        "Moving to slide two, the recurrence relation runs in O of N times M time complexity. "
        "Notice that using banded diagonal search reduces computational runtime significantly."
    )

    # Synthesize audio using edge-tts
    await tts_service.synthesize(speech_text, speaker="Host", output_path=temp_audio)

    ffmpeg_bin = Config.get_ffmpeg_binary()
    # Create 30s blue canvas video combined with speech audio
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", "color=c=0x1E3A8A:s=960x540:r=24",
        "-i", str(temp_audio),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if temp_audio.exists():
        temp_audio.unlink()

    return output_path


def main():
    """Generate all sample materials in ./sample_data/."""
    sample_dir = Path(__file__).resolve().parent.parent / "sample_data"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample lecture slides PDF...")
    pdf_path = create_sample_slides_pdf(sample_dir / "sample_slides.pdf")
    print(f"Created: {pdf_path}")

    print("Generating sample handwritten notes image...")
    notes_path = create_sample_notes_image(sample_dir / "sample_notes.png")
    print(f"Created: {notes_path}")

    print("Generating sample lecture video with audio...")
    vid_path = asyncio.run(create_sample_lecture_video(sample_dir / "sample_video.mp4"))
    print(f"Created: {vid_path}")

    print("\n[OK] All sample data successfully generated in ./sample_data/!")


if __name__ == "__main__":
    main()
