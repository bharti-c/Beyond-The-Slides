"""End-to-end smoke test for Lecture-to-Living-Course pipeline.

Executes all 10 LangGraph nodes sequentially using the sample lecture materials
in ./sample_data/ and verifies output integrity across all stages.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.graph import run_pipeline
from models import PipelineGraphState


def test_end_to_end_pipeline_run():
    """Run full pipeline over generated sample lecture data."""
    sample_dir = Path(__file__).resolve().parent.parent / "sample_data"
    video_path = sample_dir / "sample_video.mp4"
    slides_path = sample_dir / "sample_slides.pdf"
    notes_path = sample_dir / "sample_notes.png"

    assert video_path.is_file(), "Sample video file must exist"
    assert slides_path.is_file(), "Sample slides PDF must exist"
    assert notes_path.is_file(), "Sample notes photo must exist"

    # Execute pipeline
    result: PipelineGraphState = run_pipeline(
        video_path=str(video_path),
        slides_pdf_path=str(slides_path),
        notes_paths=[str(notes_path)],
        lecture_id="test_smoke_lecture",
    )

    # 1. Ingest verification
    assert result["lecture_id"] == "test_smoke_lecture"
    assert result["video_duration"] > 0

    # 2. Transcribe verification
    assert "transcript" in result
    assert len(result["transcript"]) > 0
    # Every segment has start, end, text
    for seg in result["transcript"]:
        assert seg.start <= seg.end
        assert len(seg.text) > 0

    # 3. Slide parse verification
    assert "slides" in result
    assert len(result["slides"]) == 2  # 2 slides in sample_slides.pdf

    # 4. Notes parse verification
    assert "notes" in result
    assert len(result["notes"]) == 1

    # 5. Multimodal Alignment verification
    timeline = result.get("timeline")
    assert timeline is not None
    assert len(timeline.slide_alignments) == 2
    # Check monotonic alignment constraint
    assert timeline.slide_alignments[0].start_time <= timeline.slide_alignments[1].start_time

    # 6. Verbal Gap Detection verification
    assert "gaps" in result
    # Gaps should have valid classification
    for g in result["gaps"]:
        assert g.gap_type in ("example", "caveat", "correction", "emphasis")
        assert g.start_time <= g.end_time

    # 7. Mind Map verification
    assert result["mindmap"] is not None
    assert result["mindmap"].root is not None
    assert len(result["mindmap"].root.children) > 0

    # 8. Quiz verification
    assert result["quiz"] is not None
    assert len(result["quiz"].questions) > 0
    # At least some questions flagged for notes/supplementary content
    verbal_q = [q for q in result["quiz"].questions if q.is_verbal_only]
    assert len(verbal_q) >= 1

    # 9. Citation Verification
    for q in result["quiz"].questions:
        assert q.verification_status in ("supported", "partially", "unsupported")
        assert len(q.citations) >= 1
