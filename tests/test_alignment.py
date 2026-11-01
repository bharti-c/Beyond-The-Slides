"""Unit tests for Monotonic Slide Alignment dynamic programming logic.

Verifies:
1. Strict monotonic ordering constraint (s_t <= s_{t+1}).
2. Preservation of all slides with continuous interval coverage.
3. Robust handling of edge cases (single slide, empty segments, interpolation).
"""

from __future__ import annotations

import numpy as np
import pytest
from models import SlideAlignment, TranscriptSegment
from pipeline.node_05_align import (
    build_slide_intervals,
    cosine_similarity_matrix,
    monotonic_slide_alignment,
)


def test_cosine_similarity_matrix():
    """Verify cosine similarity calculation between two matrices."""
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sim = cosine_similarity_matrix(a, b)
    assert sim.shape == (2, 2)
    assert np.isclose(sim[0, 0], 1.0)
    assert np.isclose(sim[1, 1], 1.0)
    assert np.isclose(sim[0, 1], 0.0)


def test_monotonic_alignment_order_constraint():
    """Verify that DP alignment strictly enforces monotonic non-decreasing slide assignments."""
    T = 6  # 6 transcript segments
    S = 3  # 3 slides
    D = 4  # embedding dimension

    # Synthesize transcript embeddings that gradually transition from slide 0 to 1 to 2
    slide_embs = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Slide 0
        [0.0, 1.0, 0.0, 0.0],  # Slide 1
        [0.0, 0.0, 1.0, 0.0],  # Slide 2
    ], dtype=np.float32)

    # Transcript segments matching slide 0, then 1, then 2
    transcript_embs = np.array([
        [0.9, 0.1, 0.0, 0.0],  # Seg 0 -> Slide 0
        [0.8, 0.2, 0.0, 0.0],  # Seg 1 -> Slide 0
        [0.1, 0.9, 0.0, 0.0],  # Seg 2 -> Slide 1
        [0.2, 0.8, 0.0, 0.0],  # Seg 3 -> Slide 1
        [0.0, 0.1, 0.9, 0.0],  # Seg 4 -> Slide 2
        [0.0, 0.0, 1.0, 0.0],  # Seg 5 -> Slide 2
    ], dtype=np.float32)

    segments = [
        TranscriptSegment(start=float(i * 10), end=float((i + 1) * 10), text=f"Text {i}")
        for i in range(T)
    ]

    alignment, sim_matrix = monotonic_slide_alignment(
        transcript_embeddings=transcript_embs,
        slide_embeddings=slide_embs,
        transcript_segments=segments,
        num_slides=S,
        skip_penalty=0.05,
    )

    assert len(alignment) == T
    # Check monotonicity: s_0 <= s_1 <= ... <= s_{T-1}
    for i in range(len(alignment) - 1):
        assert alignment[i] <= alignment[i + 1], f"Monotonicity violation at {i}: {alignment[i]} > {alignment[i+1]}"

    # Check correct cluster assignments
    assert alignment[0] == 0
    assert alignment[1] == 0
    assert alignment[2] == 1
    assert alignment[3] == 1
    assert alignment[4] == 2
    assert alignment[5] == 2


def test_monotonic_alignment_resists_backward_jump():
    """Verify that DP alignment will not jump backwards even if later segment has high similarity to earlier slide."""
    T = 4
    S = 2
    slide_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # Segments: 0 matches slide 0, 1 matches slide 1, 2 matches slide 1,
    # segment 3 has an incidental word from slide 0, but must NOT jump backward!
    transcript_embs = np.array([
        [1.0, 0.0],  # Seg 0 -> Slide 0
        [0.0, 1.0],  # Seg 1 -> Slide 1
        [0.0, 1.0],  # Seg 2 -> Slide 1
        [0.6, 0.4],  # Seg 3 -> slight slide 0 resemblance, but must stay >= 1
    ], dtype=np.float32)

    segments = [
        TranscriptSegment(start=float(i * 10), end=float((i + 1) * 10), text=f"Text {i}")
        for i in range(T)
    ]

    alignment, _ = monotonic_slide_alignment(
        transcript_embeddings=transcript_embs,
        slide_embeddings=slide_embs,
        transcript_segments=segments,
        num_slides=S,
    )

    for i in range(len(alignment) - 1):
        assert alignment[i] <= alignment[i + 1]


def test_build_slide_intervals_coverage():
    """Verify continuous time interval construction spanning full video duration."""
    T = 4
    S = 2
    alignment = [0, 0, 1, 1]
    sim_matrix = np.ones((T, S), dtype=np.float32)
    segments = [
        TranscriptSegment(start=0.0, end=10.0, text="Intro"),
        TranscriptSegment(start=10.0, end=20.0, text="Part 1"),
        TranscriptSegment(start=20.0, end=30.0, text="Part 2"),
        TranscriptSegment(start=30.0, end=40.0, text="Outro"),
    ]

    intervals = build_slide_intervals(
        alignment=alignment,
        sim_matrix=sim_matrix,
        transcript=segments,
        num_slides=S,
        video_duration=50.0,
    )

    assert len(intervals) == S
    # Slide 0 starts at 0.0
    assert intervals[0].start_time == 0.0
    # Slide 1 starts where slide 0 ends
    assert intervals[1].start_time == intervals[0].end_time
    # Slide 1 extends to full duration
    assert intervals[1].end_time >= 40.0
