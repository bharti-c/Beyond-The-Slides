"""Node 5: Align.

Performs multimodal alignment onto ONE master timeline:
1. Samples video frames every 5-10 seconds using ffmpeg.
2. Uses local sentence-transformers ('all-MiniLM-L6-v2') to embed:
   - Slide text & visual descriptions
   - Transcript segment texts
   - Handwritten notes
3. Enforces a monotonic dynamic programming constraint so slides appear in chronological order.
4. Aligns handwritten notes to transcript segments by semantic similarity.
5. Produces a UnifiedTimeline object.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import Config
from models import (
    NoteAlignment,
    NoteContent,
    PipelineGraphState,
    SlideAlignment,
    SlideContent,
    TranscriptSegment,
    UnifiedTimeline,
)

logger = logging.getLogger(__name__)

# Cached model holder to prevent reloading model weights repeatedly
_EMBEDDING_MODEL = None


def get_embedding_model():
    """Lazy loader for sentence-transformers embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model '{Config.EMBEDDING_MODEL_NAME}'...")
            _EMBEDDING_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using mock embedder.")

            class MockEmbedder:
                def encode(self, texts, **kwargs):
                    # Deterministic hash-based pseudo embeddings
                    rng = np.random.default_rng(42)
                    return rng.standard_normal((len(texts), 384)).astype(np.float32)

            _EMBEDDING_MODEL = MockEmbedder()
    return _EMBEDDING_MODEL


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between matrices a (M x D) and b (N x D)."""
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    a_normalized = a / norm_a
    b_normalized = b / norm_b
    return np.dot(a_normalized, b_normalized.T)


def sample_video_frames(video_path: Path, frames_dir: Path, sample_interval: int = 8) -> List[Path]:
    """Sample video frames at regular intervals using ffmpeg."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(frames_dir.glob("frame_*.jpg"))
    if existing_frames:
        return existing_frames

    ffmpeg_bin = Config.get_ffmpeg_binary()
    out_pattern = frames_dir / "frame_%04d.jpg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{sample_interval}",
        "-q:v", "3",
        str(out_pattern),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return sorted(frames_dir.glob("frame_*.jpg"))
    except Exception as e:
        logger.warning(f"Failed to sample video frames: {e}")
        return []


def monotonic_slide_alignment(
    transcript_embeddings: np.ndarray,
    slide_embeddings: np.ndarray,
    transcript_segments: List[TranscriptSegment],
    num_slides: int,
    skip_penalty: float = 0.05,
) -> Tuple[List[int], np.ndarray]:
    """Monotonic Dynamic Programming Alignment.

    Enforces that slides appear in chronological order: s_0 <= s_1 <= ... <= s_{T-1}.
    Args:
        transcript_embeddings: (T, D) array
        slide_embeddings: (S, D) array
        transcript_segments: List of transcript segments
        num_slides: Number of slides S
        skip_penalty: Penalty for advancing forward to a new slide

    Returns:
        alignment: List of length T where alignment[t] is the assigned slide index
        sim_matrix: (T, S) cosine similarity matrix
    """
    T = transcript_embeddings.shape[0]
    S = num_slides

    if S == 0 or T == 0:
        return [0] * T, np.zeros((T, max(1, S)))

    # Compute cosine similarity matrix: shape (T, S)
    sim = cosine_similarity_matrix(transcript_embeddings, slide_embeddings)

    # DP tables: dp[t, s] = max score matching prefix t segments ending at slide s
    dp = np.full((T, S), -1e9, dtype=np.float32)
    parent = np.full((T, S), -1, dtype=np.int32)

    # Base case: t = 0 (can start at slide 0, or slide s with a penalty)
    for s in range(S):
        dp[0, s] = sim[0, s] - (s * skip_penalty)

    # Induction: t = 1 to T-1
    for t in range(1, T):
        for s in range(S):
            best_prev_score = -1e9
            best_prev_s = 0
            # Monotonic constraint: prev_s must be <= s
            for prev_s in range(s + 1):
                transition_cost = 0.0 if prev_s == s else (s - prev_s) * skip_penalty
                score = dp[t - 1, prev_s] - transition_cost
                if score > best_prev_score:
                    best_prev_score = score
                    best_prev_s = prev_s
            dp[t, s] = sim[t, s] + best_prev_score
            parent[t, s] = best_prev_s

    # Backtracking: choose best ending slide at T-1
    best_final_s = int(np.argmax(dp[T - 1, :]))
    alignment = [0] * T
    curr_s = best_final_s

    for t in range(T - 1, -1, -1):
        alignment[t] = curr_s
        curr_s = int(parent[t, curr_s]) if t > 0 else 0

    return alignment, sim


def build_slide_intervals(
    alignment: List[int],
    sim_matrix: np.ndarray,
    transcript: List[TranscriptSegment],
    num_slides: int,
    video_duration: float,
) -> List[SlideAlignment]:
    """Convert segment-level slide assignments into continuous time intervals."""
    if num_slides == 0:
        return []

    slide_to_segments: Dict[int, List[int]] = {s: [] for s in range(num_slides)}
    for seg_idx, slide_idx in enumerate(alignment):
        slide_to_segments[slide_idx].append(seg_idx)

    slide_alignments: List[SlideAlignment] = []

    # Calculate preliminary boundaries based on assigned transcript segments
    boundaries: List[Tuple[float, float, float]] = []  # (start, end, avg_conf)

    for s in range(num_slides):
        segs = slide_to_segments[s]
        if segs:
            s_start = transcript[segs[0]].start
            s_end = transcript[segs[-1]].end
            conf = float(np.mean([sim_matrix[idx, s] for idx in segs]))
            boundaries.append((s_start, s_end, max(0.0, min(1.0, conf))))
        else:
            boundaries.append((-1.0, -1.0, 0.0))

    # Fill in unassigned slides via monotonic interpolation
    for s in range(num_slides):
        if boundaries[s][0] < 0:
            # Interpolate from adjacent slides
            prev_time = 0.0
            for ps in range(s - 1, -1, -1):
                if boundaries[ps][1] >= 0:
                    prev_time = boundaries[ps][1]
                    break
            next_time = video_duration
            for ns in range(s + 1, num_slides):
                if boundaries[ns][0] >= 0:
                    next_time = boundaries[ns][0]
                    break
            interp_start = prev_time
            interp_end = max(interp_start + 5.0, next_time)
            boundaries[s] = (interp_start, interp_end, 0.3)

    # Ensure full coverage without gaps: stretch first and last slide
    for s in range(num_slides):
        s_start, s_end, conf = boundaries[s]
        if s == 0:
            s_start = 0.0
        if s == num_slides - 1:
            s_end = max(s_end, video_duration)
        if s > 0:
            # Snap start to previous slide's end
            s_start = slide_alignments[s - 1].end_time

        # Ensure end >= start
        s_end = max(s_start + 1.0, min(s_end, video_duration))

        slide_alignments.append(
            SlideAlignment(
                slide_index=s,
                start_time=round(s_start, 2),
                end_time=round(s_end, 2),
                confidence=round(conf, 3),
                matched_segments=slide_to_segments[s],
            )
        )

    return slide_alignments


def align_notes_to_timeline(
    notes: List[NoteContent],
    transcript: List[TranscriptSegment],
    transcript_embeddings: np.ndarray,
    video_duration: float,
) -> List[NoteAlignment]:
    """Align student handwritten notes pages to transcript segments via semantic similarity."""
    if not notes or not transcript:
        return []

    embedder = get_embedding_model()
    note_texts = [
        f"{n.transcribed_text}\n{n.diagram_description or ''}".strip()
        for n in notes
    ]
    note_embeddings = np.array(embedder.encode(note_texts, show_progress_bar=False), dtype=np.float32)

    # Similarity: shape (len(notes), len(transcript))
    sim = cosine_similarity_matrix(note_embeddings, transcript_embeddings)

    note_alignments: List[NoteAlignment] = []
    num_segs = len(transcript)

    for n_idx, note in enumerate(notes):
        # Find top matching transcript segments (window of 3 segments)
        best_seg_idx = int(np.argmax(sim[n_idx]))
        conf = float(sim[n_idx, best_seg_idx])

        start_seg = max(0, best_seg_idx - 1)
        end_seg = min(num_segs - 1, best_seg_idx + 1)

        s_time = transcript[start_seg].start
        e_time = transcript[end_seg].end

        note_alignments.append(
            NoteAlignment(
                page_index=note.page_index,
                start_time=round(s_time, 2),
                end_time=round(e_time, 2),
                confidence=round(max(0.0, min(1.0, conf)), 3),
                matched_segments=list(range(start_seg, end_seg + 1)),
            )
        )

    return note_alignments


def align_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Align node entrypoint for LangGraph."""
    logger.info("Executing Node 5: Align")
    video_path = Path(state["video_path"])
    output_dir = Path(state["output_dir"])
    frames_dir = output_dir / "frames"

    transcript: List[TranscriptSegment] = state.get("transcript", [])
    slides: List[SlideContent] = state.get("slides", [])
    notes: List[NoteContent] = state.get("notes", [])
    duration = state.get("video_duration", 180.0)
    errors = list(state.get("errors", []))

    if not transcript:
        errors.append("Cannot align: transcript is empty.")
        timeline = UnifiedTimeline(
            lecture_id=state.get("lecture_id", "unknown"),
            video_duration=duration,
            transcript=[],
            slide_alignments=[],
            note_alignments=[],
        )
        return {
            "timeline": timeline,
            "current_stage": "align_failed",
            "errors": errors,
        }

    # Sample video frames
    try:
        sample_video_frames(video_path, frames_dir, sample_interval=Config.SLIDE_SAMPLE_RATE_SEC)
    except Exception as e:
        logger.warning(f"Frame sampling warning: {e}")

    # Compute text embeddings for transcript segments
    embedder = get_embedding_model()
    transcript_texts = [seg.text for seg in transcript]
    transcript_embs = np.array(embedder.encode(transcript_texts, show_progress_bar=False), dtype=np.float32)

    # Compute text embeddings for slides
    if slides:
        slide_texts = [
            f"{s.text}\n{s.visual_description or ''}".strip()
            for s in slides
        ]
        slide_embs = np.array(embedder.encode(slide_texts, show_progress_bar=False), dtype=np.float32)

        # Monotonic dynamic programming alignment
        alignment_seq, sim_matrix = monotonic_slide_alignment(
            transcript_embs,
            slide_embs,
            transcript,
            num_slides=len(slides),
        )
        slide_alignments = build_slide_intervals(
            alignment_seq,
            sim_matrix,
            transcript,
            num_slides=len(slides),
            video_duration=duration,
        )
    else:
        slide_alignments = []

    # Notes semantic alignment
    note_alignments = align_notes_to_timeline(notes, transcript, transcript_embs, duration)

    timeline = UnifiedTimeline(
        lecture_id=state.get("lecture_id", "unknown"),
        video_duration=duration,
        transcript=transcript,
        slide_alignments=slide_alignments,
        note_alignments=note_alignments,
    )

    logger.info(
        f"Alignment complete: {len(slide_alignments)} slides and {len(note_alignments)} notes aligned."
    )
    return {
        "timeline": timeline,
        "current_stage": "align_complete",
        "errors": errors,
    }
