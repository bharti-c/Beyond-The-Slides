"""Node 6: Gap Detect.

Detects professor's verbal-only additions that are NOT covered on the slides or notes.
Identifies examples, caveats, corrections, and exam hints/emphases with exact timestamps
and one-line summaries.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from config import Config
from groq_client import groq_client
from models import (
    GapType,
    KnowledgeGap,
    PipelineGraphState,
    SlideContent,
    TranscriptSegment,
    UnifiedTimeline,
)
from pipeline.node_05_align import cosine_similarity_matrix, get_embedding_model

logger = logging.getLogger(__name__)

GAP_CLASSIFICATION_PROMPT = """You are analyzing verbal-only lecture content spoken by a professor that does NOT appear on the lecture slides.

Analyze the spoken transcript segment below and categorize it:
1. "gap_type": Must be one of:
   - "example": practical demo, real-world scenario, code walkthrough, or analogy
   - "caveat": boundary condition, exception, limitation, or warning
   - "correction": fixing a verbal slip, correcting a slide typo, or rectifying an equation
   - "emphasis": professor stressing an exam hint, key formula, or crucial takeaway
2. "summary": A crisp, high-value one-line summary (max 15 words) explaining the point.
3. "context": Brief explanation of why this verbal addition matters for student understanding or exams.

Spoken transcript quote:
\"\"\"{quote}\"\"\"

Return ONLY a valid JSON object matching this schema:
{{
  "gap_type": "example" | "caveat" | "correction" | "emphasis",
  "summary": "...",
  "context": "..."
}}
"""


def cluster_gap_segments(
    gap_indices: List[int],
    transcript: List[TranscriptSegment],
    max_gap_separation_sec: float = 12.0,
) -> List[List[int]]:
    """Group contiguous or closely spaced gap segments into coherent knowledge gap clusters."""
    if not gap_indices:
        return []

    clusters: List[List[int]] = []
    current_cluster: List[int] = [gap_indices[0]]

    for i in range(1, len(gap_indices)):
        prev_idx = gap_indices[i - 1]
        curr_idx = gap_indices[i]

        prev_seg = transcript[prev_idx]
        curr_seg = transcript[curr_idx]

        # If adjacent indices or separated by less than threshold
        if (curr_idx == prev_idx + 1) or (curr_seg.start - prev_seg.end <= max_gap_separation_sec):
            current_cluster.append(curr_idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [curr_idx]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def gap_detect_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Gap detect node entrypoint for LangGraph."""
    logger.info("Executing Node 6: Gap Detect")
    timeline: UnifiedTimeline = state.get("timeline")
    slides: List[SlideContent] = state.get("slides", [])
    errors = list(state.get("errors", []))

    if not timeline or not timeline.transcript:
        logger.warning("No transcript available for gap detection.")
        return {
            "gaps": [],
            "current_stage": "gap_detect_complete",
            "errors": errors,
        }

    transcript = timeline.transcript
    embedder = get_embedding_model()

    # Create slide lookup
    slide_lookup = {s.slide_index: s for s in slides}

    # Embed transcript
    transcript_texts = [seg.text for seg in transcript]
    transcript_embs = np.array(embedder.encode(transcript_texts, show_progress_bar=False), dtype=np.float32)

    # Embed slides
    slide_embs_dict = {}
    for s in slides:
        text = f"{s.text}\n{s.visual_description or ''}".strip()
        slide_embs_dict[s.slide_index] = np.array(
            embedder.encode([text], show_progress_bar=False), dtype=np.float32
        )

    # Map each segment to its active slide
    seg_to_slide = {}
    for sa in timeline.slide_alignments:
        for seg_idx in sa.matched_segments:
            seg_to_slide[seg_idx] = sa.slide_index

    gap_segment_indices: List[int] = []

    for t_idx, seg in enumerate(transcript):
        assigned_slide = seg_to_slide.get(t_idx)

        # If no slide assigned, or slide has no content, candidate gap
        if assigned_slide is None or assigned_slide not in slide_embs_dict:
            gap_segment_indices.append(t_idx)
            continue

        s_emb = slide_embs_dict[assigned_slide]
        t_emb = transcript_embs[t_idx : t_idx + 1]

        sim = float(cosine_similarity_matrix(t_emb, s_emb)[0, 0])

        # If similarity is below threshold, the professor is speaking off-slide
        if sim < Config.GAP_SIMILARITY_THRESHOLD:
            gap_segment_indices.append(t_idx)

    # Cluster gap segments
    clusters = cluster_gap_segments(gap_segment_indices, transcript)
    logger.info(f"Identified {len(gap_segment_indices)} off-slide segments clustered into {len(clusters)} verbal gaps.")

    gaps: List[KnowledgeGap] = []

    for c_idx, cluster in enumerate(clusters):
        start_time = transcript[cluster[0]].start
        end_time = transcript[cluster[-1]].end
        combined_quote = " ".join(transcript[idx].text for idx in cluster).strip()

        # Ignore tiny filler utterances (< 25 characters)
        if len(combined_quote) < 25:
            continue

        gap_type = GapType.EXAMPLE
        summary = f"Professor explains verbal concept at {start_time:.0f}s"
        context = "Spoken elaboration omitted from the written lecture slides."

        if Config.validate_groq_key():
            prompt = GAP_CLASSIFICATION_PROMPT.format(quote=combined_quote)
            try:
                raw_json = groq_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                cleaned = raw_json.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]

                parsed = json.loads(cleaned.strip())
                gt_str = str(parsed.get("gap_type", "example")).lower()
                gap_type = (
                    GapType(gt_str)
                    if gt_str in GapType._value2member_map_
                    else GapType.EXAMPLE
                )
                summary = str(parsed.get("summary", summary)).strip()
                context = str(parsed.get("context", context)).strip()
            except Exception as e:
                logger.warning(f"Error classifying gap {c_idx + 1}: {e}")
                errors.append(f"Gap classification error: {e}")
        else:
            # Deterministic heuristic categorization for offline/testing mode
            lower_q = combined_quote.lower()
            if any(w in lower_q for w in ("exam", "test", "important", "remember", "key")):
                gap_type = GapType.EMPHASIS
                summary = f"Exam emphasis on: {combined_quote[:45]}..."
            elif any(w in lower_q for w in ("careful", "beware", "except", "note that", "however")):
                gap_type = GapType.CAVEAT
                summary = f"Caveat regarding: {combined_quote[:45]}..."
            elif any(w in lower_q for w in ("mistake", "correction", "i meant", "typo", "error")):
                gap_type = GapType.CORRECTION
                summary = f"Correction: {combined_quote[:45]}..."
            else:
                gap_type = GapType.EXAMPLE
                summary = f"Verbal addition: {combined_quote[:45]}..."

        gaps.append(
            KnowledgeGap(
                gap_id=f"gap_{c_idx + 1:02d}",
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                gap_type=gap_type,
                summary=summary,
                transcript_quote=combined_quote,
                context=context,
            )
        )

    logger.info(f"Detected {len(gaps)} categorized verbal knowledge gaps.")
    return {
        "gaps": gaps,
        "current_stage": "gap_detect_complete",
        "errors": errors,
    }
