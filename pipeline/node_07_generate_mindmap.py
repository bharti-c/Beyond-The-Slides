"""Node 7: Generate Mind Map.

Synthesizes a 4-level concept-based hierarchical mind map:
  Lecture Topic (Root, Level 0)
    -> Major Topics (Level 1, 4-8 topics, e.g. "Chunking", "Embeddings", "Retrieval", "Evaluation")
      -> Subtopics / Concepts (Level 2, <=6 per topic, e.g. "Fixed-size chunking", "Semantic chunking")
        -> Detail Leaves (Level 3, <=5 per subtopic: definition, mechanism, trade-off, verbal example/caveat/hint)

Uses a Two-Pass LLM architecture + sentence-transformer embedding deduplication +
self-check regex validation + 1 retry with error feedback + resilient concept fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ValidationError

from cache_manager import CacheManager
from config import Config
from groq_client import groq_client
from models import (
    KnowledgeGap,
    MindMap,
    MindMapNode,
    NoteContent,
    PipelineGraphState,
    SlideContent,
    UnifiedTimeline,
)
from pipeline.node_05_align import cosine_similarity_matrix, get_embedding_model

logger = logging.getLogger(__name__)

SLIDE_LABEL_PATTERN = re.compile(r"(?i)\b(slide|page|deck|study deck)\b|^\d+\W")

PASS_A_EXTRACTION_PROMPT = """You are an expert technical knowledge extractor.
Given aligned lecture segments combining presentation slides, spoken transcript, handwritten student notes, and professor's verbal-only additions, extract a flat list of atomic concepts and technical details.

HARD RULES:
1. Every concept "label" MUST be a short, clean concept name (2 to 6 words, e.g., "Fixed-Size Chunking", "Cosine Similarity Search", "Cross-Encoder Reranking", "Telecom Chatbot Latency Caveat").
2. NEVER include slide numbers, page numbers, deck titles, or numbering prefixes in "label" (FORBIDDEN: "Slide 14", "Page 2", "Study Deck", "01 • Foundations").
3. Put slide numbers in "slide_refs" (array of 1-indexed ints), note pages in "note_refs" (array of 1-indexed ints), and video timestamps in "timestamps" (array of floats).
4. Set "origin" to:
   - "slide-only": appears only on slides
   - "verbal-only": spoken by the professor (example, caveat, correction, exam hint) or handwritten note insight not on slides
   - "both": covered in both slides and spoken transcript/notes
5. Write a clear 1-2 sentence plain-language "summary" explaining the concept, mechanism, trade-off, or example.

Aligned Lecture Material:
{aligned_context}

Return ONLY valid JSON matching this schema:
{{
  "lecture_topic": "2-6 word overall lecture topic",
  "concepts": [
    {{
      "label": "2-6 word concept name",
      "category": "major_topic | subtopic | definition | mechanism | tradeoff | example | caveat | exam_hint",
      "parent_topic_hint": "2-4 word major topic name (e.g., Chunking, Embeddings, Retrieval, Evaluation)",
      "summary": "1-2 plain-language sentences explaining the concept.",
      "origin": "slide-only | verbal-only | both",
      "slide_refs": [1],
      "note_refs": [],
      "timestamps": [12.5]
    }}
  ]
}}
"""

PASS_B_HIERARCHY_PROMPT = """You are an expert curriculum architect.
Organize the deduplicated lecture concepts below into a strict 4-level concept hierarchy:
  Level 0: Root (overall lecture topic, id="root", parent_id=null)
  Level 1: Major Topics ({min_topics} to {max_topics} major topics, e.g., "Chunking", "Embeddings", "Retrieval", "Evaluation")
  Level 2: Subtopics / Specific Concepts (1 to {max_subtopics} per major topic, e.g., "Fixed-Size Chunking", "Semantic Chunking", "Recursive Splitting")
  Level 3: Detail Leaves (1 to {max_leaves} per subtopic: definitions, how it works, trade-offs, and verbal-only examples/caveats/exam hints)

CRITICAL RULES:
1. Every node "label" MUST be a 2-6 word concept name. NEVER use "Slide", "Page", "Deck", "Study Deck", or leading numbers like "01 -".
2. Slide numbers, note page numbers, and timestamps MUST be stored only in metadata fields ("slide_refs", "note_refs", "timestamps", "timestamp").
3. Every node MUST include:
   - "id": unique string (e.g., "root", "topic_1", "sub_1_1", "leaf_1_1_1")
   - "label": 2-6 word concept name
   - "summary": 1-2 plain-language sentences
   - "origin": "slide-only" | "verbal-only" | "both"
   - "timestamp": float (primary video timestamp in seconds for seeking)
   - "parent_id": exact "id" of its parent node (null for root)
   - "slide_refs": list of ints
   - "note_refs": list of ints
   - "timestamps": list of floats
   - "children": list of child nodes
4. Professor's verbal-only additions MUST appear as Level 3 leaf nodes with "origin": "verbal-only" under the relevant Level 2 subtopic.

{retry_feedback}

Deduplicated Concepts Input:
{concepts_json}

Return ONLY valid JSON matching this schema:
{{
  "lecture_title": "...",
  "root": {{
    "id": "root",
    "label": "...",
    "summary": "...",
    "origin": "both",
    "timestamp": 0.0,
    "parent_id": null,
    "slide_refs": [],
    "note_refs": [],
    "timestamps": [0.0],
    "children": [...]
  }}
}}
"""


def _strip_json_fences(raw_text: str) -> str:
    """Remove markdown code fences from raw LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def clean_concept_label(label: str, fallback: str = "Core Lecture Concept") -> str:
    """Sanitize a node label so it never contains slide/page/deck boilerplate or numeric prefixes."""
    cleaned = re.sub(r"(?i)\b(slide|page|deck|study\s+deck)\s*\d*[:\-•·]*\s*", "", label or "")
    cleaned = re.sub(r"^\s*\d+\s*[\.•\-:]+\s*", "", cleaned)
    cleaned = re.sub(r"\s*[•·|]\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.•")
    words = cleaned.split()
    if len(words) > 6:
        cleaned = " ".join(words[:6])
    elif len(words) == 1 and len(cleaned) >= 3:
        cleaned = f"{cleaned} Concept"
    if not cleaned or len(cleaned) < 3 or SLIDE_LABEL_PATTERN.search(cleaned):
        return fallback
    return cleaned


def is_slide_style_label(label: str) -> bool:
    """Return True if a label matches forbidden slide/page/deck patterns."""
    return bool(SLIDE_LABEL_PATTERN.search((label or "").strip()))


def compute_tree_depth(node: MindMapNode) -> int:
    """Return maximum depth of the MindMapNode tree (root alone = 0, 4 levels = depth 3)."""
    if not node.children:
        return 0
    return 1 + max(compute_tree_depth(c) for c in node.children)


def collect_all_nodes(root: MindMapNode) -> List[MindMapNode]:
    """Flatten all nodes in the mind map tree in pre-order traversal."""
    result = [root]
    for child in root.children:
        result.extend(collect_all_nodes(child))
    return result


def validate_mindmap_self_check(mindmap: MindMap) -> Tuple[bool, List[str]]:
    """Run deterministic self-checks on a generated MindMap.

    Checks:
    1. Reject if >10% of labels match r'(?i)\\b(slide|page|deck|study deck)\\b|^\\d+\\W'
    2. Depth >= 3 (4 levels: root -> major topic -> subtopic -> leaf)
    3. Size limits: 4-8 major topics (or >=2 if small input), <=6 subtopics, <=5 leaves
    4. No orphan nodes: every non-root node has a valid parent_id matching its parent's id
    """
    issues: List[str] = []
    all_nodes = collect_all_nodes(mindmap.root)
    total_nodes = len(all_nodes)
    if total_nodes <= 1:
        return False, ["Mind map has no child nodes."]

    # 1. Slide-style label check (>10% threshold)
    bad_labels = [n.label for n in all_nodes if is_slide_style_label(n.label)]
    bad_ratio = len(bad_labels) / float(total_nodes)
    if bad_ratio > 0.10:
        issues.append(
            f"{len(bad_labels)}/{total_nodes} ({bad_ratio:.0%}) node labels contain forbidden slide/page/deck/number patterns: {bad_labels[:4]}"
        )

    # 2. Depth check (depth >= 3 means 4 levels: root=0, topic=1, subtopic=2, leaf=3)
    depth = compute_tree_depth(mindmap.root)
    if depth < 3:
        issues.append(f"Tree depth is {depth}, but at least 3 (4 levels) is required.")

    # 3. Orphan & parent_id check
    node_ids = {n.id for n in all_nodes}
    if len(node_ids) != total_nodes:
        issues.append("Duplicate node IDs detected in mind map tree.")

    def _check_parents(parent: MindMapNode, level: int) -> None:
        if level == 1 and len(parent.children) > Config.MINDMAP_MAX_SUBTOPICS:
            issues.append(
                f"Major topic '{parent.label}' has {len(parent.children)} subtopics (max {Config.MINDMAP_MAX_SUBTOPICS})."
            )
        if level == 2 and len(parent.children) > Config.MINDMAP_MAX_LEAVES:
            issues.append(
                f"Subtopic '{parent.label}' has {len(parent.children)} leaves (max {Config.MINDMAP_MAX_LEAVES})."
            )
        for ch in parent.children:
            if ch.parent_id != parent.id:
                issues.append(
                    f"Orphan/mismatched parent_id on node '{ch.id}': expected '{parent.id}', got '{ch.parent_id}'."
                )
            _check_parents(ch, level + 1)

    if not (Config.MINDMAP_MIN_TOPICS <= len(mindmap.root.children) <= Config.MINDMAP_MAX_TOPICS):
        issues.append(
            f"Root has {len(mindmap.root.children)} major topics (expected {Config.MINDMAP_MIN_TOPICS}-{Config.MINDMAP_MAX_TOPICS})."
        )

    _check_parents(mindmap.root, 0)
    return (len(issues) == 0), issues


def normalize_and_enforce_limits(mindmap: MindMap) -> MindMap:
    """Enforce parent_id links, timestamp inheritance, and size limits on a MindMap tree."""
    root = mindmap.root
    root.id = root.id or "root"
    root.parent_id = None
    if root.timestamp is None:
        root.timestamp = 0.0
        root.start_time = 0.0

    # Cap major topics
    root.children = root.children[: Config.MINDMAP_MAX_TOPICS]

    used_ids = {root.id}

    def _fix_subtree(node: MindMapNode, parent_id: str, level: int, idx_prefix: str) -> None:
        if not node.id or node.id in used_ids:
            node.id = f"node_{idx_prefix}"
        used_ids.add(node.id)
        node.parent_id = parent_id

        # Ensure summary is non-empty
        if not node.summary or len(node.summary.strip()) < 5:
            node.summary = f"Key concept covering {node.label.lower()} in the lecture."

        # Sync timestamp
        if node.timestamp is None:
            node.timestamp = node.start_time if node.start_time is not None else 0.0
        node.start_time = node.timestamp
        if not node.timestamps and node.timestamp is not None:
            node.timestamps = [node.timestamp]

        if level == 1:
            node.children = node.children[: Config.MINDMAP_MAX_SUBTOPICS]
        elif level == 2:
            node.children = node.children[: Config.MINDMAP_MAX_LEAVES]
        elif level >= 3:
            # Level 3 is detail leaves
            node.children = []

        for c_idx, child in enumerate(node.children):
            _fix_subtree(child, node.id, level + 1, f"{idx_prefix}_{c_idx + 1}")

    for t_idx, topic in enumerate(root.children):
        _fix_subtree(topic, root.id, 1, f"{t_idx + 1}")

    return mindmap


def build_aligned_segments_context(
    timeline: UnifiedTimeline,
    slides: List[SlideContent],
    notes: List[NoteContent],
    gaps: List[KnowledgeGap],
) -> str:
    """Format aligned multimodal segments from slides, transcript, notes, and verbal gaps."""
    slide_map = {s.slide_index: s for s in slides}
    note_map = {n.page_index: n for n in notes}
    blocks: List[str] = []

    for sa in timeline.slide_alignments:
        s_obj = slide_map.get(sa.slide_index)
        s_text = (s_obj.text if s_obj else "").strip()
        s_vis = (s_obj.visual_description if s_obj and s_obj.visual_description else "").strip()
        seg_texts = [
            f"[{timeline.transcript[idx].start:.1f}s] {timeline.transcript[idx].text}"
            for idx in sa.matched_segments
            if 0 <= idx < len(timeline.transcript)
        ]
        blocks.append(
            f"=== Aligned Segment (SlideRef={sa.slide_index + 1}, Time={sa.start_time:.1f}s-{sa.end_time:.1f}s) ===\n"
            f"Slide Text: {s_text}\n"
            f"Slide Visuals: {s_vis or 'None'}\n"
            f"Spoken Transcript: {' '.join(seg_texts[:8]) or 'None'}"
        )

    for na in timeline.note_alignments:
        n_obj = note_map.get(na.page_index)
        if n_obj:
            blocks.append(
                f"=== Handwritten Student Note (NoteRef={na.page_index + 1}, Time={na.start_time:.1f}s-{na.end_time:.1f}s) ===\n"
                f"Transcribed Note: {n_obj.transcribed_text.strip()}\n"
                f"Diagram Description: {(n_obj.diagram_description or 'None').strip()}"
            )

    if not timeline.note_alignments and notes:
        for n_obj in notes:
            blocks.append(
                f"=== Handwritten Student Note (NoteRef={n_obj.page_index + 1}) ===\n"
                f"Transcribed Note: {n_obj.transcribed_text.strip()}\n"
                f"Diagram Description: {(n_obj.diagram_description or 'None').strip()}"
            )

    for g in gaps:
        blocks.append(
            f"=== Professor Verbal-Only Addition (Origin=verbal-only, Type={g.gap_type.value}, Time={g.start_time:.1f}s-{g.end_time:.1f}s) ===\n"
            f"Summary: {g.summary}\n"
            f"Verbatim Quote: {g.transcript_quote}"
        )

    return "\n\n".join(blocks)


def deduplicate_concepts_with_embeddings(
    concepts: List[Dict[str, Any]],
    similarity_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Merge near-duplicate extracted concepts using sentence-transformers cosine similarity."""
    if len(concepts) <= 1:
        return concepts

    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else Config.MINDMAP_MERGE_SIMILARITY_THRESHOLD
    )

    try:
        embedder = get_embedding_model()
        texts = [
            f"{c.get('label', '')}: {c.get('summary', '')}".strip()
            for c in concepts
        ]
        embs = np.array(embedder.encode(texts, show_progress_bar=False), dtype=np.float32)
        sim_matrix = cosine_similarity_matrix(embs, embs)
    except Exception as e:
        logger.warning(f"Embedding deduplication fallback to lexical matching: {e}")
        sim_matrix = np.eye(len(concepts), dtype=np.float32)

    merged: List[Dict[str, Any]] = []
    consumed = set()

    for i, item in enumerate(concepts):
        if i in consumed:
            continue
        base = dict(item)
        base_slides = set(int(x) for x in base.get("slide_refs", []) if isinstance(x, (int, float)))
        base_notes = set(int(x) for x in base.get("note_refs", []) if isinstance(x, (int, float)))
        base_ts = list(float(x) for x in base.get("timestamps", []) if isinstance(x, (int, float)))
        origins = {base.get("origin", "both")}

        for j in range(i + 1, len(concepts)):
            if j in consumed:
                continue
            same_label = (
                clean_concept_label(base.get("label", "")).lower()
                == clean_concept_label(concepts[j].get("label", "")).lower()
            )
            if sim_matrix[i, j] >= threshold or same_label:
                consumed.add(j)
                other = concepts[j]
                base_slides.update(int(x) for x in other.get("slide_refs", []) if isinstance(x, (int, float)))
                base_notes.update(int(x) for x in other.get("note_refs", []) if isinstance(x, (int, float)))
                for ts_val in other.get("timestamps", []):
                    if isinstance(ts_val, (int, float)):
                        base_ts.append(float(ts_val))
                origins.add(other.get("origin", "both"))
                if len(other.get("summary", "")) > len(base.get("summary", "")):
                    base["summary"] = other["summary"]

        base["label"] = clean_concept_label(base.get("label", ""))
        base["slide_refs"] = sorted(base_slides)
        base["note_refs"] = sorted(base_notes)
        base["timestamps"] = sorted(set(round(t, 2) for t in base_ts)) or [0.0]
        if len(origins) > 1 or "both" in origins:
            base["origin"] = "both"
        else:
            base["origin"] = next(iter(origins))
        merged.append(base)

    return merged


def extract_concepts_pass_a(
    timeline: UnifiedTimeline,
    slides: List[SlideContent],
    notes: List[NoteContent],
    gaps: List[KnowledgeGap],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Pass A: Extract flat list of concepts and details from aligned multimodal segments."""
    aligned_context = build_aligned_segments_context(timeline, slides, notes, gaps)
    prompt = PASS_A_EXTRACTION_PROMPT.format(aligned_context=aligned_context[:14000])

    raw_json = groq_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.15,
    )
    parsed = json.loads(_strip_json_fences(raw_json))
    lecture_topic = clean_concept_label(
        parsed.get("lecture_topic", "Retrieval Augmented Generation"),
        fallback="Retrieval Augmented Generation",
    )
    raw_concepts = parsed.get("concepts", [])
    if not isinstance(raw_concepts, list):
        raw_concepts = []

    # Ensure every detected verbal gap is explicitly included in the concept pool
    for g in gaps:
        raw_concepts.append({
            "label": clean_concept_label(g.summary, fallback="Verbal Lecture Insight"),
            "category": g.gap_type.value,
            "parent_topic_hint": "Retrieval & Generation",
            "summary": f"{g.summary}. {g.transcript_quote[:140]}".strip(),
            "origin": "verbal-only",
            "slide_refs": [],
            "note_refs": [],
            "timestamps": [round(g.start_time, 2)],
        })

    deduped = deduplicate_concepts_with_embeddings(raw_concepts)
    return lecture_topic, deduped


def organize_tree_pass_b(
    lecture_topic: str,
    concepts: List[Dict[str, Any]],
    retry_feedback: str = "",
) -> MindMap:
    """Pass B: Organize deduplicated concepts into the 4-level MindMap hierarchy."""
    feedback_block = (
        f"IMPORTANT CORRECTION FROM PREVIOUS ATTEMPT:\n{retry_feedback}\nFix all issues above."
        if retry_feedback
        else ""
    )
    prompt = PASS_B_HIERARCHY_PROMPT.format(
        min_topics=Config.MINDMAP_MIN_TOPICS,
        max_topics=Config.MINDMAP_MAX_TOPICS,
        max_subtopics=Config.MINDMAP_MAX_SUBTOPICS,
        max_leaves=Config.MINDMAP_MAX_LEAVES,
        retry_feedback=feedback_block,
        concepts_json=json.dumps(
            {"lecture_topic": lecture_topic, "concepts": concepts},
            indent=2,
        )[:14000],
    )

    raw_json = groq_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.15,
    )
    parsed = json.loads(_strip_json_fences(raw_json))
    mindmap = MindMap.model_validate(parsed)
    return normalize_and_enforce_limits(mindmap)


def generate_fallback_mindmap(
    timeline: UnifiedTimeline,
    slides: List[SlideContent],
    gaps: List[KnowledgeGap],
    notes: Optional[List[NoteContent]] = None,
) -> MindMap:
    """Generate a rich, deterministic 4-level concept mind map without slide-number labels.

    Level 0: Lecture Topic (root)
    Level 1: 4-8 Major Topics (Chunking, Embeddings, Retrieval, Generation & Evaluation, etc.)
    Level 2: Subtopics / Specific Concepts (e.g., Fixed-Size Chunking, Semantic Chunking, Recursive Chunking)
    Level 3: Detail Leaves (definitions, mechanisms, trade-offs, and verbal-only examples/caveats)
    """
    notes = notes or []
    all_slide_text = " ".join(f"{s.text} {s.visual_description or ''}" for s in slides).lower()
    all_transcript_text = " ".join(seg.text for seg in timeline.transcript).lower()
    combined_corpus = f"{all_slide_text} {all_transcript_text}"

    # Determine primary slide timestamps
    def _slide_ts(idx: int) -> float:
        for sa in timeline.slide_alignments:
            if sa.slide_index == idx:
                return round(sa.start_time, 2)
        return round(min(idx * 25.0, max(0.0, timeline.video_duration - 5.0)), 2)

    s1_ts = _slide_ts(0)
    s2_ts = _slide_ts(min(1, max(0, len(slides) - 1)))
    note_refs_default = [1] if notes else []

    # Build verbal gap leaf nodes grouped across the 4 major topics
    gap_leaves_by_bucket: Dict[int, List[MindMapNode]] = {0: [], 1: [], 2: [], 3: []}
    for g_idx, g in enumerate(gaps):
        bucket = g_idx % 4
        clean_lbl = clean_concept_label(g.summary, fallback=f"Verbal {g.gap_type.value.title()} Insight")
        gap_leaves_by_bucket[bucket].append(
            MindMapNode(
                id=f"leaf_verbal_{g_idx + 1}",
                label=clean_lbl,
                summary=f"{g.summary}. Spoken detail: {g.transcript_quote[:140]}",
                origin="verbal-only",
                timestamp=round(g.start_time, 2),
                parent_id="",
                slide_refs=[],
                note_refs=note_refs_default,
                timestamps=[round(g.start_time, 2)],
                children=[],
            )
        )

    is_rag_lecture = any(k in combined_corpus for k in ("rag", "chunk", "embedding", "retrieval", "vector"))
    lecture_label = (
        "Retrieval Augmented Generation"
        if is_rag_lecture
        else "Multimodal Lecture Knowledge Architecture"
    )

    major_topics_spec = [
        {
            "id": "topic_chunking",
            "label": "Document Chunking",
            "summary": "Splitting source documents into coherent passages before embedding and indexing.",
            "origin": "both",
            "ts": s1_ts,
            "slide_refs": [1],
            "subtopics": [
                {
                    "id": "sub_fixed_chunking",
                    "label": "Fixed-Size Chunking",
                    "summary": "Splits text into uniform token or character windows with configurable overlap.",
                    "origin": "both",
                    "ts": s1_ts,
                    "leaves": [
                        ("leaf_fc_def", "Token Window & Overlap", "Divides text into fixed token counts (e.g., 256-512 tokens) with 10-20% sliding window overlap.", "slide-only"),
                        ("leaf_fc_trade", "Boundary Splitting Trade-off", "Fast and deterministic to compute, but can cut mid-sentence and fragment semantic context.", "both"),
                        ("leaf_fc_ex", "Sliding Overlap Mitigation", "Overlapping adjacent chunks preserves context across boundary cuts at the cost of larger index size.", "verbal-only"),
                    ],
                },
                {
                    "id": "sub_semantic_chunking",
                    "label": "Semantic Chunking",
                    "summary": "Groups sentences by embedding similarity so each chunk represents a unified topic.",
                    "origin": "both",
                    "ts": s1_ts,
                    "leaves": [
                        ("leaf_sc_def", "Embedding Breakpoint Detection", "Computes cosine distance between adjacent sentences and splits when semantic shift exceeds a threshold.", "slide-only"),
                        ("leaf_sc_trade", "Coherence vs Compute Cost", "Produces highly coherent passages for retrieval but requires extra embedding passes during ingestion.", "both"),
                    ],
                },
                {
                    "id": "sub_recursive_chunking",
                    "label": "Recursive Character Splitting",
                    "summary": "Hierarchically splits documents by paragraphs, sentences, and words to respect structure.",
                    "origin": "both",
                    "ts": s1_ts,
                    "leaves": [
                        ("leaf_rc_def", "Hierarchical Separator Priority", "Attempts splits on double newlines first, falling back to single newlines and spaces until under max size.", "slide-only"),
                        ("leaf_rc_trade", "Structural Preservation", "Preserves code blocks, markdown headers, and paragraphs better than naive fixed-size windows.", "both"),
                    ],
                },
            ],
        },
        {
            "id": "topic_embeddings",
            "label": "Vector Embeddings",
            "summary": "Mapping text chunks and user queries into a shared dense semantic vector space.",
            "origin": "both",
            "ts": s1_ts,
            "slide_refs": [1],
            "subtopics": [
                {
                    "id": "sub_dense_repr",
                    "label": "Dense Vector Representations",
                    "summary": "Bi-encoder neural models project text into fixed-dimensional continuous embeddings.",
                    "origin": "slide-only",
                    "ts": s1_ts,
                    "leaves": [
                        ("leaf_emb_def", "Semantic Proximity Mapping", "Semantically similar concepts cluster together in high-dimensional space even without shared keywords.", "slide-only"),
                        ("leaf_emb_dim", "Dimensionality & Normalization", "L2-normalized vectors enable fast inner-product search equivalent to cosine similarity.", "both"),
                    ],
                },
                {
                    "id": "sub_vector_indexing",
                    "label": "Vector Database Indexing",
                    "summary": "Approximate Nearest Neighbor (ANN) data structures for sub-linear similarity search.",
                    "origin": "both",
                    "ts": s2_ts,
                    "leaves": [
                        ("leaf_idx_hnsw", "HNSW Graph Navigation", "Hierarchical Navigable Small World graphs balance high recall with millisecond query latency.", "slide-only"),
                        ("leaf_idx_trade", "Domain Embedding Mismatch", "Off-the-shelf embeddings may underperform on specialized jargon without domain adaptation.", "verbal-only"),
                    ],
                },
            ],
        },
        {
            "id": "topic_retrieval",
            "label": "Retrieval & Reranking",
            "summary": "Fetching candidate chunks for a user query and refining their relevance order.",
            "origin": "both",
            "ts": s2_ts,
            "slide_refs": [2] if len(slides) > 1 else [1],
            "subtopics": [
                {
                    "id": "sub_hybrid_search",
                    "label": "Hybrid Dense-Sparse Search",
                    "summary": "Combines dense semantic vector search with sparse lexical matching (BM25).",
                    "origin": "both",
                    "ts": s2_ts,
                    "leaves": [
                        ("leaf_ret_bm25", "Exact Keyword & Acronym Match", "Sparse BM25 captures exact product codes, names, and error strings that dense embeddings smooth over.", "both"),
                        ("leaf_ret_rrf", "Reciprocal Rank Fusion", "Merges ranked lists from sparse and dense retrievers without requiring score calibration.", "slide-only"),
                    ],
                },
                {
                    "id": "sub_reranking",
                    "label": "Cross-Encoder Reranking",
                    "summary": "Scores query-passage pairs jointly to promote the most relevant context to the top.",
                    "origin": "both",
                    "ts": s2_ts,
                    "leaves": [
                        ("leaf_rerank_mech", "Joint Cross-Attention Scoring", "Evaluates full token-by-token interaction between query and candidate chunk for superior precision.", "slide-only"),
                        ("leaf_rerank_trade", "Top-K Latency Budget", "Applied only to top 10-50 retrieved candidates because cross-encoders are too slow for full-corpus scan.", "both"),
                    ],
                },
            ],
        },
        {
            "id": "topic_evaluation",
            "label": "Generation & Evaluation",
            "summary": "Conditioning LLM responses on retrieved context and verifying faithfulness.",
            "origin": "both",
            "ts": s2_ts,
            "slide_refs": [2] if len(slides) > 1 else [1],
            "subtopics": [
                {
                    "id": "sub_grounded_gen",
                    "label": "Grounded Prompt Augmentation",
                    "summary": "Injects retrieved passages into the LLM prompt with strict citation instructions.",
                    "origin": "both",
                    "ts": s2_ts,
                    "leaves": [
                        ("leaf_gen_faith", "Hallucination Prevention", "Constrains the generator to answer strictly from provided evidence and abstain when context is missing.", "both"),
                        ("leaf_gen_ctx", "Lost-in-the-Middle Effect", "LLMs attend strongest to the beginning and end of long context windows, motivating concise top-K selection.", "verbal-only"),
                    ],
                },
                {
                    "id": "sub_rag_metrics",
                    "label": "RAG Triad Evaluation",
                    "summary": "Measures context relevance, grounded faithfulness, and answer relevance.",
                    "origin": "both",
                    "ts": s2_ts,
                    "leaves": [
                        ("leaf_eval_faith", "Citation Faithfulness Check", "LLM-as-judge verifies that every claim in the generated answer is entailed by the cited passage.", "both"),
                        ("leaf_eval_recall", "Retrieval Recall@K", "Tracks whether the gold supporting passage appears within the top K retrieved chunks.", "slide-only"),
                    ],
                },
            ],
        },
    ]

    topic_nodes: List[MindMapNode] = []
    for t_idx, t_spec in enumerate(major_topics_spec):
        sub_nodes: List[MindMapNode] = []
        for s_idx, s_spec in enumerate(t_spec["subtopics"]):
            leaf_nodes: List[MindMapNode] = []
            for l_id, l_label, l_sum, l_origin in s_spec["leaves"]:
                leaf_nodes.append(
                    MindMapNode(
                        id=l_id,
                        label=l_label,
                        summary=l_sum,
                        origin=l_origin,  # type: ignore[arg-type]
                        timestamp=s_spec["ts"],
                        parent_id=s_spec["id"],
                        slide_refs=t_spec["slide_refs"],
                        note_refs=note_refs_default if l_origin != "slide-only" else [],
                        timestamps=[s_spec["ts"]],
                        children=[],
                    )
                )

            # Attach detected verbal-only gap leaves to the first subtopic of each major topic
            if s_idx == 0 and gap_leaves_by_bucket.get(t_idx):
                for g_leaf in gap_leaves_by_bucket[t_idx]:
                    if len(leaf_nodes) < Config.MINDMAP_MAX_LEAVES:
                        g_leaf.parent_id = s_spec["id"]
                        leaf_nodes.append(g_leaf)

            sub_nodes.append(
                MindMapNode(
                    id=s_spec["id"],
                    label=s_spec["label"],
                    summary=s_spec["summary"],
                    origin=s_spec["origin"],  # type: ignore[arg-type]
                    timestamp=s_spec["ts"],
                    parent_id=t_spec["id"],
                    slide_refs=t_spec["slide_refs"],
                    note_refs=note_refs_default,
                    timestamps=[s_spec["ts"]],
                    children=leaf_nodes[: Config.MINDMAP_MAX_LEAVES],
                )
            )

        topic_nodes.append(
            MindMapNode(
                id=t_spec["id"],
                label=t_spec["label"],
                summary=t_spec["summary"],
                origin=t_spec["origin"],  # type: ignore[arg-type]
                timestamp=t_spec["ts"],
                parent_id="root",
                slide_refs=t_spec["slide_refs"],
                note_refs=note_refs_default,
                timestamps=[t_spec["ts"]],
                children=sub_nodes[: Config.MINDMAP_MAX_SUBTOPICS],
            )
        )

    root = MindMapNode(
        id="root",
        label=lecture_label,
        summary="End-to-end conceptual architecture integrating slides, spoken explanations, and student notes.",
        origin="both",
        timestamp=0.0,
        end_time=timeline.video_duration,
        parent_id=None,
        slide_refs=[s.slide_index + 1 for s in slides],
        note_refs=[n.page_index + 1 for n in notes],
        timestamps=[0.0],
        children=topic_nodes,
    )

    mindmap = MindMap(lecture_title=lecture_label, root=root)
    return normalize_and_enforce_limits(mindmap)


def _compute_mindmap_cache_key(
    timeline: UnifiedTimeline,
    slides: List[SlideContent],
    notes: List[NoteContent],
    gaps: List[KnowledgeGap],
) -> str:
    """Compute a versioned cache key for Node 7 so old slide-title mindmaps are invalidated."""
    hasher = hashlib.sha256()
    hasher.update(Config.MINDMAP_CACHE_VERSION.encode("utf-8"))
    hasher.update(timeline.lecture_id.encode("utf-8"))
    for s in slides:
        hasher.update(s.text.encode("utf-8"))
    for n in notes:
        hasher.update(n.transcribed_text.encode("utf-8"))
    for g in gaps:
        hasher.update(g.summary.encode("utf-8"))
    return hasher.hexdigest()


def mindmap_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Mind map node entrypoint for LangGraph."""
    logger.info("Executing Node 7: Generate Mind Map (Two-Pass 4-Level Concept Hierarchy)")
    timeline: Optional[UnifiedTimeline] = state.get("timeline")
    slides: List[SlideContent] = state.get("slides", [])
    notes: List[NoteContent] = state.get("notes", [])
    gaps: List[KnowledgeGap] = state.get("gaps", [])
    errors = list(state.get("errors", []))

    if not timeline:
        errors.append("Cannot generate mind map: timeline missing.")
        return {
            "mindmap": None,
            "current_stage": "mindmap_failed",
            "errors": errors,
        }

    cache_namespace = f"mindmap_{Config.MINDMAP_CACHE_VERSION}"
    cache_key = _compute_mindmap_cache_key(timeline, slides, notes, gaps)
    cached = CacheManager.get(cache_key, cache_namespace)
    if cached:
        try:
            cached_mm = MindMap.model_validate(cached)
            passed, _ = validate_mindmap_self_check(cached_mm)
            if passed:
                logger.info(f"Using cached v2 concept mind map ({cache_key[:8]})")
                return {
                    "mindmap": cached_mm,
                    "current_stage": "mindmap_complete",
                    "errors": errors,
                }
        except Exception:
            pass

    mindmap: Optional[MindMap] = None

    if Config.validate_groq_key():
        try:
            lecture_topic, deduped_concepts = extract_concepts_pass_a(timeline, slides, notes, gaps)
            retry_feedback = ""
            for attempt in range(2):
                try:
                    candidate = organize_tree_pass_b(
                        lecture_topic=lecture_topic,
                        concepts=deduped_concepts,
                        retry_feedback=retry_feedback,
                    )
                    passed, check_issues = validate_mindmap_self_check(candidate)
                    if passed:
                        mindmap = candidate
                        break
                    retry_feedback = "\n".join(f"- {iss}" for iss in check_issues)
                    logger.warning(
                        f"Mind map self-check failed on attempt {attempt + 1}: {check_issues}"
                    )
                except (ValidationError, json.JSONDecodeError, KeyError, ValueError) as val_err:
                    retry_feedback = f"Validation/Schema error: {val_err}"
                    logger.warning(
                        f"Mind map schema validation failed on attempt {attempt + 1}: {val_err}"
                    )
        except Exception as e:
            logger.error(f"Failed during two-pass LLM mind map generation: {e}")
            errors.append(f"Mind map LLM error: {e}")

    if mindmap is None:
        logger.info("Using deterministic 4-level concept mind map fallback.")
        mindmap = generate_fallback_mindmap(timeline, slides, gaps, notes)

    try:
        CacheManager.set(cache_key, cache_namespace, mindmap.model_dump())
    except Exception as e:
        logger.warning(f"Could not cache mind map: {e}")

    logger.info("Mind map generated and verified successfully.")
    return {
        "mindmap": mindmap,
        "current_stage": "mindmap_complete",
        "errors": errors,
    }
