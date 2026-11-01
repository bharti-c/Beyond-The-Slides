"""Unit tests for 4-level Concept Mind Map structure, self-checks, and retry logic.

Verifies:
- Tree depth >= 3 (4 levels: root -> major topic -> subtopic -> detail leaf)
- No slide-style labels (e.g., "Slide 14", "STUDY DECK", "01 • Foundations")
- No orphan nodes (every non-root node has a valid parent_id matching its parent)
- Size limits enforced (4-8 major topics, <=6 subtopics, <=5 leaves)
- Mocked two-pass LLM retry when Pass B returns slide-style labels
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from config import Config
from evals.evaluators import concept_label_rate
from models import (
    GapType,
    KnowledgeGap,
    MindMap,
    MindMapNode,
    NoteContent,
    SlideAlignment,
    SlideContent,
    TranscriptSegment,
    UnifiedTimeline,
)
from pipeline.node_07_generate_mindmap import (
    collect_all_nodes,
    compute_tree_depth,
    generate_fallback_mindmap,
    is_slide_style_label,
    mindmap_node,
    validate_mindmap_self_check,
)


def _sample_fixture_state():
    timeline = UnifiedTimeline(
        lecture_id="test_rag_mm",
        video_duration=120.0,
        transcript=[
            TranscriptSegment(
                start=0.0,
                end=30.0,
                text="Retrieval Augmented Generation splits documents into chunks using fixed-size or semantic chunking.",
            ),
            TranscriptSegment(
                start=30.0,
                end=60.0,
                text="In our telecom customer care chatbot, we use hybrid search and cross-encoder reranking.",
            ),
        ],
        slide_alignments=[
            SlideAlignment(slide_index=0, start_time=0.0, end_time=30.0, confidence=0.9, matched_segments=[0]),
            SlideAlignment(slide_index=1, start_time=30.0, end_time=60.0, confidence=0.9, matched_segments=[1]),
        ],
        note_alignments=[],
    )
    slides = [
        SlideContent(slide_index=0, text="Chunking & Embeddings: Fixed-size, semantic, recursive splitting"),
        SlideContent(slide_index=1, text="Retrieval & Evaluation: Hybrid BM25 + Dense search, RAG Triad"),
    ]
    notes = [
        NoteContent(
            page_index=0,
            image_path="sample_notes.png",
            transcribed_text="Use 15% chunk overlap to preserve context across boundaries.",
        )
    ]
    gaps = [
        KnowledgeGap(
            gap_id="gap_01",
            start_time=30.0,
            end_time=60.0,
            gap_type=GapType.EXAMPLE,
            summary="Telecom chatbot hybrid retrieval latency",
            transcript_quote="In our telecom customer care chatbot, we use hybrid search and cross-encoder reranking.",
            context="Real-world production architecture.",
        )
    ]
    return {
        "lecture_id": "test_rag_mm",
        "timeline": timeline,
        "slides": slides,
        "notes": notes,
        "gaps": gaps,
        "errors": [],
    }


def test_mindmap_depth_no_orphans_and_size_limits():
    """Verify fallback and validated mind maps have depth >= 3, no orphans, and obey size limits."""
    state = _sample_fixture_state()
    mm = generate_fallback_mindmap(
        timeline=state["timeline"],
        slides=state["slides"],
        gaps=state["gaps"],
        notes=state["notes"],
    )

    # 1. Depth >= 3 (4 levels: root -> major topic -> subtopic -> detail leaf)
    depth = compute_tree_depth(mm.root)
    assert depth >= 3, f"Expected tree depth >= 3, got {depth}"

    # 2. Size limits
    assert Config.MINDMAP_MIN_TOPICS <= len(mm.root.children) <= Config.MINDMAP_MAX_TOPICS
    for topic in mm.root.children:
        assert 1 <= len(topic.children) <= Config.MINDMAP_MAX_SUBTOPICS
        for sub in topic.children:
            assert 1 <= len(sub.children) <= Config.MINDMAP_MAX_LEAVES

    # 3. No orphan nodes (every non-root node has valid parent_id)
    all_nodes = collect_all_nodes(mm.root)
    id_map = {n.id: n for n in all_nodes}
    assert mm.root.parent_id is None
    for n in all_nodes:
        if n.id != "root":
            assert n.parent_id is not None, f"Node {n.id} is missing parent_id"
            assert n.parent_id in id_map, f"Node {n.id} has orphan parent_id {n.parent_id}"
            parent = id_map[n.parent_id]
            assert any(ch.id == n.id for ch in parent.children)

    # 4. No slide-style labels
    for n in all_nodes:
        assert not is_slide_style_label(n.label), f"Forbidden slide-style label found: {n.label}"
        assert len(n.summary.strip()) > 0
        assert n.origin in ("slide-only", "verbal-only", "both")

    # 5. Evaluator concept_label_rate == 1.0
    eval_res = concept_label_rate(mm)
    assert eval_res["score"] == 1.0

    # 6. Check that "Document Chunking" has Fixed-Size / Semantic / Recursive children
    chunking_topic = next((t for t in mm.root.children if "chunking" in t.label.lower()), None)
    assert chunking_topic is not None
    sub_labels = " ".join(s.label.lower() for s in chunking_topic.children)
    assert "fixed-size" in sub_labels
    assert "semantic" in sub_labels
    assert "recursive" in sub_labels


def test_self_check_rejects_slide_style_labels():
    """Verify validate_mindmap_self_check rejects trees where >10% of labels look like slide titles."""
    bad_root = MindMapNode(
        id="root",
        label="Slide 1: RAG BASICS • STUDY DECK",
        summary="Root summary",
        origin="both",
        timestamp=0.0,
        parent_id=None,
        children=[
            MindMapNode(
                id=f"t_{i}",
                label=f"Slide {i + 2}: 01 • FOUNDATIONS",
                summary="Topic summary",
                origin="slide-only",
                timestamp=10.0,
                parent_id="root",
                children=[
                    MindMapNode(
                        id=f"s_{i}",
                        label=f"Page {i}: Study Deck Section",
                        summary="Sub summary",
                        origin="slide-only",
                        timestamp=10.0,
                        parent_id=f"t_{i}",
                        children=[
                            MindMapNode(
                                id=f"l_{i}",
                                label="Valid Concept Leaf",
                                summary="Leaf summary",
                                origin="both",
                                timestamp=10.0,
                                parent_id=f"s_{i}",
                                children=[],
                            )
                        ],
                    )
                ],
            )
            for i in range(4)
        ],
    )
    bad_mm = MindMap(lecture_title="Bad Map", root=bad_root)
    passed, issues = validate_mindmap_self_check(bad_mm)
    assert not passed
    assert any("slide/page/deck/number" in iss for iss in issues)


def test_mindmap_node_two_pass_llm_with_retry_on_bad_labels(monkeypatch):
    """Verify mindmap_node retries Pass B when first LLM response has slide-style labels."""
    state = _sample_fixture_state()
    state["lecture_id"] = "test_mm_retry_mock"

    monkeypatch.setattr(Config, "GROQ_API_KEY", "gsk_test_valid_key_12345")

    pass_a_response = json.dumps({
        "lecture_topic": "Retrieval Augmented Generation",
        "concepts": [
            {
                "label": "Fixed-Size Chunking",
                "category": "subtopic",
                "parent_topic_hint": "Document Chunking",
                "summary": "Splits text into fixed token windows with sliding overlap.",
                "origin": "both",
                "slide_refs": [1],
                "note_refs": [1],
                "timestamps": [5.0],
            }
        ],
    })

    # First Pass B attempt has forbidden "Slide 14" labels -> should trigger retry!
    bad_pass_b_response = json.dumps({
        "lecture_title": "RAG Course",
        "root": {
            "id": "root",
            "label": "Slide 14 RAG BASICS • STUDY DECK",
            "summary": "Overview",
            "origin": "both",
            "timestamp": 0.0,
            "parent_id": None,
            "children": [],
        },
    })

    # Second Pass B attempt returns a valid 4-level concept tree
    valid_mm = generate_fallback_mindmap(
        state["timeline"], state["slides"], state["gaps"], state["notes"]
    )
    good_pass_b_response = json.dumps(valid_mm.model_dump())

    call_responses = [pass_a_response, bad_pass_b_response, good_pass_b_response]

    with patch("pipeline.node_07_generate_mindmap.groq_client.chat_completion", side_effect=call_responses) as mock_llm:
        out = mindmap_node(state)  # type: ignore[arg-type]
        assert mock_llm.call_count == 3  # 1 Pass A + 2 Pass B attempts
        mm = out["mindmap"]
        assert mm is not None
        passed, issues = validate_mindmap_self_check(mm)
        assert passed, f"Expected valid mindmap after retry, got issues: {issues}"

