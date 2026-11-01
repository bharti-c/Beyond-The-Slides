"""Unit tests for Concept Quiz generation, quality rubric, and targeted retry.

Verifies:
- Forbidden patterns ("Slide 2", "according to the presentation", "None of the above", etc.) are rejected
- Duplicate stems and duplicate concept_ids are rejected
- >= 30% (>= 3/10) verbal-only questions enforced
- Targeted retry in Node 9 + Node 10 only regenerates failing questions while leaving passing questions untouched
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from config import Config
from evals.evaluators import quiz_concept_rate
from models import (
    GapType,
    KnowledgeGap,
    NoteContent,
    Quiz,
    QuizCitation,
    QuizQuestion,
    SlideAlignment,
    SlideContent,
    TranscriptSegment,
    UnifiedTimeline,
)
from pipeline.node_07_generate_mindmap import generate_fallback_mindmap
from pipeline.node_09_generate_quiz import (
    generate_fallback_quiz,
    has_forbidden_quiz_text,
    quiz_node,
)
from pipeline.node_10_verify_citations import (
    evaluate_deterministic_quality,
    verify_citations_node,
)


def _sample_quiz_state():
    timeline = UnifiedTimeline(
        lecture_id="test_quiz_quality_lec",
        video_duration=120.0,
        transcript=[
            TranscriptSegment(
                start=0.0,
                end=30.0,
                text="Fixed-size chunking uses sliding window overlap to preserve semantic continuity across boundaries.",
            ),
            TranscriptSegment(
                start=30.0,
                end=60.0,
                text="In our telecom chatbot, hybrid search combines BM25 lexical matching with dense vector embeddings.",
            ),
        ],
        slide_alignments=[
            SlideAlignment(slide_index=0, start_time=0.0, end_time=30.0, confidence=0.9, matched_segments=[0]),
            SlideAlignment(slide_index=1, start_time=30.0, end_time=60.0, confidence=0.9, matched_segments=[1]),
        ],
        note_alignments=[],
    )
    slides = [
        SlideContent(slide_index=0, text="Chunking strategies: Fixed-size, semantic, and recursive splitting."),
        SlideContent(slide_index=1, text="Retrieval & Evaluation: Hybrid search, Cross-encoder reranking, RAG Triad."),
    ]
    notes = [
        NoteContent(
            page_index=0,
            image_path="sample_notes.png",
            transcribed_text="Cross-encoder reranking is applied only to top-K candidates due to quadratic attention cost.",
        )
    ]
    gaps = [
        KnowledgeGap(
            gap_id="gap_01",
            start_time=30.0,
            end_time=60.0,
            gap_type=GapType.EXAMPLE,
            summary="Telecom chatbot hybrid BM25 + dense search",
            transcript_quote="In our telecom chatbot, hybrid search combines BM25 lexical matching with dense vector embeddings.",
            context="Production RAG example.",
        )
    ]
    mindmap = generate_fallback_mindmap(timeline, slides, gaps, notes)
    return {
        "lecture_id": "test_quiz_quality_lec",
        "timeline": timeline,
        "slides": slides,
        "notes": notes,
        "gaps": gaps,
        "mindmap": mindmap,
        "retries": 0,
        "errors": [],
    }


def test_forbidden_patterns_rejected():
    """Verify stems and options mentioning slides, pages, decks, or 'None of the above' are flagged."""
    bad_q = QuizQuestion(
        id="q_bad",
        concept_id="sub_fixed_chunking",
        question_type="definition",
        question="According to the presentation slides (Slide 2), which of the following is a key point?",
        options=[
            "01 • FOUNDATIONS",
            "Fixed-size chunking",
            "In the material on page 4",
            "None of the above",
        ],
        correct_index=1,
        explanation="Explanation text.",
        difficulty="easy",
        is_verbal_only=False,
        citations=[QuizCitation(start=0.0, end=10.0, quote="Fixed-size chunking")],
    )

    hits = has_forbidden_quiz_text(bad_q)
    assert len(hits) >= 2
    qual = evaluate_deterministic_quality(bad_q)
    assert not qual.concept_based
    assert not qual.self_contained
    assert not qual.plausible_distractors
    assert not qual.passed


def test_duplicate_stems_and_concepts_rejected():
    """Verify duplicate stems or duplicate concept_ids fail the quality gate."""
    q1 = QuizQuestion(
        id="q_01",
        concept_id="sub_fixed_chunking",
        question_type="mechanism",
        question="How does token overlap between consecutive fixed-size chunks preserve semantic continuity?",
        options=["Option A", "Option B", "Option C", "Option D"],
        correct_index=0,
        explanation="Detailed explanation of why A is right and B is wrong.",
        difficulty="medium",
        is_verbal_only=False,
        citations=[QuizCitation(start=0.0, end=10.0, quote="Token overlap preserves context across boundaries.")],
    )
    q2_dup_stem = q1.model_copy(deep=True)
    q2_dup_stem.id = "q_02"
    q2_dup_stem.concept_id = "sub_semantic_chunking"

    seen_stems = set()
    seen_concepts = set()

    check1 = evaluate_deterministic_quality(q1, seen_stems=seen_stems, seen_concepts=seen_concepts)
    assert check1.passed

    check2 = evaluate_deterministic_quality(q2_dup_stem, seen_stems=seen_stems, seen_concepts=seen_concepts)
    assert not check2.not_duplicate
    assert not check2.passed


def test_fallback_quiz_meets_all_concept_and_verbal_rules():
    """Verify generated concept quiz has 10 unique questions, >=30% verbal-only, and zero forbidden phrases."""
    state = _sample_quiz_state()
    quiz = generate_fallback_quiz(
        slides=state["slides"],
        notes=state["notes"],
        gaps=state["gaps"],
        mindmap=state["mindmap"],
        timeline=state["timeline"],
    )

    assert len(quiz.questions) == 10
    verbal_count = sum(1 for q in quiz.questions if q.is_verbal_only)
    assert verbal_count >= 3, f"Expected >= 3 verbal-only questions, got {verbal_count}"

    stems = {q.question for q in quiz.questions}
    concepts = {q.concept_id for q in quiz.questions}
    assert len(stems) == 10, "All 10 question stems must be unique"
    assert len(concepts) == 10, "All 10 concept_ids must be unique"

    eval_res = quiz_concept_rate(quiz)
    assert eval_res["score"] == 1.0


def test_targeted_retry_only_touches_failing_questions(monkeypatch):
    """Verify Node 10 flags only failing questions and Node 9 retry regenerates ONLY those questions."""
    state = _sample_quiz_state()
    quiz = generate_fallback_quiz(
        slides=state["slides"],
        notes=state["notes"],
        gaps=state["gaps"],
        mindmap=state["mindmap"],
        timeline=state["timeline"],
    )

    # Corrupt ONLY q_03 and q_07 with slide-location stems and "None of the above"
    quiz.questions[2].question = "According to Slide 2, which of the following is a key point?"
    quiz.questions[2].options[3] = "None of the above"
    quiz.questions[6].question = quiz.questions[0].question  # Duplicate stem of q_01!

    # Snapshot passing questions before retry
    passing_ids = [q.id for q in quiz.questions if q.id not in ("q_03", "q_07")]
    passing_before = {q.id: q.question for q in quiz.questions if q.id in passing_ids}

    state["quiz"] = quiz
    monkeypatch.setattr(Config, "GROQ_API_KEY", "")

    # Run Node 10: should flag ONLY q_03 and q_07
    v_out = verify_citations_node(state)  # type: ignore[arg-type]
    assert v_out["current_stage"] == "verify_retry_needed"
    assert v_out["retries"] == 1
    failed_map = v_out["failed_question_issues"]
    assert set(failed_map.keys()) == {"q_03", "q_07"}

    # Now run Node 9 with mocked LLM returning targeted replacements for q_03 and q_07
    state.update(v_out)
    monkeypatch.setattr(Config, "GROQ_API_KEY", "gsk_test_valid_key_12345")

    mock_retry_payload = json.dumps({
        "questions": [
            {
                "id": "q_03",
                "concept_id": "sub_recursive_chunking_retry",
                "question_type": "scenario",
                "question": "When indexing markdown files with fenced code blocks, why is recursive splitting preferred over fixed token slicing?",
                "options": [
                    "It respects structural paragraph and code-block delimiters before falling back to finer splits",
                    "It compresses vector dimensions to 16 bits automatically",
                    "It skips indexing code blocks entirely during ingestion",
                    "It replaces dense embeddings with static regex rules",
                ],
                "correct_index": 0,
                "explanation": "Recursive splitting preserves natural document hierarchy, unlike fixed slicing which cuts mid-block.",
                "difficulty": "hard",
                "is_verbal_only": False,
                "citations": [{"start": 5.0, "end": 20.0, "quote": "Recursive splitting preserves structure."}],
            },
            {
                "id": "q_07",
                "concept_id": "sub_reranking_retry",
                "question_type": "comparison",
                "question": "How does a cross-encoder reranker achieve higher relevance precision than a bi-encoder on candidate passages?",
                "options": [
                    "By computing joint token-level cross-attention across the concatenated query and candidate passage",
                    "By hashing each document into a single 8-bit integer prior to query execution",
                    "By discarding the user query and ranking passages solely by character length",
                    "By restricting retrieval to documents containing fewer than twenty words",
                ],
                "correct_index": 0,
                "explanation": "Joint cross-attention models fine-grained query-passage interactions, unlike bi-encoders which compress each into a single vector.",
                "difficulty": "hard",
                "is_verbal_only": False,
                "citations": [{"start": 30.0, "end": 55.0, "quote": "Cross-encoder reranking scores query-passage pairs jointly."}],
            },
        ]
    })

    with patch("pipeline.node_09_generate_quiz.groq_client.chat_completion", return_value=mock_retry_payload) as mock_llm:
        q_out = quiz_node(state)  # type: ignore[arg-type]
        assert mock_llm.call_count == 1
        retried_quiz: Quiz = q_out["quiz"]

        # Verify the 8 passing questions were NOT touched at all
        for pid in passing_ids:
            q_after = next(q for q in retried_quiz.questions if q.id == pid)
            assert q_after.question == passing_before[pid]

        # Verify q_03 and q_07 were replaced with the clean concept questions
        q03_after = next(q for q in retried_quiz.questions if q.id == "q_03")
        q07_after = next(q for q in retried_quiz.questions if q.id == "q_07")
        assert "recursive splitting preferred" in q03_after.question
        assert "cross-encoder reranker achieve" in q07_after.question
        assert quiz_concept_rate(retried_quiz)["score"] == 1.0

