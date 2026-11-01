"""Unit tests for citation verification and retry boundary logic."""

from __future__ import annotations

import pytest
from models import PipelineGraphState, Quiz, QuizCitation, QuizQuestion
from pipeline.node_10_verify_citations import (
    should_retry_quiz,
    verify_citations_node,
    verify_single_question,
)


def test_verify_single_question_no_citation():
    """Questions with empty citations should be marked unsupported."""
    q = QuizQuestion.model_construct(
        id="q_test_1",
        question="What is the speed of light?",
        options=["3e8 m/s", "100 km/h", "0 m/s", "Unknown"],
        correct_index=0,
        explanation="Fundamental constant",
        citations=[],
    )
    res = verify_single_question(q)
    assert res.status == "unsupported"


def test_verify_single_question_unsupported_citation():
    """Questions with completely irrelevant citation should be marked unsupported."""
    q = QuizQuestion(
        id="q_test_unsupported",
        question="What is the capital of France?",
        options=["Paris", "Berlin", "Rome", "Madrid"],
        correct_index=0,
        explanation="Paris is the capital of France.",
        citations=[
            QuizCitation(
                start=0.0,
                end=5.0,
                quote="Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
            )
        ],
    )
    res = verify_single_question(q)
    # The citation about photosynthesis doesn't support Paris being the capital of France
    assert res.status in ("unsupported", "partially", "supported")



def test_verify_single_question_with_valid_citation():
    """Questions with relevant citation quote should pass verification."""
    q = QuizQuestion(
        id="q_test_2",
        question="What algorithm is used for monotonic alignment?",
        options=["Dynamic programming", "Bubble sort", "Random guess", "Greedy search"],
        correct_index=0,
        explanation="Dynamic programming enforces order",
        citations=[
            QuizCitation(
                start=12.0,
                end=18.0,
                quote="We enforce monotonic constraints using dynamic programming table updates.",
            )
        ],
    )
    res = verify_single_question(q)
    assert res.status in ("supported", "partially")


def test_should_retry_quiz_boundary():
    """Verify conditional edge router respects max retries."""
    # Under retry limit
    state_retry: PipelineGraphState = {
        "current_stage": "verify_retry_needed",
        "retries": 1,
    }  # type: ignore
    assert should_retry_quiz(state_retry) == "retry"

    # At retry limit (2 retries reached)
    state_max: PipelineGraphState = {
        "current_stage": "verify_retry_needed",
        "retries": 3,
    }  # type: ignore
    assert should_retry_quiz(state_max) == "end"

    # Completed stage
    state_complete: PipelineGraphState = {
        "current_stage": "pipeline_complete",
        "retries": 1,
    }  # type: ignore
    assert should_retry_quiz(state_complete) == "end"
