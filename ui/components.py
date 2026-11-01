"""Reusable UI components for Lecture-to-Living-Course.

Contains cards, badges, interactive quiz widgets, feedback controls,
and progress indicators styled in a clean, high-contrast theme without emojis.
"""

from __future__ import annotations

import streamlit as st
from typing import Dict, List, Optional
from models import KnowledgeGap, Quiz, QuizQuestion, SlideContent, UnifiedTimeline
from pipeline.graph import log_langsmith_feedback
from ui.video_player import format_seconds, render_timestamp_seek_button


def render_feedback_controls(run_id: str, artifact_name: str, key_prefix: str) -> None:
    """Render Helpful / Needs Work feedback buttons that log scores directly to LangSmith."""
    col1, col2, col3 = st.columns([6, 1.2, 1.2])
    with col1:
        st.caption(f"Was this {artifact_name} accurate and helpful?")
    with col2:
        if st.button("Helpful", key=f"{key_prefix}_helpful", help="Mark as helpful"):
            log_langsmith_feedback(run_id=run_id, key=f"{artifact_name}_feedback", score=1.0)
            st.toast("Thank you! Positive feedback recorded in LangSmith.")
    with col3:
        if st.button("Needs Work", key=f"{key_prefix}_unhelpful", help="Needs improvement"):
            log_langsmith_feedback(run_id=run_id, key=f"{artifact_name}_feedback", score=0.0)
            st.toast("Feedback recorded in LangSmith. We'll improve this artifact.")


def render_gap_card(gap: KnowledgeGap, run_id: str, index: int) -> None:
    """Render a single Knowledge Gap card for professor's verbal additions."""
    with st.container():
        st.markdown(
            f"""
            <div class="relect-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <div>
                        <strong style="font-size: 1.05rem; color: #0F172A;">{gap.summary}</strong>
                    </div>
                </div>
                <div class="citation-box">
                    <em>"{gap.transcript_quote}"</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_btn, _ = st.columns([2.5, 4.5])
        with col_btn:
            render_timestamp_seek_button(
                gap.start_time,
                label=f"Play at {format_seconds(gap.start_time)}",
                key=f"gap_seek_{index}_{gap.gap_id}",
            )


def render_quiz_tab(quiz: Quiz, run_id: str) -> None:
    """Render 10-question multiple choice quiz testing lecture concepts and verbal insights."""
    if not quiz or not quiz.questions:
        st.info("Quiz has not yet been generated.")
        return

    st.subheader(f"Beyond the slides Concept Quiz ({len(quiz.questions)} Questions)")
    st.caption("Self-contained conceptual questions across definitions, mechanisms, comparisons, scenarios, and trade-offs.")
    render_feedback_controls(run_id=run_id, artifact_name="quiz", key_prefix="quiz_tab")
    st.divider()

    for idx, q in enumerate(quiz.questions):
        st.markdown(f"### Question {idx + 1}")
        
        origin_badge = (
            '<span class="badge-caveat">Verbal-Only Insight</span>'
            if q.is_verbal_only
            else '<span class="badge-emphasis">Core Concept</span>'
        )
        qtype_badge = f"<span class='badge-correction'>{q.question_type}</span>"
        difficulty_color = {"easy": "#059669", "medium": "#D97706", "hard": "#DC2626"}.get(q.difficulty, "#2563EB")
        diff_badge = f"<span style='color: {difficulty_color}; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;'>[{q.difficulty}]</span>"

        st.markdown(f"{origin_badge} &nbsp; {qtype_badge} &nbsp; {diff_badge}", unsafe_allow_html=True)
        st.markdown(f"**{q.question}**")

        user_choice = st.radio(
            "Select an answer:",
            options=range(len(q.options)),
            format_func=lambda i: q.options[i],
            key=f"quiz_opt_{q.id}_{idx}",
            index=None,
        )

        with st.expander("Show Answer, Explanation & Verbatim Citation", expanded=False):
            if user_choice is not None:
                if user_choice == q.correct_index:
                    st.success("Correct!")
                else:
                    st.error(f"Incorrect. The correct answer was: {q.options[q.correct_index]}")

            st.markdown(f"**Explanation:** {q.explanation}")
            
            for cit in q.citations:
                st.markdown(
                    f"""
                    <div class="citation-box">
                        <strong>Supporting Lecture Evidence ({format_seconds(cit.start)} - {format_seconds(cit.end)}):</strong><br>
                        <em>"{cit.quote}"</em><br>
                        <small style="color: #059669; font-weight: 600;">Status: {q.verification_status.upper()} (Verified by LLM-as-Judge)</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.divider()
