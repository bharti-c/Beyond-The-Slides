"""Node 10: Verify Citations & Quiz Quality Gate.

Acts as an LLM-as-judge + deterministic quality gate to verify:
1. Citation verification: whether the cited excerpt supports the correct answer
   ('supported' | 'partially' | 'unsupported').
2. Quiz quality rubric per question:
   {
     "concept_based": bool,
     "self_contained": bool,
     "single_correct_answer": bool,
     "plausible_distractors": bool,
     "not_duplicate": bool,
     "answerable_from_citation": bool,
     "issues": [str, ...]
   }
3. Deterministic regex check for forbidden slide/page/deck/"None of the above" phrases
   and duplicate stems / concept_ids.
4. Triggers a targeted retry loop (max 2 retries) passing `failed_question_issues` back
   to `generate_quiz` so ONLY failing questions are regenerated.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Set

from langsmith import Client

from config import Config
from groq_client import groq_client
from models import (
    PipelineGraphState,
    Quiz,
    QuizQualityCheck,
    QuizQuestion,
    VerificationResult,
)
from pipeline.node_09_generate_quiz import has_forbidden_quiz_text

logger = logging.getLogger(__name__)

CITATION_AND_QUALITY_JUDGE_PROMPT = """You are an exacting academic exam reviewer and LLM-as-judge.
Evaluate the multiple-choice question below on BOTH citation support and conceptual question quality.

Question ID: {question_id}
Concept ID: {concept_id}
Question Stem: {question}
Options:
0: {opt_0}
1: {opt_1}
2: {opt_2}
3: {opt_3}
Marked Correct Index: {correct_index} ({correct_answer})
Explanation: {explanation}
Cited Quote: \"\"\"{quote}\"\"\"

Evaluate against this strict rubric:
1. "status":
   - "supported": The cited quote directly supports the correct answer.
   - "partially": The cited quote is on-topic and relevant to the answer.
   - "unsupported": The quote is missing, irrelevant, or contradicts the answer.
2. "concept_based" (bool): True ONLY if the question tests understanding of a domain concept/mechanism/trade-off, NOT where something appeared on a slide or page.
3. "self_contained" (bool): True ONLY if a student who never saw the slides can understand what is being asked.
4. "single_correct_answer" (bool): True ONLY if there is a single defensible correct option.
5. "plausible_distractors" (bool): True ONLY if all distractors are plausible and free of "None of the above" / "All of the above" / joke options.
6. "not_duplicate" (bool): True if the stem is distinct from other questions.
7. "answerable_from_citation" (bool): True if the question is supported by the cited excerpt.
8. "issues" (array of strings): List specific flaws if any boolean above is false or status is "unsupported" (empty array [] if everything passes).

Return ONLY a valid JSON object matching this schema:
{{
  "status": "supported" | "partially" | "unsupported",
  "reasoning": "Concise explanation of citation support and quality.",
  "concept_based": true,
  "self_contained": true,
  "single_correct_answer": true,
  "plausible_distractors": true,
  "not_duplicate": true,
  "answerable_from_citation": true,
  "issues": []
}}
"""


def _strip_json_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def evaluate_deterministic_quality(
    q: QuizQuestion,
    seen_stems: Optional[Set[str]] = None,
    seen_concepts: Optional[Set[str]] = None,
) -> QuizQualityCheck:
    """Run deterministic regex and uniqueness checks on a single question."""
    issues: List[str] = []
    forbidden_hits = has_forbidden_quiz_text(q)
    concept_based = len(forbidden_hits) == 0
    self_contained = len(forbidden_hits) == 0
    issues.extend(forbidden_hits)

    # Check distractors for "None/All of the above" or empty options
    plausible_distractors = True
    for idx, opt in enumerate(q.options):
        if re.search(r"(?i)\b(none|all)\s+of\s+the\s+above\b", opt or ""):
            plausible_distractors = False
            issues.append(f"Option {idx + 1} uses forbidden 'None/All of the above'.")
        if len((opt or "").strip()) < 3:
            plausible_distractors = False
            issues.append(f"Option {idx + 1} is too short or empty.")

    single_correct = 0 <= q.correct_index < len(q.options) and len(set(q.options)) == 4
    if not single_correct:
        issues.append("Options must contain 4 distinct choices with valid correct_index.")

    not_duplicate = True
    norm_stem = re.sub(r"\W+", " ", (q.question or "").lower()).strip()
    if seen_stems is not None:
        if norm_stem in seen_stems:
            not_duplicate = False
            issues.append("Duplicate question stem detected in quiz.")
        seen_stems.add(norm_stem)

    cid = (q.concept_id or "").strip()
    if seen_concepts is not None and cid:
        if cid in seen_concepts:
            not_duplicate = False
            issues.append(f"Duplicate concept_id '{cid}' tested by multiple questions.")
        seen_concepts.add(cid)

    has_citation = bool(q.citations and any(len(c.quote.strip()) > 5 for c in q.citations))
    if not has_citation:
        issues.append("Missing or empty citation quote.")

    return QuizQualityCheck(
        question_id=q.id,
        concept_based=concept_based,
        self_contained=self_contained,
        single_correct_answer=single_correct,
        plausible_distractors=plausible_distractors,
        not_duplicate=not_duplicate,
        answerable_from_citation=has_citation,
        issues=issues,
    )


def verify_single_question(
    q: QuizQuestion,
    seen_stems: Optional[Set[str]] = None,
    seen_concepts: Optional[Set[str]] = None,
) -> VerificationResult:
    """Evaluate a single question's citation support and conceptual quality rubric."""
    det_check = evaluate_deterministic_quality(q, seen_stems=seen_stems, seen_concepts=seen_concepts)

    if not q.citations:
        det_check.answerable_from_citation = False
        if "Question has no citations." not in det_check.issues:
            det_check.issues.append("Question has no citations.")
        return VerificationResult(
            question_id=q.id,
            status="unsupported",
            reasoning="Question has no citations.",
            quality=det_check,
        )

    correct_option = q.options[q.correct_index] if 0 <= q.correct_index < len(q.options) else ""
    combined_quote = " ".join(c.quote for c in q.citations).strip()

    if not Config.validate_groq_key():
        status_val: Literal["supported", "partially", "unsupported"] = (
            "supported" if det_check.answerable_from_citation else "unsupported"
        )
        return VerificationResult(
            question_id=q.id,
            status=status_val,
            reasoning="Deterministic quality & citation verification complete.",
            quality=det_check,
        )

    prompt = CITATION_AND_QUALITY_JUDGE_PROMPT.format(
        question_id=q.id,
        concept_id=q.concept_id or "unspecified",
        question=q.question,
        opt_0=q.options[0] if len(q.options) > 0 else "",
        opt_1=q.options[1] if len(q.options) > 1 else "",
        opt_2=q.options[2] if len(q.options) > 2 else "",
        opt_3=q.options[3] if len(q.options) > 3 else "",
        correct_index=q.correct_index,
        correct_answer=correct_option,
        explanation=q.explanation,
        quote=combined_quote,
    )

    try:
        raw_json = groq_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = json.loads(_strip_json_fences(raw_json))
        status_val_str = str(parsed.get("status", "supported")).lower()
        if status_val_str not in ("supported", "partially", "unsupported"):
            status_val_str = "supported"

        llm_issues = [str(x) for x in parsed.get("issues", []) if x]
        merged_issues = list(dict.fromkeys(det_check.issues + llm_issues))

        concept_based = det_check.concept_based and bool(parsed.get("concept_based", True))
        self_contained = det_check.self_contained and bool(parsed.get("self_contained", True))
        single_correct = det_check.single_correct_answer and bool(parsed.get("single_correct_answer", True))
        plausible_dist = det_check.plausible_distractors and bool(parsed.get("plausible_distractors", True))
        not_dup = det_check.not_duplicate and bool(parsed.get("not_duplicate", True))
        answerable = (
            det_check.answerable_from_citation
            and bool(parsed.get("answerable_from_citation", True))
            and status_val_str != "unsupported"
        )

        if not (concept_based and self_contained and single_correct and plausible_dist and not_dup and answerable):
            if not merged_issues:
                merged_issues.append("Failed one or more LLM-as-judge quality rubric checks.")

        quality_obj = QuizQualityCheck(
            question_id=q.id,
            concept_based=concept_based,
            self_contained=self_contained,
            single_correct_answer=single_correct,
            plausible_distractors=plausible_dist,
            not_duplicate=not_dup,
            answerable_from_citation=answerable,
            issues=merged_issues,
        )

        return VerificationResult(
            question_id=q.id,
            status=status_val_str,  # type: ignore[arg-type]
            reasoning=str(parsed.get("reasoning", "")).strip(),
            quality=quality_obj,
        )
    except Exception as e:
        logger.warning(f"Verification LLM call failed on question {q.id}: {e}")
        return VerificationResult(
            question_id=q.id,
            status="supported" if det_check.passed else "unsupported",
            reasoning=f"Verification fallback to deterministic checks: {e}",
            quality=det_check,
        )


def verify_citations_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Verify citations and quiz quality gate entrypoint for LangGraph."""
    logger.info("Executing Node 10: Verify Citations & Quiz Quality Gate")
    quiz: Optional[Quiz] = state.get("quiz")
    errors = list(state.get("errors", []))
    retries = state.get("retries", 0)
    run_id = state.get("run_id")

    if not quiz or not quiz.questions:
        errors.append("Cannot verify citations: quiz missing.")
        return {
            "current_stage": "verify_citations_failed",
            "errors": errors,
        }

    verified_questions: List[QuizQuestion] = []
    failed_question_issues: Dict[str, List[str]] = {}
    quality_checks_list: List[Dict[str, Any]] = []

    seen_stems: Set[str] = set()
    seen_concepts: Set[str] = set()

    for q in quiz.questions:
        result = verify_single_question(q, seen_stems=seen_stems, seen_concepts=seen_concepts)
        q.verification_status = result.status
        q_qual = result.quality or evaluate_deterministic_quality(q)
        q.quality_issues = list(q_qual.issues)
        quality_checks_list.append(q_qual.model_dump())

        if result.status == "unsupported" or not q_qual.passed:
            reasons = list(q_qual.issues)
            if result.status == "unsupported" and result.reasoning:
                reasons.append(result.reasoning)
            failed_question_issues[q.id] = reasons or ["Failed quality or citation check."]
            logger.info(f"Question {q.id} flagged by quality/citation gate: {failed_question_issues[q.id]}")

        verified_questions.append(q)

    total_q = max(1, len(verified_questions))
    passed_count = total_q - len(failed_question_issues)
    quality_report = {
        "total_questions": len(verified_questions),
        "passed_questions": passed_count,
        "failed_question_ids": list(failed_question_issues.keys()),
        "quiz_concept_rate": round(passed_count / total_q, 3),
        "citation_support_rate": round(
            sum(1 for q in verified_questions if q.verification_status in ("supported", "partially")) / total_q,
            3,
        ),
        "verbal_only_rate": round(
            sum(1 for q in verified_questions if q.is_verbal_only) / total_q,
            3,
        ),
        "per_question_checks": quality_checks_list,
    }

    # Log quality scores to LangSmith as run metadata if configured
    if run_id and Config.validate_langsmith_key():
        try:
            client = Client(api_key=Config.LANGSMITH_API_KEY)
            client.update_run(
                run_id=run_id,
                metadata={
                    "quiz_concept_rate": quality_report["quiz_concept_rate"],
                    "citation_support_rate": quality_report["citation_support_rate"],
                    "verbal_only_rate": quality_report["verbal_only_rate"],
                    "quiz_quality_passed": passed_count,
                },
            )
        except Exception as e:
            logger.debug(f"Could not log quiz quality metadata to LangSmith: {e}")

    # Trigger targeted retry if any question failed and retries < MAX_CITATIONS_VERIFICATION_RETRIES
    if failed_question_issues and retries < Config.MAX_CITATIONS_VERIFICATION_RETRIES:
        logger.warning(
            f"Found {len(failed_question_issues)} failing questions ({list(failed_question_issues.keys())}). "
            f"Triggering targeted retry #{retries + 1}."
        )
        return {
            "quiz": Quiz(questions=verified_questions),
            "retries": retries + 1,
            "failed_question_issues": failed_question_issues,
            "quality_report": quality_report,
            "current_stage": "verify_retry_needed",
            "errors": errors,
        }

    # If retries exhausted, keep supported/passing questions (or all if < 5 pass)
    final_questions = [
        q for q in verified_questions if q.verification_status != "unsupported" and not q.quality_issues
    ]
    if len(final_questions) < 5:
        final_questions = verified_questions

    logger.info(
        f"Verification & quality gate complete. Passing: {passed_count}/{len(verified_questions)}."
    )
    return {
        "quiz": Quiz(questions=final_questions),
        "failed_question_issues": {},
        "quality_report": quality_report,
        "current_stage": "pipeline_complete",
        "errors": errors,
    }


def should_retry_quiz(state: PipelineGraphState) -> Literal["retry", "end"]:
    """Conditional edge router for LangGraph to decide if quiz needs targeted regeneration."""
    stage = state.get("current_stage", "")
    retries = state.get("retries", 0)
    if stage == "verify_retry_needed" and retries <= Config.MAX_CITATIONS_VERIFICATION_RETRIES:
        return "retry"
    return "end"
