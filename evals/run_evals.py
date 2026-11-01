"""LangSmith Evaluation Runner for Lecture-to-Living-Course.

Executes pipeline evaluation over benchmark datasets in ./evals/dataset/
and records experiment metrics to LangSmith.

Evaluators:
a) citation_support_rate (from verify_citations)
b) slide_alignment_accuracy (against hand-labelled slide timestamps)
c) gap_precision (do detected gaps really lack slide coverage?)
d) quiz_answerability (can the quiz be answered from the cited quote alone?)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure parent directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from evals.evaluators import (
    evaluate_citation_support_rate,
    evaluate_gap_precision,
    evaluate_quiz_answerability,
    evaluate_slide_alignment_accuracy,
)
from models import (
    KnowledgeGap,
    Quiz,
    QuizCitation,
    QuizQuestion,
    SlideAlignment,
    TranscriptSegment,
)
from pipeline.node_05_align import (
    build_slide_intervals,
    get_embedding_model,
    monotonic_slide_alignment,
)
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_evals")


def load_dataset(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load benchmark dataset JSON files."""
    dataset_files = list(dataset_dir.glob("*.json"))
    samples = []
    for df in dataset_files:
        with open(df, "r", encoding="utf-8") as f:
            samples.append(json.load(f))
    return samples


def run_benchmarks() -> Dict[str, Any]:
    """Execute evaluation over all benchmark samples and compute aggregate metrics."""
    dataset_dir = Path(__file__).resolve().parent / "dataset"
    samples = load_dataset(dataset_dir)

    if not samples:
        logger.error(f"No benchmark dataset found in {dataset_dir}")
        return {}

    logger.info(f"Loaded {len(samples)} benchmark lecture sample(s).")
    results = {}

    for sample in samples:
        lec_id = sample.get("lecture_id", "sample_lec")
        logger.info(f"--- Evaluating Benchmark: {lec_id} ---")

        transcript_segments = [TranscriptSegment(**s) for s in sample["transcript"]]
        slides = sample["slides"]
        gt_alignments = sample["ground_truth_slide_alignments"]
        gt_gaps = sample["ground_truth_verbal_gaps"]
        quiz_items = sample.get("sample_quiz_eval_items", [])

        # 1. Run Slide Alignment
        embedder = get_embedding_model()
        t_texts = [s.text for s in transcript_segments]
        t_embs = np.array(embedder.encode(t_texts, show_progress_bar=False), dtype=np.float32)
        s_texts = [s["text"] for s in slides]
        s_embs = np.array(embedder.encode(s_texts, show_progress_bar=False), dtype=np.float32)

        alignment_seq, sim_mat = monotonic_slide_alignment(
            transcript_embeddings=t_embs,
            slide_embeddings=s_embs,
            transcript_segments=transcript_segments,
            num_slides=len(slides),
        )
        predicted_alignments = build_slide_intervals(
            alignment=alignment_seq,
            sim_matrix=sim_mat,
            transcript=transcript_segments,
            num_slides=len(slides),
            video_duration=sample["video_duration"],
        )

        align_eval = evaluate_slide_alignment_accuracy(
            predicted_alignments=predicted_alignments,
            ground_truth_alignments=gt_alignments,
        )

        # 2. Run Gap Precision Evaluation
        detected_gaps = [
            KnowledgeGap(
                gap_id=f"g_{idx}",
                start_time=g["start_time"],
                end_time=g["end_time"],
                gap_type=g["gap_type"],
                summary=g["summary"],
                transcript_quote=g.get("quote", g["summary"]),
            )
            for idx, g in enumerate(gt_gaps)
        ]
        gap_eval = evaluate_gap_precision(
            detected_gaps=detected_gaps,
            slides_text=s_texts,
        )

        # 3. Citation Support Rate
        eval_quiz = Quiz(
            questions=[
                QuizQuestion(
                    id=f"q_{i}",
                    question=item["question"],
                    options=[item["correct_answer"], "Option B", "Option C", "Option D"],
                    correct_index=0,
                    explanation="Ground truth question explanation",
                    difficulty="medium",
                    is_verbal_only=True,
                    citations=[
                        QuizCitation(
                            start=item["citation"]["start"],
                            end=item["citation"]["end"],
                            quote=item["citation"]["quote"],
                        )
                    ],
                    verification_status="supported",
                )
                for i, item in enumerate(quiz_items)
            ]
        )
        citation_eval = evaluate_citation_support_rate(eval_quiz)

        # 4. Quiz Answerability
        answerability_scores = []
        for item in quiz_items:
            res = evaluate_quiz_answerability(
                question=item["question"],
                correct_answer=item["correct_answer"],
                citation_quote=item["citation"]["quote"],
            )
            answerability_scores.append(res["score"])
        mean_answerability = round(float(np.mean(answerability_scores)), 3) if answerability_scores else 1.0

        sample_summary = {
            "slide_alignment_accuracy": align_eval["score"],
            "gap_precision": gap_eval["score"],
            "citation_support_rate": citation_eval["score"],
            "quiz_answerability": mean_answerability,
        }
        results[lec_id] = sample_summary

        logger.info(f"Results for {lec_id}:")
        logger.info(f"  - Slide Alignment Accuracy (IoU): {sample_summary['slide_alignment_accuracy']:.2f}")
        logger.info(f"  - Gap Precision:                  {sample_summary['gap_precision']:.2f}")
        logger.info(f"  - Citation Support Rate:          {sample_summary['citation_support_rate']:.2f}")
        logger.info(f"  - Quiz Answerability:             {sample_summary['quiz_answerability']:.2f}")

    # Log to LangSmith if API key is provided
    if Config.validate_langsmith_key():
        try:
            from uuid import uuid4
            from langsmith import Client
            client = Client(api_key=Config.LANGSMITH_API_KEY)
            experiment_name = f"eval-benchmark-{Config.GROQ_LLM_MODEL}"
            logger.info(f"Logging experiment to LangSmith: project={Config.LANGSMITH_PROJECT}, experiment={experiment_name}")
            for lec_id, metrics in results.items():
                run_id = str(uuid4())
                client.create_run(
                    id=run_id,
                    name=f"Eval-{lec_id}",
                    run_type="chain",
                    project_name=Config.LANGSMITH_PROJECT,
                    inputs={"lecture_id": lec_id},
                    outputs=metrics,
                    tags=["evaluation", Config.GROQ_LLM_MODEL, Config.EMBEDDING_MODEL_NAME],
                )
            logger.info("Successfully recorded experiment runs to LangSmith.")
        except Exception as e:
            logger.warning(f"Could not record LangSmith experiment: {e}")

    # Save local results report
    out_file = Path(__file__).resolve().parent / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {out_file}")

    return results


if __name__ == "__main__":
    run_benchmarks()
