"""LangGraph Orchestration and LangSmith Tracing for Lecture-to-Living-Course.

Configures a 10-node StateGraph with conditional retry edge for citation verification.
Instruments every node as a named span and attaches metadata and run tags for LangSmith.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langsmith import Client, traceable

from config import Config
from models import PipelineGraphState
from pipeline.node_01_ingest import ingest_node
from pipeline.node_02_transcribe import transcribe_node
from pipeline.node_03_parse_slides import parse_slides_node
from pipeline.node_04_parse_notes import parse_notes_node
from pipeline.node_05_align import align_node
from pipeline.node_06_gap_detect import gap_detect_node
from pipeline.node_07_generate_mindmap import mindmap_node
from pipeline.node_09_generate_quiz import quiz_node
from pipeline.node_10_verify_citations import should_retry_quiz, verify_citations_node

logger = logging.getLogger(__name__)

# Ensure LangSmith environment variables are configured
if Config.LANGSMITH_TRACING and Config.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = Config.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = Config.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = Config.LANGSMITH_ENDPOINT


def build_lecture_pipeline_graph() -> Any:
    """Construct the LangGraph StateGraph connecting all pipeline nodes."""
    workflow = StateGraph(PipelineGraphState)

    # 1. Add pipeline nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("transcribe", transcribe_node)
    workflow.add_node("parse_slides", parse_slides_node)
    workflow.add_node("parse_notes", parse_notes_node)
    workflow.add_node("align", align_node)
    workflow.add_node("gap_detect", gap_detect_node)
    workflow.add_node("generate_mindmap", mindmap_node)
    workflow.add_node("generate_quiz", quiz_node)
    workflow.add_node("verify_citations", verify_citations_node)

    # 2. Define sequential pipeline flow
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "transcribe")
    workflow.add_edge("transcribe", "parse_slides")
    workflow.add_edge("parse_slides", "parse_notes")
    workflow.add_edge("parse_notes", "align")
    workflow.add_edge("align", "gap_detect")
    workflow.add_edge("gap_detect", "generate_mindmap")
    workflow.add_edge("generate_mindmap", "generate_quiz")
    workflow.add_edge("generate_quiz", "verify_citations")

    # 3. Add conditional edge for LLM-as-judge citation verification retry loop
    workflow.add_conditional_edges(
        "verify_citations",
        should_retry_quiz,
        {
            "retry": "generate_quiz",
            "end": END,
        },
    )

    return workflow.compile()


# Compile graph instance
pipeline_app = build_lecture_pipeline_graph()


def run_pipeline(
    video_path: str,
    slides_pdf_path: str,
    notes_paths: Optional[List[str]] = None,
    lecture_id: Optional[str] = None,
    on_step_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end lecture processing pipeline.

    Configures LangSmith distributed tracing with run tags and metadata.
    """
    notes_paths = notes_paths or []
    run_id = str(uuid4())

    initial_state: PipelineGraphState = {
        "lecture_id": lecture_id or "",
        "video_path": video_path,
        "slides_pdf_path": slides_pdf_path,
        "notes_paths": notes_paths,
        "video_duration": 0.0,
        "output_dir": "",
        "transcript": [],
        "slides": [],
        "notes": [],
        "timeline": None,
        "gaps": [],
        "mindmap": None,
        "podcast": None,
        "quiz": None,
        "retries": 0,
        "errors": [],
        "current_stage": "started",
        "run_id": run_id,
    }

    # LangSmith tracing tags and metadata
    tags = [
        f"lecture_id:{lecture_id or 'auto'}",
        f"llm:{Config.GROQ_LLM_MODEL}",
        f"vision:{Config.GROQ_VISION_MODEL}",
        f"stt:{Config.GROQ_STT_MODEL}",
    ]

    metadata = {
        "run_id": run_id,
        "notes_count": len(notes_paths),
        "embedding_model": Config.EMBEDDING_MODEL_NAME,
        "gap_threshold": Config.GAP_SIMILARITY_THRESHOLD,
    }

    config = {
        "run_name": f"LecturePipeline-{lecture_id or run_id[:8]}",
        "tags": tags,
        "metadata": metadata,
    }

    logger.info(f"Starting pipeline run {run_id}...")
    final_state = dict(initial_state)

    for event in pipeline_app.stream(initial_state, config=config):
        for node_name, node_state in event.items():
            logger.info(f"Completed node: {node_name}")
            final_state.update(node_state)
            if on_step_callback:
                on_step_callback(node_name, final_state)

    # Attach final summary metadata to LangSmith run if client available
    if Config.validate_langsmith_key():
        try:
            client = Client(api_key=Config.LANGSMITH_API_KEY)
            q_rep = final_state.get("quality_report") or {}
            client.update_run(
                run_id=run_id,
                metadata={
                    "num_transcript_segments": len(final_state.get("transcript", [])),
                    "num_slides": len(final_state.get("slides", [])),
                    "num_gaps": len(final_state.get("gaps", [])),
                    "num_retries": final_state.get("retries", 0),
                    "video_duration": final_state.get("video_duration", 0.0),
                    "quiz_concept_rate": q_rep.get("quiz_concept_rate", 1.0),
                    "citation_support_rate": q_rep.get("citation_support_rate", 1.0),
                },
            )
        except Exception as e:
            logger.warning(f"Could not update LangSmith run metadata: {e}")

    return final_state


def log_langsmith_feedback(run_id: str, key: str, score: float, comment: Optional[str] = None) -> bool:
    """Log user feedback (e.g., thumbs up/down: 1.0 or 0.0) to LangSmith against a run."""
    if not Config.validate_langsmith_key():
        logger.info(f"[Offline] Feedback logged locally: run={run_id}, key={key}, score={score}")
        return True

    try:
        client = Client(api_key=Config.LANGSMITH_API_KEY)
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        logger.info(f"LangSmith feedback recorded: {key}={score} on run {run_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to record LangSmith feedback: {e}")
        return False
