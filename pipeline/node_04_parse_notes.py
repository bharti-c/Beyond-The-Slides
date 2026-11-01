"""Node 4: Parse Notes.

Sends each student handwritten notes photo to the Groq Vision model.
Extracts transcribed text, describes diagrams/sketches, and identifies low-confidence
or illegible words/formulas. Caches parsed results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import Config
from cache_manager import CacheManager
from groq_client import groq_client
from models import NoteContent, PipelineGraphState

logger = logging.getLogger(__name__)

PARSE_NOTES_PROMPT = """You are analyzing a student's handwritten lecture notes photo.
Analyze this image carefully and return a JSON object with exactly these fields:
1. "transcribed_text": The complete text transcribed from the handwriting, preserving order and formatting.
2. "diagram_description": A clear description of any sketches, diagrams, flowcharts, or equations present, or null if none.
3. "low_confidence_notes": A list of specific words, phrases, or equations that are smudged, ambiguous, or hard to read.

Return ONLY a valid JSON object:
{
  "transcribed_text": "...",
  "diagram_description": "..." or null,
  "low_confidence_notes": ["...", "..."]
}
"""


def parse_notes_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Parse notes node entrypoint for LangGraph."""
    logger.info("Executing Node 4: Parse Notes")
    notes_paths = [Path(p) for p in state.get("notes_paths", [])]
    errors = list(state.get("errors", []))

    if not notes_paths:
        logger.info("No handwritten notes uploaded; continuing with empty notes list.")
        return {
            "notes": [],
            "current_stage": "parse_notes_complete",
            "errors": errors,
        }

    # Cache check
    notes_hash = CacheManager.compute_multi_file_hash(notes_paths)
    cached_notes = CacheManager.get(notes_hash, "notes")
    if cached_notes:
        logger.info(f"Using cached notes parse for {notes_hash[:8]}")
        notes = [NoteContent(**nc) for nc in cached_notes]
        return {
            "notes": notes,
            "current_stage": "parse_notes_complete",
            "errors": errors,
        }

    notes: List[NoteContent] = []

    for idx, np in enumerate(notes_paths):
        if not np.is_file():
            logger.warning(f"Note image not found: {np}")
            continue

        if Config.validate_groq_key():
            logger.info(f"Transcribing handwritten note {idx + 1}/{len(notes_paths)} ({np.name})...")
            try:
                response_text = groq_client.vision_analyze(np, PARSE_NOTES_PROMPT)
                cleaned = response_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]

                parsed_json = json.loads(cleaned.strip())
                notes.append(
                    NoteContent(
                        page_index=idx,
                        image_path=str(np),
                        transcribed_text=str(parsed_json.get("transcribed_text", "")).strip(),
                        diagram_description=parsed_json.get("diagram_description"),
                        low_confidence_notes=parsed_json.get("low_confidence_notes", []) or [],
                    )
                )
            except Exception as e:
                logger.error(f"Failed to parse note {np.name}: {e}")
                errors.append(f"Notes vision parsing error on {np.name}: {e}")
                notes.append(
                    NoteContent(
                        page_index=idx,
                        image_path=str(np),
                        transcribed_text=f"Student handwritten notes for section {idx + 1}.",
                        diagram_description="Handwritten diagrams and notes.",
                        low_confidence_notes=["Partially illegible handwriting"],
                    )
                )
        else:
            logger.warning("GROQ_API_KEY not set. Using fallback placeholder for handwritten notes.")
            notes.append(
                NoteContent(
                    page_index=idx,
                    image_path=str(np),
                    transcribed_text=f"Student handwritten notes for page {idx + 1}.",
                    diagram_description=None,
                    low_confidence_notes=[],
                )
            )

    # Cache parsed notes
    if notes:
        CacheManager.set(
            notes_hash,
            "notes",
            [n.model_dump() for n in notes],
        )

    logger.info(f"Parsed {len(notes)} handwritten notes pages.")
    return {
        "notes": notes,
        "current_stage": "parse_notes_complete",
        "errors": errors,
    }
