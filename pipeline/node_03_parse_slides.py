"""Node 3: Parse Slides.

Extracts text per page using PyMuPDF. If a page has minimal text or contains diagrams,
renders the slide to an image and uses the Groq Vision model to describe its visual content.
Caches slide data and rendered preview images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from config import Config
from cache_manager import CacheManager
from groq_client import groq_client
from models import PipelineGraphState, SlideContent

logger = logging.getLogger(__name__)


def parse_slides_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Parse slides node entrypoint for LangGraph."""
    logger.info("Executing Node 3: Parse Slides")
    slides_pdf_path = Path(state["slides_pdf_path"])
    output_dir = Path(state["output_dir"])
    slides_img_dir = output_dir / "slides"
    slides_img_dir.mkdir(parents=True, exist_ok=True)

    errors = list(state.get("errors", []))

    if not slides_pdf_path.is_file():
        errors.append(f"Slides PDF not found: {slides_pdf_path}")
        return {
            "slides": [],
            "current_stage": "parse_slides_failed",
            "errors": errors,
        }

    # Check cache
    pdf_hash = CacheManager.compute_file_hash(slides_pdf_path)
    cached_slides = CacheManager.get(pdf_hash, "slides")
    if cached_slides:
        logger.info(f"Using cached slide parse for {pdf_hash[:8]}")
        slides = [SlideContent(**sc) for sc in cached_slides]
        return {
            "slides": slides,
            "current_stage": "parse_slides_complete",
            "errors": errors,
        }

    slides: List[SlideContent] = []

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(slides_pdf_path)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            extracted_text = page.get_text("text").strip()

            # Always render slide image for UI display & visual analysis
            img_path = slides_img_dir / f"slide_{page_idx:03d}.png"
            pix = page.get_pixmap(dpi=150)
            pix.save(str(img_path))

            visual_description = None
            has_rich_text = len(extracted_text) >= 40

            # If slide has little text, call vision model for visual description
            if not has_rich_text:
                if Config.validate_groq_key():
                    logger.info(f"Slide {page_idx + 1} has low text ({len(extracted_text)} chars); querying vision model...")
                    prompt = (
                        "This is a lecture slide. Describe all key concepts, diagrams, equations, "
                        "tables, and text visible on this slide in detail so a student can learn from it."
                    )
                    try:
                        visual_description = groq_client.vision_analyze(img_path, prompt)
                    except Exception as e:
                        logger.warning(f"Vision model call failed on slide {page_idx + 1}: {e}")
                        errors.append(f"Vision analysis failed on slide {page_idx + 1}: {e}")
                        visual_description = f"Visual diagram or chart on slide {page_idx + 1}."
                else:
                    visual_description = f"Diagram or visual slide content for slide {page_idx + 1}."

            slides.append(
                SlideContent(
                    slide_index=page_idx,
                    text=extracted_text,
                    visual_description=visual_description,
                    image_path=str(img_path),
                    has_rich_text=has_rich_text,
                )
            )

        doc.close()

    except Exception as e:
        logger.error(f"Failed to parse slides PDF: {e}")
        errors.append(f"Slide parsing error: {e}")

    # Cache results
    if slides:
        CacheManager.set(
            pdf_hash,
            "slides",
            [s.model_dump() for s in slides],
        )

    logger.info(f"Parsed {len(slides)} slides.")
    return {
        "slides": slides,
        "current_stage": "parse_slides_complete",
        "errors": errors,
    }
