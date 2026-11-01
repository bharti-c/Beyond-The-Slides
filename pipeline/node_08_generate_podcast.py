"""Node 8: Generate Podcast.

Produces a two-host podcast audio recap:
- Host (Alex): Explains primary concepts from the lecture slides.
- Curious Student (Sam): Inquires about nuances and specifically highlights the professor's verbal-only gaps.
Synthesizes speech using the pluggable TTS layer (EdgeTTS) and embeds clickable timestamp references.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import Config
from groq_client import groq_client
from models import KnowledgeGap, PipelineGraphState, PodcastDialogue, PodcastScript, SlideContent, UnifiedTimeline
from tts import tts_service

logger = logging.getLogger(__name__)

PODCAST_PROMPT = """You are a podcast producer creating a 2-host audio recap of a college lecture.
Hosts:
1. "Host": The lead host (Alex), structured, insightful, explains the core slide takeaways.
2. "Curious Student": Inquisitive co-host (Sam), asks probing questions, points out what wasn't on the slides, and highlights the professor's verbal examples, caveats, corrections, and exam hints.

Lecture Slide Highlights:
{slides_summary}

Professor's Verbal-Only Additions (Gaps):
{gaps_summary}

Requirements:
1. Write 8 to 14 dialogue turns alternating between Host and Curious Student.
2. The Curious Student MUST explicitly call out at least 2 verbal-only points that weren't on the slides.
3. In each dialogue line, provide "referenced_timestamps" (a list of video timestamps in seconds that correspond to the topic being discussed).
4. Tone: lively, engaging, educational, conversational, never robotic.

Return ONLY a valid JSON object matching this schema:
{{
  "episode_title": "...",
  "dialogue": [
    {{
      "speaker": "Host" | "Curious Student",
      "text": "...",
      "referenced_timestamps": [12.0, 45.0]
    }}
  ]
}}
"""


def generate_fallback_podcast(
    timeline: UnifiedTimeline,
    slides: List[SlideContent],
    gaps: List[KnowledgeGap],
) -> PodcastScript:
    """Generate a clean fallback podcast script when offline or without API key."""
    title = "Living Course Audio Recap: Key Insights & Gaps"
    dialogue: List[PodcastDialogue] = [
        PodcastDialogue(
            speaker="Host",
            text="Welcome to the Lecture Recap! Today we're breaking down the key concepts covered in class.",
            referenced_timestamps=[0.0],
        ),
        PodcastDialogue(
            speaker="Curious Student",
            text="I went through the slides, but the professor definitely dropped some crucial verbal explanations that weren't on the screen!",
            referenced_timestamps=[gaps[0].start_time if gaps else 15.0],
        ),
    ]

    for idx, s in enumerate(slides[:3]):
        sa_time = timeline.slide_alignments[idx].start_time if idx < len(timeline.slide_alignments) else 10.0 * (idx + 1)
        dialogue.append(
            PodcastDialogue(
                speaker="Host",
                text=f"Looking at Slide {s.slide_index + 1}, the central concept introduced was {s.text[:50].strip() or 'the foundational theory'}.",
                referenced_timestamps=[sa_time],
            )
        )
        if idx < len(gaps):
            gap = gaps[idx]
            dialogue.append(
                PodcastDialogue(
                    speaker="Curious Student",
                    text=f"Right, and notice how the professor added a crucial {gap.gap_type.value}: \"{gap.summary}\"!",
                    referenced_timestamps=[gap.start_time],
                )
            )

    dialogue.append(
        PodcastDialogue(
            speaker="Host",
            text="That's why reviewing both the slides and the professor's verbal commentary makes all the difference. See you in the next recap!",
            referenced_timestamps=[timeline.video_duration],
        )
    )

    return PodcastScript(episode_title=title, dialogue=dialogue)


def podcast_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Podcast node entrypoint for LangGraph."""
    logger.info("Executing Node 8: Generate Podcast")
    timeline: UnifiedTimeline = state.get("timeline")
    slides: List[SlideContent] = state.get("slides", [])
    gaps: List[KnowledgeGap] = state.get("gaps", [])
    output_dir = Path(state["output_dir"])
    podcast_dir = output_dir / "podcast"
    podcast_dir.mkdir(parents=True, exist_ok=True)
    combined_audio_path = podcast_dir / "podcast_recap.mp3"

    errors = list(state.get("errors", []))

    if not timeline:
        errors.append("Cannot generate podcast: timeline missing.")
        return {
            "podcast": None,
            "current_stage": "podcast_failed",
            "errors": errors,
        }

    podcast_script: PodcastScript

    if Config.validate_groq_key():
        slides_summary = "\n".join(
            f"- Slide {s.slide_index + 1}: {s.text[:120].strip() or s.visual_description or 'Visual diagram'}"
            for s in slides[:10]
        )
        gaps_summary = "\n".join(
            f"- [{g.gap_type.value} @ {g.start_time:.0f}s]: {g.summary} (Quote: \"{g.transcript_quote[:80]}\")"
            for g in gaps
        )
        prompt = PODCAST_PROMPT.format(slides_summary=slides_summary, gaps_summary=gaps_summary)

        try:
            raw_json = groq_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            cleaned = raw_json.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            parsed = json.loads(cleaned.strip())
            podcast_script = PodcastScript.model_validate(parsed)
        except Exception as e:
            logger.error(f"Podcast script LLM generation failed: {e}")
            errors.append(f"Podcast LLM error: {e}")
            podcast_script = generate_fallback_podcast(timeline, slides, gaps)
    else:
        logger.warning("GROQ_API_KEY not set. Using structured fallback podcast script.")
        podcast_script = generate_fallback_podcast(timeline, slides, gaps)

    # Synthesize audio with pluggable TTS layer (EdgeTTS)
    try:
        script_dict_list = [
            {"speaker": d.speaker, "text": d.text}
            for d in podcast_script.dialogue
        ]
        # Run async synthesis synchronously in the node
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()

        audio_file = loop.run_until_complete(
            tts_service.synthesize_dialogue(
                script=script_dict_list,
                output_dir=podcast_dir / "clips",
                combined_output_path=combined_audio_path,
            )
        )
        podcast_script.full_audio_path = audio_file
    except Exception as e:
        logger.warning(f"TTS audio synthesis encountered an issue: {e}")
        errors.append(f"TTS synthesis warning: {e}")

    logger.info("Podcast generation complete.")
    return {
        "podcast": podcast_script,
        "current_stage": "podcast_complete",
        "errors": errors,
    }
