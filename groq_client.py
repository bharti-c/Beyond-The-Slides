"""Resilient Groq Client wrapper.

Features:
- Dynamic model resolution from environment variables (never hardcoded).
- Automatic retries with exponential backoff and jitter on 429 RateLimitErrors and transient API errors.
- Support for Chat, Vision (base64 encoded images), and Audio (Whisper STT with verbose_json).
- Structured JSON parsing with Pydantic model validation.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel
from config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GroqClientWrapper:
    """Wrapper around the Groq Python SDK with exponential backoff and dynamic models."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or Config.GROQ_API_KEY
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Groq client."""
        if self._client is None:
            from groq import Groq
            if not self.api_key or self.api_key.startswith("your_groq"):
                raise ValueError(
                    "GROQ_API_KEY is not configured. Please set a valid key in your .env file."
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _execute_with_retry(self, func, max_retries: int = 5, initial_delay: float = 1.5, *args, **kwargs) -> Any:
        """Execute a Groq API call with exponential backoff for rate limits and transient errors."""
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = "429" in err_msg or "rate limit" in err_msg or "rate_limit" in err_msg
                is_transient = "500" in err_msg or "503" in err_msg or "connection" in err_msg or "timeout" in err_msg

                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    jitter = random.uniform(0.5, 1.5)
                    sleep_time = delay + jitter
                    logger.warning(
                        f"Groq API error ({e}). Attempt {attempt + 1}/{max_retries}. Backing off for {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= 2.0
                else:
                    logger.error(f"Groq API call failed permanently after {attempt + 1} attempts: {e}")
                    raise

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Call LLM with dynamic model from config. Never hardcoded."""
        target_model = model or Config.GROQ_LLM_MODEL

        def _call():
            kwargs = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return self._execute_with_retry(_call)

    def structured_output(
        self,
        messages: List[Dict[str, Any]],
        schema: Type[T],
        temperature: float = 0.1,
        model: Optional[str] = None,
    ) -> T:
        """Generate structured response matching a Pydantic model schema."""
        target_model = model or Config.GROQ_LLM_MODEL
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        augmented_messages = list(messages)
        system_instructions = (
            f"\n\nYou MUST respond ONLY with a valid JSON object matching this JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any text before or after the JSON."
        )

        if augmented_messages and augmented_messages[0].get("role") == "system":
            augmented_messages[0]["content"] += system_instructions
        else:
            augmented_messages.insert(0, {"role": "system", "content": system_instructions.strip()})

        raw_json = self.chat_completion(
            messages=augmented_messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            model=target_model,
        )

        cleaned = raw_json.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        parsed = json.loads(cleaned.strip())
        return schema.model_validate(parsed)

    def vision_analyze(
        self,
        image_path: str | Path,
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """Analyze an image using Groq Vision model dynamically from config."""
        target_model = model or Config.GROQ_VISION_MODEL
        img_p = Path(image_path)
        if not img_p.is_file():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Determine MIME type
        ext = img_p.suffix.lower()
        mime_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        with open(img_p, "rb") as f:
            encoded_bytes = base64.b64encode(f.read()).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{encoded_bytes}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        def _call():
            resp = self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=0.1,
            )
            return resp.choices[0].message.content or ""

        return self._execute_with_retry(_call)

    def transcribe_audio_chunk(
        self,
        audio_chunk_path: str | Path,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe an audio chunk using Groq Whisper with verbose_json timestamps."""
        target_model = model or Config.GROQ_STT_MODEL
        chunk_p = Path(audio_chunk_path)
        if not chunk_p.is_file():
            raise FileNotFoundError(f"Audio chunk not found at {audio_chunk_path}")

        def _call():
            with open(chunk_p, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    file=f,
                    model=target_model,
                    response_format="verbose_json",
                    temperature=0.0,
                )
            # transcription.segments is a list of segments with start, end, text
            if hasattr(transcription, "segments") and transcription.segments:
                return [
                    {
                        "start": float(seg.get("start", 0.0) if isinstance(seg, dict) else seg.start),
                        "end": float(seg.get("end", 0.0) if isinstance(seg, dict) else seg.end),
                        "text": str(seg.get("text", "") if isinstance(seg, dict) else seg.text).strip(),
                    }
                    for seg in transcription.segments
                ]
            else:
                full_text = getattr(transcription, "text", "") or ""
                return [{"start": 0.0, "end": 0.0, "text": full_text.strip()}]

        return self._execute_with_retry(_call)


# Shared singleton instance
groq_client = GroqClientWrapper()
