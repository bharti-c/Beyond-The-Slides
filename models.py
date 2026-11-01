"""Data models for Lecture-to-Living-Course.

Uses Pydantic v2 for robust validation and structured LLM outputs.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Literal, Dict, Any
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech with absolute video timestamps."""
    start: float = Field(..., description="Start timestamp in seconds from video start")
    end: float = Field(..., description="End timestamp in seconds from video start")
    text: str = Field(..., description="Transcribed spoken text")


class SlideContent(BaseModel):
    """Extracted text and visual representation of a single slide."""
    slide_index: int = Field(..., description="0-indexed slide position")
    text: str = Field(default="", description="Text extracted directly from PDF")
    visual_description: Optional[str] = Field(
        default=None,
        description="Visual description from vision model if slide contains diagrams/equations",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Rendered slide image path on disk",
    )
    has_rich_text: bool = Field(
        default=True,
        description="Whether PDF slide had sufficient extracted text or needed vision fallback",
    )


class NoteContent(BaseModel):
    """Transcribed text and diagram analysis from a handwritten note photo."""
    page_index: int = Field(..., description="0-indexed photo/page number")
    image_path: str = Field(..., description="Local path to uploaded note photo")
    transcribed_text: str = Field(..., description="Transcribed student handwriting")
    diagram_description: Optional[str] = Field(
        default=None,
        description="Description of any sketches, flowcharts, or diagrams in notes",
    )
    low_confidence_notes: List[str] = Field(
        default_factory=list,
        description="Phrases, equations, or words that were uncertain or illegible",
    )


class SlideAlignment(BaseModel):
    """Mapping of a slide to its corresponding active video time interval."""
    slide_index: int = Field(..., description="0-indexed slide position")
    start_time: float = Field(..., description="Estimated time slide first appears on screen")
    end_time: float = Field(..., description="Estimated time slide transitions away")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Alignment confidence score")
    matched_segments: List[int] = Field(
        default_factory=list,
        description="Indices of transcript segments aligned to this slide",
    )


class NoteAlignment(BaseModel):
    """Mapping of a student note page to the video timeline."""
    page_index: int = Field(..., description="0-indexed note photo position")
    start_time: float = Field(..., description="Estimated lecture timestamp corresponding to note")
    end_time: float = Field(..., description="Estimated end timestamp corresponding to note")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Alignment confidence score")
    matched_segments: List[int] = Field(
        default_factory=list,
        description="Indices of transcript segments aligned to this note",
    )


class UnifiedTimeline(BaseModel):
    """The master timeline integrating video, transcript, slides, and notes."""
    lecture_id: str
    video_duration: float
    transcript: List[TranscriptSegment] = Field(default_factory=list)
    slide_alignments: List[SlideAlignment] = Field(default_factory=list)
    note_alignments: List[NoteAlignment] = Field(default_factory=list)


class GapType(str, Enum):
    """Classification of verbal-only content missing from slides."""
    EXAMPLE = "example"
    CAVEAT = "caveat"
    CORRECTION = "correction"
    EMPHASIS = "emphasis"


class KnowledgeGap(BaseModel):
    """A verbal-only concept spoken by the professor not found on the slides."""
    gap_id: str = Field(..., description="Unique ID for this gap")
    start_time: float = Field(..., description="Start timestamp in video")
    end_time: float = Field(..., description="End timestamp in video")
    gap_type: GapType = Field(..., description="Classification: example, caveat, correction, emphasis")
    summary: str = Field(..., description="One-line summary of the verbal-only point")
    transcript_quote: str = Field(..., description="Verbatim quote from the transcript")
    context: str = Field(default="", description="Surrounding context or why it matters for exams/understanding")


class MindMapNode(BaseModel):
    """Node in the 4-level hierarchical concept mind map."""
    id: str = Field(..., description="Node unique identifier")
    label: str = Field(..., description="Short concept name (2-6 words), never slide/page numbers")
    summary: str = Field(
        default="",
        description="1-2 plain-language sentences summarizing this concept or detail",
    )
    origin: Literal["slide-only", "verbal-only", "both"] = Field(
        default="both",
        description="Origin tag indicating if concept came from slide-only, verbal-only, or both",
    )
    source_type: Literal["slide-only", "verbal-only", "both"] = Field(
        default="both",
        description="Alias for origin tag kept for backward compatibility",
    )
    timestamp: Optional[float] = Field(
        default=None,
        description="Primary video timestamp in seconds for seeking",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Primary video start timestamp in seconds for seeking",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End video timestamp in seconds",
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="ID of parent node (None only for root)",
    )
    slide_refs: List[int] = Field(
        default_factory=list,
        description="1-indexed slide numbers where this concept appears (metadata)",
    )
    note_refs: List[int] = Field(
        default_factory=list,
        description="1-indexed student note page numbers where this concept appears (metadata)",
    )
    timestamps: List[float] = Field(
        default_factory=list,
        description="All video timestamps in seconds associated with this node",
    )
    children: List[MindMapNode] = Field(
        default_factory=list,
        description="Child subtopics or detail leaves",
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields_before(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Sync origin <-> source_type
        origin_val = data.get("origin") or data.get("source_type") or "both"
        if origin_val not in ("slide-only", "verbal-only", "both"):
            origin_val = "both"
        data["origin"] = origin_val
        data["source_type"] = origin_val

        # Sync timestamp <-> start_time <-> timestamps
        ts = data.get("timestamp")
        st_time = data.get("start_time")
        ts_list = data.get("timestamps") or []
        primary_ts = ts if ts is not None else (st_time if st_time is not None else (ts_list[0] if ts_list else None))
        if primary_ts is not None:
            primary_ts = float(primary_ts)
            data["timestamp"] = primary_ts
            data["start_time"] = primary_ts
            if not ts_list:
                data["timestamps"] = [primary_ts]
        return data

    @model_validator(mode="after")
    def _sync_fields_after(self) -> MindMapNode:
        # Ensure origin and source_type match
        if self.origin != self.source_type:
            self.source_type = self.origin
        if self.timestamp is not None and self.start_time is None:
            self.start_time = self.timestamp
        elif self.start_time is not None and self.timestamp is None:
            self.timestamp = self.start_time
        if self.timestamp is not None and not self.timestamps:
            self.timestamps = [self.timestamp]
        return self


class MindMap(BaseModel):
    """Complete hierarchical mind map representation."""
    lecture_title: str
    root: MindMapNode


class PodcastDialogue(BaseModel):
    """Single line of speech in the podcast recap."""
    speaker: Literal["Host", "Curious Student"] = Field(..., description="Speaker persona")
    text: str = Field(..., description="Dialogue text spoken")
    referenced_timestamps: List[float] = Field(
        default_factory=list,
        description="Timestamps in the video referenced in this turn",
    )
    audio_path: Optional[str] = Field(
        default=None,
        description="Path to synthesized audio clip for this turn",
    )


class PodcastScript(BaseModel):
    """Two-host podcast recap covering the lecture and highlighting verbal gaps."""
    episode_title: str
    dialogue: List[PodcastDialogue] = Field(default_factory=list)
    full_audio_path: Optional[str] = Field(
        default=None,
        description="Path to final merged audio file",
    )


class QuizCitation(BaseModel):
    """Timestamped transcript citation backing a quiz question/answer."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    quote: str = Field(..., description="Exact or near-exact transcript/slide/note excerpt")


QuestionTypeLiteral = Literal["definition", "mechanism", "comparison", "scenario", "tradeoff"]


class QuizQualityCheck(BaseModel):
    """LLM-as-judge and deterministic quality rubric result for a single quiz question."""
    question_id: str = Field(default="", description="Question ID")
    concept_based: bool = Field(default=True, description="Tests a lecture concept rather than slide/page location")
    self_contained: bool = Field(default=True, description="Makes sense without looking at slides")
    single_correct_answer: bool = Field(default=True, description="Exactly one defensible correct option")
    plausible_distractors: bool = Field(default=True, description="Distractors are plausible, no 'None of the above'")
    not_duplicate: bool = Field(default=True, description="Does not duplicate another stem or concept_id")
    answerable_from_citation: bool = Field(default=True, description="Supported and answerable from the cited quote")
    issues: List[str] = Field(default_factory=list, description="Specific quality failures to fix on retry")

    @property
    def passed(self) -> bool:
        """Return True if all quality criteria pass."""
        return (
            self.concept_based
            and self.self_contained
            and self.single_correct_answer
            and self.plausible_distractors
            and self.not_duplicate
            and self.answerable_from_citation
            and len(self.issues) == 0
        )


class QuizQuestion(BaseModel):
    """Multiple choice question grounded in a mind-map concept with required citation."""
    id: str = Field(..., description="Question identifier")
    concept_id: str = Field(
        default="",
        description="ID of the mind-map concept node tested by this question",
    )
    question_type: QuestionTypeLiteral = Field(
        default="mechanism",
        description="Cognitive question type: definition, mechanism, comparison, scenario, tradeoff",
    )
    question: str = Field(..., description="Self-contained concept question prompt")
    options: List[str] = Field(..., min_length=4, max_length=4, description="Four plausible options")
    correct_index: int = Field(..., ge=0, le=3, description="Index of correct option (0-3)")
    explanation: str = Field(
        ...,
        description="Educational explanation of why the correct option is right and why the top distractor is wrong",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")
    is_verbal_only: bool = Field(
        default=False,
        description="True if derived from professor's verbal-only additions",
    )
    citations: List[QuizCitation] = Field(
        default_factory=list,
        min_length=1,
        description="One or more citations with {start, end, quote}",
    )
    verification_status: Literal["supported", "partially", "unsupported"] = Field(
        default="supported",
        description="LLM-as-judge verification result",
    )
    quality_issues: List[str] = Field(
        default_factory=list,
        description="Any quality rubric issues detected by Node 10",
    )


class Quiz(BaseModel):
    """Complete quiz containing 10 questions."""
    questions: List[QuizQuestion] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """LLM-as-judge citation verification and quality outcome for a single question."""
    question_id: str
    status: Literal["supported", "partially", "unsupported"]
    reasoning: str
    quality: Optional[QuizQualityCheck] = None


class PipelineGraphState(TypedDict):
    """LangGraph state passed through all nodes in the pipeline."""
    lecture_id: str
    video_path: str
    slides_pdf_path: str
    notes_paths: List[str]
    video_duration: float
    output_dir: str

    # Pipeline artifacts
    transcript: List[TranscriptSegment]
    slides: List[SlideContent]
    notes: List[NoteContent]
    timeline: Optional[UnifiedTimeline]
    gaps: List[KnowledgeGap]
    mindmap: Optional[MindMap]
    podcast: Optional[PodcastScript]
    quiz: Optional[Quiz]

    # Quality & Verification tracking
    retries: int
    errors: List[str]
    current_stage: str
    run_id: Optional[str]
    quality_report: NotRequired[Optional[Dict[str, Any]]]
    failed_question_issues: NotRequired[Optional[Dict[str, List[str]]]]
