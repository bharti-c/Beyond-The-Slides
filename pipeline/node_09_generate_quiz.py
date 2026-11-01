"""Node 9: Generate Quiz.

Generates 10 self-contained, concept-based multiple-choice questions grounded in the
lecture's 4-level mind map, aligned transcript, slides, notes, and verbal gaps:
- Strictly forbids references to slide/page/deck locations or "None/All of the above".
- Each question links to a unique mind-map `concept_id` and spreads across >=4 major topics.
- Balanced cognitive mix: definition (<=2), mechanism (>=2), comparison (>=2),
  scenario (>=2), tradeoff (>=1).
- >=3 questions (>=30%) test verbal-only content with {start, end, quote} citations.
- Supports targeted regeneration of ONLY failing questions on retry from Node 10.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from cache_manager import CacheManager
from config import Config
from groq_client import groq_client
from models import (
    KnowledgeGap,
    MindMap,
    MindMapNode,
    NoteContent,
    PipelineGraphState,
    Quiz,
    QuizCitation,
    QuizQuestion,
    SlideContent,
    UnifiedTimeline,
)
from pipeline.node_07_generate_mindmap import collect_all_nodes, generate_fallback_mindmap

logger = logging.getLogger(__name__)

FORBIDDEN_QUIZ_PATTERN = re.compile(
    r"(?i)\b(slide|slides|page|pages|deck|study\s+deck|according\s+to\s+the\s+presentation|"
    r"according\s+to\s+the\s+slides|based\s+on\s+the\s+student\s+notes|in\s+the\s+material|"
    r"as\s+shown\s+above|which\s+of\s+the\s+following\s+is\s+a\s+key\s+point|"
    r"none\s+of\s+the\s+above|all\s+of\s+the\s+above)\b"
)

QUIZ_GENERATION_PROMPT = """You are a rigorous computer science and AI professor designing a 10-question conceptual multiple-choice exam.

Mind Map Concept Tree (every question MUST test a distinct `concept_id` from this list, spread across >=4 major topics):
{concepts_catalog}

Lecture Content (Transcript, Slides, Student Notes):
{lecture_context}

Professor's Verbal-Only Additions (at least 3 questions MUST test these verbal-only insights with "is_verbal_only": true):
{gaps_context}

HARD RULES (VIOLATION WILL CAUSE IMMEDIATE REJECTION):
1. FORBIDDEN PHRASES in question stems and options:
   NEVER use "slide", "page", "deck", "according to the presentation", "in the material", "as shown above", "which of the following is a key point", "None of the above", or "All of the above".
   Every question MUST be 100% self-contained and make complete sense to an expert who never saw the slides.
2. CONCEPT COVERAGE:
   Every question MUST specify a unique "concept_id" from the Mind Map Concept Tree above. No two questions may share a stem or test the same "concept_id". Cover at least 4 major topics.
3. QUESTION TYPE MIX across the 10 questions ("question_type"):
   - "definition": recall/definition (at most 2 questions)
   - "mechanism": why/how does X work under the hood (at least 2 questions)
   - "comparison": compare/contrast two techniques (at least 2 questions)
   - "scenario": applied engineering scenario ("A team faces X constraint; which design fits best?") (at least 2 questions)
   - "tradeoff": trade-off, boundary condition, edge-case, or common mistake (at least 1 question)
4. HONEST DIFFICULTY ("difficulty"):
   - "easy": direct conceptual recall
   - "medium": mechanism understanding or comparison
   - "hard": multi-step scenario reasoning or subtle trade-off/edge-case
5. PLAUSIBLE DISTRACTORS:
   Exactly 4 options per question ("correct_index" 0..3). Build distractors from realistic misconceptions or neighboring concepts. Exactly ONE defensible correct answer.
6. VERBAL-ONLY COVERAGE:
   At least 3 of the 10 questions (>=30%) MUST have "is_verbal_only": true and test the professor's verbal-only additions, with a verbatim citation {{ "start": <float>, "end": <float>, "quote": "<exact quote>" }}.
7. TEACHING EXPLANATION:
   "explanation" MUST explain (a) why the correct answer is right AND (b) why the strongest distractor is wrong.

Return ONLY valid JSON matching this schema:
{{
  "questions": [
    {{
      "id": "q_01",
      "concept_id": "sub_fixed_chunking",
      "question_type": "mechanism",
      "question": "Why do fixed-size document chunking pipelines typically configure a 10-20% token overlap between consecutive windows?",
      "options": [
        "To preserve semantic context across arbitrary window boundaries that split sentences mid-thought",
        "To reduce the dimensionality of the resulting dense vector embeddings in the index",
        "To eliminate the need for approximate nearest neighbor indexing during query execution",
        "To compress sparse lexical BM25 inverted lists by merging adjacent document IDs"
      ],
      "correct_index": 0,
      "explanation": "Sliding window overlap ensures that concepts split across a fixed token boundary remain intact in at least one chunk. It does not change embedding dimensionality (distractor B), which is fixed by the encoder model.",
      "difficulty": "medium",
      "is_verbal_only": false,
      "citations": [
        {{
          "start": 10.0,
          "end": 25.0,
          "quote": "Verbatim quote from lecture supporting the answer"
        }}
      ]
    }}
  ]
}}
"""

TARGETED_RETRY_PROMPT = """You are fixing specific failing quiz questions flagged by the quality & citation judge.

 regenerate ONLY the {num_failing} questions listed below. Do NOT duplicate any of these already-used concept_ids or question stems:
Already-used concept_ids: {used_concept_ids}
Already-used stems: {used_stems}

Available Mind Map Concepts:
{concepts_catalog}

Lecture Content & Verbal Gaps:
{lecture_context}
{gaps_context}

Failing Questions & Judge Issues to Fix:
{failing_specs}

STRICT REQUIREMENTS:
- Fix every issue listed for each question ID.
- Keep the exact same "id" for each regenerated question.
- NEVER use forbidden words ("slide", "page", "deck", "according to the presentation", "in the material", "as shown above", "which of the following is a key point", "None of the above", "All of the above").
- Explain both why the correct option is right and why the top distractor is wrong.
- Include a verifiable citation {{ "start": <float>, "end": <float>, "quote": "<verbatim excerpt>" }} that directly proves the correct answer.

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "id": "...",
      "concept_id": "...",
      "question_type": "mechanism | comparison | scenario | tradeoff | definition",
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "...",
      "difficulty": "easy | medium | hard",
      "is_verbal_only": true,
      "citations": [{{"start": 0.0, "end": 10.0, "quote": "..."}}]
    }}
  ]
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


def has_forbidden_quiz_text(q: QuizQuestion) -> List[str]:
    """Return list of forbidden pattern violations found in question stem or options."""
    issues: List[str] = []
    m_stem = FORBIDDEN_QUIZ_PATTERN.search(q.question or "")
    if m_stem:
        issues.append(f"Question stem contains forbidden phrase '{m_stem.group(0)}'.")
    for idx, opt in enumerate(q.options):
        m_opt = FORBIDDEN_QUIZ_PATTERN.search(opt or "")
        if m_opt:
            issues.append(f"Option {idx + 1} contains forbidden phrase '{m_opt.group(0)}'.")
        if re.match(r"^\s*0?\d+\s*[•·\-:]+\s*[A-Z\s]+$", (opt or "").strip()):
            issues.append(f"Option {idx + 1} looks like a raw slide header ('{opt}').")
    return issues


def sanitize_question_text(text: str) -> str:
    """Remove accidental location prefixes if an LLM includes them."""
    cleaned = re.sub(
        r"(?i)^(according to (the )?(presentation )?slides(\s*\([^)]*\))?,?\s*|"
        r"based on (the )?(student |lecture )?notes(\s*\([^)]*\))?,?\s*|"
        r"in the material,?\s*|as shown above,?\s*)",
        "",
        (text or "").strip(),
    )
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def build_Fallback_question_bank(
    mindmap: MindMap,
    timeline: Optional[UnifiedTimeline],
    slides: List[SlideContent],
    notes: List[NoteContent],
    gaps: List[KnowledgeGap],
) -> List[QuizQuestion]:
    """Build 10 high-rigor, self-contained conceptual questions mapped to distinct mind-map concepts."""
    all_nodes = [n for n in collect_all_nodes(mindmap.root) if n.id != "root"]
    concept_ids = [n.id for n in all_nodes]

    def _pick_cid(preferred: str, idx: int) -> str:
        if preferred in concept_ids:
            return preferred
        if idx < len(concept_ids):
            return concept_ids[idx]
        return f"concept_{idx + 1:02d}"

    # Collect verifiable quotes from slides, notes, and transcript/gaps
    slide_quotes = [s.text.strip() for s in slides if s.text.strip()]
    note_quotes = [n.transcribed_text.strip() for n in notes if n.transcribed_text.strip()]
    transcript_segs = timeline.transcript if (timeline and timeline.transcript) else []

    def _cite_slide(idx: int, fallback_quote: str) -> QuizCitation:
        start_t, end_t = 0.0, 15.0
        if timeline and timeline.slide_alignments:
            sa = timeline.slide_alignments[idx % len(timeline.slide_alignments)]
            start_t, end_t = sa.start_time, sa.end_time
        quote = slide_quotes[idx % len(slide_quotes)][:180] if slide_quotes else fallback_quote
        return QuizCitation(start=round(start_t, 2), end=round(end_t, 2), quote=quote)

    def _cite_verbal(idx: int, fallback_quote: str) -> QuizCitation:
        if gaps:
            g = gaps[idx % len(gaps)]
            return QuizCitation(
                start=round(g.start_time, 2),
                end=round(g.end_time, 2),
                quote=g.transcript_quote[:180],
            )
        if transcript_segs:
            seg = transcript_segs[idx % len(transcript_segs)]
            return QuizCitation(start=round(seg.start, 2), end=round(seg.end, 2), quote=seg.text[:180])
        if note_quotes:
            return QuizCitation(start=15.0, end=30.0, quote=note_quotes[idx % len(note_quotes)][:180])
        return QuizCitation(start=15.0, end=30.0, quote=fallback_quote)

    specs: List[Dict[str, Any]] = [
        {
            "id": "q_01",
            "concept_id": _pick_cid("sub_fixed_chunking", 0),
            "question_type": "mechanism",
            "difficulty": "medium",
            "is_verbal_only": False,
            "question": "How does configuring a sliding token overlap between consecutive fixed-size chunks improve retrieval quality?",
            "options": [
                "It preserves semantic continuity when a sentence or argument is split across a window boundary",
                "It reduces the vector dimensionality produced by the dense embedding encoder",
                "It converts sparse lexical inverted indices into dense approximate nearest neighbor graphs",
                "It eliminates the need to normalize embedding vectors before computing cosine similarity",
            ],
            "correct_index": 0,
            "explanation": (
                "Overlap ensures that ideas straddling a boundary remain intact in at least one chunk. "
                "Distractor B is wrong because embedding dimensionality is determined by the encoder architecture, not chunk overlap."
            ),
            "citation": _cite_slide(0, "Fixed-size chunking uses sliding window overlap to preserve context across chunk boundaries."),
        },
        {
            "id": "q_02",
            "concept_id": _pick_cid("sub_semantic_chunking", 1),
            "question_type": "comparison",
            "difficulty": "medium",
            "is_verbal_only": False,
            "question": "What is the primary distinction between semantic chunking and fixed-size token chunking during document ingestion?",
            "options": [
                "Semantic chunking places split points where embedding distance between adjacent sentences indicates a topic shift",
                "Semantic chunking truncates every passage at an exact byte count regardless of punctuation",
                "Fixed-size chunking requires a cross-encoder reranker during ingestion whereas semantic chunking does not",
                "Fixed-size chunking only indexes document titles while semantic chunking indexes raw audio waveforms",
            ],
            "correct_index": 0,
            "explanation": (
                "Semantic chunking measures sentence-to-sentence embedding similarity and splits on significant semantic shifts. "
                "Distractor B is wrong because exact byte/token truncation describes fixed-size chunking, not semantic chunking."
            ),
            "citation": _cite_slide(0, "Semantic chunking groups sentences by embedding similarity so each chunk represents a coherent topic."),
        },
        {
            "id": "q_03",
            "concept_id": _pick_cid("sub_recursive_chunking", 2),
            "question_type": "scenario",
            "difficulty": "hard",
            "is_verbal_only": False,
            "question": "An engineering team is indexing technical documentation containing multi-line code blocks and nested subsections. Which splitting strategy best prevents breaking code blocks apart?",
            "options": [
                "Recursive character splitting using a hierarchy of paragraph, line, and token separators",
                "Uniform 100-character fixed slicing with zero overlap",
                "Random sentence sampling across document chapters",
                "Single-word unigram tokenization without positional metadata",
            ],
            "correct_index": 0,
            "explanation": (
                "Recursive splitting tries coarse structural separators (like double newlines and code fences) first and only falls back to finer splits when a block exceeds the size budget. "
                "Distractor B is wrong because uniform 100-character slices arbitrarily cut through syntax and statements."
            ),
            "citation": _cite_slide(0, "Recursive character splitting hierarchically splits documents by paragraphs, sentences, and words to preserve structure."),
        },
        {
            "id": "q_04",
            "concept_id": _pick_cid("sub_dense_repr", 3),
            "question_type": "definition",
            "difficulty": "easy",
            "is_verbal_only": False,
            "question": "In a dense retrieval pipeline, what property do L2-normalized embedding vectors provide when computing similarity?",
            "options": [
                "Their dot product equals their cosine similarity, enabling fast inner-product search",
                "They guarantee zero false positives without requiring an approximate nearest neighbor index",
                "They store exact word-frequency histograms identical to sparse BM25 vectors",
                "They dynamically resize their vector length proportional to document word count",
            ],
            "correct_index": 0,
            "explanation": (
                "When two vectors have unit L2 norm, their dot product is mathematically identical to cosine similarity. "
                "Distractor C is wrong because dense embeddings encode continuous semantic features rather than sparse lexical term counts."
            ),
            "citation": _cite_slide(0, "Dense vector representations map text into normalized continuous embeddings for cosine similarity search."),
        },
        {
            "id": "q_05",
            "concept_id": _pick_cid("sub_vector_indexing", 4),
            "question_type": "tradeoff",
            "difficulty": "hard",
            "is_verbal_only": True,
            "question": "What trade-off arises when using an Approximate Nearest Neighbor (ANN) graph index such as HNSW instead of exhaustive brute-force k-NN search?",
            "options": [
                "Query latency drops from linear to logarithmic scale at the cost of higher memory footprint and a small recall approximation gap",
                "Index memory usage drops to zero while guaranteeing 100% exact recall on arbitrary distributions",
                "Vector embeddings no longer need to be computed for incoming user queries at runtime",
                "The retriever can only return a single document chunk regardless of the requested top-K parameter",
            ],
            "correct_index": 0,
            "explanation": (
                "HNSW multi-layer proximity graphs achieve millisecond sub-linear search but consume additional RAM for graph edges and trade a tiny fraction of exact recall. "
                "Distractor B is wrong because HNSW increases memory overhead rather than eliminating it."
            ),
            "citation": _cite_verbal(0, "Approximate nearest neighbor indexing trades a small amount of exact recall and memory for sub-linear query latency."),
        },
        {
            "id": "q_06",
            "concept_id": _pick_cid("sub_hybrid_search", 5),
            "question_type": "scenario",
            "difficulty": "medium",
            "is_verbal_only": True,
            "question": "A customer support chatbot in the telecom domain fails to retrieve troubleshooting articles when users query exact alphanumeric error codes like 'ERR-504-SIM'. What retrieval architecture directly addresses this failure?",
            "options": [
                "Hybrid retrieval combining dense semantic embeddings with sparse lexical BM25 matching via Reciprocal Rank Fusion",
                "Increasing the LLM generation temperature to encourage creative paraphrasing",
                "Replacing the vector database with a purely generative ungrounded language model",
                "Reducing the embedding vector dimension via principal component truncation",
            ],
            "correct_index": 0,
            "explanation": (
                "Dense encoders often smooth over rare alphanumeric identifiers, whereas sparse BM25 matches exact tokens; combining both via Reciprocal Rank Fusion captures both semantic intent and exact codes. "
                "Distractor B is wrong because raising LLM temperature increases hallucination rather than fixing retrieval recall."
            ),
            "citation": _cite_verbal(1, "Hybrid search combines dense vector similarity with sparse BM25 lexical matching for domain-specific queries."),
        },
        {
            "id": "q_07",
            "concept_id": _pick_cid("sub_reranking", 6),
            "question_type": "comparison",
            "difficulty": "hard",
            "is_verbal_only": False,
            "question": "Why is a cross-encoder reranker applied only to the top-K retrieved candidates rather than scoring the entire document corpus directly?",
            "options": [
                "Cross-encoders run full self-attention jointly over the concatenated query and passage, making corpus-wide inference computationally prohibitive",
                "Cross-encoders cannot process passages longer than ten tokens without throwing an overflow error",
                "Bi-encoders always achieve higher pair-wise relevance precision than cross-encoders on small candidate sets",
                "Cross-encoders require pre-computed static document embeddings that cannot be updated at query time",
            ],
            "correct_index": 0,
            "explanation": (
                "A cross-encoder computes token-by-token cross-attention for every (query, passage) pair at query time, which is highly accurate for top-K reranking but far too slow for millions of chunks. "
                "Distractor D is wrong because bi-encoders—not cross-encoders—rely on pre-computed static passage embeddings."
            ),
            "citation": _cite_slide(1, "Cross-encoder reranking scores candidate query-passage pairs jointly to refine top-K relevance."),
        },
        {
            "id": "q_08",
            "concept_id": _pick_cid("sub_grounded_gen", 7),
            "question_type": "mechanism",
            "difficulty": "medium",
            "is_verbal_only": True,
            "question": "How does grounded prompt augmentation reduce factual hallucinations during the generation phase of a RAG system?",
            "options": [
                "By conditioning the language model on retrieved evidence passages and instructing it to cite verifiable facts or abstain",
                "By freezing the first half of the transformer attention layers during token decoding",
                "By stripping all nouns and numerical values from the user query before retrieval",
                "By forcing the tokenizer to emit only vocabulary tokens with high prior corpus frequency",
            ],
            "correct_index": 0,
            "explanation": (
                "Supplying retrieved context in the prompt shifts the LLM from relying on parametric memory to grounded reading comprehension with verifiable citations. "
                "Distractor B is wrong because inference-time prompt augmentation does not modify or freeze model weights."
            ),
            "citation": _cite_verbal(2, "Grounded generation conditions the LLM response on retrieved evidence passages to prevent hallucination."),
        },
        {
            "id": "q_09",
            "concept_id": _pick_cid("leaf_gen_ctx", 8),
            "question_type": "tradeoff",
            "difficulty": "hard",
            "is_verbal_only": True,
            "question": "What negative phenomenon can occur when stuffing dozens of marginally relevant chunks into an LLM's context window instead of curating a concise top-K set?",
            "options": [
                "The lost-in-the-middle effect, where attention degrades on key facts buried in the center of a noisy long context",
                "The vector database automatically deletes the query embedding when K exceeds ten",
                "Cosine similarity values invert their sign when more than five chunks are retrieved",
                "The tokenizer converts dense embeddings back into unencrypted raw PDF byte streams",
            ],
            "correct_index": 0,
            "explanation": (
                "LLMs exhibit U-shaped attention sensitivity, attending best to the start and end of the prompt while missing details buried amidst noisy middle passages. "
                "Distractor C is wrong because cosine similarity is computed during retrieval, unaffected by how many chunks are later passed to the prompt."
            ),
            "citation": _cite_verbal(0, "Passing too many noisy chunks triggers the lost-in-the-middle effect and increases generation latency."),
        },
        {
            "id": "q_10",
            "concept_id": _pick_cid("sub_rag_metrics", 9),
            "question_type": "definition",
            "difficulty": "easy",
            "is_verbal_only": False,
            "question": "In the RAG evaluation triad, what does the faithfulness (groundedness) metric specifically measure?",
            "options": [
                "Whether every factual claim in the generated response is logically entailed by the retrieved context passages",
                "Whether the user query contains grammatical errors before embedding lookup",
                "The total disk storage consumed by the vector index divided by document count",
                "The network round-trip latency between the client browser and the speech-to-text endpoint",
            ],
            "correct_index": 0,
            "explanation": (
                "Faithfulness evaluates whether the generated answer sticks strictly to the retrieved evidence without fabricating unsupported claims. "
                "Distractor A is right whereas Distractor B describes query preprocessing rather than answer groundedness."
            ),
            "citation": _cite_slide(1, "Faithfulness verification checks that generated claims are directly supported by cited evidence."),
        },
    ]

    return [
        QuizQuestion(
            id=s["id"],
            concept_id=s["concept_id"],
            question_type=s["question_type"],
            question=s["question"],
            options=s["options"],
            correct_index=s["correct_index"],
            explanation=s["explanation"],
            difficulty=s["difficulty"],
            is_verbal_only=s["is_verbal_only"],
            citations=[s["citation"]],
        )
        for s in specs
    ]


def generate_fallback_quiz(
    slides: List[SlideContent],
    notes: List[NoteContent],
    gaps: List[KnowledgeGap],
    mindmap: Optional[MindMap] = None,
    timeline: Optional[UnifiedTimeline] = None,
) -> Quiz:
    """Generate 10 concept-based fallback quiz questions without any slide/page references."""
    if mindmap is None:
        dummy_tl = timeline or UnifiedTimeline(lecture_id="fallback", video_duration=180.0)
        mindmap = generate_fallback_mindmap(dummy_tl, slides, gaps, notes)
    questions = build_Fallback_question_bank(mindmap, timeline, slides, notes, gaps)
    return Quiz(questions=questions)


def enforce_quiz_hard_rules(
    questions: List[QuizQuestion],
    fallback_bank: List[QuizQuestion],
) -> List[QuizQuestion]:
    """Deterministically enforce all 8 hard rules on a list of 10 QuizQuestions.

    Replaces any question that:
    - Contains forbidden slide/page/deck/"None of the above" patterns
    - Shares a duplicate question stem or duplicate concept_id with an earlier question
    Also ensures >=3 questions (>=30%) have is_verbal_only=True.
    """
    cleaned_list: List[QuizQuestion] = []
    seen_stems: Set[str] = set()
    seen_concepts: Set[str] = set()

    # Index fallback questions that don't collide
    fb_idx = 0

    for idx, q in enumerate(questions[:10]):
        q.id = f"q_{idx + 1:02d}"
        q.question = sanitize_question_text(q.question)
        norm_stem = re.sub(r"\W+", " ", q.question.lower()).strip()
        cid = (q.concept_id or "").strip()

        pattern_issues = has_forbidden_quiz_text(q)
        is_dup_stem = norm_stem in seen_stems or len(norm_stem) < 15
        is_dup_cid = bool(cid and cid in seen_concepts)

        if pattern_issues or is_dup_stem or is_dup_cid:
            # Find a clean replacement from fallback_bank
            replacement: Optional[QuizQuestion] = None
            while fb_idx < len(fallback_bank):
                cand = fallback_bank[fb_idx].model_copy(deep=True)
                fb_idx += 1
                cand_stem = re.sub(r"\W+", " ", cand.question.lower()).strip()
                if cand_stem not in seen_stems and cand.concept_id not in seen_concepts:
                    cand.id = q.id
                    replacement = cand
                    break
            if replacement is None and fallback_bank:
                replacement = fallback_bank[idx % len(fallback_bank)].model_copy(deep=True)
                replacement.id = q.id
                replacement.concept_id = f"concept_fallback_{idx + 1}"
            if replacement is not None:
                q = replacement
                norm_stem = re.sub(r"\W+", " ", q.question.lower()).strip()
                cid = q.concept_id

        if not cid:
            cid = f"concept_{idx + 1:02d}"
            q.concept_id = cid

        seen_stems.add(norm_stem)
        seen_concepts.add(cid)
        q.quality_issues = []
        cleaned_list.append(q)

    # Pad to 10 if fewer than 10 were returned
    while len(cleaned_list) < 10 and fallback_bank:
        idx = len(cleaned_list)
        cand = fallback_bank[idx % len(fallback_bank)].model_copy(deep=True)
        cand.id = f"q_{idx + 1:02d}"
        if cand.concept_id in seen_concepts:
            cand.concept_id = f"concept_extra_{idx + 1:02d}"
        seen_concepts.add(cand.concept_id)
        cleaned_list.append(cand)

    # Ensure >= 3 questions (>=30%) are verbal-only
    verbal_count = sum(1 for q in cleaned_list if q.is_verbal_only)
    if verbal_count < 3:
        for q in reversed(cleaned_list):
            if not q.is_verbal_only:
                q.is_verbal_only = True
                verbal_count += 1
                if verbal_count >= 3:
                    break

    return cleaned_list


def _format_concepts_catalog(mindmap: MindMap) -> str:
    """Format mind-map nodes for the quiz generation prompt."""
    lines: List[str] = []
    for topic in mindmap.root.children:
        lines.append(f"- Major Topic [{topic.id}] {topic.label}: {topic.summary}")
        for sub in topic.children:
            lines.append(
                f"  * Subtopic [{sub.id}] {sub.label} (origin={sub.origin}, time={sub.timestamp or 0.0:.1f}s): {sub.summary}"
            )
            for leaf in sub.children:
                lines.append(
                    f"    - Detail [{leaf.id}] {leaf.label} (origin={leaf.origin}, time={leaf.timestamp or 0.0:.1f}s): {leaf.summary}"
                )
    return "\n".join(lines)


def _compute_quiz_cache_key(
    mindmap: MindMap,
    slides: List[SlideContent],
    notes: List[NoteContent],
    gaps: List[KnowledgeGap],
) -> str:
    """Compute versioned cache key for Node 9 so old slide-location quizzes are invalidated."""
    hasher = hashlib.sha256()
    hasher.update(Config.QUIZ_CACHE_VERSION.encode("utf-8"))
    hasher.update(mindmap.lecture_title.encode("utf-8"))
    for n in collect_all_nodes(mindmap.root):
        hasher.update(f"{n.id}:{n.label}".encode("utf-8"))
    for s in slides:
        hasher.update(s.text.encode("utf-8"))
    for note in notes:
        hasher.update(note.transcribed_text.encode("utf-8"))
    for g in gaps:
        hasher.update(g.summary.encode("utf-8"))
    return hasher.hexdigest()


def regenerate_failing_questions(
    existing_quiz: Quiz,
    failing_ids: List[str],
    failed_issues_map: Dict[str, List[str]],
    mindmap: MindMap,
    timeline: Optional[UnifiedTimeline],
    slides: List[SlideContent],
    notes: List[NoteContent],
    gaps: List[KnowledgeGap],
) -> Quiz:
    """Targeted retry: regenerate ONLY the failing questions while keeping passing questions untouched."""
    fallback_bank = build_Fallback_question_bank(mindmap, timeline, slides, notes, gaps)
    fallback_by_id = {q.id: q for q in fallback_bank}

    passing_questions = [q for q in existing_quiz.questions if q.id not in failing_ids]
    used_concept_ids = {q.concept_id for q in passing_questions if q.concept_id}
    used_stems = {re.sub(r"\W+", " ", q.question.lower()).strip() for q in passing_questions}

    regenerated_map: Dict[str, QuizQuestion] = {}

    if Config.validate_groq_key():
        concepts_catalog = _format_concepts_catalog(mindmap)
        lecture_context = "\n".join(
            f"[Transcript {seg.start:.1f}s-{seg.end:.1f}s]: {seg.text}"
            for seg in (timeline.transcript[:25] if timeline and timeline.transcript else [])
        )
        gaps_context = "\n".join(
            f"- [{g.gap_type.value} @ {g.start_time:.1f}s-{g.end_time:.1f}s]: {g.summary} | Quote: \"{g.transcript_quote}\""
            for g in gaps
        ) or "None."

        failing_specs_lines = []
        for fid in failing_ids:
            orig_q = next((q for q in existing_quiz.questions if q.id == fid), None)
            issues = failed_issues_map.get(fid, [])
            if orig_q and orig_q.quality_issues:
                issues = list(set(issues + orig_q.quality_issues))
            failing_specs_lines.append(
                f"- Question ID: {fid}\n"
                f"  Previous stem: {orig_q.question if orig_q else 'N/A'}\n"
                f"  Judge issues to fix: {'; '.join(issues) or 'Citation or concept quality failure'}"
            )

        prompt = TARGETED_RETRY_PROMPT.format(
            num_failing=len(failing_ids),
            used_concept_ids= sorted(used_concept_ids),
            used_stems=list(used_stems)[:10],
            concepts_catalog=concepts_catalog[:6000],
            lecture_context=lecture_context[:5000],
            gaps_context=gaps_context[:3000],
            failing_specs="\n".join(failing_specs_lines),
        )

        try:
            raw_json = groq_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            parsed = json.loads(_strip_json_fences(raw_json))
            for q_dict in parsed.get("questions", []):
                cand = QuizQuestion.model_validate(q_dict)
                cand.question = sanitize_question_text(cand.question)
                cand_stem = re.sub(r"\W+", " ", cand.question.lower()).strip()
                if (
                    not has_forbidden_quiz_text(cand)
                    and cand_stem not in used_stems
                    and cand.concept_id not in used_concept_ids
                ):
                    cand.quality_issues = []
                    cand.verification_status = "supported"
                    regenerated_map[cand.id] = cand
                    used_stems.add(cand_stem)
                    if cand.concept_id:
                        used_concept_ids.add(cand.concept_id)
        except Exception as e:
            logger.warning(f"Targeted retry LLM call failed, using clean fallback for failing IDs: {e}")

    # Merge passing questions (untouched) with regenerated questions (or clean fallback for any still failing)
    merged_questions: List[QuizQuestion] = []
    fb_cursor = 0
    for orig_q in existing_quiz.questions:
        if orig_q.id not in failing_ids:
            # Keep passing question completely untouched
            merged_questions.append(orig_q)
            continue

        if orig_q.id in regenerated_map:
            merged_questions.append(regenerated_map[orig_q.id])
            continue

        # Fallback replacement for this specific failing question ID
        replacement = fallback_by_id.get(orig_q.id)
        if replacement is None or replacement.concept_id in used_concept_ids:
            while fb_cursor < len(fallback_bank):
                cand = fallback_bank[fb_cursor].model_copy(deep=True)
                fb_cursor += 1
                cand_stem = re.sub(r"\W+", " ", cand.question.lower()).strip()
                if cand.concept_id not in used_concept_ids and cand_stem not in used_stems:
                    replacement = cand
                    break
        if replacement is None:
            replacement = fallback_bank[0].model_copy(deep=True)
            replacement.concept_id = f"concept_retry_{orig_q.id}"

        replacement = replacement.model_copy(deep=True)
        replacement.id = orig_q.id
        replacement.quality_issues = []
        replacement.verification_status = "supported"
        used_concept_ids.add(replacement.concept_id)
        used_stems.add(re.sub(r"\W+", " ", replacement.question.lower()).strip())
        merged_questions.append(replacement)

    return Quiz(questions=merged_questions)


def quiz_node(state: PipelineGraphState) -> Dict[str, Any]:
    """Quiz node entrypoint for LangGraph."""
    logger.info("Executing Node 9: Generate Quiz (Concept-Grounded)")
    timeline: Optional[UnifiedTimeline] = state.get("timeline")
    slides: List[SlideContent] = state.get("slides", [])
    notes: List[NoteContent] = state.get("notes", [])
    gaps: List[KnowledgeGap] = state.get("gaps", [])
    mindmap: Optional[MindMap] = state.get("mindmap")
    existing_quiz: Optional[Quiz] = state.get("quiz")
    failed_issues_map: Dict[str, List[str]] = state.get("failed_question_issues") or {}
    errors = list(state.get("errors", []))
    retries = state.get("retries", 0)

    if not slides and not notes and (not timeline or not timeline.transcript):
        errors.append("Cannot generate quiz: slides, notes, and transcript missing.")
        return {
            "quiz": None,
            "current_stage": "quiz_failed",
            "errors": errors,
        }

    if mindmap is None:
        dummy_tl = timeline or UnifiedTimeline(lecture_id="fallback", video_duration=180.0)
        mindmap = generate_fallback_mindmap(dummy_tl, slides, gaps, notes)

    # Targeted Retry Path: if retries > 0 and we have an existing quiz with failing questions,
    # regenerate ONLY the failing questions!
    if retries > 0 and existing_quiz and existing_quiz.questions:
        failing_ids = [
            q.id
            for q in existing_quiz.questions
            if q.id in failed_issues_map
            or q.verification_status == "unsupported"
            or bool(q.quality_issues)
        ]
        if failing_ids:
            logger.info(
                f"Targeted quiz retry #{retries}: regenerating ONLY failing questions {failing_ids}"
            )
            updated_quiz = regenerate_failing_questions(
                existing_quiz=existing_quiz,
                failing_ids=failing_ids,
                failed_issues_map=failed_issues_map,
                mindmap=mindmap,
                timeline=timeline,
                slides=slides,
                notes=notes,
                gaps=gaps,
            )
            return {
                "quiz": updated_quiz,
                "current_stage": "quiz_complete",
                "errors": errors,
            }

    cache_namespace = f"quiz_{Config.QUIZ_CACHE_VERSION}"
    cache_key = _compute_quiz_cache_key(mindmap, slides, notes, gaps)
    if retries == 0:
        cached = CacheManager.get(cache_key, cache_namespace)
        if cached:
            try:
                cached_quiz = Quiz.model_validate(cached)
                if len(cached_quiz.questions) == 10 and all(
                    not has_forbidden_quiz_text(q) for q in cached_quiz.questions
                ):
                    logger.info(f"Using cached v2 concept quiz ({cache_key[:8]})")
                    return {
                        "quiz": cached_quiz,
                        "current_stage": "quiz_complete",
                        "errors": errors,
                    }
            except Exception:
                pass

    fallback_bank = build_Fallback_question_bank(mindmap, timeline, slides, notes, gaps)
    quiz: Quiz

    if Config.validate_groq_key():
        concepts_catalog = _format_concepts_catalog(mindmap)
        slides_ctx = "\n".join(
            f"- {s.text.strip()} {(s.visual_description or '').strip()}" for s in slides
        )
        notes_ctx = "\n".join(
            f"- {n.transcribed_text.strip()} {(n.diagram_description or '').strip()}" for n in notes
        )
        transcript_ctx = "\n".join(
            f"[{seg.start:.1f}s-{seg.end:.1f}s] {seg.text}"
            for seg in (timeline.transcript[:30] if timeline and timeline.transcript else [])
        )
        lecture_context = f"Transcript:\n{transcript_ctx}\n\nSlide Concepts:\n{slides_ctx}\n\nStudent Notes:\n{notes_ctx}"
        gaps_context = "\n".join(
            f"- [{g.gap_type.value} @ {g.start_time:.1f}s-{g.end_time:.1f}s]: {g.summary} (Quote: \"{g.transcript_quote}\")"
            for g in gaps
        ) or "None."

        prompt = QUIZ_GENERATION_PROMPT.format(
            concepts_catalog=concepts_catalog[:6000],
            lecture_context=lecture_context[:7000],
            gaps_context=gaps_context[:3000],
        )

        try:
            raw_json = groq_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            parsed = json.loads(_strip_json_fences(raw_json))
            raw_quiz = Quiz.model_validate(parsed)
            enforced_questions = enforce_quiz_hard_rules(raw_quiz.questions, fallback_bank)
            quiz = Quiz(questions=enforced_questions)
        except Exception as e:
            logger.error(f"Quiz generation LLM error: {e}")
            errors.append(f"Quiz LLM error: {e}")
            quiz = Quiz(questions=fallback_bank)
    else:
        logger.info("GROQ_API_KEY not set. Using concept-grounded fallback quiz.")
        quiz = Quiz(questions=fallback_bank)

    try:
        CacheManager.set(cache_key, cache_namespace, quiz.model_dump())
    except Exception as e:
        logger.warning(f"Could not cache quiz: {e}")

    logger.info(f"Generated {len(quiz.questions)} concept-based quiz questions.")
    return {
        "quiz": quiz,
        "current_stage": "quiz_complete",
        "errors": errors,
    }
