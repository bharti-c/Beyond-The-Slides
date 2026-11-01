"""Beyond the slides: Main Streamlit Application.

An interactive AI agent transforming lecture recordings, slides, and handwritten notes
into a verified course with synchronized video seeking, mind map, and quiz.
Styled in a warm Peach & Coral (#FF8B66 / #FFF2EB / #FFE5D9 / #FFFBF9) Raleway theme.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# Configure page metadata - sidebar collapsed by default for full-canvas modern view
st.set_page_config(
    page_title="Beyond the slides",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import importlib
from models import PipelineGraphState
from pipeline.graph import run_pipeline
import ui.styles
importlib.reload(ui.styles)
from ui.styles import WHITE_BLUISH_THEME_CSS
from ui.video_player import format_seconds, render_synchronized_video_player, render_timestamp_seek_button
import ui.mindmap_viewer
importlib.reload(ui.mindmap_viewer)
from ui.mindmap_viewer import render_interactive_mindmap
import ui.components
importlib.reload(ui.components)
from ui.components import (
    render_gap_card,
    render_quiz_tab,
    render_feedback_controls,
)

# Inject custom Peach & Coral CSS styling
st.markdown(WHITE_BLUISH_THEME_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize app session state variables."""
    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None
    if "video_seek_time" not in st.session_state:
        st.session_state["video_seek_time"] = 0.0
    if "current_lecture_id" not in st.session_state:
        st.session_state["current_lecture_id"] = None
    if "use_sample_data" not in st.session_state:
        st.session_state["use_sample_data"] = False


init_session_state()

# -------------------------------------------------------------
# 2.5D Isometric Hero Illustration (viewBox 0 0 1600 900)
# Palette: #8A4F7D (Grape Soda), #887880 (Rosy Granite), #88A096 (Muted Teal)
# Single-line, comment-free SVG to prevent Markdown code-block escaping
# -------------------------------------------------------------
HERO_ISOMETRIC_SVG = (
    '<div style="display: flex; justify-content: center; width: 100%; max-width: 1000px; margin: 1rem auto 0 auto; overflow: hidden; border-radius: 16px; background: transparent;">'
    '<svg viewBox="0 0 1600 900" width="100%" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<radialGradient id="ambientGlow" cx="50%" cy="46%" r="48%">'
    '<stop offset="0%" stop-color="#8A4F7D" stop-opacity="0.35"/>'
    '<stop offset="42%" stop-color="#88A096" stop-opacity="0.22"/>'
    '<stop offset="76%" stop-color="#887880" stop-opacity="0.08"/>'
    '<stop offset="100%" stop-color="#887880" stop-opacity="0"/>'
    '</radialGradient>'
    '<radialGradient id="bulbHalo" cx="50%" cy="42%" r="45%">'
    '<stop offset="0%" stop-color="#8A4F7D" stop-opacity="0.50"/>'
    '<stop offset="55%" stop-color="#88A096" stop-opacity="0.25"/>'
    '<stop offset="100%" stop-color="#887880" stop-opacity="0"/>'
    '</radialGradient>'
    '<radialGradient id="bulbSphere" cx="46%" cy="38%" r="58%">'
    '<stop offset="0%" stop-color="#F4F1F3"/>'
    '<stop offset="48%" stop-color="#88A096"/>'
    '<stop offset="85%" stop-color="#8A4F7D"/>'
    '<stop offset="100%" stop-color="#6B3A60"/>'
    '</radialGradient>'
    '<linearGradient id="centerPlatTop" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#B8C8C1"/>'
    '<stop offset="100%" stop-color="#88A096"/>'
    '</linearGradient>'
    '<linearGradient id="centerPlatSide" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#8A4F7D"/>'
    '<stop offset="100%" stop-color="#887880"/>'
    '</linearGradient>'
    '<linearGradient id="subPlatTop" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#F4F1F3"/>'
    '<stop offset="100%" stop-color="#88A096"/>'
    '</linearGradient>'
    '<linearGradient id="subPlatSide" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#887880"/>'
    '<stop offset="100%" stop-color="#8A4F7D"/>'
    '</linearGradient>'
    '</defs>'
    '<ellipse cx="800" cy="450" rx="680" ry="390" fill="url(#ambientGlow)"/>'
    '<path d="M800 180 L1340 460 L800 780 L260 460 Z" stroke="#8A4F7D" stroke-width="5" stroke-linejoin="round" opacity="0.65"/>'
    '<path d="M800 192 L1322 460 L800 766 L278 460 Z" stroke="#88A096" stroke-width="2.5" stroke-linejoin="round" opacity="0.75"/>'
    '<path d="M560 345 L710 432 M1040 345 L890 432 M560 595 L710 508 M1040 595 L890 508" stroke="#8A4F7D" stroke-width="4.5" stroke-linecap="round"/>'
    '<path d="M568 353 L718 440 M1032 353 L882 440 M568 603 L718 516 M1032 603 L882 524" stroke="#88A096" stroke-width="2" stroke-linecap="round"/>'
    '<g transform="translate(270, 145)">'
    '<polygon points="210,126 400,224 210,322 20,224" fill="url(#subPlatSide)" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="210,110 400,208 210,306 20,208" fill="url(#subPlatTop)" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="92,182 225,112 225,20 92,90" fill="#8A4F7D" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="115,196 252,124 252,32 115,104" fill="#FFFFFF" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="132,164 176,141 176,102 132,125" fill="#88A096" stroke="#231920" stroke-width="2" stroke-linejoin="round"/>'
    '<path d="M192 102 L232 81 M192 118 L232 97 M192 134 L222 118" stroke="#231920" stroke-width="3.5" stroke-linecap="round"/>'
    '<polygon points="282,118 354,156 354,96 282,58" fill="#8A4F7D" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="310,94 310,122 332,116" fill="#FFFFFF" stroke="#231920" stroke-width="2" stroke-linejoin="round"/>'
    '<polygon points="210,192 292,234 234,276 152,234" fill="#FFFFFF" stroke="#231920" stroke-width="2.2" stroke-linejoin="round"/>'
    '<path d="M185 228 Q202 220 215 229 T245 226 M198 242 Q215 234 228 243 T258 240 M212 255 Q226 248 242 256" stroke="#231920" stroke-width="2.2" stroke-linecap="round" fill="none"/>'
    '</g>'
    '<g transform="translate(935, 165)">'
    '<polygon points="210,116 400,214 210,312 20,214" fill="url(#subPlatSide)" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="210,100 400,198 210,296 20,198" fill="url(#subPlatTop)" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<path d="M210 196 L205 138 M210 196 L112 204 M210 196 L308 198 M210 196 L222 256" stroke="#231920" stroke-width="3" stroke-linecap="round"/>'
    '<ellipse cx="210" cy="196" rx="28" ry="16" fill="#8A4F7D" stroke="#231920" stroke-width="2.4"/>'
    '<ellipse cx="205" cy="134" rx="21" ry="12" fill="#88A096" stroke="#231920" stroke-width="2.2"/>'
    '<ellipse cx="108" cy="204" rx="21" ry="12" fill="#88A096" stroke="#231920" stroke-width="2.2"/>'
    '<ellipse cx="312" cy="198" rx="21" ry="12" fill="#88A096" stroke="#231920" stroke-width="2.2"/>'
    '<ellipse cx="222" cy="258" rx="21" ry="12" fill="#88A096" stroke="#231920" stroke-width="2.2"/>'
    '</g>'
    '<g transform="translate(270, 495)">'
    '<polygon points="210,126 400,224 210,322 20,224" fill="url(#subPlatSide)" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="210,110 400,208 210,306 20,208" fill="url(#subPlatTop)" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<ellipse cx="172" cy="222" rx="30" ry="16" fill="#8A4F7D" stroke="#231920" stroke-width="2.4"/>'
    '<rect x="166" y="182" width="12" height="38" fill="#8A4F7D" stroke="#231920" stroke-width="2.2"/>'
    '<path d="M142 142 C142 184, 202 184, 202 142" stroke="#231920" stroke-width="5" stroke-linecap="round" fill="none"/>'
    '<rect x="148" y="76" width="48" height="86" rx="24" fill="#887880" stroke="#231920" stroke-width="2.4"/>'
    '<path d="M148 122 C148 148, 196 148, 196 122 L196 138 C196 162, 148 162, 148 138 Z" fill="#88A096" stroke="#231920" stroke-width="2.2"/>'
    '<path d="M225 144 L225 154 M236 132 L236 168 M247 118 L247 182 M258 138 L258 166 M269 128 L269 178 M280 112 L280 194 M291 136 L291 174 M302 150 L302 166" stroke="#231920" stroke-width="4.5" stroke-linecap="round"/>'
    '</g>'
    '<g transform="translate(935, 505)">'
    '<polygon points="210,126 400,224 210,322 20,224" fill="url(#subPlatSide)" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="210,110 400,208 210,306 20,208" fill="url(#subPlatTop)" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="185,136 286,188 188,242 86,190" fill="#8A4F7D" stroke="#231920" stroke-width="2.4" stroke-linejoin="round"/>'
    '<polygon points="185,144 272,188 188,233 100,189" fill="#FFFFFF" stroke="#231920" stroke-width="2" stroke-linejoin="round"/>'
    '<polygon points="222,146 258,165 246,174 210,155" fill="#887880" stroke="#231920" stroke-width="2" stroke-linejoin="round"/>'
    '<path d="M168 182 L176 198 L216 184" stroke="#88A096" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    '<path d="M168 182 L176 198 L216 184" stroke="#231920" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    '<circle cx="272" cy="236" r="24" fill="#8A4F7D" stroke="#231920" stroke-width="2.4"/>'
    '<circle cx="272" cy="236" r="9" fill="#88A096" stroke="#231920" stroke-width="2.2"/>'
    '</g>'
    '<g transform="translate(530, 80)">'
    '<ellipse cx="270" cy="240" rx="210" ry="200" fill="url(#bulbHalo)"/>'
    '<polygon points="270,302 520,432 270,562 20,432" fill="url(#centerPlatSide)" stroke="#231920" stroke-width="2.6" stroke-linejoin="round"/>'
    '<polygon points="270,282 520,412 270,542 20,412" fill="url(#centerPlatTop)" stroke="#231920" stroke-width="2.6" stroke-linejoin="round"/>'
    '<ellipse cx="270" cy="364" rx="38" ry="18" fill="#887880" stroke="#231920" stroke-width="2"/>'
    '<ellipse cx="270" cy="348" rx="44" ry="20" fill="#8A4F7D" stroke="#231920" stroke-width="2"/>'
    '<ellipse cx="270" cy="332" rx="48" ry="21" fill="#88A096" stroke="#231920" stroke-width="2"/>'
    '<path d="M270 42 C182 42, 134 104, 134 176 C134 234, 182 268, 206 312 C214 326, 236 336, 270 336 C304 336, 326 326, 334 312 C358 268, 406 234, 406 176 C406 104, 358 42, 270 42 Z" fill="url(#bulbSphere)" stroke="#231920" stroke-width="2.2"/>'
    '</g>'
    '</svg>'
    '</div>'
)

# -------------------------------------------------------------
# 1. Glassmorphic Hero, Green Gradient Typography & 2.5D Isometric Hero
# -------------------------------------------------------------
hero_html = (
    '<div class="landing-hero">'
    '<div class="landing-title-wrap">'
    '<h1 class="landing-title">Beyond the <span class="highlight-box">slides</span></h1>'
    '</div>'
    '<div class="glass-description-box">'
    'Synchronize lecture video, presentation slides, and handwritten student notes onto one unified master timeline. '
    'Discover what the professor said that is <strong>NOT</strong> on the slides, generating concept mind maps and citation-verified quizzes.'
    '</div>'
    f'{HERO_ISOMETRIC_SVG}'
    '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)

st.divider()

# -------------------------------------------------------------
# 2. Three-Column Dashed Upload Section & Top-Right Coral Preload Button
# -------------------------------------------------------------
sample_dir = Path(__file__).resolve().parent / "sample_data"
sample_video_file = sample_dir / "sample_video.mp4"
sample_slides_file = sample_dir / "sample_slides.pdf"
sample_notes_file = sample_dir / "sample_notes.png"
has_sample_files = sample_video_file.is_file() and sample_slides_file.is_file()

col_up_head, col_up_demo = st.columns([3.5, 1.2])
with col_up_head:
    st.markdown("### Upload Lecture Materials")
    if st.session_state["use_sample_data"]:
        st.caption("Sample lecture files preloaded (`sample_video.mp4`, `sample_slides.pdf`, `sample_notes.png`).")
    else:
        st.caption("Drop your lecture recording, slide deck PDF, and handwritten notes below.")

with col_up_demo:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if has_sample_files:
        if not st.session_state["use_sample_data"]:
            if st.button("Preload Demo Files", use_container_width=True, type="secondary"):
                st.session_state["use_sample_data"] = True
                st.rerun()
        else:
            if st.button("Clear Demo Files", use_container_width=True, type="secondary"):
                st.session_state["use_sample_data"] = False
                st.rerun()

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        '<div class="upload-zone-card">'
        '<span class="upload-zone-badge">Step 01</span>'
        '<div class="upload-title">Lecture Video</div>'
        '<div class="upload-hint">MP4, MKV, WebM, or M4A recording</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    uploaded_video = st.file_uploader(
        "Choose video file",
        type=["mp4", "mkv", "webm", "m4a"],
        help="Full video recording of the lecture",
        key="file_video",
        label_visibility="collapsed",
    )
    if uploaded_video:
        st.caption(f"Ready: `{uploaded_video.name}` ({round(uploaded_video.size / (1024*1024), 2)} MB)")
    elif st.session_state["use_sample_data"]:
        st.caption("Demo: `sample_video.mp4` loaded")

with col2:
    st.markdown(
        '<div class="upload-zone-card">'
        '<span class="upload-zone-badge">Step 02</span>'
        '<div class="upload-title">Presentation Slides</div>'
        '<div class="upload-hint">Official PDF Slide Deck (PPT)</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    uploaded_slides = st.file_uploader(
        "Choose PDF slides",
        type=["pdf"],
        help="Professor's official slide deck",
        key="file_slides",
        label_visibility="collapsed",
    )
    if uploaded_slides:
        st.caption(f"Ready: `{uploaded_slides.name}` ({round(uploaded_slides.size / 1024, 1)} KB)")
    elif st.session_state["use_sample_data"]:
        st.caption("Demo: `sample_slides.pdf` loaded")

with col3:
    st.markdown(
        '<div class="upload-zone-card">'
        '<span class="upload-zone-badge">Step 03</span>'
        '<div class="upload-title">Student Notes</div>'
        '<div class="upload-hint">Photos / Scans of Handwritten Notes</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    uploaded_notes = st.file_uploader(
        "Choose notes photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Photos or scans of student handwritten notebook",
        key="file_notes",
        label_visibility="collapsed",
    )
    if uploaded_notes:
        st.caption(f"Ready: {len(uploaded_notes)} note page(s) uploaded")
    elif st.session_state["use_sample_data"]:
        st.caption("Demo: `sample_notes.png` loaded")

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

ready_to_launch = bool(
    st.session_state["use_sample_data"]
    or (uploaded_video is not None and uploaded_slides is not None)
)

col_btn, col_info = st.columns([2.2, 4.8])
with col_btn:
    start_processing = st.button(
        "Launch Beyond the slides Pipeline",
        type="primary",
        disabled=not ready_to_launch,
        use_container_width=True,
    )

with col_info:
    if ready_to_launch:
        st.markdown(
            '<div style="padding-top: 8px; color: #231920; font-weight: 700; font-size: 0.94rem;">'
            'All materials ready. Click Launch to run the multimodal pipeline.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding-top: 8px; color: #231920; font-size: 0.9rem;">'
            'Provide at least a lecture video and slides PDF, or click <strong>Preload Demo Files</strong>.'
            '</div>',
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------
# 3. Pipeline Execution Workflow
# -------------------------------------------------------------
if start_processing:
    temp_dir = Path(tempfile.mkdtemp(prefix="relect_upload_"))

    if st.session_state["use_sample_data"] and (uploaded_video is None or uploaded_slides is None):
        vid_path = sample_video_file
        pdf_path = sample_slides_file
        notes_file_paths = [str(sample_notes_file)] if sample_notes_file.is_file() else []
    else:
        vid_path = temp_dir / uploaded_video.name
        with open(vid_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        pdf_path = temp_dir / uploaded_slides.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_slides.getbuffer())

        notes_file_paths = []
        if uploaded_notes:
            for idx, n_file in enumerate(uploaded_notes):
                n_path = temp_dir / f"note_{idx}_{n_file.name}"
                with open(n_path, "wb") as f:
                    f.write(n_file.getbuffer())
                notes_file_paths.append(str(n_path))

    progress_bar = st.progress(0, text="Initializing LangGraph pipeline...")
    status_box = st.empty()

    stages = [
        ("ingest", "Validating & Ingesting Files", 0.10),
        ("transcribe", "Extracting Audio & Transcribing via Groq Whisper", 0.25),
        ("parse_slides", "Extracting Text & Visuals from PDF Slides", 0.40),
        ("parse_notes", "Transcribing Handwritten Notes via Groq Vision", 0.55),
        ("align", "Computing Monotonic Dynamic Programming Alignment", 0.70),
        ("gap_detect", "Detecting Professor's Off-Slide Verbal Additions", 0.80),
        ("generate_mindmap", "Synthesizing Hierarchical Mind Map", 0.90),
        ("generate_quiz", "Generating 10 Slides & Notes Quiz Questions", 0.96),
        ("verify_citations", "Verifying Citations with LLM-as-Judge", 1.00),
    ]

    def on_step_update(node_name: str, current_state: dict):
        match = next((item for item in stages if item[0] == node_name), None)
        if match:
            _, desc, prog = match
            progress_bar.progress(prog, text=f"Stage: {desc}...")
            status_box.info(f"Processing Node: **{node_name}** - {desc}")

    try:
        with st.spinner("Executing multimodal pipeline..."):
            result_state = run_pipeline(
                video_path=str(vid_path),
                slides_pdf_path=str(pdf_path),
                notes_paths=notes_file_paths,
                on_step_callback=on_step_update,
            )
            st.session_state["pipeline_result"] = result_state
            st.session_state["current_lecture_id"] = result_state.get("lecture_id")
            progress_bar.progress(1.0, text="Pipeline processing complete!")
            status_box.success("Beyond the slides course successfully built.")
    except Exception as e:
        st.error(f"Pipeline encountered an error: {e}")
        st.exception(e)

# -------------------------------------------------------------
# 4. Results Presentation (4 Interactive Tabs)
# -------------------------------------------------------------
result: Optional[PipelineGraphState] = st.session_state.get("pipeline_result")

if result:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.divider()

    timeline = result.get("timeline")
    slides = result.get("slides", [])
    notes = result.get("notes", [])
    gaps = result.get("gaps", [])
    mindmap = result.get("mindmap")
    quiz = result.get("quiz")
    run_id = result.get("run_id", "run_local")

    st.markdown("### Synchronized Video Player")
    st.caption("Click any timestamp pill across the tabs below to instantly jump the video to that moment.")
    render_synchronized_video_player(result["video_path"])

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    tab_timeline, tab_gaps, tab_mindmap, tab_quiz = st.tabs([
        "Master Timeline",
        f"Verbal Gaps ({len(gaps)})",
        "Mind Map",
        f"Quiz ({len(quiz.questions) if quiz else 0})",
    ])

    with tab_timeline:
        st.subheader("Unified Multimodal Master Timeline")
        st.caption("Slides, handwritten student notes, and transcript aligned chronologically with monotonic order constraints.")
        render_feedback_controls(run_id=run_id, artifact_name="timeline", key_prefix="timeline_tab")
        st.divider()

        col_t1, col_t2 = st.columns([1, 1])

        with col_t1:
            st.markdown("#### Aligned Slide Deck")
            if timeline and timeline.slide_alignments:
                for sa in timeline.slide_alignments:
                    s_idx = sa.slide_index
                    slide_obj = slides[s_idx] if s_idx < len(slides) else None

                    with st.container():
                        st.markdown(
                            f'<div class="relect-card">'
                            f'<strong>Slide {s_idx + 1}</strong> &nbsp;'
                            f'<span style="color: #8A4F7D; font-weight: 700; font-family: \'JetBrains Mono\', monospace;">[{format_seconds(sa.start_time)} - {format_seconds(sa.end_time)}]</span><br>'
                            f'<small style="color: #5C4E56;">Alignment Confidence: {int(sa.confidence * 100)}%</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if slide_obj and slide_obj.image_path and Path(slide_obj.image_path).is_file():
                            st.image(slide_obj.image_path, use_container_width=True)
                        elif slide_obj and slide_obj.text:
                            st.info(slide_obj.text[:150] + "...")

                        render_timestamp_seek_button(
                            sa.start_time,
                            label=f"Jump to Slide {s_idx + 1} ({format_seconds(sa.start_time)})",
                            key=f"slide_seek_{s_idx}",
                        )
            else:
                st.info("No slide alignments available.")

        with col_t2:
            st.markdown("#### Aligned Handwritten Notes & Transcript")
            if timeline and timeline.note_alignments:
                for na in timeline.note_alignments:
                    n_idx = na.page_index
                    note_obj = notes[n_idx] if n_idx < len(notes) else None

                    with st.container():
                        st.markdown(
                            f'<div class="relect-card">'
                            f'<strong>Handwritten Note Page {n_idx + 1}</strong> &nbsp;'
                            f'<span style="color: #8A4F7D; font-weight: 700; font-family: \'JetBrains Mono\', monospace;">[{format_seconds(na.start_time)} - {format_seconds(na.end_time)}]</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if note_obj and note_obj.image_path and Path(note_obj.image_path).is_file():
                            st.image(note_obj.image_path, width=280)
                        if note_obj:
                            st.markdown(f"**Transcribed:** *{note_obj.transcribed_text[:120]}...*")
                            if note_obj.low_confidence_notes:
                                st.caption(f"Low confidence notes: {', '.join(note_obj.low_confidence_notes)}")

                        render_timestamp_seek_button(
                            na.start_time,
                            label=f"Jump to Note {n_idx + 1} ({format_seconds(na.start_time)})",
                            key=f"note_seek_{n_idx}",
                        )
            else:
                st.caption("No handwritten notes uploaded or aligned.")

            with st.expander("View Full Spoken Transcript Segments"):
                if timeline and timeline.transcript:
                    for t_idx, seg in enumerate(timeline.transcript):
                        col_ts, col_txt = st.columns([1, 4])
                        with col_ts:
                            render_timestamp_seek_button(
                                seg.start,
                                label=format_seconds(seg.start),
                                key=f"tr_seek_{t_idx}",
                            )
                        with col_txt:
                            st.write(seg.text)

    with tab_gaps:
        st.subheader("Professor's Off-Slide Additions (Verbal Gaps)")
        st.caption("Concepts, caveats, and key points that the professor said verbally but were NOT on the slides.")
        render_feedback_controls(run_id=run_id, artifact_name="gaps", key_prefix="gaps_tab")
        st.divider()

        if gaps:
            for idx, gap in enumerate(gaps):
                render_gap_card(gap, run_id=run_id, index=idx)
        else:
            st.info("No off-slide verbal gaps were detected.")

    with tab_mindmap:
        st.subheader("Interactive Hierarchical Mind Map")
        st.caption("Topic -> Subtopics -> Points with source tags (slide-only, verbal-only, both) and video seeking.")
        render_feedback_controls(run_id=run_id, artifact_name="mindmap", key_prefix="mindmap_tab")
        st.divider()

        if mindmap:
            render_interactive_mindmap(mindmap)
        else:
            st.info("Mind map is not yet generated.")

    with tab_quiz:
        render_quiz_tab(quiz, run_id=run_id)

# Subtle Branding Footer
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align: center; color: #231920; font-size: 0.82rem; padding: 1.5rem 0; border-top: 1px solid rgba(255, 255, 255, 0.5);">'
    '<strong>Beyond the slides</strong> &nbsp;•&nbsp; Powered by LangGraph, Groq Llama 3.3/3.2, and Whisper'
    '</div>',
    unsafe_allow_html=True,
)
