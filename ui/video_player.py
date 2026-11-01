"""Video Player & Timestamp Synchronization Components.

Renders video with precision seeking when timestamp badges or quiz citations are clicked.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Optional


def format_seconds(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    total_sec = max(0, int(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_synchronized_video_player(video_path: str | Path, key: str = "main_player") -> None:
    """Render the main lecture video player with programmatic seeking support."""
    p = Path(video_path)
    if not p.is_file():
        st.warning(f"Video file not found at {p.name}")
        return

    # Retrieve current seek timestamp from session state
    seek_time = int(st.session_state.get("video_seek_time", 0))

    col1, col2 = st.columns([3, 1])
    with col1:
        st.video(str(p), start_time=seek_time)
    with col2:
        st.markdown(
            f"""
            <div class="video-hud-card">
                <div class="video-hud-label">Synchronized Playhead</div>
                <div class="video-hud-time">{format_seconds(seek_time)}</div>
                <div class="video-hud-desc">Click any timestamp pill across the 5 tabs below to instantly jump the video to that exact second.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_timestamp_seek_button(
    seconds: float,
    label: Optional[str] = None,
    key: Optional[str] = None,
) -> None:
    """Render a clickable timestamp button that seeks the embedded video player."""
    time_str = format_seconds(seconds)
    display_label = label or time_str
    button_key = key or f"btn_seek_{seconds}_{label or ''}"

    if st.button(display_label, key=button_key, help=f"Seek video to {time_str}"):
        st.session_state["video_seek_time"] = float(seconds)
        st.rerun()
