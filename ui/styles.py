"""Custom CSS Design System for Beyond the slides.

Palette:
- #8A4F7D (Grape Soda)
- #887880 (Rosy Granite)
- #88A096 (Muted Teal)

Features Raleway typography, a moving Grape Soda / Rosy Granite / Muted Teal
mesh gradient, frosted glassmorphism containers, gradient typography, and
well-proportioned, aligned UI buttons.
"""

WHITE_BLUISH_THEME_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Raleway:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --font-sans: 'Raleway', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Courier New', monospace;
    --grape-soda: #8A4F7D;
    --rosy-granite: #887880;
    --muted-teal: #88A096;
    --text-primary: #231920;
    --text-secondary: #3E3339;
    --text-muted: #5C4E56;
    --glass-bg: rgba(255, 255, 255, 0.72);
    --glass-bg-hover: rgba(255, 255, 255, 0.88);
    --glass-border: rgba(255, 255, 255, 0.78);
    --shadow-glass: 0 8px 32px rgba(35, 25, 32, 0.12);
    --shadow-glass-hover: 0 12px 36px rgba(138, 79, 125, 0.22);
}

.stApp,
.stApp p,
.stApp label,
.stApp input,
.stApp button {
    font-family: var(--font-sans) !important;
}

/* Preserve Material Symbols Icon Font so icon ligatures never render as raw text like "upload" */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
span[translate="no"] {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal !important;
    font-style: normal !important;
}

/* -------------------------------------------------------------
   1. Dynamic Moving Palette Mesh Gradient Background (10s Loop)
      Colors: #8A4F7D (Grape Soda), #887880 (Rosy Granite), #88A096 (Muted Teal)
   ------------------------------------------------------------- */
@keyframes paletteMeshShift {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

.stApp,
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(
        -45deg,
        #8A4F7D 0%,
        #887880 38%,
        #88A096 72%,
        #887880 88%,
        #8A4F7D 100%
    ) !important;
    background-size: 200% 200% !important;
    background-attachment: fixed !important;
    animation: paletteMeshShift 10s ease-in-out infinite !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stBottom"],
.main,
header[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1240px !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* -------------------------------------------------------------
   2. Frosted Glassmorphism Containers (Hero & Description Box)
   ------------------------------------------------------------- */
.landing-hero {
    text-align: center;
    padding: 2.2rem 2rem 1.2rem 2rem;
    margin-bottom: 1.8rem;
    background: var(--glass-bg) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border-radius: 20px !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--shadow-glass) !important;
    color: var(--text-primary) !important;
}

.landing-title-wrap {
    display: inline-block;
    margin: 0 auto 1.1rem auto;
    max-width: 100%;
}

.landing-title,
h1.landing-title {
    font-family: var(--font-sans) !important;
    font-size: 4.8rem !important;
    font-weight: 800 !important;
    color: #231920 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.03em !important;
    margin: 0 !important;
    padding: 0.08em 0.06em !important;
    display: inline-block;
    overflow: hidden;
    white-space: nowrap;
    border-right: 4px solid #8A4F7D;
    width: 0;
    animation:
        headerTypewriter 1.6s steps(17, end) 0.15s forwards,
        headerCaretBlink 0.7s step-end 5 forwards,
        headerCaretHide 0.2s 3.6s forwards;
}

@keyframes headerTypewriter {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes headerCaretBlink {
    from, to { border-color: transparent; }
    50% { border-color: #8A4F7D; }
}

@keyframes headerCaretHide {
    to { border-color: transparent; }
}

/* -------------------------------------------------------------
   3. Grape Soda to Muted Teal Gradient Typography on "slides"
   ------------------------------------------------------------- */
.highlight-box {
    display: inline;
    background: linear-gradient(90deg, #8A4F7D 0%, #647D73 100%) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.highlight-box::before,
.highlight-box::after {
    display: none !important;
    content: none !important;
}

/* Torn-Paper Note Description Box */
.glass-description-box {
    position: relative !important;
    max-width: 720px;
    margin: 1.2rem auto 1.8rem auto;
    padding: 1.45rem 2.2rem 1.45rem 2.8rem;
    background-color: #FAF8F9 !important;
    background-image:
        linear-gradient(90deg, transparent 34px, rgba(138, 79, 125, 0.35) 34px, rgba(138, 79, 125, 0.35) 36px, transparent 36px),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 27px,
            rgba(136, 160, 150, 0.25) 27px,
            rgba(136, 160, 150, 0.25) 28px
        ) !important;
    border-left: 1px solid rgba(136, 120, 128, 0.28) !important;
    border-right: 1px solid rgba(136, 120, 128, 0.28) !important;
    border-top: none !important;
    border-bottom: none !important;
    border-radius: 2px !important;
    filter: drop-shadow(0 8px 18px rgba(35, 25, 32, 0.16)) !important;
    color: #231920 !important;
    font-size: 1.04rem !important;
    font-weight: 500 !important;
    line-height: 1.75 !important;
    text-align: center;
}

.glass-description-box::before {
    content: "";
    position: absolute;
    top: -10px;
    left: 0;
    right: 0;
    height: 10px;
    background-image:
        linear-gradient(135deg, transparent 33.33%, #FAF8F9 33.33%, #FAF8F9 66.67%, transparent 66.67%),
        linear-gradient(45deg, transparent 33.33%, #FAF8F9 33.33%, #FAF8F9 66.67%, transparent 66.67%);
    background-size: 18px 20px;
    background-repeat: repeat-x;
}

.glass-description-box::after {
    content: "";
    position: absolute;
    bottom: -10px;
    left: 0;
    right: 0;
    height: 10px;
    background-image:
        linear-gradient(-45deg, transparent 33.33%, #FAF8F9 33.33%, #FAF8F9 66.67%, transparent 66.67%),
        linear-gradient(-135deg, transparent 33.33%, #FAF8F9 33.33%, #FAF8F9 66.67%, transparent 66.67%);
    background-size: 18px 20px;
    background-repeat: repeat-x;
}

/* -------------------------------------------------------------
   4. Glassmorphic Cards & Upload Drop-Zones
   ------------------------------------------------------------- */
.category-sidebar,
.course-grid-card,
.relect-card,
.video-hud-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    box-shadow: var(--shadow-glass) !important;
    color: var(--text-primary) !important;
    transition: all 0.22s ease !important;
}

.course-grid-card:hover,
.relect-card:hover {
    background: var(--glass-bg-hover) !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-glass-hover) !important;
}

.relect-card {
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
}

.upload-zone-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1.5px solid rgba(138, 79, 125, 0.45) !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.55rem !important;
    min-height: 100px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: flex-start !important;
    box-shadow: var(--shadow-glass) !important;
    color: var(--text-primary) !important;
    transition: all 0.2s ease;
}

.upload-zone-card:hover {
    background: var(--glass-bg-hover) !important;
    border-color: #8A4F7D !important;
    transform: translateY(-2px);
}

.upload-zone-badge {
    display: inline-block;
    background: rgba(136, 160, 150, 0.28);
    color: #8A4F7D !important;
    border: 1px solid #88A096;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 2px 10px;
    border-radius: 999px;
    margin-bottom: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.upload-title {
    font-weight: 800;
    font-size: 1.04rem;
    color: #231920;
    margin-bottom: 0.15rem;
}

.upload-hint {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 0;
}

/* -------------------------------------------------------------
   5. Adjusted, Balanced UI Buttons & File Uploader Controls
   ------------------------------------------------------------- */
.stButton > button {
    border-radius: 12px !important;
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.58rem 1.35rem !important;
    min-height: 44px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Primary Button (Launch Beyond the slides Pipeline - black text as requested) */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #88A096 0%, #A5BAB1 50%, #8A4F7D 100%) !important;
    background-color: #88A096 !important;
    border: 1.5px solid #8A4F7D !important;
    color: #000000 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(138, 79, 125, 0.28) !important;
}

.stButton > button[kind="primary"] *,
button[data-testid="baseButton-primary"] *,
button[data-testid="stBaseButton-primary"] * {
    color: #000000 !important;
    fill: #000000 !important;
    font-weight: 800 !important;
}

.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #9BB3A9 0%, #88A096 60%, #9E6391 100%) !important;
    border-color: #6B3A60 !important;
    color: #000000 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(138, 79, 125, 0.38) !important;
}

.stButton > button[kind="primary"]:disabled,
button[data-testid="baseButton-primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled {
    background: rgba(255, 255, 255, 0.68) !important;
    border: 1.5px solid #887880 !important;
    color: #000000 !important;
    opacity: 0.78 !important;
    box-shadow: none !important;
}

.stButton > button[kind="primary"]:disabled *,
button[data-testid="baseButton-primary"]:disabled *,
button[data-testid="stBaseButton-primary"]:disabled * {
    color: #000000 !important;
}

/* Secondary Button (Preload Demo Files & Timestamp Seek Buttons) */
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, 0.9) !important;
    color: #231920 !important;
    border: 1.5px solid #8A4F7D !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 6px rgba(35, 25, 32, 0.08) !important;
}

.stButton > button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
button[data-testid="stBaseButton-secondary"] * {
    color: #231920 !important;
    font-weight: 700 !important;
}

.stButton > button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    background: #88A096 !important;
    border-color: #8A4F7D !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(138, 79, 125, 0.22) !important;
}

/* File Uploader Dropzones & Browse Button */
section[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.78) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1.5px dashed #8A4F7D !important;
    border-radius: 16px !important;
    padding: 1.1rem 1rem !important;
    box-shadow: var(--shadow-glass) !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #88A096 !important;
    background: rgba(255, 255, 255, 0.92) !important;
}

section[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #88A096 0%, #8A4F7D 100%) !important;
    color: #000000 !important;
    border-radius: 10px !important;
    border: 1.5px solid #6B3A60 !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    padding: 0.45rem 0.95rem !important;
    min-height: 38px !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    box-shadow: 0 2px 6px rgba(138, 79, 125, 0.25) !important;
    transition: all 0.18s ease !important;
}

/* Render a crisp black '+' sign inside the file uploader button without raw ligature text */
section[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0 !important;
    line-height: 1 !important;
    color: transparent !important;
}

section[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"]::after,
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"]::after {
    content: "+" !important;
    font-family: var(--font-sans) !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: #000000 !important;
    line-height: 1 !important;
}

/* Keep the delete/remove (x) icon and any SVG icons on uploaded file pills black and visible */
[data-testid="stFileUploaderFile"] button [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDeleteBtn"] [data-testid="stIconMaterial"],
section[data-testid="stFileUploaderDropzone"] button svg,
[data-testid="stFileUploader"] button svg {
    font-family: 'Material Symbols Rounded' !important;
    font-size: 1.15rem !important;
    color: #000000 !important;
    fill: #000000 !important;
}

section[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, #9BB3A9 0%, #88A096 100%) !important;
    border-color: #231920 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(138, 79, 125, 0.35) !important;
}

section[data-testid="stFileUploaderDropzone"] button p,
section[data-testid="stFileUploaderDropzone"] button span:not([data-testid="stIconMaterial"]),
[data-testid="stFileUploader"] button p {
    color: #000000 !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

/* Make inline code (e.g., uploaded filenames in Ready: captions) and captions black instead of light green */
.stApp code,
[data-testid="stCaptionContainer"] code,
[data-testid="stMarkdownContainer"] code,
code {
    color: #000000 !important;
    background-color: rgba(255, 255, 255, 0.65) !important;
    border: 1px solid rgba(35, 25, 32, 0.2) !important;
    border-radius: 6px !important;
    padding: 0.12rem 0.4rem !important;
    font-weight: 700 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Progress Bar & Glassmorphic Tabs */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #8A4F7D 0%, #88A096 100%) !important;
    border-radius: 9999px;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-radius: 16px;
    padding: 6px;
    gap: 8px;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-glass);
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    color: var(--text-secondary) !important;
    font-weight: 700;
    font-size: 0.92rem;
    padding: 10px 20px;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8A4F7D 0%, #887880 100%) !important;
    color: #FFFFFF !important;
}

.stTabs [aria-selected="true"] * {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

.badge-example,
.badge-emphasis {
    display: inline-flex;
    background-color: rgba(136, 160, 150, 0.28);
    color: #231920;
    border: 1px solid #88A096;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-caveat {
    display: inline-flex;
    background-color: rgba(138, 79, 125, 0.16);
    color: #8A4F7D;
    border: 1px solid #8A4F7D;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-correction {
    display: inline-flex;
    background-color: rgba(136, 120, 128, 0.22);
    color: #3E3339;
    border: 1px solid #887880;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
}

.citation-box {
    background: rgba(255, 255, 255, 0.76);
    border-left: 4px solid #8A4F7D;
    padding: 0.9rem 1.2rem;
    border-radius: 0 12px 12px 0;
    margin-top: 0.75rem;
    font-size: 0.92rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

.video-hud-card {
    padding: 1.25rem 1.4rem;
}

.video-hud-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
}

.video-hud-time {
    font-size: 2.2rem;
    font-weight: 800;
    font-family: var(--font-mono);
    color: #8A4F7D;
    margin: 0.2rem 0;
}

.video-hud-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
}

[data-testid="stExpander"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
}
</style>"""
