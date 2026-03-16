from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.sign_translator.config import MODELS_DIR, load_labels
from src.sign_translator.video_inference import analyze_video_bytes

st.set_page_config(
    page_title="AI Sign Language Translator",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe6;
            --panel: rgba(255, 250, 243, 0.88);
            --panel-strong: rgba(255, 248, 237, 0.98);
            --ink: #1f2e2a;
            --muted: #576b63;
            --accent: #d96c3f;
            --accent-deep: #8b3f25;
            --teal: #1c7c74;
            --gold: #d6a64f;
            --border: rgba(92, 78, 60, 0.12);
            --shadow: 0 20px 50px rgba(47, 31, 16, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214, 166, 79, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(28, 124, 116, 0.18), transparent 25%),
                linear-gradient(180deg, #fbf6ee 0%, var(--bg) 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1f4037 0%, #274e46 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f7f4ec !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-shell {
            padding: 2.2rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(255, 249, 240, 0.92), rgba(247, 239, 227, 0.85)),
                linear-gradient(120deg, rgba(217, 108, 63, 0.08), rgba(28, 124, 116, 0.10));
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            margin-bottom: 1.3rem;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(28, 124, 116, 0.12);
            color: var(--teal);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title {
            font-size: 3rem;
            line-height: 1;
            margin: 0.8rem 0 0.6rem 0;
            color: var(--ink);
            font-weight: 800;
        }

        .hero-copy {
            max-width: 760px;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
            margin-bottom: 0;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.6rem;
        }

        .feature-card, .panel-card, .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
        }

        .feature-card {
            padding: 1.1rem 1rem;
        }

        .feature-title {
            font-weight: 700;
            color: var(--accent-deep);
            margin-bottom: 0.35rem;
        }

        .feature-copy {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0;
        }

        .panel-card {
            padding: 1.35rem;
            margin-top: 1rem;
            background: var(--panel-strong);
        }

        .panel-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 0.5rem;
        }

        .panel-copy {
            color: var(--muted);
            line-height: 1.6;
            margin-bottom: 0;
        }

        .metric-card {
            padding: 1rem 1.1rem;
            min-height: 120px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--ink);
            line-height: 1.1;
        }

        .metric-note {
            color: var(--muted);
            margin-top: 0.35rem;
            font-size: 0.92rem;
        }

        .sentence-card {
            padding: 1.4rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(28, 124, 116, 0.95), rgba(20, 92, 102, 0.92));
            color: #f6fbf9;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .sentence-label {
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.78;
        }

        .sentence-text {
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.35;
            margin-top: 0.35rem;
        }

        .upload-card {
            padding: 1.3rem;
            border-radius: 24px;
            border: 1px dashed rgba(217, 108, 63, 0.45);
            background: rgba(255, 251, 246, 0.82);
        }

        .stDataFrame, .stAlert, .stVideo {
            border-radius: 18px;
            overflow: hidden;
        }

        @media (max-width: 900px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 2.3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    labels = load_labels()
    model_path = MODELS_DIR / "sign_translator.keras"

    st.sidebar.markdown("## Project Console")
    st.sidebar.write("A deployment-friendly interface for your NNDL sign language translator.")
    st.sidebar.markdown("---")
    st.sidebar.write(f"Configured labels: {', '.join(labels)}")
    st.sidebar.write(f"Model available: {'Yes' if model_path.exists() else 'No'}")
    st.sidebar.write("Recommended clip length: 2 to 5 seconds")
    st.sidebar.write("Best input: one signer, clean background, good lighting")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Workflow")
    st.sidebar.write("1. Upload a short sign video")
    st.sidebar.write("2. Let the model analyze frames")
    st.sidebar.write("3. Review the translated output and confidence")


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <div class="eyebrow">NNDL Project Demo</div>
            <h1 class="hero-title">AI Sign Language Translator</h1>
            <p class="hero-copy">
                A clean web interface for video-based sign recognition using MediaPipe hand landmarks
                and an LSTM sequence model. Upload a short clip, inspect confidence trends, and present
                your project with a polished, academic-demo friendly UI.
            </p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-title">Landmark-Based Recognition</div>
                    <p class="feature-copy">Extracts hand keypoints frame by frame instead of relying only on raw images.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-title">Temporal Deep Learning</div>
                    <p class="feature-copy">Uses an LSTM sequence model to capture gesture motion across time.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-title">Deployment Ready</div>
                    <p class="feature-copy">Runs as a Streamlit app with video upload, metrics, and result previews.</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_upload_section() -> Path | None:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Upload Test Video</div>
            <p class="panel-copy">
                Record a short video showing one or more trained signs. Keep your full hand visible,
                stay close to the same angle used during data collection, and avoid busy backgrounds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file


def metric_card(column, label: str, value: str, note: str) -> None:
    column.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(result: dict) -> None:
    st.markdown(
        f"""
        <div class="sentence-card">
            <div class="sentence-label">Translated Output</div>
            <div class="sentence-text">{result['sentence']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_card(metric_col1, "Frames Processed", str(result["processed_frames"]), "Total video frames analyzed")
    metric_card(metric_col2, "Hand Detections", str(result["detected_frames"]), "Frames where landmarks were found")
    metric_card(metric_col3, "Dominant Sign", str(result["dominant_label"]).replace("_", " "), "Most repeated stable prediction")
    metric_card(metric_col4, "Average Confidence", f"{result['average_confidence']:.2f}", "Mean confidence across accepted predictions")

    left_col, right_col = st.columns([1.2, 0.8])

    with left_col:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-title">Prediction Timeline</div>
                <p class="panel-copy">Frame-level accepted predictions after the stability filter removes uncertain outputs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["predictions"]:
            timeline = pd.DataFrame(result["predictions"])
            st.dataframe(timeline, use_container_width=True, hide_index=True)
        else:
            st.warning("No stable prediction was produced. Try a clearer video with your hand fully visible.")

    with right_col:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-title">Model Reading</div>
                <p class="panel-copy">These preview frames show what the landmark detector observed while the app analyzed your video.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["preview_frames"]:
            for frame in result["preview_frames"]:
                st.image(frame, use_container_width=True)
        else:
            st.info("Preview frames will appear here when hand landmarks are detected.")


def render_status_banner(model_exists: bool) -> None:
    labels = load_labels()
    label_text = ", ".join(labels)
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">Current Model Status</div>
            <p class="panel-copy">
                Trained labels: {label_text}. Model file available: {'Yes' if model_exists else 'No'}.
                This deployed version is optimized for uploaded videos rather than live desktop webcam capture.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    render_sidebar()
    render_hero()

    model_path = MODELS_DIR / "sign_translator.keras"
    model_exists = model_path.exists()
    render_status_banner(model_exists)

    if not model_exists:
        st.error(f"No trained model found at {model_path}. Train the model first with `python train.py`.")
        return

    top_left, top_right = st.columns([1.15, 0.85])
    with top_left:
        uploaded_file = render_upload_section()
    with top_right:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-title">Presentation Notes</div>
                <p class="panel-copy">
                    Use this app during your project demo to explain the full pipeline:
                    video input, landmark extraction, LSTM sequence modeling, and final text output.
                    It is a much better showcase than a raw terminal window.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-title">Best Testing Pattern</div>
                <p class="panel-copy">
                    Upload one sign per clip first. After verifying each individual sign, try multi-sign clips
                    to demonstrate sentence assembly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        return

    st.video(uploaded_file)
    suffix = Path(uploaded_file.name).suffix or ".mp4"

    with st.spinner("Analyzing video and translating signs..."):
        result = analyze_video_bytes(uploaded_file.getvalue(), suffix)

    if not result["is_trained"]:
        st.warning("The app is running without a trained model. Predictions are heuristic only.")

    render_results(result)


if __name__ == "__main__":
    main()
