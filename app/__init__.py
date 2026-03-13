"""
TfL Delay Predictor — Streamlit dashboard package.

Splitting the dashboard into focused modules:
- constants   — brand colours, line lists
- styles      — CSS injection
- data_loading — cached artifact & collection loaders
- charts      — Plotly chart builders
- components  — header, sidebar
- tabs        — one renderer per dashboard tab
"""

import streamlit as st
from datetime import datetime
from typing import Dict

from app.styles import apply_custom_css
from app.data_loading import load_artifacts
from app.components import render_header, render_sidebar
from app.tabs import (
    render_prediction_tab,
    render_performance_tab,
    render_line_comparison_tab,
    render_trends_tab,
    render_data_collection_tab,
    render_about_tab,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from utils import get_latest_run_id


def main() -> None:
    """
    My main application function. I orchestrate session state, CSS, and
    tab rendering from here, keeping each concern in its own function.
    """
    # I initialise session state defaults on first load
    if "dark_mode"     not in st.session_state:
        st.session_state["dark_mode"]     = False
    if "model_choice"  not in st.session_state:
        st.session_state["model_choice"]  = "Best"
    if "selected_line" not in st.session_state:
        st.session_state["selected_line"] = "Central"

    dark = st.session_state["dark_mode"]

    # I apply CSS before rendering anything else so styles load cleanly
    apply_custom_css(dark_mode=dark)

    # ── Load config & latest run ─────────────────────────────────────────────
    config = get_config()
    latest_run_id = get_latest_run_id(config.paths.artifacts_dir)

    if latest_run_id is None:
        render_header()
        st.error("⚠️ No training runs found. Please run `python train.py` first.")
        st.stop()

    artifact_dir = config.paths.artifacts_dir / latest_run_id

    with st.spinner("Loading model artifacts…"):
        artifacts = load_artifacts(str(artifact_dir))

    if not artifacts:
        render_header()
        st.error("⚠️ Failed to load artifacts. Check your training output.")
        st.stop()

    # ── Header ───────────────────────────────────────────────────────────────
    render_header()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    model_choice, model_col, selected_line, date_range = render_sidebar(artifacts, config)

    # Re-read dark_mode after sidebar (toggle may have changed it)
    dark = st.session_state["dark_mode"]
    apply_custom_css(dark_mode=dark)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Predictions",
        "📊 Performance",
        "🚇 Line Comparison",
        "📈 Trends",
        "💾 Data Collection",
        "ℹ️ About",
    ])

    with tab1:
        render_prediction_tab(
            artifacts, selected_line, model_col, model_choice, date_range, dark
        )

    with tab2:
        render_performance_tab(artifacts, model_col, model_choice, dark)

    with tab3:
        render_line_comparison_tab(artifacts, model_col, dark)

    with tab4:
        render_trends_tab(artifacts, model_col, selected_line, dark)

    with tab5:
        render_data_collection_tab(config, dark)

    with tab6:
        render_about_tab(artifacts)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="dashboard-footer">
        <strong>London Underground Delay Predictor</strong> &nbsp;·&nbsp;
        COMP1682 Dissertation &nbsp;·&nbsp; University of Greenwich &nbsp;·&nbsp;
        Built with Streamlit + Plotly &nbsp;·&nbsp;
        Last rendered: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)
