"""
Reusable UI components: hero header and sidebar controls.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Optional, Tuple

from app.constants import ALL_LINES, LINE_COLOURS
from utils import get_latest_run_id


def render_header() -> None:
    """
    Render the hero header with TfL roundel branding and a live timestamp.
    The layout is kept compact to maximise vertical space for content.
    """
    now_str = datetime.now().strftime("%A %d %B %Y  ·  %H:%M")
    st.markdown(f"""
    <div class="hero-card fade-in">
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="font-size:3rem; line-height:1;">🚇</div>
            <div>
                <p class="hero-title">London Underground Delay Predictor</p>
                <p class="hero-subtitle">
                    ML-Powered Disruption Intelligence  ·  {now_str}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(artifacts: Dict, config) -> Tuple[str, str, str, Optional[Tuple]]:
    """
    Build the sidebar controls: model selector, line filter, date range picker,
    and dark mode toggle.  User selections are returned as a tuple for clean
    propagation to each tab renderer.
    """
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.2rem;">
        <span style="font-size:2rem;">🚇</span><br>
        <span style="font-size:1.1rem; font-weight:700; color:white; letter-spacing:0.03em;">TfL ML Dashboard</span><br>
        <span style="font-size:0.75rem; color:#b8ccff;">COMP1682 Dissertation</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Dark mode
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False))
    st.session_state["dark_mode"] = dark_mode

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 CONTROLS**")

    # Model selector
    model_choice = st.sidebar.selectbox(
        "Model",
        ["Best", "Ridge", "Naive"],
        index=0,
        help="Choose which trained model's predictions to display",
    )
    st.session_state["model_choice"] = model_choice

    pred_col_map = {"Naive": "pred_naive", "Ridge": "pred_ridge", "Best": "pred_best"}
    model_col = pred_col_map[model_choice]

    # Line selector
    test_preds = artifacts.get("test_predictions")
    lines = sorted(test_preds["line"].unique()) if test_preds is not None else ALL_LINES
    default_line_idx = lines.index("Central") if "Central" in lines else 0
    selected_line = st.sidebar.selectbox(
        "Tube Line",
        lines,
        index=default_line_idx,
        help="Filter visualisations to a specific Underground line",
    )
    st.session_state["selected_line"] = selected_line

    # Date range
    date_range = None
    if test_preds is not None:
        min_d = test_preds["timestamp"].min().date()
        max_d = test_preds["timestamp"].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )

    st.sidebar.markdown("---")

    # Run info
    latest_run = get_latest_run_id(config.paths.artifacts_dir)
    if latest_run:
        ts = latest_run.replace("run_", "")
        fmt = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%d %b %Y  %H:%M")
        st.sidebar.markdown(f"""
        <div style="background:rgba(255,255,255,0.08); border-radius:8px; padding:0.8rem; font-size:0.78rem;">
            <div style="color:#b8ccff; font-weight:700; margin-bottom:0.4rem;">LATEST RUN</div>
            <div style="color:white;">{latest_run}</div>
            <div style="color:#b8ccff; margin-top:0.2rem;">{fmt}</div>
        </div>
        """, unsafe_allow_html=True)

    # Best model badge
    best_name = artifacts.get("best_model_name", "lightgbm").upper()
    st.sidebar.markdown(f"""
    <div style="background:rgba(0,177,64,0.2); border:1px solid #00B140;
                border-radius:8px; padding:0.8rem; margin-top:0.6rem; font-size:0.78rem;">
        <div style="color:#b8ccff; font-weight:700; margin-bottom:0.3rem;">BEST MODEL</div>
        <div style="color:#00B140; font-weight:800; font-size:1rem;">✨ {best_name}</div>
    </div>
    """, unsafe_allow_html=True)

    # Download
    if test_preds is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📥 EXPORT**")
        csv_bytes = test_preds.to_csv(index=False).encode()
        st.sidebar.download_button(
            label="⬇ Download Predictions CSV",
            data=csv_bytes,
            file_name=f"predictions_{selected_line}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    return model_choice, model_col, selected_line, date_range
