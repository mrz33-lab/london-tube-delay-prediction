"""
Tab renderers for each section of the Streamlit dashboard.

Each function receives pre-loaded artifacts and user selections,
keeping the main() function clean and declarative.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

from app.constants import LINE_COLOURS, STATUS_COLOURS, ALL_LINES, DATA_COLLECTION_TARGET
from app.charts import (
    create_gauge_chart,
    create_forecast_chart,
    create_line_heatmap,
    create_model_comparison_bar,
    create_feature_importance_chart,
    create_error_distribution,
    create_scatter_actual_vs_pred,
    create_line_perf_bar,
    create_collection_progress_chart,
    create_confusion_matrix_chart,
)
from app.data_loading import load_collection_status


def render_prediction_tab(
    artifacts: Dict,
    selected_line: str,
    model_col: str,
    model_choice: str,
    date_range,
    dark: bool,
) -> None:
    """
    Render the Predictions tab: KPI cards, a delay gauge, a 24-hour forecast,
    an hour-of-day heatmap, and a per-line summary statistics table.
    """
    test_preds = artifacts.get("test_predictions")
    metrics    = artifacts.get("metrics", {})

    if test_preds is None:
        st.warning("No test predictions found. Please run `python train.py` first.")
        return

    # Apply date filter
    filtered = test_preds.copy()
    if date_range and len(date_range) == 2:
        s, e = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= s) &
            (filtered["timestamp"].dt.date <= e)
        ]

    line_df = filtered[filtered["line"] == selected_line]

    # ── KPI row ──────────────────────────────────────────────────────────────
    m_key = model_choice.lower()
    m_data = metrics.get(m_key, {})
    naive_data = metrics.get("naive", {})

    mae  = m_data.get("test_mae",  0)
    rmse = m_data.get("test_rmse", 0)
    r2   = m_data.get("test_r2",   0)
    impr = (1 - mae / naive_data["test_mae"]) * 100 if naive_data.get("test_mae") else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("📉 Test MAE", f"{mae:.2f} min",
                  delta=f"{-impr:.1f}% vs Naive", delta_color="inverse")
    with k2:
        st.metric("📊 Test RMSE", f"{rmse:.2f} min")
    with k3:
        st.metric("📈 R² Score", f"{r2:.3f}")
    with k4:
        avg_delay = float(line_df["actual"].mean()) if not line_df.empty else 0.0
        st.metric(f"🚇 {selected_line} Avg Delay", f"{avg_delay:.1f} min")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + forecast side by side ────────────────────────────────────────
    pred_col, gauge_col = st.columns([2, 1])

    with gauge_col:
        # The mean prediction for the selected line is used as the gauge input.
        mean_pred = float(line_df[model_col].mean()) if not line_df.empty else 0.0
        st.markdown("**Current Prediction**")
        st.plotly_chart(
            create_gauge_chart(mean_pred, dark=dark),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Status badge
        if mean_pred < 2:
            badge_col, badge_txt = "#00B140", "Good Service"
        elif mean_pred < 5:
            badge_col, badge_txt = "#FFD300", "Minor Delays"
        elif mean_pred < 10:
            badge_col, badge_txt = "#FF6600", "Moderate Delays"
        else:
            badge_col, badge_txt = "#DC241F", "Severe Delays"

        line_col = LINE_COLOURS.get(selected_line, "#003688")
        st.markdown(f"""
        <div style="text-align:center; margin-top:0.5rem;">
            <span class="line-pill" style="background:{line_col};">{selected_line}</span>
            <span class="status-badge" style="background:{badge_col}20; color:{badge_col};
                  border:1.5px solid {badge_col};">
                {badge_txt}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with pred_col:
        st.markdown("**24-Hour Forecast**")
        st.plotly_chart(
            create_forecast_chart(filtered, selected_line, model_col, dark=dark),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Heatmap ──────────────────────────────────────────────────────────────
    with st.expander("🗓 Hour-of-Day Delay Heatmap (all lines)", expanded=False):
        st.plotly_chart(
            create_line_heatmap(filtered, model_col, dark=dark),
            use_container_width=True,
        )

    # ── Raw stats ────────────────────────────────────────────────────────────
    with st.expander(f"📋 Summary Statistics — {selected_line}", expanded=False):
        if line_df.empty:
            st.info("No data in selected date range.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Actual Delays**")
                s = line_df["actual"]
                st.table(pd.DataFrame({
                    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                    "Value":  [f"{s.mean():.2f} min", f"{s.median():.2f} min",
                               f"{s.std():.2f} min",  f"{s.min():.2f} min", f"{s.max():.2f} min"],
                }))
            with c2:
                st.markdown(f"**Predicted Delays ({model_choice})**")
                p = line_df[model_col]
                st.table(pd.DataFrame({
                    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                    "Value":  [f"{p.mean():.2f} min", f"{p.median():.2f} min",
                               f"{p.std():.2f} min",  f"{p.min():.2f} min", f"{p.max():.2f} min"],
                }))


def render_performance_tab(artifacts: Dict, model_col: str, model_choice: str, dark: bool) -> None:
    """
    Render the Performance tab: model comparison bars, detailed metrics table,
    error analysis charts, per-line MAE, and SHAP feature importance.
    """
    metrics    = artifacts.get("metrics", {})
    test_preds = artifacts.get("test_predictions")
    feat_imp   = artifacts.get("feature_importance")
    comp_df    = artifacts.get("model_comparison")

    # ── Model comparison bars ─────────────────────────────────────────────────
    st.subheader("Model Comparison")
    if metrics:
        st.plotly_chart(
            create_model_comparison_bar(metrics, dark=dark),
            use_container_width=True,
        )
    else:
        st.info("Metrics not available.")

    # ── Comparison table ─────────────────────────────────────────────────────
    if comp_df is not None:
        st.markdown("**Detailed Metrics Table**")

        # Improvement over naive is computed and appended as a column.
        # "Test MAE" is the actual CSV column name; "MAE" would not match.
        if "Test MAE" in comp_df.columns and len(comp_df) > 1:
            naive_mae = comp_df[comp_df["Model"].str.lower() == "naive"]["Test MAE"].values
            if len(naive_mae):
                comp_df = comp_df.copy()
                comp_df["Improvement vs Naive"] = comp_df["Test MAE"].apply(
                    lambda x: f"{(1 - x / naive_mae[0]) * 100:+.1f}%"
                )

        def _colour_row(row):
            if "best" in str(row.get("Model", "")).lower():
                return ["background-color: rgba(0,177,64,0.15)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            comp_df.style.apply(_colour_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    # ── Error analysis ────────────────────────────────────────────────────────
    if test_preds is not None:
        st.subheader("Error Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                create_error_distribution(test_preds, model_col, dark=dark),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                create_scatter_actual_vs_pred(test_preds, model_col, dark=dark),
                use_container_width=True,
            )

        st.subheader("Performance by Tube Line")
        st.plotly_chart(
            create_line_perf_bar(test_preds, model_col, dark=dark),
            use_container_width=True,
        )

        st.subheader("Service Status Confusion Matrix")
        st.info(
            "This matrix shows how well the model predicts the categorical TfL service status "
            "derived from predicted delay minutes."
        )
        st.plotly_chart(
            create_confusion_matrix_chart(test_preds, model_col, dark=dark),
            use_container_width=True,
        )

    # ── Feature importance ────────────────────────────────────────────────────
    if feat_imp is not None:
        st.subheader("Feature Importance (SHAP)")
        st.info(
            "SHAP (SHapley Additive exPlanations) values quantify each feature's "
            "contribution to individual predictions.  Higher absolute values indicate "
            "greater influence on the predicted delay."
        )
        st.plotly_chart(
            create_feature_importance_chart(feat_imp, dark=dark),
            use_container_width=True,
        )
    else:
        st.info("Feature importance data not found. Run `python explain.py` to generate it.")


def render_line_comparison_tab(artifacts: Dict, model_col: str, dark: bool) -> None:
    """
    Render the Line Comparison tab as a grid of mini-cards, one per line,
    providing an immediate network-wide situational overview.
    """
    test_preds = artifacts.get("test_predictions")

    st.subheader("All Lines — Current Snapshot")
    st.caption("Mean predicted delay across the latest available data window for each line.")

    if test_preds is None:
        st.warning("No prediction data available.")
        return

    # Build per-line summary
    records = []
    for line in ALL_LINES:
        ldf = test_preds[test_preds["line"] == line]
        if ldf.empty:
            records.append({"line": line, "pred": 0.0, "actual": 0.0, "mae": 0.0, "n": 0})
        else:
            records.append({
                "line":   line,
                "pred":   float(ldf[model_col].mean()),
                "actual": float(ldf["actual"].mean()),
                "mae":    float(np.abs(ldf["actual"] - ldf[model_col]).mean()),
                "n":      len(ldf),
            })

    records.sort(key=lambda x: x["pred"], reverse=True)

    # Render grid (3 columns)
    cols = st.columns(3)
    for i, rec in enumerate(records):
        line  = rec["line"]
        pred  = rec["pred"]
        mae   = rec["mae"]
        lc    = LINE_COLOURS.get(line, "#003688")

        if pred < 2:
            sc, sl = "#00B140", "Good Service"
        elif pred < 5:
            sc, sl = "#FFD300", "Minor Delays"
        elif pred < 10:
            sc, sl = "#FF6600", "Moderate Delays"
        else:
            sc, sl = "#DC241F", "Severe Delays"

        with cols[i % 3]:
            st.markdown(f"""
            <div style="background: {'#21262d' if dark else '#ffffff'};
                        border: 1px solid {'#30363d' if dark else '#dee2e6'};
                        border-left: 5px solid {lc};
                        border-radius: 12px;
                        padding: 1.1rem 1.2rem;
                        margin-bottom: 0.8rem;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.07);">
                <div style="font-weight:800; font-size:0.95rem; color:{'#e6edf3' if dark else '#1a1a2e'};">
                    {line}
                </div>
                <div style="font-size:2rem; font-weight:800; color:{lc}; margin:0.2rem 0;">
                    {pred:.1f}<span style="font-size:1rem; font-weight:400;"> min</span>
                </div>
                <div>
                    <span style="background:{sc}20; color:{sc}; border:1px solid {sc};
                                 border-radius:12px; padding:0.15rem 0.6rem;
                                 font-size:0.72rem; font-weight:700;">
                        {sl}
                    </span>
                </div>
                <div style="font-size:0.75rem; color:{'#8b949e' if dark else '#6c757d'}; margin-top:0.5rem;">
                    MAE: {mae:.2f} min  ·  n={rec['n']:,}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Also show sortable dataframe
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 View as sortable table", expanded=False):
