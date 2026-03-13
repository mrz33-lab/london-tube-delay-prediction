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
        df_view = pd.DataFrame(records).rename(columns={
            "line": "Line", "pred": "Avg Predicted (min)",
            "actual": "Avg Actual (min)", "mae": "MAE (min)", "n": "Records",
        })
        st.dataframe(
            df_view.style.background_gradient(subset=["Avg Predicted (min)"], cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True,
        )


def render_trends_tab(artifacts: Dict, model_col: str, selected_line: str, dark: bool) -> None:
    """
    Render the Historical Trends tab: time-series of predicted vs actual delays,
    residuals over time, and an average-delay-by-hour-of-day profile.
    """
    test_preds = artifacts.get("test_predictions")
    if test_preds is None:
        st.warning("No prediction data available.")
        return

    st.subheader(f"Historical Accuracy — {selected_line} Line")

    line_df = test_preds[test_preds["line"] == selected_line].sort_values("timestamp")

    if line_df.empty:
        st.info(f"No data found for {selected_line}.")
        return

    # ── Time series ──────────────────────────────────────────────────────────
    lc = LINE_COLOURS.get(selected_line, "#003688")
    paper_bg = "#0d1117" if dark else "#ffffff"
    plot_bg  = "#161b22" if dark else "#fafbfc"
    font_col = "#e6edf3" if dark else "#1a1a2e"
    grid_col = "#30363d" if dark else "#e9ecef"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_df["timestamp"], y=line_df["actual"],
        mode="lines", name="Actual", line=dict(color="#6c757d", width=1.5, dash="dot"),
        hovertemplate="Actual: %{y:.1f} min<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=line_df["timestamp"], y=line_df[model_col],
        mode="lines", name="Predicted", line=dict(color=lc, width=2),
        hovertemplate="Predicted: %{y:.1f} min<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
        font=dict(color=font_col), margin=dict(l=20, r=20, t=50, b=20),
        title=dict(text="Predictions vs Actuals Over Time", font=dict(size=15, color=font_col), x=0.01),
        xaxis=dict(title="", gridcolor=grid_col),
        yaxis=dict(title="Delay (minutes)", gridcolor=grid_col),
        hovermode="x unified", height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Residuals over time ──────────────────────────────────────────────────
    residuals = line_df["actual"] - line_df[model_col]
    fig2 = go.Figure()
    fig2.add_hline(y=0, line=dict(color="#FFD300", dash="dash", width=1.5))
    fig2.add_trace(go.Scatter(
        x=line_df["timestamp"], y=residuals,
        mode="markers", name="Residual",
        marker=dict(
            size=4, opacity=0.55,
            color=residuals,
            colorscale=[[0, "#00B140"], [0.5, "#FFD300"], [1, "#DC241F"]],
        ),
        hovertemplate="Residual: %{y:.1f} min<extra></extra>",
    ))
    fig2.update_layout(
        paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
        font=dict(color=font_col), margin=dict(l=20, r=20, t=50, b=20),
        title=dict(text="Residuals Over Time", font=dict(size=15, color=font_col), x=0.01),
        xaxis=dict(gridcolor=grid_col),
        yaxis=dict(title="Residual (min)", gridcolor=grid_col),
        height=280, showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Hour-of-day pattern ──────────────────────────────────────────────────
    st.subheader("Seasonal Pattern — Hour of Day")
    hourly = line_df.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly_agg = hourly.groupby("hour").agg(
        actual_mean=("actual", "mean"),
        pred_mean=(model_col, "mean"),
    ).reset_index()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=hourly_agg["hour"], y=hourly_agg["actual_mean"],
        mode="lines+markers", name="Actual (avg)",
        line=dict(color="#6c757d", width=2), marker=dict(size=5),
    ))
    fig3.add_trace(go.Scatter(
        x=hourly_agg["hour"], y=hourly_agg["pred_mean"],
        mode="lines+markers", name="Predicted (avg)",
        line=dict(color=lc, width=2), marker=dict(size=5),
    ))
    fig3.update_layout(
        paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
        font=dict(color=font_col), margin=dict(l=20, r=20, t=50, b=20),
        title=dict(text="Average Delay by Hour of Day", font=dict(size=15, color=font_col), x=0.01),
        xaxis=dict(title="Hour", tickvals=list(range(0, 24, 2)), gridcolor=grid_col),
        yaxis=dict(title="Avg Delay (min)", gridcolor=grid_col),
        height=320, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)


def render_data_collection_tab(config, dark: bool) -> None:
    """
    Render the Data Collection tab with live progress metrics sourced from
    the merged CSV, showing collection rate, ETA, and per-line coverage.
    """
    st.subheader("Real-Time Data Collection Status")

    with st.spinner("Reading collection data…"):
        status = load_collection_status(str(config.paths.data_dir))

    # ── Status header ────────────────────────────────────────────────────────
    active_icon = "🟢 Active" if status["is_active"] else "🔴 Inactive"
    h_col = "#00B140" if status["is_active"] else "#DC241F"
    st.markdown(f"""
    <div style="background:{'#21262d' if dark else '#f8f9fa'};
                border:1px solid {'#30363d' if dark else '#dee2e6'};
                border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem;">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <span style="font-size:1.4rem; font-weight:800;">Data Collection</span>
                <span style="margin-left:1rem; font-size:0.85rem; font-weight:700;
                             color:{h_col};">● {active_icon}</span>
            </div>
            <div style="font-size:0.78rem; color:{'#8b949e' if dark else '#6c757d'};">
                Target: {DATA_COLLECTION_TARGET:,} records (2 weeks @ 15 min)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not status["has_data"]:
        st.info(
            "No data collected yet. Start collection with:\n"
            "```\npython data_collection.py\n```"
        )
        return

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Records Collected", f"{status['record_count']:,}")
    with c2:
        rate = status["rate_per_hour"]
        st.metric("Collection Rate", f"{rate:.0f} rec/hr")
    with c3:
        if status["first_ts"] and status["last_ts"]:
            elapsed = status["last_ts"] - status["first_ts"]
            d, s = elapsed.days, elapsed.seconds
            elapsed_str = f"{d}d {s//3600}h {(s%3600)//60}m"
        else:
            elapsed_str = "—"
        st.metric("Time Elapsed", elapsed_str)
    with c4:
        if status["eta_hours"] is not None:
            eta_d = int(status["eta_hours"] // 24)
            eta_h = int(status["eta_hours"] % 24)
            eta_str = f"{eta_d}d {eta_h}h"
        else:
            eta_str = "Complete"
        st.metric("ETA to Target", eta_str)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Progress ring + bar ──────────────────────────────────────────────────
    ring_col, bar_col = st.columns([1, 2])
    with ring_col:
        st.plotly_chart(
            create_collection_progress_chart(status, dark=dark),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with bar_col:
        st.markdown(f"**Progress: {status['record_count']:,} / {status['target']:,} records**")
        st.progress(min(status["pct"] / 100, 1.0))
        st.caption(f"{status['pct']:.1f}% complete")

        if status["first_ts"]:
            st.markdown(f"**First record:** {status['first_ts'].strftime('%Y-%m-%d %H:%M')}")
        if status["last_ts"]:
            st.markdown(f"**Last record:** {status['last_ts'].strftime('%Y-%m-%d %H:%M')}")

    # ── Lines present ─────────────────────────────────────────────────────────
    if status["lines_present"]:
        st.markdown("**Lines with collected data:**")
        pills = "".join(
            f'<span class="line-pill" style="background:{LINE_COLOURS.get(l, "#003688")};">{l}</span>'
            for l in status["lines_present"]
        )
        st.markdown(f'<div style="margin:0.5rem 0;">{pills}</div>', unsafe_allow_html=True)

    # ── How to run ───────────────────────────────────────────────────────────
    with st.expander("📖 How to run data collection", expanded=False):
        st.markdown("""
        **Start continuous collection (every 15 minutes):**
        ```bash
        python data_collection.py
        ```

        **Test a single collection cycle:**
        ```bash
        python data_collection.py --once
        ```

        **Monitor progress:**
        ```bash
        python scripts/check_collection_progress.py
        ```
