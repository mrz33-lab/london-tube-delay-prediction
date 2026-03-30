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

        **Data is stored at:** `data/tfl_merged.csv`

        Collection requires:
        - `TFL_APP_KEY` (optional but recommended for higher rate limits)
        - `OPENWEATHERMAP_API_KEY` (required for weather data)

        Store keys in `.env` file (see `.env.example`).
        """)


def render_about_tab(artifacts: Dict) -> None:
    """
    Render the About tab: a structured project overview covering methodology,
    limitations, and technology stack aimed at a non-technical audience.
    """
    best = artifacts.get("best_model_name", "lightgbm").upper()
    metrics = artifacts.get("metrics", {})
    best_key = artifacts.get("best_model_name", "best")
    mae  = metrics.get(best_key, {}).get("test_mae",  "—")
    r2   = metrics.get(best_key, {}).get("test_r2",   "—")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #003688 0%, #0098D4 100%);
                border-radius: 14px; padding: 2rem; color: white; margin-bottom: 1.5rem;">
        <h2 style="margin:0; font-weight:800;">🚇 London Underground Delay Predictor</h2>
        <p style="margin:0.4rem 0 0; opacity:0.9;">
            COMP1682 Final Year Project · University of Greenwich · 2026
        </p>
        <div style="margin-top:1rem; display:flex; gap:2rem; flex-wrap:wrap;">
            <div><div style="font-size:0.75rem; opacity:0.8;">BEST MODEL</div>
                 <div style="font-size:1.3rem; font-weight:700;">{best}</div></div>
            <div><div style="font-size:0.75rem; opacity:0.8;">TEST MAE</div>
                 <div style="font-size:1.3rem; font-weight:700;">
                     {f"{mae:.2f} min" if isinstance(mae, float) else mae}</div></div>
            <div><div style="font-size:0.75rem; opacity:0.8;">R² SCORE</div>
                 <div style="font-size:1.3rem; font-weight:700;">
                     {f"{r2:.3f}" if isinstance(r2, float) else r2}</div></div>
            <div><div style="font-size:0.75rem; opacity:0.8;">LINES MODELLED</div>
                 <div style="font-size:1.3rem; font-weight:700;">11</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_a, tab_b, tab_c, tab_d = st.tabs(["🎯 Project", "🔧 Methodology", "📦 Tech Stack", "⚠️ Ethics"])

    with tab_a:
        st.markdown("""
        ### Purpose
        I built this system to predict London Underground delay severity (in minutes)
        using machine learning, helping transport planners anticipate disruptions
        before they escalate.

        ### Research Questions
        1. Can ML models outperform a simple naive baseline for short-term delay prediction?
        2. Which features (weather, temporal, crowding) are most predictive?
        3. How does model performance vary across different tube lines?

        ### Key Contributions
        - **Rigorous temporal validation** — I use a strict chronological 80/20 train/test
          split with no look-ahead, preventing data leakage.
        - **Multi-model comparison** — Naive baseline vs Ridge regression vs LightGBM.
        - **SHAP explainability** — Every prediction is backed by feature-attribution scores.
        - **Real data collection** — A 2-week TfL + weather data pipeline I built from scratch.
        - **This dashboard** — Interactive, production-quality interface for dissertation demo.
        """)

    with tab_b:
        st.markdown("""
        ### Data Sources
        | Source | Content | Frequency |
        |--------|---------|-----------| 
        | TfL Unified API | Line status, disruptions | Every 15 min |
        | OpenWeatherMap | Temperature, rain, humidity | Every 15 min |
        | `holidays` library | UK bank holiday calendar | Static |

        ### Feature Engineering
        I apply the following transformations with leakage protection:
        - **Lag features** — delay at t-1 h and t-3 h (per line)
        - **Rolling statistics** — mean & std over 3 h and 12 h windows
        - **Weather deltas** — rate of change in temperature and precipitation
        - **Temporal one-hots** — hour, day-of-week, month, peak/off-peak flag
        - **Crowding index** — proxy derived from known peak/off-peak patterns

        ### Models
        | Model | Description |
        |-------|-------------|
        | Naive | Persistence: last observed delay per line |
        | Ridge | L2-regularised linear regression |
        | LightGBM | Gradient-boosted trees with RandomisedSearchCV tuning |

        ### Evaluation
        - Metric: **MAE** (primary), RMSE, R²
        - Validation: 5-fold `TimeSeriesSplit` cross-validation
        - Test: Held-out final 20% of chronological data
        """)

    with tab_c:
        cols = st.columns(3)
        techs = [
            ("🐍 Python 3.x",       "Core language"),
            ("🐼 pandas / numpy",   "Data wrangling"),
            ("🤖 scikit-learn",     "Ridge, CV, preprocessing"),
            ("⚡ LightGBM",         "Best model candidate"),
            ("🧠 SHAP",             "Explainability"),
            ("📊 Plotly",           "Interactive charts"),
            ("🌐 Streamlit",        "This dashboard"),
            ("⚡ FastAPI",          "Production REST API"),
            ("🗓 holidays",         "UK calendar"),
            ("☁️ OpenWeatherMap",   "Weather API"),
            ("🚇 TfL API",          "Line status API"),
            ("📦 joblib",           "Model serialisation"),
        ]
        for i, (name, desc) in enumerate(techs):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:#f8f9fa; border-radius:8px; padding:0.7rem 0.9rem; margin:0.3rem 0;">
                    <div style="font-weight:700; font-size:0.9rem;">{name}</div>
                    <div style="color:#6c757d; font-size:0.78rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_d:
        st.markdown("""
        ### Ethical Considerations

        **Privacy**
        - I use zero personal passenger data. All metrics are fully aggregated at line level.

        **Transparency**
        - Every prediction is backed by SHAP feature attributions — no black-box decisions.
        - My methodology is fully documented and reproducible via the public codebase.

        **Limitations**
        - Predictions are estimates, not guarantees. Major incidents (strikes, engineering
          works) are not represented in training data and may degrade accuracy.
        - Model performance varies by line; I recommend consulting per-line MAE before
          operational use.

        **Human Oversight**
        - This system is designed as **decision support**, not automated action.
          Human review is required before acting on any prediction.

        **Data Quality**
        - Synthetic data was used for initial development; real data collection
          is ongoing. Dissertation results are clearly labelled by data source.
        """)
