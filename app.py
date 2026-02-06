"""
Streamlit dashboard for the London Underground Delay Prediction system.

TfL-branded interface with live predictions, training diagnostics,
per-line comparisons, and data collection status.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from utils import get_latest_run_id
from train import NaiveBaselineModel

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="TfL Delay Predictor | ML Dashboard",
    page_icon="🚇",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# BRAND CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LINE_COLOURS: Dict[str, str] = {
    "Bakerloo":          "#B36305",
    "Central":           "#E32017",
    "Circle":            "#FFD300",
    "District":          "#00782A",
    "Hammersmith & City":"#F3A9BB",
    "Jubilee":           "#A0A5A9",
    "Metropolitan":      "#9B0056",
    "Northern":          "#000000",
    "Piccadilly":        "#003688",
    "Victoria":          "#0098D4",
    "Waterloo & City":   "#95CDBA",
}

STATUS_COLOURS: Dict[str, str] = {
    "Good Service":   "#00B140",
    "Minor Delays":   "#FFD300",
    "Moderate Delays":"#FF6600",
    "Severe Delays":  "#DC241F",
}

ALL_LINES: List[str] = list(LINE_COLOURS.keys())
DATA_COLLECTION_TARGET: int = 14_784


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
def apply_custom_css(dark_mode: bool = False) -> None:
    """Inject TfL-branded CSS. Dark mode swaps root colour variables."""
    if dark_mode:
        bg_primary   = "#0d1117"
        bg_secondary = "#161b22"
        bg_card      = "#21262d"
        text_primary = "#e6edf3"
        text_muted   = "#8b949e"
        border_col   = "#30363d"
        metric_bg    = "#161b22"
    else:
        bg_primary   = "#f0f4f8"
        bg_secondary = "#ffffff"
        bg_card      = "#ffffff"
        text_primary = "#1a1a2e"
        text_muted   = "#6c757d"
        border_col   = "#dee2e6"
        metric_bg    = "#f8f9fa"

    st.markdown(f"""
    <style>
        /* ── Root variables ── */
        :root {{
            --bg-primary:   {bg_primary};
            --bg-secondary: {bg_secondary};
            --bg-card:      {bg_card};
            --text-primary: {text_primary};
            --text-muted:   {text_muted};
            --border:       {border_col};
            --accent:       #003688;
            --accent2:      #0098D4;
            --radius:       12px;
            --shadow:       0 4px 16px rgba(0,0,0,0.10);
            --shadow-hover: 0 8px 28px rgba(0,0,0,0.18);
        }}

        /* ── App background ── */
        .stApp {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, Oxygen, Ubuntu, sans-serif;
        }}

        /* ── Main content padding ── */
        .block-container {{
            padding: 1.5rem 2rem 3rem;
            max-width: 1400px;
        }}

        /* ── Metric cards ── */
        div[data-testid="metric-container"] {{
            background-color: {metric_bg};
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.2rem 1.4rem;
            box-shadow: var(--shadow);
            transition: box-shadow 0.25s;
        }}
        div[data-testid="metric-container"]:hover {{
            box-shadow: var(--shadow-hover);
        }}
        div[data-testid="metric-container"] label {{
            color: var(--text-muted) !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
        }}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            width: 100%;
            border-radius: 8px;
            height: 2.8rem;
            font-weight: 600;
            font-size: 0.9rem;
            background: linear-gradient(135deg, #003688 0%, #0098D4 100%);
            color: white;
            border: none;
            box-shadow: 0 2px 8px rgba(0,54,136,0.3);
            transition: all 0.25s ease;
            letter-spacing: 0.02em;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0,54,136,0.45);
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #003688 0%, #002060 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: #e0e8ff !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stDateInput label,
        section[data-testid="stSidebar"] .stCheckbox label {{
            color: #b8ccff !important;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.15) !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox > div > div {{
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            color: white !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: transparent;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            font-size: 0.88rem;
            color: var(--text-muted);
            background: transparent;
            border: none;
            transition: all 0.2s;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #003688 0%, #0098D4 100%);
            color: white !important;
            box-shadow: 0 4px 12px rgba(0,54,136,0.25);
        }}

        /* ── Expander ── */
        .streamlit-expanderHeader {{
            background-color: var(--bg-card);
            border-radius: var(--radius);
            border: 1px solid var(--border);
            font-weight: 600;
            color: var(--text-primary);
        }}
        .streamlit-expanderContent {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 var(--radius) var(--radius);
        }}

        /* ── Dataframe ── */
        .stDataFrame {{
            border-radius: var(--radius);
            overflow: hidden;
            box-shadow: var(--shadow);
        }}

        /* ── Progress bar ── */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #003688, #0098D4);
            border-radius: 4px;
        }}

        /* ── Custom hero card ── */
        .hero-card {{
            background: linear-gradient(135deg, #003688 0%, #0098D4 100%);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            color: white;
            box-shadow: 0 12px 40px rgba(0,54,136,0.35);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-card::before {{
            content: '';
            position: absolute;
            top: -40px; right: -40px;
            width: 200px; height: 200px;
            background: rgba(255,255,255,0.06);
            border-radius: 50%;
        }}
        .hero-card::after {{
            content: '';
            position: absolute;
            bottom: -60px; left: -30px;
            width: 250px; height: 250px;
            background: rgba(255,255,255,0.04);
            border-radius: 50%;
        }}
        .hero-title {{
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .hero-subtitle {{
            font-size: 1rem;
            opacity: 0.85;
            margin-top: 0.3rem;
        }}

        /* ── Status badge ── */
        .status-badge {{
            display: inline-block;
            padding: 0.3rem 0.9rem;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        /* ── Line pill ── */
        .line-pill {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            color: white;
            margin: 2px;
        }}

        /* ── Section card ── */
        .section-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }}

        /* ── Fade-in animation ── */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .fade-in {{
            animation: fadeInUp 0.45s ease-out both;
        }}

        /* ── Footer ── */
        .dashboard-footer {{
            text-align: center;
            padding: 1.5rem;
            color: var(--text-muted);
            font-size: 0.82rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
        ::-webkit-scrollbar-thumb {{
            background: #003688;
            border-radius: 4px;
        }}

        /* ── Info / warning / success boxes ── */
        .stAlert {{
            border-radius: var(--radius);
        }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_artifacts(artifact_dir: str) -> Dict:
    """Load training artifacts from disk (cached)."""
    path = Path(artifact_dir)
    artifacts: Dict = {}

    try:
        for name in ("naive", "ridge", "best"):
            p = path / f"{name}_model.pkl"
            if p.exists():
                artifacts[f"{name}_model"] = joblib.load(p)

        metrics_p = path / "all_metrics.json"
        if metrics_p.exists():
            with open(metrics_p) as f:
                artifacts["metrics"] = json.load(f)

        pred_p = path / "test_predictions.csv"
        if pred_p.exists():
            df = pd.read_csv(pred_p)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            artifacts["test_predictions"] = df

        comp_p = path / "model_comparison.csv"
        if comp_p.exists():
            artifacts["model_comparison"] = pd.read_csv(comp_p)

        feat_p = path / "feature_importance.csv"
        if feat_p.exists():
            artifacts["feature_importance"] = pd.read_csv(feat_p)

        best_p = path / "best_model_name.txt"
        if best_p.exists():
            artifacts["best_model_name"] = best_p.read_text().strip()

    except Exception as exc:
        st.error(f"Error loading artifacts: {exc}")

    return artifacts


@st.cache_data(show_spinner=False, ttl=60)
def load_collection_status(data_dir: str) -> Dict:
    """Read real data CSV and compute collection progress. TTL=60s."""
    result: Dict = {
        "record_count": 0,
        "target":       DATA_COLLECTION_TARGET,
        "pct":          0.0,
        "rate_per_hour":0.0,
        "first_ts":     None,
        "last_ts":      None,
        "is_active":    False,
        "eta_hours":    None,
        "lines_present":[],
        "has_data":     False,
    }

    csv_path = Path(data_dir) / "tfl_merged.csv"
    if not csv_path.exists():
        return result

    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        n = len(df)
        result["record_count"] = n
        result["has_data"] = n > 0

        if n > 0:
            result["pct"] = min(n / DATA_COLLECTION_TARGET * 100, 100)
            result["first_ts"] = df["timestamp"].min()
            result["last_ts"]  = df["timestamp"].max()

            elapsed_h = (result["last_ts"] - result["first_ts"]).total_seconds() / 3600
            if elapsed_h > 0:
                result["rate_per_hour"] = n / elapsed_h
                remaining = DATA_COLLECTION_TARGET - n
                if result["rate_per_hour"] > 0:
                    result["eta_hours"] = remaining / result["rate_per_hour"]

            if "line" in df.columns:
                result["lines_present"] = sorted(df["line"].unique().tolist())

            # Collection is considered active if the most recent record is less than 30 minutes old.
            cutoff = datetime.now() - timedelta(minutes=30)
            result["is_active"] = result["last_ts"].to_pydatetime().replace(tzinfo=None) > cutoff

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _plotly_layout(fig: go.Figure, title: str = "", dark: bool = False) -> go.Figure:
    """Apply consistent TfL-branded layout to a Plotly figure."""
    paper_bg = "#0d1117" if dark else "#ffffff"
    plot_bg  = "#161b22" if dark else "#fafbfc"
    font_col = "#e6edf3" if dark else "#1a1a2e"
    grid_col = "#30363d" if dark else "#e9ecef"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=font_col, family="'Segoe UI', sans-serif"), x=0.01),
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_col, family="'Segoe UI', sans-serif", size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        hoverlabel=dict(bgcolor="#003688", font_color="white", font_size=13, bordercolor="#003688"),
        xaxis=dict(gridcolor=grid_col, linecolor=grid_col, zeroline=False),
        yaxis=dict(gridcolor=grid_col, linecolor=grid_col, zeroline=False),
        legend=dict(
            bgcolor="rgba(0,0,0,0.04)",
            bordercolor=grid_col,
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    return fig


def create_gauge_chart(delay_minutes: float, dark: bool = False) -> go.Figure:
    """Dial gauge showing predicted delay severity with TfL colour thresholds."""
    if delay_minutes < 2:
        colour = STATUS_COLOURS["Good Service"]
        label  = "Good Service"
    elif delay_minutes < 5:
        colour = STATUS_COLOURS["Minor Delays"]
        label  = "Minor Delays"
    elif delay_minutes < 10:
        colour = STATUS_COLOURS["Moderate Delays"]
        label  = "Moderate Delays"
    else:
        colour = STATUS_COLOURS["Severe Delays"]
        label  = "Severe Delays"

    paper_bg = "#0d1117" if dark else "#ffffff"
    font_col = "#e6edf3" if dark else "#1a1a2e"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(delay_minutes, 1),
        number=dict(suffix=" min", font=dict(size=38, color=font_col)),
        delta=dict(reference=5, increasing=dict(color="#DC241F"), decreasing=dict(color="#00B140")),
        gauge=dict(
            axis=dict(
                range=[0, 20],
                tickwidth=2,
                tickcolor=font_col,
                tickvals=[0, 2, 5, 10, 20],
                ticktext=["0", "2", "5", "10", "20+"],
                tickfont=dict(size=11),
            ),
            bar=dict(color=colour, thickness=0.65),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  2],  color="#e8f8ef"),
                dict(range=[2,  5],  color="#fff9e6"),
                dict(range=[5,  10], color="#fff0e0"),
                dict(range=[10, 20], color="#fce8e8"),
            ],
            threshold=dict(
                line=dict(color=colour, width=4),
                thickness=0.85,
                value=delay_minutes,
            ),
        ),
        title=dict(text=f"<b>{label}</b>", font=dict(size=14, color=colour)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor=paper_bg,
        font=dict(color=font_col),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
    )
    return fig


def create_forecast_chart(
    predictions: pd.DataFrame,
    selected_line: str,
    model_col: str,
    dark: bool = False,
) -> go.Figure:
    """24-hour forecast chart with ±1 MAE confidence band."""
    line_df = (
        predictions[predictions["line"] == selected_line]
        .sort_values("timestamp")
        .tail(96)   # last 24 h at 15-min intervals
    )

    if line_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected line", showarrow=False)
        return _plotly_layout(fig, dark=dark)

    mae = float(np.abs(line_df["actual"] - line_df[model_col]).mean())

    line_colour = LINE_COLOURS.get(selected_line, "#003688")
    r, g, b = int(line_colour[1:3], 16), int(line_colour[3:5], 16), int(line_colour[5:7], 16)
    rgba_fill = f"rgba({r},{g},{b},0.15)"

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([line_df["timestamp"], line_df["timestamp"].iloc[::-1]]),
        y=pd.concat([
            line_df[model_col] + mae,
            (line_df[model_col] - mae).iloc[::-1],
        ]),
        fill="toself",
        fillcolor=rgba_fill,
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 MAE band",
        hoverinfo="skip",
    ))

    # Actual delays
    fig.add_trace(go.Scatter(
        x=line_df["timestamp"],
        y=line_df["actual"],
        mode="lines",
        name="Actual Delay",
        line=dict(color="#6c757d", width=1.5, dash="dot"),
        hovertemplate="<b>Actual</b> %{y:.1f} min @ %{x|%H:%M}<extra></extra>",
    ))

    # Predicted delays
    fig.add_trace(go.Scatter(
        x=line_df["timestamp"],
        y=line_df[model_col],
        mode="lines+markers",
        name="Predicted",
        line=dict(color=line_colour, width=2.5),
        marker=dict(size=4, color=line_colour),
        hovertemplate="<b>Predicted</b> %{y:.1f} min @ %{x|%H:%M}<extra></extra>",
    ))

    fig = _plotly_layout(fig, title=f"Delay Forecast — {selected_line} Line", dark=dark)
    fig.update_layout(
        height=360,
        hovermode="x unified",
        xaxis=dict(title="Time", tickformat="%H:%M"),
        yaxis=dict(title="Delay (minutes)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_line_heatmap(predictions: pd.DataFrame, model_col: str, dark: bool = False) -> go.Figure:
    """Hour-of-day × tube-line heatmap of average predicted delay."""
    if predictions.empty:
        return go.Figure()

    predictions = predictions.copy()
    predictions["hour"] = predictions["timestamp"].dt.hour

    pivot = predictions.pivot_table(
        values=model_col, index="line", columns="hour", aggfunc="mean"
    ).reindex(columns=range(24))

    paper_bg = "#0d1117" if dark else "#ffffff"
    font_col = "#e6edf3" if dark else "#1a1a2e"

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  "#00B140"],
            [0.25, "#FFD300"],
            [0.55, "#FF6600"],
            [1.0,  "#DC241F"],
        ],
        colorbar=dict(title="Avg Delay (min)", tickfont=dict(color=font_col)),
        hovertemplate="<b>%{y}</b><br>Hour: %{x}<br>Avg Delay: %{z:.1f} min<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
        font=dict(color=font_col, family="'Segoe UI', sans-serif"),
        margin=dict(l=140, r=20, t=50, b=60),
        title=dict(text="Average Delay by Line & Hour of Day", font=dict(size=15, color=font_col), x=0.01),
        xaxis=dict(title="Hour of Day", tickangle=-45),
        yaxis=dict(title=""),
        height=420,
    )
    return fig


def create_model_comparison_bar(metrics: Dict, dark: bool = False) -> go.Figure:
    """Grouped bar chart comparing MAE, RMSE, R² across models."""
    model_names = [k for k in ("naive", "ridge", "best") if k in metrics]
    display_names = {"naive": "Naive Baseline", "ridge": "Ridge Regression", "best": "Best Model"}
    bar_colours   = {"naive": "#6c757d",         "ridge": "#0098D4",           "best": "#003688"}

    mae_vals  = [metrics[m].get("test_mae",  0) for m in model_names]
    rmse_vals = [metrics[m].get("test_rmse", 0) for m in model_names]
    r2_vals   = [metrics[m].get("test_r2",   0) for m in model_names]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Mean Absolute Error (↓ better)", "RMSE (↓ better)", "R² Score (↑ better)"),
        shared_yaxes=False,
    )

    for i, (m, mae, rmse, r2) in enumerate(zip(model_names, mae_vals, rmse_vals, r2_vals)):
        col = bar_colours[m]
        name = display_names[m]
        fig.add_trace(go.Bar(name=name, x=[name], y=[mae],  marker_color=col,
                             hovertemplate=f"MAE: %{{y:.3f}}<extra></extra>",
                             showlegend=(i == 0)), row=1, col=1)
        fig.add_trace(go.Bar(name=name, x=[name], y=[rmse], marker_color=col,
                             hovertemplate=f"RMSE: %{{y:.3f}}<extra></extra>",
                             showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name=name, x=[name], y=[r2],   marker_color=col,
                             hovertemplate=f"R²: %{{y:.4f}}<extra></extra>",
                             showlegend=False), row=1, col=3)

    paper_bg = "#0d1117" if dark else "#ffffff"
    plot_bg  = "#161b22" if dark else "#fafbfc"
    font_col = "#e6edf3" if dark else "#1a1a2e"
    grid_col = "#30363d" if dark else "#e9ecef"

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_col, family="'Segoe UI', sans-serif"),
        barmode="group",
        showlegend=False,
        height=380,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=grid_col, linecolor=grid_col)

    return fig


def create_feature_importance_chart(feat_df: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Horizontal bar chart of top 12 SHAP feature importances."""
    top = feat_df.sort_values("importance", ascending=False).head(12)
    top = top.sort_values("importance", ascending=True)

    colours = [
        f"rgba(0,{int(54 + 180 * (v / top['importance'].max()))},136,0.85)"
        for v in top["importance"]
    ]

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker_color=colours,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))

    fig = _plotly_layout(fig, title="Top Feature Importances (SHAP)", dark=dark)
    fig.update_layout(
        height=420,
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        margin=dict(l=160, r=20, t=50, b=20),
    )
    return fig


def create_error_distribution(predictions: pd.DataFrame, model_col: str, dark: bool = False) -> go.Figure:
    """Residual histogram with KDE overlay."""
    errors = predictions["actual"] - predictions[model_col]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=60,
        name="Residuals",
        marker_color="#0098D4",
        opacity=0.75,
        hovertemplate="Error: %{x:.1f} min<br>Count: %{y}<extra></extra>",
    ))

    # Simple KDE-like overlay using Plotly's smoothed scatter
    hist_vals, bin_edges = np.histogram(errors.dropna(), bins=60, density=True)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    scale = len(errors) * (bin_edges[1] - bin_edges[0])
    fig.add_trace(go.Scatter(
        x=bin_centres,
        y=hist_vals * scale,
        mode="lines",
        name="Density",
        line=dict(color="#DC241F", width=2.5),
        hoverinfo="skip",
    ))

    fig.add_vline(x=0, line=dict(color="#FFD300", width=2, dash="dash"), annotation_text="Zero error")

    fig = _plotly_layout(fig, title="Prediction Error Distribution", dark=dark)
    fig.update_layout(
        height=340,
        xaxis_title="Prediction Error (minutes)",
        yaxis_title="Count",
        bargap=0.05,
    )
    return fig


def create_scatter_actual_vs_pred(predictions: pd.DataFrame, model_col: str, dark: bool = False) -> go.Figure:
    """
    Build a predicted-vs-actual scatter with a perfect-prediction diagonal.
    Tight clustering around the diagonal indicates low systematic bias.
    """
    errors = np.abs(predictions["actual"] - predictions[model_col])
    max_val = float(max(predictions["actual"].max(), predictions[model_col].max())) * 1.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predictions["actual"],
        y=predictions[model_col],
        mode="markers",
        name="Predictions",
        marker=dict(
            size=4,
            color=errors,
            colorscale=[[0, "#00B140"], [0.5, "#FFD300"], [1, "#DC241F"]],
            colorbar=dict(title="Error (min)"),
            opacity=0.55,
        ),
        hovertemplate="Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="#6c757d", dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    fig = _plotly_layout(fig, title="Predicted vs Actual Delays", dark=dark)
    fig.update_layout(
        height=420,
        xaxis=dict(title="Actual Delay (min)", range=[0, max_val]),
        yaxis=dict(title="Predicted Delay (min)", range=[0, max_val]),
    )
    return fig


def create_line_perf_bar(predictions: pd.DataFrame, model_col: str, dark: bool = False) -> go.Figure:
    """
    Compute per-line MAE and render it as a sorted bar chart, coloured by
    severity so the worst-performing lines are immediately identifiable.
    """
    perf = (
        predictions.groupby("line")
        .apply(lambda x: np.abs(x["actual"] - x[model_col]).mean())
        .reset_index()
        .rename(columns={0: "MAE"})
        .sort_values("MAE", ascending=False)
    )

    colours = [
        STATUS_COLOURS["Severe Delays"]  if v >= 10 else
        STATUS_COLOURS["Moderate Delays"] if v >= 5  else
        STATUS_COLOURS["Minor Delays"]   if v >= 2  else
        STATUS_COLOURS["Good Service"]
        for v in perf["MAE"]
    ]

    fig = go.Figure(go.Bar(
        x=perf["line"],
        y=perf["MAE"],
        marker_color=colours,
        hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f} min<extra></extra>",
    ))

    fig = _plotly_layout(fig, title="Mean Absolute Error by Tube Line", dark=dark)
    fig.update_layout(
        height=340,
        xaxis_title="",
        yaxis_title="MAE (minutes)",
        xaxis_tickangle=-30,
    )
    return fig


def create_collection_progress_chart(status: Dict, dark: bool = False) -> go.Figure:
    """
    Build a radial ring chart showing data collection completion percentage.
    The donut layout communicates progress more intuitively than a plain
    progress bar at a glance.
    """
    pct = status.get("pct", 0)
    paper_bg = "#0d1117" if dark else "#ffffff"
    font_col = "#e6edf3" if dark else "#1a1a2e"

    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct],
        hole=0.72,
        marker_colors=["#003688", "#e9ecef" if not dark else "#30363d"],
        hoverinfo="none",
        textinfo="none",
        direction="clockwise",
        sort=False,
    ))
    fig.add_annotation(
        text=f"<b>{pct:.1f}%</b>",
        font=dict(size=30, color=font_col),
        showarrow=False,
        x=0.5, y=0.52,
    )
    fig.add_annotation(
        text="complete",
        font=dict(size=12, color=font_col, family="'Segoe UI', sans-serif"),
        showarrow=False,
        x=0.5, y=0.38,
    )
    fig.update_layout(
        paper_bgcolor=paper_bg,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=200,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# TAB RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
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


if __name__ == "__main__":
    main()
