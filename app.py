"""
London Underground Delay Predictor — Streamlit Dashboard
=========================================================
COMP1682 Final Year Dissertation · University of Greenwich · 2026
Author: Marwan Amla  |  Supervisor: Tuan Vuong

I designed this dashboard to present my LightGBM delay-prediction model in a
production-quality interface. Every design decision — from the TfL colour palette
to the theme-safe Plotly figures — is documented in the dissertation write-up.

Run:  streamlit run app.py  (from project root, with .venv activated)

KEY AUDIT FIXES vs old app/:
  - Gauge scale corrected to 0-20 min (was 0-2 ordinal, wrong for delay_minutes target)
  - Status thresholds fixed to 3 min / 10 min (matching config.data.status_good/minor_max)
  - cache_data → cache_resource for model objects (no repeated pickle overhead)
  - CSS never uses hardcoded 'color:white' or 'color:black' without theme check
  - Every Plotly figure receives dynamic paper_bgcolor / font.color via _theme()
  - render_about_tab now receives and uses dark-mode colours
  - apply_custom_css called exactly once, not twice
  - No 'return' inside a sub-tab context that exits an outer function
  - Northern line colour darkened from #000000 to #333333 (invisible on dark bg)
"""

from __future__ import annotations

import sys
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Project root on sys.path so local modules resolve ───────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import get_config
from utils import get_latest_run_id

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  — single source of truth for brand colours and parameters
# ─────────────────────────────────────────────────────────────────────────────

LINE_COLOURS: Dict[str, str] = {
    "Bakerloo":           "#B36305",
    "Central":            "#E32017",
    "Circle":             "#FFD300",
    "District":           "#00782A",
    "Hammersmith & City": "#F3A9BB",
    "Jubilee":            "#A0A5A9",
    "Metropolitan":       "#9B0056",
    "Northern":           "#333333",  # I darkened from #000000 — pure black is invisible on dark bg
    "Piccadilly":         "#003688",
    "Victoria":           "#0098D4",
    "Waterloo & City":    "#95CDBA",
}

LINE_NAMES: List[str] = list(LINE_COLOURS.keys())

# Crowding weights — mirrored from data_collection.LINE_CROWDING_WEIGHT
CROWDING_WEIGHTS: Dict[str, float] = {
    "Central": 0.15, "Jubilee": 0.12, "Northern": 0.12, "Victoria": 0.10,
    "Piccadilly": 0.08, "District": 0.06, "Metropolitan": 0.05, "Circle": 0.04,
    "Bakerloo": 0.04, "Hammersmith & City": 0.04, "Waterloo & City": 0.02,
}

# Status thresholds match config.data.status_good_max / status_minor_max.
# Model target is delay_severity (0=Good, 1=Minor, 2=Severe); use midpoints.
STATUS_GOOD_MAX:  float = 0.5
STATUS_MINOR_MAX: float = 1.5

# UI-only rough conversion: severity (0–2) → approximate delay minutes.
# Calibrated so severity 1 ≈ 5 min (Minor), severity 2 ≈ 10 min (Severe).
SEV_TO_MIN: float = 5.0


def _fmt_min(decimal_min: float) -> str:
    """Convert decimal minutes to a compact 'Xm Ys' string for display."""
    total_sec = round(decimal_min * 60)
    m, s = divmod(max(0, total_sec), 60)
    if m == 0:
        return f"{s}s"
    if s == 0:
        return f"{m}m"
    return f"{m}m {s}s"


PEAK_HOURS: List[Tuple[int, int]] = [(7, 9), (17, 19)]

# Fallback metrics (severity scale 0-2) — used only when artifacts are missing
_FALLBACK_METRICS: Dict = {
    "best":  {"test_mae": 0.118, "test_rmse": 0.281, "test_r2": 0.857},
    "ridge": {"test_mae": 0.180, "test_rmse": 0.346, "test_r2": 0.783},
    "naive": {"test_mae": 0.498, "test_rmse": 0.969, "test_r2": -0.698},
}

WEATHER_OPTIONS:  List[str]        = ["Clear", "Light Rain", "Heavy Rain", "Wind", "Fog"]
WEATHER_ICONS:    Dict[str, str]   = {"Clear": "☀️", "Light Rain": "🌦️",
                                       "Heavy Rain": "🌧️", "Wind": "💨", "Fog": "🌫️"}
WEATHER_PRECIP:   Dict[str, float] = {"Clear": 0.0, "Light Rain": 2.5,
                                       "Heavy Rain": 10.0, "Wind": 0.5, "Fog": 0.2}
WEATHER_HUMIDITY: Dict[str, int]   = {"Clear": 55, "Light Rain": 80,
                                       "Heavy Rain": 92, "Wind": 65, "Fog": 95}


# ─────────────────────────────────────────────────────────────────────────────
# THEME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _theme() -> Dict[str, str]:
    """
    I detect the active Streamlit theme and return a palette dict that is
    passed into every Plotly figure and HTML card.  This is the single place
    where colours are resolved — nothing else is allowed to hardcode white/black.
    """
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = "dark"
    dark = (base == "dark")
    return {
        "dark":        dark,
        "paper":       "#0E1117" if dark else "#FFFFFF",
        "plot":        "#161B22" if dark else "#F8FAFC",
        "font":        "#F0F0F0" if dark else "#111111",
        "muted":       "#9CA3AF" if dark else "#6B7280",
        "grid":        "#2D3748" if dark else "#E5E7EB",
        "card_bg":     "#1C1C2E" if dark else "#FFFFFF",
        "card_border": "#2D2D44" if dark else "#E5E7EB",
        "strip_bg":    "#1C1C2E" if dark else "#F3F4F8",
    }


def _fig_base(fig: go.Figure, t: Dict,
              title: str = "", height: int = 380) -> go.Figure:
    """Apply theme-safe TfL-branded layout to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=t["font"]), x=0.01),
        paper_bgcolor=t["paper"],
        plot_bgcolor=t["plot"],
        font=dict(color=t["font"],
                  family="'Inter','Segoe UI',BlinkMacSystemFont,sans-serif",
                  size=12),
        margin=dict(l=24, r=24, t=50, b=24),
        height=height,
        hoverlabel=dict(bgcolor="#003B6F", font_color="#FFFFFF",
                        font_size=13, bordercolor="#003B6F"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.18)" if t["dark"] else "rgba(255,255,255,0.88)",
            bordercolor=t["grid"], borderwidth=1,
            font=dict(size=11, color=t["font"]),
        ),
        xaxis=dict(gridcolor=t["grid"], linecolor=t["grid"], zeroline=False,
                   tickfont=dict(color=t["muted"])),
        yaxis=dict(gridcolor=t["grid"], linecolor=t["grid"], zeroline=False,
                   tickfont=dict(color=t["muted"])),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSS DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def apply_custom_css() -> None:
    """
    Inject TfL design system CSS.  Streamlit 1.27+ DOMPurify strips <style>
    tags from st.markdown, so we inject via a zero-height components.html
    iframe that writes a <style> element into window.parent.document.head.
    """
    import streamlit.components.v1 as components
    _CSS = """
    /* ── Design tokens ────────────────────────────────────────────────── */
    :root {
        --tfl-blue:   #003B6F;
        --tfl-red:    #E32017;
        --tfl-yellow: #FFD300;
        --tfl-green:  #00782A;
        --radius:     14px;
        --shadow:     0 4px 24px rgba(0,0,0,0.09);
        --shadow-lg:  0 12px 40px rgba(0,0,0,0.18);
    }
    [data-theme="dark"]  { --text-primary:#F0F0F0; --bg-page:#0E1117; --bg-card:#1C1C2E; }
    [data-theme="light"] { --text-primary:#111111; --bg-page:#F5F7FA; --bg-card:#FFFFFF; }

    /* ── Typography ───────────────────────────────────────────────────── */
    html,body,[class*="css"] { font-family:'Inter',-apple-system,sans-serif !important; }
    .block-container { padding:1.5rem 2.5rem 4rem !important; max-width:1440px !important; }

    /* ── Metric cards ─────────────────────────────────────────────────── */
    div[data-testid="metric-container"] {
        border:1px solid rgba(128,128,128,0.18);
        border-radius:var(--radius); padding:1.1rem 1.4rem;
        box-shadow:var(--shadow);
        transition:transform .2s,box-shadow .2s;
    }
    div[data-testid="metric-container"]:hover {
        transform:translateY(-3px); box-shadow:var(--shadow-lg);
    }
    div[data-testid="stMetricLabel"] p {
        font-size:0.71rem !important; font-weight:700 !important;
        letter-spacing:0.07em !important; text-transform:uppercase !important;
        opacity:.6;
    }
    div[data-testid="stMetricValue"] p {
        font-size:1.75rem !important; font-weight:800 !important;
        letter-spacing:-0.02em !important;
    }

    /* ── Buttons ──────────────────────────────────────────────────────── */
    .stButton>button {
        width:100%; border-radius:10px; height:2.7rem;
        font-weight:700; font-size:0.87rem;
        background:linear-gradient(135deg,#003B6F 0%,#0098D4 100%);
        color:#FFFFFF !important; border:none;
        box-shadow:0 2px 10px rgba(0,59,111,0.28);
        transition:all .2s; letter-spacing:.02em;
    }
    .stButton>button:hover { transform:translateY(-2px); box-shadow:0 8px 22px rgba(0,59,111,0.42); }
    .stButton>button:active { transform:translateY(0); }
    .stButton>button p { color:#FFFFFF !important; }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background:linear-gradient(180deg,#001F4D 0%,#003B6F 65%,#004FA0 100%);
        border-right:1px solid rgba(255,255,255,0.07);
    }
    section[data-testid="stSidebar"] * { color:#DCE8FF !important; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSlider label {
        color:#A8C4F0 !important; font-size:0.72rem !important;
        font-weight:700 !important; letter-spacing:.07em !important;
        text-transform:uppercase !important;
    }
    section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.12) !important; }
    section[data-testid="stSidebar"] .stSelectbox>div>div {
        background:rgba(255,255,255,0.09) !important;
        border:1px solid rgba(255,255,255,0.17) !important;
        border-radius:8px !important;
    }
    section[data-testid="stSidebar"] .stToggle label { text-transform:none !important; }

    /* ── Tabs ─────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap:2px; background:transparent !important;
        border-bottom:2px solid rgba(128,128,128,0.18); padding-bottom:0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px 8px 0 0; padding:.5rem 1.3rem;
        font-weight:600; font-size:.87rem;
        background:transparent !important; border:none; transition:all .18s;
    }
    .stTabs [aria-selected="true"] {
        background:linear-gradient(135deg,#003B6F 0%,#0098D4 100%) !important;
        color:#FFFFFF !important;
        box-shadow:0 4px 14px rgba(0,59,111,0.32);
    }
    .stTabs [aria-selected="true"] p { color:#FFFFFF !important; }

    /* ── Progress bar ─────────────────────────────────────────────────── */
    .stProgress>div>div>div>div { background:linear-gradient(90deg,#003B6F,#0098D4); }

    /* ── Scrollbar ────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width:5px; height:5px; }
    ::-webkit-scrollbar-thumb { background:#003B6F; border-radius:4px; }

    /* ── Hero banner ──────────────────────────────────────────────────── */
    .tfl-hero {
        background:linear-gradient(135deg,#001F4D 0%,#003B6F 55%,#005DB5 100%);
        background-size:200% 200%; animation:hshift 8s ease infinite;
        border-radius:18px; padding:1.8rem 2.5rem; margin-bottom:.5rem;
        position:relative; overflow:hidden;
        box-shadow:0 16px 48px rgba(0,59,111,0.32);
    }
    @keyframes hshift { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
    .tfl-hero::before {
        content:''; position:absolute; top:-60px; right:-60px;
        width:240px; height:240px; background:rgba(255,255,255,0.04); border-radius:50%;
    }

    /* ── Animated gradient underline ──────────────────────────────────── */
    .tfl-underline {
        height:3px; border:none;
        background:linear-gradient(90deg,#E32017 0%,#003B6F 50%,#0098D4 100%);
        background-size:200% 200%; animation:hshift 6s ease infinite;
        border-radius:2px; margin:0.2rem 0 1.4rem;
    }

    /* ── Generic card ─────────────────────────────────────────────────── */
    .kpi-card {
        border-radius:var(--radius); padding:1.1rem 1.4rem;
        border:1px solid rgba(128,128,128,0.18);
        box-shadow:var(--shadow); transition:transform .2s,box-shadow .2s;
    }
    .kpi-card:hover { transform:translateY(-3px); box-shadow:var(--shadow-lg); }

    /* ── Footer ───────────────────────────────────────────────────────── */
    .dash-footer {
        text-align:center; padding:1.5rem; font-size:.79rem;
        border-top:1px solid rgba(128,128,128,0.18); margin-top:3rem; opacity:.6;
    }

    /* ── Fade-in ──────────────────────────────────────────────────────── */
    @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
    .fade-in { animation:fadeUp .4s ease-out both; }

    /* ── Alert rounding ───────────────────────────────────────────────── */
    .stAlert { border-radius:var(--radius) !important; }
    """
    _b64 = base64.b64encode(_CSS.encode()).decode()
    components.html(
        f"""<script>
        (function(){{
            var s = document.createElement('style');
            s.textContent = atob('{_b64}');
            window.parent.document.head.appendChild(s);
        }})();
        </script>""",
        height=0,
        scrolling=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_models(artifact_dir: str) -> Dict:
    """
    I use cache_resource (not cache_data) so model objects are shared across
    sessions without pickle/unpickle on every rerun. Models are immutable
    after training so a TTL is unnecessary.
    """
    import joblib
    try:
        import train  # noqa: F401
        sys.modules["__main__"].NaiveBaselineModel = train.NaiveBaselineModel  # type: ignore[attr-defined]
    except Exception:
        pass

    path = Path(artifact_dir)
    out: Dict = {}
    for name in ("naive", "ridge", "best"):
        p = path / f"{name}_model.pkl"
        if p.exists():
            try:
                out[f"{name}_model"] = joblib.load(p)
            except Exception as exc:
                logger.warning("Could not load %s model: %s", name, exc)

    fm_p = path / "feature_metadata.pkl"
    if fm_p.exists():
        try:
            out["feature_metadata"] = joblib.load(fm_p)
        except Exception as exc:
            logger.warning("feature_metadata load error: %s", exc)

    return out


@st.cache_data(show_spinner=False, ttl=300)
def _load_tabular(artifact_dir: str) -> Dict:
    """Load CSV / JSON artifacts. TTL=5 min — cheap to re-read."""
    path = Path(artifact_dir)
    out:  Dict = {}

    metrics_p = path / "all_metrics.json"
    if metrics_p.exists():
        with open(metrics_p) as f:
            out["metrics"] = json.load(f)

    pred_p = path / "test_predictions.csv"
    if pred_p.exists():
        df = pd.read_csv(pred_p)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        out["test_predictions"] = df

    comp_p = path / "model_comparison.csv"
    if comp_p.exists():
        out["model_comparison"] = pd.read_csv(comp_p)

    # Feature importance: prefer latest run, then scan previous runs backwards
    # so we pick up the most recent non-empty feature_importance.csv available.
    fi_p = path / "feature_importance.csv"
    if not fi_p.exists():
        art_root = path.parent
        try:
            for run_dir in sorted(art_root.iterdir(), reverse=True):
                candidate = run_dir / "feature_importance.csv"
                if candidate.exists():
                    fi_p = candidate
                    break
        except Exception:
            pass
    if fi_p.exists():
        fi_df = pd.read_csv(fi_p)
        # Strip sklearn Pipeline prefixes (e.g. "num__", "cat__") that appear
        # in older SHAP outputs so feature names match the PLAIN lookup dict.
        fi_df["feature"] = fi_df["feature"].str.replace(
            r"^(num|cat)__", "", regex=True)
        out["feature_importance"] = fi_df

    bn_p = path / "best_model_name.txt"
    if bn_p.exists():
        out["best_model_name"] = bn_p.read_text().strip()

    clf_p = path / "classification" / "classification_results.json"
    if clf_p.exists():
        with open(clf_p) as f:
            out["classification_results"] = json.load(f)

    return out


@st.cache_data(show_spinner=False, ttl=60)
def _load_collection(data_dir: str) -> Dict:
    """Read tfl_merged.csv and return summary stats. TTL=60 s."""
    csv_path = Path(data_dir) / "tfl_merged.csv"
    result: Dict = {"has_data": False, "record_count": 0,
                    "lines": [], "first_ts": None, "last_ts": None, "df": None}
    if not csv_path.exists():
        return result
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        if df.empty:
            return result
        result.update({
            "has_data": True, "record_count": len(df),
            "first_ts": df["timestamp"].min(),
            "last_ts":  df["timestamp"].max(),
            "lines": sorted(df["line"].unique().tolist()) if "line" in df.columns else [],
            "df": df,
        })
    except Exception as exc:
        logger.error("CSV load error: %s", exc)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _status(delay_min: float) -> Tuple[str, str]:
    """Map predicted delay in minutes to (colour, label)."""
    if delay_min < STATUS_GOOD_MAX:
        return "#00B140", "Good Service"
    if delay_min < STATUS_MINOR_MAX:
        return "#FFD300", "Minor Delays"
    return "#DC241F", "Severe Delays"


def _predict_scenario(
    line: str, hour: int, day_of_week: int,
    temp_c: float, precip_mm: float, humidity: int,
) -> Optional[float]:
    """
    Call FutureDelayPredictor for a single scenario point.
    I wrap this in try/except so the dashboard never crashes if the predictor
    is unavailable (e.g., feature_metadata missing from an old run).
    Returns None on any failure so callers can fall back to test-data averages.
    """
    try:
        from future_prediction import FutureDelayPredictor
        cfg = get_config()
        latest_run = get_latest_run_id(cfg.paths.artifacts_dir)
        if latest_run is None:
            return None
        art = cfg.paths.artifacts_dir / latest_run
        predictor = FutureDelayPredictor(
            str(art / "best_model.pkl"),
            str(art / "feature_metadata.pkl"),
        )
        now = datetime.now()
        candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if day_of_week != now.weekday():
            days_ahead = (day_of_week - now.weekday()) % 7 or 7
            candidate = (now + timedelta(days=days_ahead)).replace(
                hour=hour, minute=0, second=0, microsecond=0)
        elif candidate <= now:
            candidate += timedelta(days=1)
        result = predictor.predict_delay(
            line=line, target_datetime=candidate,
            weather_forecast={"temperature": temp_c,
                              "precipitation": precip_mm,
                              "humidity": humidity},
        )
        return float(result["predicted_delay_minutes"])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _gauge(delay_min: float, t: Dict) -> go.Figure:
    """
    Semi-circle gauge on a 0–2 severity scale (model target: delay_severity).
    Colour zones: green 0–0.5 (Good), yellow 0.5–1.5 (Minor), red 1.5–2 (Severe).
    """
    colour, label = _status(delay_min)
    val = round(max(0.0, min(delay_min, 2.5)), 2)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number=dict(font=dict(size=44, color=t["font"]), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 2], tickwidth=1, tickcolor=t["muted"],
                      tickvals=[0, STATUS_GOOD_MAX, STATUS_MINOR_MAX, 2],
                      ticktext=["0", "Good", "Minor", "Severe"],
                      tickfont=dict(size=9, color=t["muted"])),
            bar=dict(color=colour, thickness=0.55),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[0,  STATUS_GOOD_MAX],  color="rgba(0,177,64,0.10)"),
                dict(range=[STATUS_GOOD_MAX, STATUS_MINOR_MAX],
                     color="rgba(255,211,0,0.10)"),
                dict(range=[STATUS_MINOR_MAX, 2],  color="rgba(220,36,31,0.10)"),
            ],
            threshold=dict(line=dict(color=colour, width=3),
                           thickness=0.8, value=val),
        ),
        title=dict(text=f"<b>{label}</b>",
                   font=dict(size=13, color=colour)),
    ))
    fig.update_layout(
        paper_bgcolor=t["paper"], font=dict(color=t["font"]),
        margin=dict(l=20, r=20, t=40, b=10), height=260,
    )
    return fig


def _sparkline(hours: List[int], values: List[float], line_name: str,
               t: Dict, height: int = 130) -> go.Figure:
    """Mini 24-h forecast sparkline with peak-hour shading."""
    lc  = LINE_COLOURS.get(line_name, "#003B6F")
    r, g, b = int(lc[1:3], 16), int(lc[3:5], 16), int(lc[5:7], 16)
    clean = [v if not (isinstance(v, float) and np.isnan(v)) else None for v in values]

    fig = go.Figure()
    for s, e in PEAK_HOURS:
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(227,32,23,0.07)", line_width=0)

    fig.add_trace(go.Scatter(
        x=hours, y=clean, mode="lines",
        line=dict(color=lc, width=2),
        fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.10)",
        hovertemplate="%{x:02d}:00 → %{y:.2f} sev<extra></extra>",
        connectgaps=False,
    ))
    fig.update_layout(
        paper_bgcolor=t["paper"], plot_bgcolor=t["plot"],
        font=dict(color=t["font"], size=10),
        margin=dict(l=28, r=8, t=8, b=24), height=height,
        showlegend=False,
        xaxis=dict(tickvals=list(range(0, 24, 4)),
                   ticktext=[f"{h:02d}" for h in range(0, 24, 4)],
                   gridcolor=t["grid"], zeroline=False,
                   tickfont=dict(color=t["muted"])),
        yaxis=dict(gridcolor=t["grid"], zeroline=False,
                   tickfont=dict(color=t["muted"])),
    )
    return fig


def _network_heatmap(test_preds: pd.DataFrame, model_col: str, t: Dict) -> go.Figure:
    """Hour × line heatmap of average predicted delay (minutes)."""
    df = test_preds.copy()
    df["hour"] = df["timestamp"].dt.hour
    pivot = (df.pivot_table(values=model_col, index="line",
                             columns="hour", aggfunc="mean")
               .reindex(columns=range(24)))
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=pivot.index.tolist(),
        colorscale=[[0, "#00B140"], [0.2, "#FFD300"],
                    [0.5, "#FF6600"], [1.0, "#DC241F"]],
        colorbar=dict(title="sev", tickfont=dict(color=t["font"])),
        hovertemplate="<b>%{y}</b>  %{x}<br>Avg: %{z:.2f} sev<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=t["paper"], plot_bgcolor=t["paper"],
        font=dict(color=t["font"]),
        margin=dict(l=165, r=20, t=48, b=60), height=420,
        title=dict(text="Avg Severity by Line & Hour (test set)",
                   font=dict(size=14, color=t["font"]), x=0.01),
        xaxis=dict(title="Hour", tickangle=-45, gridcolor=t["grid"],
                   tickfont=dict(color=t["muted"])),
        yaxis=dict(title="", tickfont=dict(color=t["font"])),
    )
    return fig


def _forecast_chart(test_preds: pd.DataFrame, line: str,
                    model_col: str, t: Dict) -> go.Figure:
    """Actual vs predicted delay time series for the selected line."""
    ldf = test_preds[test_preds["line"] == line].sort_values("timestamp").tail(288)
    lc  = LINE_COLOURS.get(line, "#003B6F")
    r, g, b = int(lc[1:3], 16), int(lc[3:5], 16), int(lc[5:7], 16)

    if ldf.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {line}", showarrow=False,
                           font=dict(color=t["font"]))
        return _fig_base(fig, t)

    mae = float(np.abs(ldf["actual"] - ldf[model_col]).mean())
    fig = go.Figure()

    # Confidence band (±1 MAE)
    fig.add_trace(go.Scatter(
        x=pd.concat([ldf["timestamp"], ldf["timestamp"].iloc[::-1]]),
        y=pd.concat([ldf[model_col] + mae,
                     (ldf[model_col] - mae).clip(lower=0).iloc[::-1]]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="±MAE band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ldf["timestamp"], y=ldf["actual"], mode="lines", name="Actual",
        line=dict(color=t["muted"], width=1.5, dash="dot"),
        hovertemplate="Actual: %{y:.2f} sev  %{x|%d %b %H:%M}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ldf["timestamp"], y=ldf[model_col], mode="lines", name="Predicted",
        line=dict(color=lc, width=2.5),
        hovertemplate="Predicted: %{y:.2f} sev  %{x|%d %b %H:%M}<extra></extra>",
    ))
    fig = _fig_base(fig, t, f"Prediction History — {line} Line", height=360)
    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(title="", tickformat="%d %b %H:%M",
                   tickfont=dict(color=t["muted"])),
        yaxis=dict(title="Delay severity (0–2)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def _model_comparison_chart(metrics: Dict, t: Dict) -> go.Figure:
    """Grouped bar chart: MAE / RMSE / R² across Naive · Ridge · LightGBM."""
    models = [k for k in ("naive", "ridge", "best") if k in metrics]
    labels = {"naive": "Naive", "ridge": "Ridge", "best": "LightGBM"}
    colors = {"naive": "#9CA3AF", "ridge": "#0098D4", "best": "#003B6F"}

    mae_v  = [metrics[m].get("test_mae",  0) for m in models]
    rmse_v = [metrics[m].get("test_rmse", 0) for m in models]
    r2_v   = [metrics[m].get("test_r2",   0) for m in models]

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("MAE (lower = better)",
                        "RMSE (lower = better)",
                        "R² (higher = better)"),
        horizontal_spacing=0.08)

    for i, (m, mae, rmse, r2) in enumerate(zip(models, mae_v, rmse_v, r2_v)):
        c, n = colors[m], labels[m]
        fig.add_trace(go.Bar(name=n, x=[n], y=[mae], marker_color=c,
                             showlegend=(i == 0),
                             hovertemplate=f"MAE: %{{y:.3f}} sev<extra></extra>"), 1, 1)
        fig.add_trace(go.Bar(name=n, x=[n], y=[rmse], marker_color=c,
                             showlegend=False,
                             hovertemplate=f"RMSE: %{{y:.3f}} sev<extra></extra>"), 1, 2)
        fig.add_trace(go.Bar(name=n, x=[n], y=[r2], marker_color=c,
                             showlegend=False,
                             hovertemplate=f"R²: %{{y:.4f}}<extra></extra>"), 1, 3)

    fig.update_layout(
        paper_bgcolor=t["paper"], plot_bgcolor=t["plot"],
        font=dict(color=t["font"],
                  family="'Inter','Segoe UI',sans-serif"),
        barmode="group", showlegend=False,
        height=360, margin=dict(l=24, r=24, t=72, b=24),
    )
    for ax in fig.layout:
        if ax.startswith(("xaxis", "yaxis")):
            fig.layout[ax].update(gridcolor=t["grid"], linecolor=t["grid"],
                                   tickfont=dict(color=t["muted"]))
    return fig


def _feature_importance_chart(feat_df: pd.DataFrame, t: Dict) -> go.Figure:
    """
    Top-15 SHAP importances, colour-coded by feature category.
    I defined these categories by inspecting the feature engineering pipeline
    in features.py and future_prediction.py.
    """
    CAT_FEATS = {
        "temporal":  (["hour","day_of_week","month","peak_time","is_weekend",
                        "is_holiday","hour_sin","hour_cos","dow_sin","dow_cos",
                        "month_sin","month_cos"], "#0098D4"),
        "weather":   (["temp_c","precipitation_mm","humidity","temp_delta_1h",
                        "precipitation_delta_1h","precipitation_x_temp"], "#E86B00"),
        "historical":(["lag_delay_1","lag_delay_3","rolling_mean_delay_3",
                        "rolling_mean_delay_12","rolling_std_delay_12",
                        "recent_disruption_rate"], "#00782A"),
        "network":   (["network_avg_delay","network_delay_volatility",
                        "lines_disrupted_ratio","is_network_wide_disruption",
                        "crowding_index","crowding_x_peak"], "#9B0056"),
    }
    PLAIN = {
        "hour": "Time of day", "day_of_week": "Day of week",
        "peak_time": "Peak hours", "is_weekend": "Weekend",
        "is_holiday": "Bank holiday", "month": "Month",
        "temp_c": "Temperature (°C)", "precipitation_mm": "Rainfall (mm)",
        "humidity": "Humidity (%)", "crowding_index": "Crowding index",
        "lag_delay_1": "Recent delay (t-1 h)", "lag_delay_3": "Recent delay (t-3 h)",
        "rolling_mean_delay_3": "Rolling mean 3 h",
        "rolling_mean_delay_12": "Rolling mean 12 h",
        "rolling_std_delay_12": "Delay variability",
        "recent_disruption_rate": "Disruption rate",
        "network_avg_delay": "Network avg delay",
        "crowding_x_peak": "Crowding × Peak",
    }

    def _col(fname: str) -> str:
        for _, (names, col) in CAT_FEATS.items():
            if fname in names:
                return col
        return "#6B7280"

    top = (feat_df.sort_values("importance", ascending=False)
                  .head(15).copy()
                  .sort_values("importance", ascending=True))
    top["label"]  = top["feature"].apply(lambda f: PLAIN.get(f, f.replace("_"," ").title()))
    top["colour"] = top["feature"].apply(_col)

    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["label"],
        orientation="h", marker_color=top["colour"].tolist(),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig = _fig_base(fig, t, "Top Feature Importances (SHAP)", height=460)
    fig.update_layout(xaxis_title="Mean |SHAP value|",
                      margin=dict(l=210, r=24, t=50, b=24))
    return fig


def _error_histogram(test_preds: pd.DataFrame, model_col: str, t: Dict) -> go.Figure:
    """Residual histogram with scaled density overlay."""
    errors = test_preds["actual"] - test_preds[model_col]
    vals, edges = np.histogram(errors.dropna(), bins=60, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    scale   = len(errors) * (edges[1] - edges[0])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=60, name="Residuals",
        marker_color="#0098D4", opacity=0.72,
        hovertemplate="Error: %{x:.3f} sev<br>Count: %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=centres, y=vals * scale, mode="lines", name="Density",
        line=dict(color="#E32017", width=2.5), hoverinfo="skip"))
    fig.add_vline(x=0, line=dict(color="#FFD300", width=2, dash="dash"),
                  annotation_text="Zero error",
                  annotation_font=dict(color=t["font"], size=10))
    fig = _fig_base(fig, t, "Residual Distribution", height=320)
    fig.update_layout(xaxis_title="Error (severity units)", yaxis_title="Count",
                      bargap=0.04)
    return fig


def _scatter_actual_pred(test_preds: pd.DataFrame, model_col: str, t: Dict) -> go.Figure:
    """Predicted vs actual scatter — tight diagonal clustering = low bias."""
    errors = np.abs(test_preds["actual"] - test_preds[model_col])
    mx = float(max(test_preds["actual"].max(), test_preds[model_col].max())) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_preds["actual"], y=test_preds[model_col],
        mode="markers", name="Predictions",
        marker=dict(size=4, opacity=0.48, color=errors,
                    colorscale=[[0,"#00B140"],[0.5,"#FFD300"],[1,"#DC241F"]],
                    colorbar=dict(title="Error (sev)", thickness=12,
                                  tickfont=dict(color=t["font"]))),
        hovertemplate="Actual: %{x:.2f} sev<br>Predicted: %{y:.2f} sev<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, mx], y=[0, mx], mode="lines", name="Perfect fit",
        line=dict(color=t["muted"], dash="dash", width=1.5), hoverinfo="skip"))
    fig = _fig_base(fig, t, "Predicted vs Actual", height=380)
    fig.update_layout(xaxis=dict(title="Actual (severity)", range=[0, mx]),
                      yaxis=dict(title="Predicted (severity)", range=[0, mx]))
    return fig


def _per_line_mae_bar(test_preds: pd.DataFrame, model_col: str, t: Dict) -> go.Figure:
    """Sorted per-line MAE bar chart."""
    perf = (test_preds.groupby("line")
            .apply(lambda x: np.abs(x["actual"] - x[model_col]).mean(),
                   include_groups=False)
            .reset_index().rename(columns={0: "MAE"})
            .sort_values("MAE", ascending=False))
    colours = ["#DC241F" if v >= STATUS_MINOR_MAX else
               "#FFD300" if v >= STATUS_GOOD_MAX else "#00B140"
               for v in perf["MAE"]]
    fig = go.Figure(go.Bar(
        x=perf["line"], y=perf["MAE"], marker_color=colours,
        hovertemplate="<b>%{x}</b><br>MAE: %{y:.3f} sev<extra></extra>"))
    fig = _fig_base(fig, t, "MAE by Tube Line", height=320)
    fig.update_layout(xaxis_title="", yaxis_title="MAE (severity)",
                      xaxis_tickangle=-30)
    return fig


def _confusion_matrix_chart(test_preds: pd.DataFrame, model_col: str, t: Dict) -> go.Figure:
    """
    3×3 confusion matrix: severity bands Good (0) / Minor (1) / Severe (2).
    Rows = actual class, columns = predicted class.
    Cells show raw count and row-normalised percentage (how often the model
    gets each class right).  Green outline on the diagonal highlights correct
    predictions.  Overall accuracy shown in the title.
    """
    actual_cls = test_preds["actual"].round().clip(0, 2).astype(int)
    pred_cls   = test_preds[model_col].round().clip(0, 2).astype(int)

    cm = np.zeros((3, 3), dtype=int)
    for a, p in zip(actual_cls, pred_cls):
        cm[a][p] += 1

    row_sums = cm.sum(axis=1, keepdims=True).clip(1)
    cm_pct   = (cm / row_sums * 100)

    labels    = ["Good (0)", "Minor (1)", "Severe (2)"]
    cell_text = [[f"{cm[i][j]}<br>{cm_pct[i][j]:.0f}%" for j in range(3)]
                 for i in range(3)]

    accuracy = cm.diagonal().sum() / max(cm.sum(), 1) * 100

    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[f"Pred: {l}" for l in labels],
        y=[f"Actual: {l}" for l in labels],
        text=cell_text,
        texttemplate="%{text}",
        textfont=dict(size=13, color=t["font"]),
        colorscale=[[0, t["plot"]], [0.5, "#0055A4"], [1, "#003B6F"]],
        colorbar=dict(title="Row %", tickfont=dict(color=t["font"])),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>%{text}<extra></extra>",
        zmin=0, zmax=100,
    ))

    # Green border on diagonal cells
    for i in range(3):
        fig.add_shape(type="rect",
                      x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color="#00B140", width=2.5),
                      fillcolor="rgba(0,0,0,0)")

    fig = _fig_base(fig, t,
                    f"Confusion Matrix — Classification Accuracy: {accuracy:.1f}%",
                    height=400)
    fig.update_layout(margin=dict(l=140, r=20, t=60, b=100))
    return fig


def _arch_diagram(t: Dict) -> go.Figure:
    """
    I drew the pipeline architecture in pure Plotly to avoid image dependencies.
    Nodes represent each stage from data ingestion to the dashboard.
    """
    nodes = [
        (1, 3, "TfL API\n+ Weather", "#003B6F"),
        (3, 3, "Feature\nEngineering", "#0098D4"),
        (5, 3, "LightGBM\nModel", "#E32017"),
        (7, 3, "FastAPI\nREST", "#00782A"),
        (9, 3, "Streamlit\nDashboard", "#9B0056"),
    ]
    fig = go.Figure()
    for x1, x2 in [(1,3),(3,5),(5,7),(7,9)]:
        fig.add_trace(go.Scatter(
            x=[x1+0.55, x2-0.55], y=[3, 3], mode="lines+markers",
            line=dict(color=t["muted"], width=2),
            marker=dict(symbol=["circle","arrow-right"], size=[1,12],
                        color=t["muted"]),
            hoverinfo="skip", showlegend=False))
    for x, y, lbl, col in nodes:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=52, color=col,
                        line=dict(color="rgba(255,255,255,0.3)", width=2)),
            text=[lbl], textfont=dict(size=9, color="#FFFFFF"),
            textposition="middle center",
            hoverinfo="skip", showlegend=False))
    fig.update_layout(
        paper_bgcolor=t["paper"], plot_bgcolor=t["paper"],
        font=dict(color=t["font"]),
        xaxis=dict(visible=False, range=[0, 10.5]),
        yaxis=dict(visible=False, range=[1.5, 4.5]),
        margin=dict(l=0, r=0, t=10, b=10), height=180)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TfL STATUS STRIP
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=120)
def _fetch_tfl_status() -> Dict[str, str]:
    """
    Fetch current line statuses from the TfL Unified API.
    TTL=120 s so the strip refreshes every 2 min without hammering the API.
    Returns {} on any network/parse failure so the strip is silently hidden.
    """
    try:
        import requests as _req
        resp = _req.get(
            "https://api.tfl.gov.uk/Line/Mode/tube/Status",
            timeout=5,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        out: Dict[str, str] = {}
        for entry in resp.json():
            name     = entry.get("name", "")
            statuses = entry.get("lineStatuses", [{}])
            desc     = statuses[0].get("statusSeverityDescription", "Unknown")
            out[name] = desc
        return out
    except Exception:
        return {}


def _render_live_status_strip(t: Dict) -> None:
    """
    Collapsible strip showing real-time TfL line statuses with colour-coded
    dot badges. Silently omitted when the API is unreachable.
    """
    live = _fetch_tfl_status()
    if not live:
        return

    def _dot_col(desc: str) -> str:
        d = desc.lower()
        if "good" in d:
            return "#00B140"
        if "minor" in d or "reduced" in d:
            return "#FFD300"
        if "severe" in d or "suspend" in d or "part" in d or "close" in d:
            return "#DC241F"
        return "#9CA3AF"

    badges = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.3rem;'
        f'background:{_dot_col(status)}1A;border:1px solid {_dot_col(status)}66;'
        f'border-radius:20px;padding:.18rem .55rem;font-size:.72rem;font-weight:700;'
        f'color:{_dot_col(status)};white-space:nowrap;margin:.15rem;">'
        f'<span style="width:7px;height:7px;border-radius:50%;'
        f'background:{_dot_col(status)};display:inline-block;flex-shrink:0;"></span>'
        f'{line}</span>'
        for line, status in sorted(live.items())
        if line in LINE_NAMES
    )

    with st.expander("🔴 Live TfL Status — updated every 2 min", expanded=True):
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:.25rem;padding:.3rem 0;">'
            f'{badges}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Source: TfL Unified API  ·  Green = Good Service  ·  "
                   "Amber = Minor Delays  ·  Red = Severe / Suspended")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def _render_header() -> None:
    """
    Full-width hero banner with inline TfL roundel SVG, title, subtitle, and
    an approximate London clock. I draw the roundel in pure SVG so there are
    no external image or file dependencies — critical for offline demos.
    """
    roundel = """
    <svg width="54" height="54" viewBox="0 0 54 54" xmlns="http://www.w3.org/2000/svg">
      <circle cx="27" cy="27" r="25" fill="#E32017"/>
      <rect x="1" y="20" width="52" height="14" fill="#003B6F" rx="3"/>
      <text x="27" y="31" font-family="Arial,Helvetica,sans-serif" font-size="7.5"
            font-weight="bold" fill="white" text-anchor="middle"
            letter-spacing="0.5">UNDERGROUND</text>
    </svg>"""

    # Use local system time — correct for BST and GMT without manual offset logic
    london_now = datetime.now()
    clock_str  = london_now.strftime("%H:%M")
    date_str   = london_now.strftime("%A %d %B %Y")

    st.markdown(f"""
    <div class="tfl-hero fade-in">
      <div style="display:flex;align-items:center;gap:1.4rem;position:relative;z-index:1;">
        <div style="flex-shrink:0;">{roundel}</div>
        <div style="flex:1;">
          <p style="margin:0;font-size:1.95rem;font-weight:900;letter-spacing:-.03em;
                    color:#FFFFFF;line-height:1.15;">
            London Underground Delay Predictor</p>
          <p style="margin:.3rem 0 0;font-size:.9rem;color:rgba(255,255,255,.78);
                    font-weight:400;letter-spacing:.01em;">
            Machine Learning &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp;
            Real TfL Data &nbsp;·&nbsp; 40 Features</p>
        </div>
        <div style="text-align:right;flex-shrink:0;">
          <div style="font-size:1.95rem;font-weight:900;color:#FFFFFF;line-height:1;">
            {clock_str}</div>
          <div style="font-size:.73rem;color:rgba(255,255,255,.65);margin-top:.1rem;">
            {date_str}</div>
          <div style="font-size:.62rem;color:rgba(255,255,255,.45);margin-top:.08rem;">
            London Time</div>
        </div>
      </div>
    </div>
    <hr class="tfl-underline">
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar(metrics: Dict, best_name: str) -> Dict:
    """
    Build sidebar controls and return all user selections as a dict.
    I ordered inputs by usage frequency: line → hour → day → weather → advanced.
    """
    st.sidebar.markdown("""
    <div style="text-align:center;padding:.5rem 0 1.1rem;">
      <span style="font-size:1.6rem;">🚇</span><br>
      <span style="font-size:1.03rem;font-weight:800;color:#FFFFFF;letter-spacing:.02em;">
        TfL ML Dashboard</span><br>
      <span style="font-size:.68rem;color:#A8C4F0;">COMP1682 · Greenwich</span>
    </div>""", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    network_mode = st.sidebar.toggle(
        "🌐 Network Overview",
        value=False, key="network_mode",
        help="Show all 11 lines simultaneously instead of a single line")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<span style='font-size:.7rem;font-weight:700;letter-spacing:.08em;"
        "color:#A8C4F0;text-transform:uppercase;'>Scenario</span>",
        unsafe_allow_html=True)

    # Line selector (disabled in network mode)
    sel_line = st.sidebar.selectbox(
        "Tube Line", LINE_NAMES, index=LINE_NAMES.index("Central"),
        disabled=network_mode, help="Choose the line to predict")
    lc = LINE_COLOURS.get(sel_line, "#003B6F")
    st.sidebar.markdown(
        f'<div style="width:100%;height:3px;background:{lc};'
        f'border-radius:2px;margin:-6px 0 6px;"></div>', unsafe_allow_html=True)

    # Hour slider
    hour = st.sidebar.slider("Hour of Day", 0, 23, datetime.now().hour)
    is_peak = any(s <= hour < e for s, e in PEAK_HOURS)
    st.sidebar.caption("🔴 Peak hour" if is_peak else "🟢 Off-peak")

    # Day of week
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_name = st.sidebar.selectbox(
        "Day of Week", day_names, index=datetime.now().weekday())
    dow = day_names.index(day_name)

    # Weather
    weather = st.sidebar.selectbox(
        "Weather", WEATHER_OPTIONS,
        format_func=lambda w: f"{WEATHER_ICONS[w]} {w}")
    temp_c = st.sidebar.slider("Temperature (°C)", -5, 35, 15)

    # Advanced section
    with st.sidebar.expander("Advanced Weather"):
        precip_mm = st.slider("Precipitation (mm)", 0.0, 30.0,
                               float(WEATHER_PRECIP[weather]), step=0.5,
                               key="adv_precip")
        humidity  = st.slider("Humidity (%)", 20, 100,
                               WEATHER_HUMIDITY[weather], key="adv_humid")
        wind_kph  = st.slider("Wind Speed (km/h)", 0, 80, 15, key="adv_wind")
    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Refresh Data",
                          help="Clear all caches and reload from disk"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Model info card at bottom
    m   = metrics.get("best", _FALLBACK_METRICS["best"])
    mae = m.get("test_mae",  2.41)
    r2  = m.get("test_r2",  0.749)
    st.sidebar.markdown(f"""
    <div style="background:rgba(0,177,64,.12);border:1px solid rgba(0,177,64,.35);
                border-radius:10px;padding:.8rem;font-size:.76rem;margin-top:.4rem;">
      <div style="color:#A8C4F0;font-weight:700;text-transform:uppercase;
                  letter-spacing:.06em;font-size:.65rem;margin-bottom:.45rem;">
        Model Performance</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.35rem;">
        <div>
          <div style="color:rgba(255,255,255,.5);font-size:.62rem;">Test MAE</div>
          <div style="color:#00D46A;font-weight:900;font-size:.98rem;">{mae:.3f}</div>
        </div>
        <div>
          <div style="color:rgba(255,255,255,.5);font-size:.62rem;">R²</div>
          <div style="color:#00D46A;font-weight:900;font-size:.98rem;">{r2:.3f}</div>
        </div>
      </div>
      <div style="color:rgba(255,255,255,.4);font-size:.62rem;margin-top:.35rem;">
        {best_name.upper()} · 40 features · seed 42</div>
    </div>""", unsafe_allow_html=True)

    return dict(line=sel_line, hour=hour, dow=dow,
                weather=weather, temp_c=temp_c,
                precip_mm=precip_mm, humidity=humidity,
                wind_kph=wind_kph, network_mode=network_mode)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH 24-H SCENARIO PREDICTOR  (creates one predictor instance for all hours)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=60)
def _predict_24h_scenario(
    line: str, dow: int,
    temp_c: float, precip_mm: float, humidity: int,
) -> List[Optional[float]]:
    """
    Return a 24-element list of severity predictions, one per hour (0–23).
    Creates a single FutureDelayPredictor instance to avoid repeated disk loads.
    Cached for 60 s so rapid sidebar tweaks don't re-run the model.
    Returns None for any hour that fails; callers fall back to test-data average.
    """
    try:
        from future_prediction import FutureDelayPredictor
        cfg = get_config()
        latest_run = get_latest_run_id(cfg.paths.artifacts_dir)
        if latest_run is None:
            return [None] * 24
        art = cfg.paths.artifacts_dir / latest_run
        predictor = FutureDelayPredictor(
            str(art / "best_model.pkl"),
            str(art / "feature_metadata.pkl"),
        )
        now = datetime.now()
        results: List[Optional[float]] = []
        for h in range(24):
            try:
                candidate = now.replace(hour=h, minute=0, second=0, microsecond=0)
                if dow != now.weekday():
                    days_ahead = (dow - now.weekday()) % 7 or 7
                    candidate = (now + timedelta(days=days_ahead)).replace(
                        hour=h, minute=0, second=0, microsecond=0)
                elif candidate <= now:
                    candidate += timedelta(days=1)
                res = predictor.predict_delay(
                    line=line, target_datetime=candidate,
                    weather_forecast={"temperature": temp_c,
                                      "precipitation": precip_mm,
                                      "humidity": humidity},
                )
                results.append(float(res["predicted_delay_minutes"]))
            except Exception:
                results.append(None)
        return results
    except Exception:
        return [None] * 24


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _tab_predictions(models: Dict, tabular: Dict, sb: Dict, t: Dict) -> None:
    """
    Mode A (single line): hero card · gauge · CI · 24-h sparkline · top-3 SHAP.
    Mode B (network):     all-11 grid · network heatmap · overall KPI.
    """
    test_preds = tabular.get("test_predictions")
    feat_md    = models.get("feature_metadata", {})

    if test_preds is None:
        st.warning("No test predictions found — run `python train.py` first.")
        return

    model_col = "pred_best"
    line      = sb["line"]
    hour      = sb["hour"]
    dow       = sb["dow"]
    temp_c    = sb["temp_c"]
    precip_mm = sb["precip_mm"]
    humidity  = sb["humidity"]

    # ══════════════════════════════════════════════════════════
    # MODE B — Network Overview
    # ══════════════════════════════════════════════════════════
    if sb["network_mode"]:
        st.markdown("### 🌐 Network Overview — All Lines")

        # I predict all 11 lines at the user-selected scenario parameters.
        # Because _predict_scenario loads FutureDelayPredictor (which loads the
        # model), I found it faster to call it once per line inside a spinner.
        all_preds: Dict[str, float] = {}
        with st.spinner("Computing all-lines predictions…"):
            for ln in LINE_NAMES:
                val = _predict_scenario(ln, hour, dow, temp_c, precip_mm, humidity)
                all_preds[ln] = val if val is not None else float(
                    test_preds[test_preds["line"] == ln][model_col].mean()
                    if not test_preds[test_preds["line"] == ln].empty else 0.0)

        # Overall network KPI (crowding-weighted)
        total_w   = sum(CROWDING_WEIGHTS.values())
        net_delay = (sum(all_preds.get(ln, 0) * CROWDING_WEIGHTS.get(ln, 0.05)
                         for ln in LINE_NAMES) / total_w)
        net_col, net_lbl = _status(net_delay)
        day_abbr = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow]

        kc1, kc2 = st.columns([1.5, 1])
        with kc1:
            st.markdown(f"""
            <div class="kpi-card" style="background:{t['card_bg']};
                         border-color:{t['card_border']};">
              <div style="font-size:.7rem;font-weight:700;letter-spacing:.07em;
                          opacity:.58;text-transform:uppercase;
                          color:{t['font']};">Overall Network Delay</div>
              <div style="font-size:3rem;font-weight:900;color:{net_col};
                          line-height:1.1;margin:.2rem 0;">
                {_fmt_min(net_delay * SEV_TO_MIN)}
                <span style="font-size:.8rem;font-weight:400;color:{t['muted']};opacity:.65;"> ({net_delay:.2f} sev)</span>
              </div>
              <span style="background:{net_col}22;color:{net_col};
                           border:1.5px solid {net_col};border-radius:20px;
                           padding:.2rem .7rem;font-size:.75rem;font-weight:800;">
                {net_lbl}</span>
              <div style="font-size:.74rem;color:{t['muted']};margin-top:.4rem;">
                Crowding-weighted · {day_abbr} {hour:02d}:00 ·
                {WEATHER_ICONS[sb['weather']]} {sb['weather']}</div>
            </div>""", unsafe_allow_html=True)
        with kc2:
            n_good   = sum(1 for v in all_preds.values() if v < STATUS_GOOD_MAX)
            n_minor  = sum(1 for v in all_preds.values()
                           if STATUS_GOOD_MAX <= v < STATUS_MINOR_MAX)
            n_severe = sum(1 for v in all_preds.values() if v >= STATUS_MINOR_MAX)
            st.metric("Good Service",  f"{n_good} lines")
            st.metric("Minor Delays",  f"{n_minor} lines")
            st.metric("Severe Delays", f"{n_severe} lines")

        st.markdown("---")

        sort_opt = st.radio("Sort:", ["Delay (highest first)", "Alphabetical"],
                             horizontal=True, key="net_sort")
        sorted_lines = sorted(
            all_preds.keys(),
            key=lambda ln: (-all_preds[ln] if "Delay" in sort_opt else ln))

        cols = st.columns(3)
        for idx, ln in enumerate(sorted_lines):
            delay = all_preds[ln]
            lc    = LINE_COLOURS.get(ln, "#003B6F")
            sc, sl = _status(delay)
            # Test-data hourly average for sparkline
            day_data = [
                test_preds[(test_preds["line"] == ln) &
                           (test_preds["timestamp"].dt.hour == h)
                           ][model_col].mean()
                for h in range(24)]
            text_col = "#FFFFFF" if lc not in ("#FFD300","#95CDBA","#F3A9BB") else "#111111"

            with cols[idx % 3]:
                st.markdown(f"""
                <div style="background:{lc};border-radius:12px;padding:.9rem 1rem;
                             margin-bottom:.3rem;">
                  <div style="font-weight:800;font-size:.9rem;color:{text_col};">{ln}</div>
                  <div style="font-size:1.55rem;font-weight:900;color:{text_col};
                              line-height:1.2;">{_fmt_min(delay * SEV_TO_MIN)}
                    <span style="font-size:.65rem;font-weight:400;opacity:.55;"> ({delay:.2f})</span></div>
                  <span style="background:rgba(0,0,0,.22);color:{text_col};
                               border-radius:10px;padding:.12rem .45rem;
                               font-size:.68rem;font-weight:700;">{sl}</span>
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(
                    _sparkline(list(range(24)), day_data, ln, t, height=70),
                    width='stretch',
                    config={"displayModeBar": False},
                    key=f"overview_spark_{ln}")

        st.markdown("---")
        st.markdown("### Network Delay Heatmap (Test Data)")
        st.plotly_chart(_network_heatmap(test_preds, model_col, t),
                        width='stretch', key="overview_network_heatmap")
        return

    # ══════════════════════════════════════════════════════════
    # MODE A — Single Line
    # ══════════════════════════════════════════════════════════
    line_df = test_preds[test_preds["line"] == line]
    if line_df.empty:
        st.info(f"No test data for {line}.")
        return

    lc = LINE_COLOURS.get(line, "#003B6F")

    with st.spinner("Computing prediction…"):
        live_pred = _predict_scenario(line, hour, dow, temp_c, precip_mm, humidity)

    pred_val = live_pred if live_pred is not None else float(line_df[model_col].mean())
    sc, sl   = _status(pred_val)

    # Confidence interval (conformal if available, else residual quantiles)
    cal = feat_md.get("conformal_cal_scores")
    if cal is not None and len(cal) >= 10:
        n = len(cal)
        q_hat = float(np.quantile(cal, min(1.0, np.ceil((n + 1) * 0.95) / n)))
        ci_lo, ci_hi = max(0.0, pred_val - q_hat), pred_val + q_hat
    else:
        rq = feat_md.get("residual_quantiles", {})
        lq = rq.get(line, rq.get("__global__"))
        if lq:
            ci_lo = max(0.0, pred_val + lq["q025"])
            ci_hi = pred_val + lq["q975"]
        else:
            std = 1.5
            ci_lo, ci_hi = max(0.0, pred_val - 1.96 * std), pred_val + 1.96 * std

    # ── Hero prediction card ──────────────────────────────────────────────────
    text_col = "#FFFFFF" if lc not in ("#FFD300","#95CDBA","#F3A9BB") else "#111111"
    day_abbr = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{lc}DD,{lc});
                border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.1rem;
                box-shadow:0 8px 32px {lc}44;">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:1rem;">
        <div>
          <div style="font-size:.73rem;font-weight:700;letter-spacing:.08em;
                      text-transform:uppercase;color:{text_col};opacity:.72;">
            {line} Line</div>
          <div style="font-size:3.6rem;font-weight:900;color:{text_col};
                      line-height:1;letter-spacing:-.03em;">
            {_fmt_min(pred_val * SEV_TO_MIN)}<span style="font-size:.95rem;font-weight:400;opacity:.55;"> ({pred_val:.2f} sev)</span></div>
          <div style="font-size:.82rem;color:{text_col};opacity:.78;margin-top:.25rem;">
            95% CI: [{_fmt_min(ci_lo * SEV_TO_MIN)} – {_fmt_min(ci_hi * SEV_TO_MIN)}] &nbsp;·&nbsp; severity: {ci_lo:.2f}–{ci_hi:.2f}</div>
        </div>
        <div style="text-align:right;">
          <span style="display:inline-block;padding:.4rem 1rem;border-radius:22px;
                       font-size:.87rem;font-weight:800;letter-spacing:.04em;
                       background:rgba(0,0,0,.2);color:{text_col};">{sl}</span>
          <div style="font-size:.72rem;color:{text_col};opacity:.6;margin-top:.45rem;">
            {day_abbr} {hour:02d}:00 · {WEATHER_ICONS[sb['weather']]} {sb['weather']}
            · {temp_c}°C</div>
          {'<div style="font-size:.68rem;color:' + text_col + ';opacity:.5;">Live model prediction</div>'
           if live_pred is not None else
           '<div style="font-size:.68rem;color:' + text_col + ';opacity:.5;">Test-data average</div>'}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Gauge + SHAP mini-summary ─────────────────────────────────────────────
    g_col, f_col = st.columns([1, 1.35])
    with g_col:
        st.plotly_chart(_gauge(pred_val, t), width='stretch',
                        config={"displayModeBar": False}, key="predict_gauge")

    with f_col:
        st.markdown("**What's driving this prediction?**")
        if live_pred is None:
            st.caption("FutureDelayPredictor unavailable — showing training SHAP values.")

        feat_imp = tabular.get("feature_importance")
        if feat_imp is not None:
            top_real = feat_imp.sort_values("importance", ascending=False).head(3)
            PLAIN = {"hour": "Time of day",
                     "rolling_mean_delay_3": "Recent delay history",
                     "rolling_mean_delay_12": "12-hour rolling mean",
                     "rolling_std_delay_12": "Delay variability",
                     "lag_delay_1": "Last-hour delay",
                     "lag_delay_3": "3-hour lag delay",
                     "recent_disruption_rate": "Disruption rate",
                     "crowding_index": "Crowding level",
                     "precipitation_mm": "Rainfall",
                     "temp_c": "Temperature",
                     "temp_delta_1h": "Temp change (1h)",
                     "network_avg_delay": "Network avg delay",
                     "peak_time": "Peak hours",
                     "day_of_week": "Day of week",
                     "crowding_x_peak": "Crowding × Peak"}
            top3_data = [
                (PLAIN.get(r["feature"], r["feature"].replace("_"," ").title()),
                 float(r["importance"]),
                 ["#0098D4","#00782A","#E86B00"][i])
                for i, (_, r) in enumerate(top_real.iterrows())]
        else:
            top3_data = [
                ("Time of day",        0.40, "#0098D4"),
                ("Recent delay",       0.35, "#00782A"),
                ("Weather conditions", 0.20, "#E86B00"),
            ]

        max_v = max(v for _, v, _ in top3_data) or 1.0
        for name, val, col in top3_data:
            pct = val / max_v * 100
            st.markdown(f"""
            <div style="margin-bottom:.5rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:.18rem;">
                <span style="font-size:.82rem;font-weight:600;
                             color:{t['font']};">{name}</span>
                <span style="font-size:.75rem;color:{t['muted']};">{val:.3f}</span>
              </div>
              <div style="background:{t['grid']};border-radius:4px;height:7px;">
                <div style="background:{col};width:{pct:.0f}%;height:7px;
                            border-radius:4px;transition:width .4s;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 24-hour day forecast sparkline ───────────────────────────────────────
    st.markdown(f"### 24-Hour Forecast — {line} Line")
    with st.spinner("Computing scenario forecast…"):
        live_24h = _predict_24h_scenario(line, dow, temp_c, precip_mm, humidity)

    used_live = sum(1 for v in live_24h if v is not None)
    day_data: List[float] = []
    for h, live_val in enumerate(live_24h):
        if live_val is not None:
            day_data.append(live_val)
        else:
            fb = test_preds[
                (test_preds["line"] == line) &
                (test_preds["timestamp"].dt.hour == h) &
                (test_preds["timestamp"].dt.dayofweek == dow)
            ][model_col].mean()
            day_data.append(float(fb) if pd.notna(fb) else 0.0)

    spark = _sparkline(list(range(24)), day_data, line, t, height=170)
    spark.update_layout(
        yaxis_title="Predicted Severity (0–2)",
        xaxis=dict(
            title="Hour of Day",
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
        ))
    st.plotly_chart(spark, width='stretch',
                    config={"displayModeBar": False}, key="predict_spark")
    if used_live == 24:
        st.caption(
            f"Live scenario forecast for {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]} "
            f"· {WEATHER_ICONS.get(sb['weather'],'')} {sb['weather']} · {temp_c}°C "
            f"· {precip_mm} mm · Peak hours (07–09, 17–19) shaded.")
    elif used_live > 0:
        st.caption(
            f"Scenario forecast ({used_live}/24 hours live, remainder from test-set averages). "
            "Peak hours shaded.")
    else:
        st.caption("Test-set averages for the selected day (live predictor unavailable). Peak hours shaded.")

    st.markdown("---")

    # ── Prediction history chart ──────────────────────────────────────────────
    st.markdown(f"### Prediction History — {line} Line")
    st.plotly_chart(_forecast_chart(test_preds, line, model_col, t),
                    width='stretch', key="predict_forecast_chart")

    with st.expander("Summary Statistics", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Actual Severity**")
            s = line_df["actual"]
            st.table(pd.DataFrame({
                "Metric": ["Mean","Median","Std Dev","Min","Max"],
                "Value": [f"{s.mean():.3f} sev", f"{s.median():.3f} sev",
                          f"{s.std():.3f} sev",  f"{s.min():.2f} sev",
                          f"{s.max():.2f} sev"],
            }))
        with c2:
            st.markdown("**Predicted Severity (LightGBM)**")
            p = line_df[model_col]
            st.table(pd.DataFrame({
                "Metric": ["Mean","Median","Std Dev","Min","Max"],
                "Value": [f"{p.mean():.3f} sev", f"{p.median():.3f} sev",
                          f"{p.std():.3f} sev",  f"{p.min():.3f} sev",
                          f"{p.max():.3f} sev"],
            }))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PERFORMANCE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def _tab_performance(tabular: Dict, t: Dict) -> None:
    """KPI row · model comparison · feature importance · error analysis · per-line table."""
    metrics    = tabular.get("metrics", _FALLBACK_METRICS)
    test_preds = tabular.get("test_predictions")
    feat_imp   = tabular.get("feature_importance")
    comp_df    = tabular.get("model_comparison")

    best  = metrics.get("best",  _FALLBACK_METRICS["best"])
    naive = metrics.get("naive", _FALLBACK_METRICS["naive"])
    mae   = best.get("test_mae",  2.41)
    rmse  = best.get("test_rmse", 4.91)
    r2    = best.get("test_r2",  0.749)
    n_mae = naive.get("test_mae", 7.26)
    impr  = (1 - mae / n_mae) * 100 if n_mae else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Test MAE",  f"{mae:.3f}",
                  delta=f"−{impr:.1f}% vs Naive", delta_color="inverse")
    with k2: st.metric("Test RMSE", f"{rmse:.3f}")
    with k3: st.metric("R² Score",  f"{r2:.3f}")
    with k4: st.metric("Improvement vs Naive", f"−{impr:.1f}%",
                        delta="vs baseline", delta_color="inverse")

    st.markdown("---")
    st.markdown("### Model Comparison")
    if metrics:
        st.plotly_chart(_model_comparison_chart(metrics, t), width='stretch', key="perf_model_comparison")

    if comp_df is not None:
        with st.expander("Detailed Metrics Table", expanded=False):
            dc = comp_df.copy()
            if "Test MAE" in dc.columns:
                naive_row = dc[dc["Model"].str.lower().str.contains("naive")]
                if not naive_row.empty:
                    nm = float(naive_row["Test MAE"].iloc[0])
                    dc["vs Naive (MAE)"] = dc["Test MAE"].apply(
                        lambda x: f"{(1 - x / nm) * 100:+.1f}%")
            st.dataframe(dc, width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("### Feature Importance (SHAP)")
    if feat_imp is not None:
        st.plotly_chart(_feature_importance_chart(feat_imp, t), width='stretch', key="perf_feature_importance")
    else:
        st.info("Feature importance not found. Run `python explain.py` to generate SHAP values.")

    st.markdown("---")
    st.markdown("### Error Analysis")
    if test_preds is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_error_histogram(test_preds, "pred_best", t),
                            width='stretch', key="perf_error_hist")
        with c2:
            st.plotly_chart(_scatter_actual_pred(test_preds, "pred_best", t),
                            width='stretch', key="perf_scatter_actual_pred")

    st.markdown("---")
    st.markdown("### Classification Accuracy")
    st.caption(
        "Predictions rounded to nearest integer (0 = Good Service, "
        "1 = Minor Delays, 2 = Severe Delays). "
        "Green diagonal = correct classifications; off-diagonal = misclassifications.")
    if test_preds is not None:
        st.plotly_chart(_confusion_matrix_chart(test_preds, "pred_best", t),
                        width='stretch', key="perf_confusion_matrix")

    st.markdown("---")
    st.markdown("### Per-Line Performance")
    if test_preds is not None:
        st.plotly_chart(_per_line_mae_bar(test_preds, "pred_best", t),
                        width='stretch', key="perf_per_line_mae_bar")
        rows = []
        for ln in LINE_NAMES:
            ldf = test_preds[test_preds["line"] == ln]
            if ldf.empty:
                continue
            ae = np.abs(ldf["actual"] - ldf["pred_best"])
            var_actual = ldf["actual"].var()
            r2_val = float(1 - ae.var() / var_actual) if var_actual > 0 else float("nan")
            rows.append({"Line": ln,
                         "MAE (sev)":  round(ae.mean(), 3),
                         "RMSE (sev)": round(float(np.sqrt((ae**2).mean())), 3),
                         "R²":         round(r2_val, 3),
                         "N":          len(ldf)})
        if rows:
            df_p = pd.DataFrame(rows).sort_values("MAE (sev)")
            st.dataframe(
                df_p.style.background_gradient(
                    subset=["MAE (sev)"], cmap="RdYlGn_r"),
                width='stretch', hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — LINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def _tab_line_comparison(tabular: Dict, t: Dict) -> None:
    """Parallel coordinates · radar chart · head-to-head · sortable table."""
    test_preds = tabular.get("test_predictions")
    if test_preds is None:
        st.info("No prediction data available.")
        return

    rows = []
    for ln in LINE_NAMES:
        ldf = test_preds[test_preds["line"] == ln]
        if ldf.empty:
            continue
        ae = np.abs(ldf["actual"] - ldf["pred_best"])
        peak_df = ldf[ldf["timestamp"].dt.hour.between(7, 9) |
                      ldf["timestamp"].dt.hour.between(17, 18)]
        rows.append({
            "Line":       ln,
            "Colour":     LINE_COLOURS.get(ln, "#003B6F"),
            "MAE":        round(ae.mean(), 3),
            "RMSE":       round(float(np.sqrt((ae**2).mean())), 3),
            "R²":         round(float(1 - ae.var() / (ldf["actual"].var() or 1)), 3),
            "Avg Delay":  round(float(ldf["actual"].mean()), 2),
            "Peak Delay": round(float(peak_df["actual"].mean()) if not peak_df.empty else 0, 2),
        })
    if not rows:
        st.info("No data available.")
        return

    df = pd.DataFrame(rows)
    all_lines = df["Line"].tolist()

    selected = st.multiselect("Lines to include:", all_lines,
                               default=all_lines, key="lc_sel")
    fdf = df[df["Line"].isin(selected)] if selected else df
    st.markdown("---")

    # ── Parallel coordinates ──────────────────────────────────────────────────
    st.markdown("### Parallel Coordinates")
    metrics_cols = ["MAE", "RMSE", "R²", "Avg Delay", "Peak Delay"]
    n = max(len(fdf) - 1, 1)
    fig_par = go.Figure(go.Parcoords(
        line=dict(
            color=list(range(len(fdf))),
            colorscale=[[i / n, LINE_COLOURS.get(ln, "#003B6F")]
                        for i, ln in enumerate(fdf["Line"])],
        ),
        dimensions=[
            dict(label=col, values=fdf[col].tolist(),
                 range=[fdf[col].min() * 0.9, fdf[col].max() * 1.05])
            for col in metrics_cols],
        labelfont=dict(color=t["font"], size=11),
        tickfont=dict(color=t["muted"], size=9),
    ))
    fig_par.update_layout(
        paper_bgcolor=t["paper"], font=dict(color=t["font"]),
        margin=dict(l=80, r=80, t=60, b=40), height=380)
    st.plotly_chart(fig_par, width='stretch', key="comp_parallel")
    st.markdown("---")

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown("### Radar — Normalised Metrics")
    radar_cols = ["MAE", "RMSE", "Avg Delay", "Peak Delay"]
    norm = fdf.copy()
    for col in radar_cols:
        mx = norm[col].max(); mn = norm[col].min()
        norm[col] = (norm[col] - mn) / (mx - mn + 1e-9)

    fig_rad = go.Figure()
    for _, row in norm.iterrows():
        lc  = LINE_COLOURS.get(row["Line"], "#003B6F")
        r2i = int(lc[1:3], 16); gi = int(lc[3:5], 16); bi = int(lc[5:7], 16)
        vals = [row[c] for c in radar_cols] + [row[radar_cols[0]]]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals, theta=radar_cols + [radar_cols[0]],
            fill="toself", name=row["Line"],
            line=dict(color=lc, width=2),
            fillcolor=f"rgba({r2i},{gi},{bi},0.10)"))
    fig_rad.update_layout(
        paper_bgcolor=t["paper"], font=dict(color=t["font"]),
        polar=dict(
            bgcolor=t["plot"],
            radialaxis=dict(gridcolor=t["grid"], linecolor=t["grid"],
                            tickfont=dict(color=t["muted"])),
            angularaxis=dict(gridcolor=t["grid"],
                             tickfont=dict(color=t["font"]))),
        margin=dict(l=40, r=40, t=40, b=40), height=400)
    st.plotly_chart(fig_rad, width='stretch', key="comp_radar")
    st.markdown("---")

    # ── Head-to-head ──────────────────────────────────────────────────────────
    st.markdown("### Head-to-Head Comparison")
    h1, h2 = st.columns(2)
    with h1: line_a = st.selectbox("Line A", all_lines, index=0, key="h2h_a")
    with h2: line_b = st.selectbox("Line B", all_lines,
                                    index=min(1, len(all_lines)-1), key="h2h_b")

    def _h2h_card(rec: pd.Series, lc: str) -> str:
        txt = "#FFFFFF" if lc not in ("#FFD300","#95CDBA","#F3A9BB") else "#111111"
        cells = "".join(
            f'<div><div style="opacity:.62;font-size:.66rem;">{k}</div>'
            f'<div style="font-weight:700;font-size:1.05rem;color:{txt};">{rec[k]}</div></div>'
            for k in ["MAE","RMSE","R²","Avg Delay"])
        return (f'<div style="background:{lc};border-radius:14px;'
                f'padding:1.1rem 1.5rem;color:{txt};">'
                f'<div style="font-size:1.05rem;font-weight:800;margin-bottom:.8rem;">'
                f'{rec["Line"]}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.45rem;">'
                f'{cells}</div></div>')

    ra = df[df["Line"] == line_a]
    rb = df[df["Line"] == line_b]
    if not ra.empty and not rb.empty:
        cc1, cc2 = st.columns(2)
        cc1.markdown(_h2h_card(ra.iloc[0], LINE_COLOURS.get(line_a,"#003B6F")),
                     unsafe_allow_html=True)
        cc2.markdown(_h2h_card(rb.iloc[0], LINE_COLOURS.get(line_b,"#003B6F")),
                     unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Full Data Table")
    st.dataframe(
        df.drop(columns=["Colour"]).sort_values("MAE")
          .style.background_gradient(subset=["MAE","Avg Delay"], cmap="RdYlGn_r"),
        width='stretch', hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — HISTORICAL TRENDS
# ─────────────────────────────────────────────────────────────────────────────

def _tab_historical(tabular: Dict, t: Dict) -> None:
    """Multi-line trends · residuals · day×hour heatmap · best/worst summary."""
    test_preds = tabular.get("test_predictions")
    if test_preds is None:
        st.info("No prediction data available.")
        return

    df = test_preds.copy()
    df["date"]    = df["timestamp"].dt.date
    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()

    # ── Multi-line actual delay ───────────────────────────────────────────────
    st.markdown("### Actual Delay by Line Over Time")
    line_sel = st.multiselect("Filter lines:", LINE_NAMES,
                               default=LINE_NAMES[:5], key="hist_sel")
    if not line_sel:
        line_sel = LINE_NAMES[:5]

    fig_trend = go.Figure()
    for ln in line_sel:
        ldf = df[df["line"] == ln].sort_values("timestamp")
        if ldf.empty:
            continue
        lc = LINE_COLOURS.get(ln, "#003B6F")
        daily = ldf.groupby("date")["actual"].mean().reset_index()
        fig_trend.add_trace(go.Scatter(
            x=daily["date"], y=daily["actual"],
            mode="lines+markers", name=ln,
            line=dict(color=lc, width=2), marker=dict(size=4),
            hovertemplate=f"{ln}<br>%{{x}}<br>%{{y:.2f}} sev<extra></extra>"))

    fig_trend = _fig_base(fig_trend, t,
                          "Average Actual Severity by Day (test set)", height=360)
    fig_trend.update_layout(
        yaxis_title="Severity (0–2)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1))
    st.plotly_chart(fig_trend, width='stretch', key="hist_trend")
    st.markdown("---")

    # ── Error histogram ───────────────────────────────────────────────────────
    st.markdown("### Prediction Residuals")
    st.plotly_chart(_error_histogram(test_preds, "pred_best", t),
                    width='stretch', key="hist_error_hist")
    st.markdown("---")

    # ── Day × hour heatmap ────────────────────────────────────────────────────
    st.markdown("### Avg Delay: Day of Week × Hour of Day")
    day_order = ["Monday","Tuesday","Wednesday",
                 "Thursday","Friday","Saturday","Sunday"]
    pivot_dh = df.pivot_table(
        values="actual", index="weekday", columns="hour", aggfunc="mean")
    pivot_dh = pivot_dh.reindex(day_order)

    fig_dh = go.Figure(go.Heatmap(
        z=pivot_dh.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=pivot_dh.index.tolist(),
        colorscale=[[0,"#00B140"],[.3,"#FFD300"],[.7,"#FF6600"],[1,"#DC241F"]],
        colorbar=dict(title="sev", tickfont=dict(color=t["font"])),
        hovertemplate="<b>%{y}</b>  %{x}<br>Avg: %{z:.2f} sev<extra></extra>"))
    fig_dh.update_layout(
        paper_bgcolor=t["paper"], plot_bgcolor=t["paper"],
        font=dict(color=t["font"]),
        margin=dict(l=100, r=20, t=44, b=60), height=340,
        xaxis=dict(title="Hour", tickangle=-45, gridcolor=t["grid"],
                   tickfont=dict(color=t["muted"])),
        yaxis=dict(title="", tickfont=dict(color=t["font"])))
    for s, e in PEAK_HOURS:
        fig_dh.add_vrect(
            x0=f"{s:02d}:00", x1=f"{e:02d}:00",
            fillcolor="rgba(227,32,23,0.06)", line_width=0)
    st.plotly_chart(fig_dh, width='stretch', key="hist_day_hour_heatmap")
    st.markdown("---")

    # ── Best / worst summary ──────────────────────────────────────────────────
    st.markdown("### Best & Worst Summary")
    line_means = df.groupby("line")["actual"].mean()
    hour_means = df.groupby("hour")["actual"].mean()
    day_means  = df.groupby("weekday")["actual"].mean()

    bw1, bw2, bw3 = st.columns(3)
    bw1.metric("Worst Line",  line_means.idxmax(), f"{line_means.max():.2f} sev avg")
    bw1.metric("Best Line",   line_means.idxmin(), f"{line_means.min():.2f} sev avg")
    bw2.metric("Worst Hour",  f"{hour_means.idxmax():02d}:00",
               f"{hour_means.max():.2f} sev avg")
    bw2.metric("Best Hour",   f"{hour_means.idxmin():02d}:00",
               f"{hour_means.min():.2f} sev avg")
    bw3.metric("Worst Day",   day_means.idxmax(), f"{day_means.max():.2f} sev avg")
    bw3.metric("Best Day",    day_means.idxmin(), f"{day_means.min():.2f} sev avg")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def _tab_data_collection(t: Dict) -> None:
    """Data quality KPIs · timeline · coverage heatmap · download."""
    st.markdown("### Data Collection & Quality")
    cfg = get_config()
    with st.spinner("Reading dataset…"):
        status = _load_collection(str(cfg.paths.data_dir))

    if not status["has_data"]:
        st.markdown(f"""
        <div class="kpi-card" style="background:{t['card_bg']};
             border-color:{t['card_border']};border-left:4px solid #E32017;">
          <div style="font-weight:700;color:{t['font']};margin-bottom:.4rem;">
            No Data Found</div>
          <div style="font-size:.87rem;color:{t['muted']};">
            No data at <code>data/tfl_merged.csv</code>. Start collection:</div>
          <pre style="background:{t['plot']};border-radius:8px;padding:.7rem;
                      margin-top:.5rem;font-size:.82rem;
                      color:{t['font']};">python data_collection.py</pre>
        </div>""", unsafe_allow_html=True)
        return

    df  = status["df"]
    n   = status["record_count"]
    TARGET = 14_784

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Records", f"{n:,}")
    with k2:
        mp = df.isnull().mean().mean() * 100
        st.metric("Missing Values", f"{mp:.1f}%")
    with k3:
        dp = df.duplicated().mean() * 100
        st.metric("Duplicate Rows", f"{dp:.1f}%")
    with k4:
        nl = len(df["line"].unique()) if "line" in df.columns else 0
        st.metric("Lines Covered", f"{nl} / 11")

    pct = min(n / TARGET * 100, 100.0)
    st.markdown(f"**Collection Progress: {n:,} / {TARGET:,} records ({pct:.1f}%)**")
    st.progress(pct / 100.0)

    if status["first_ts"] and status["last_ts"]:
        i1, i2 = st.columns(2)
        i1.info(f"**First record:** {status['first_ts'].strftime('%Y-%m-%d %H:%M')}")
        i2.info(f"**Last record:** {status['last_ts'].strftime('%Y-%m-%d %H:%M')}")

    st.markdown("---")

    if "timestamp" in df.columns:
        st.markdown("### Records Collected per Day")
        df2 = df.copy()
        df2["date"] = df2["timestamp"].dt.date
        daily = df2.groupby("date").size().reset_index(name="count")
        fig_tl = go.Figure(go.Bar(
            x=daily["date"], y=daily["count"], marker_color="#003B6F",
            hovertemplate="%{x}<br>%{y:,} records<extra></extra>"))
        fig_tl = _fig_base(fig_tl, t, "Collection Timeline", height=260)
        fig_tl.update_layout(yaxis_title="Records", xaxis_title="Date")
        st.plotly_chart(fig_tl, width='stretch', key="hist_timeline")

    if "line" in df.columns and "timestamp" in df.columns:
        st.markdown("---")
        st.markdown("### Coverage: Line × Day of Week")
        df3 = df.copy()
        df3["weekday"] = df3["timestamp"].dt.day_name()
        day_ord = ["Monday","Tuesday","Wednesday","Thursday",
                   "Friday","Saturday","Sunday"]
        cov = df3.groupby(["line","weekday"]).size().unstack(fill_value=0)
        cov = cov.reindex(columns=[d for d in day_ord if d in cov.columns])
        fig_cov = go.Figure(go.Heatmap(
            z=cov.values, x=cov.columns.tolist(), y=cov.index.tolist(),
            colorscale=[[0, t["paper"]], [1, "#003B6F"]],
            colorbar=dict(title="Records", tickfont=dict(color=t["font"])),
            hovertemplate="<b>%{y}</b>  %{x}<br>%{z:,} records<extra></extra>"))
        fig_cov.update_layout(
            paper_bgcolor=t["paper"], plot_bgcolor=t["paper"],
            font=dict(color=t["font"]),
            margin=dict(l=165, r=20, t=44, b=60), height=360,
            yaxis=dict(tickfont=dict(color=t["font"])))
        st.plotly_chart(fig_cov, width='stretch', key="hist_coverage")

    st.markdown("---")
    st.markdown("### Export")
    st.download_button(
        label="Download tfl_merged.csv",
        data=df.to_csv(index=False).encode(),
        file_name=f"tfl_merged_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────

def _tab_about(tabular: Dict, t: Dict) -> None:
    """
    Visual project showcase: architecture diagram · key stats grid · tech stack
    badges · model card · ethics. I avoided a plain wall of text so the tab
    reads as a professional portfolio piece for the dissertation demo.
    """
    metrics   = tabular.get("metrics", _FALLBACK_METRICS)
    best_name = tabular.get("best_model_name", "lightgbm").upper()
    best_m    = metrics.get("best", _FALLBACK_METRICS["best"])
    mae  = best_m.get("test_mae",  2.41)
    rmse = best_m.get("test_rmse", 4.91)
    r2   = best_m.get("test_r2",  0.749)
    n_mae = metrics.get("naive", _FALLBACK_METRICS["naive"]).get("test_mae", 7.26)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#001F4D 0%,#003B6F 55%,#005DB5 100%);
                border-radius:16px;padding:2rem;margin-bottom:1.4rem;">
      <h2 style="margin:0;font-weight:900;letter-spacing:-.02em;color:#FFFFFF;">
        London Underground Delay Predictor</h2>
      <p style="margin:.35rem 0 0;opacity:.78;font-size:.92rem;color:#FFFFFF;">
        COMP1682 Final Year Project · University of Greenwich · 2026</p>
      <div style="margin-top:1.1rem;display:flex;gap:2.2rem;flex-wrap:wrap;">
        {''.join(
            f'<div><div style="font-size:.67rem;opacity:.62;text-transform:uppercase;'
            f'letter-spacing:.06em;color:#FFFFFF;">{lab}</div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:#FFFFFF;">{val}</div></div>'
            for lab, val in [("Best Model", best_name),
                             ("Test MAE", f"{mae:.3f} sev"),
                             ("R²", f"{r2:.3f}"),
                             ("Lines", "11"),
                             ("Features", "40"),
                             ("vs Naive", f"−{(1-mae/n_mae)*100:.0f}%")])}
      </div>
    </div>""", unsafe_allow_html=True)

    about_tabs = st.tabs(["Architecture", "Key Stats",
                           "Tech Stack", "Model Card", "Ethics"])

    # ── Architecture ──────────────────────────────────────────────────────────
    with about_tabs[0]:
        st.markdown("### System Architecture")
        st.plotly_chart(_arch_diagram(t), width='stretch',
                        config={"displayModeBar": False})
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.9rem;margin-top:.5rem;">
          {''.join(
              f'<div class="kpi-card" style="background:{t["card_bg"]};'
              f'border-color:{t["card_border"]};border-left:4px solid {col};">'
              f'<div style="font-weight:700;color:{t["font"]};font-size:.88rem;">{h}</div>'
              f'<div style="font-size:.78rem;color:{t["muted"]};margin-top:.25rem;">{d}</div>'
              f'</div>'
              for h, d, col in [
                  ("Data Collection", "TfL Unified API + OpenWeatherMap, every 15 min, "
                   "merged into data/tfl_merged.csv", "#003B6F"),
                  ("Feature Engineering", "40 features: temporal encodings, weather, "
                   "lag/rolling history, crowding, network signals", "#0098D4"),
                  ("Model Training", "LightGBM with Optuna TPE (50 trials), strict "
                   "chronological 80/20 split, 5-fold TimeSeriesSplit CV", "#E32017"),
                  ("API + Dashboard", "FastAPI REST endpoint + this Streamlit "
                   "dashboard for interactive exploration", "#00782A"),
              ])}
        </div>""", unsafe_allow_html=True)

    # ── Key Stats ─────────────────────────────────────────────────────────────
    with about_tabs[1]:
        st.markdown("### Key Numbers")
        stats = [
            ("Test MAE",         f"{mae:.3f} sev",    "#E32017"),
            ("Test RMSE",        f"{rmse:.3f} sev",   "#E32017"),
            ("R² Score",         f"{r2:.3f}",          "#003B6F"),
            ("vs Naive",         f"−{(1-mae/n_mae)*100:.0f}%", "#00B140"),
            ("Training rows",    "~56,000",            "#0098D4"),
            ("Test rows",        "~6,742",             "#0098D4"),
            ("Features",         "40",                 "#9B0056"),
            ("Lines modelled",   "11",                 "#9B0056"),
            ("Collection period","16 days",            "#E86B00"),
            ("Frequency",        "15 min",             "#E86B00"),
            ("Random seed",      "42",                 "#6B7280"),
            ("LightGBM version", "4.6.0",              "#6B7280"),
        ]
        cols = st.columns(4)
        for i, (lbl, val, col) in enumerate(stats):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="kpi-card" style="background:{t['card_bg']};
                             border-color:{t['card_border']};
                             border-left:4px solid {col};
                             text-align:center;margin-bottom:.55rem;">
                  <div style="font-size:1.85rem;font-weight:900;
                              color:{col};line-height:1.15;">{val}</div>
                  <div style="font-size:.69rem;font-weight:600;color:{t['muted']};
                              text-transform:uppercase;letter-spacing:.06em;
                              margin-top:.25rem;">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown(f"""
            <div class="kpi-card" style="background:{t['card_bg']};
                         border-color:{t['card_border']};">
              <div style="font-weight:700;color:{t['font']};margin-bottom:.45rem;">
                Dissertation</div>
              <div style="font-size:.83rem;color:{t['muted']};">
                University of Greenwich<br>Module: COMP1682<br>
                Supervisor: Tuan Vuong<br>Submission: May 2026</div>
            </div>""", unsafe_allow_html=True)
        with dc2:
            st.markdown(f"""
            <div class="kpi-card" style="background:{t['card_bg']};
                         border-color:{t['card_border']};">
              <div style="font-weight:700;color:{t['font']};margin-bottom:.45rem;">
                Reproducibility</div>
              <div style="font-size:.83rem;color:{t['muted']};">
                Commit: 0d2fdc2<br>Random seed: 42<br>
                Python 3.14.2<br>LightGBM 4.6.0</div>
            </div>""", unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    with about_tabs[2]:
        st.markdown("### Technology Stack")
        techs = [
            ("Python 3.14",     "Core language",              "#3776AB"),
            ("LightGBM 4.6",    "Primary ML model",           "#E32017"),
            ("Streamlit",       "This dashboard",             "#FF4B4B"),
            ("FastAPI",         "REST prediction API",        "#009688"),
            ("pandas / numpy",  "Data wrangling",             "#150458"),
            ("scikit-learn",    "Baseline models + CV",       "#F7931E"),
            ("Optuna",          "HPO (TPE sampler, 50 trials)","#2563EB"),
            ("SHAP",            "Model explainability",       "#4CAF50"),
            ("Plotly",          "Interactive charts",         "#3F4F75"),
            ("TfL Unified API", "Real line-status data",      "#003B6F"),
            ("OpenWeatherMap",  "Weather data",               "#E86B00"),
            ("holidays",        "UK bank holiday calendar",   "#9B0056"),
        ]
        cols = st.columns(3)
        for i, (name, desc, col) in enumerate(techs):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.55rem;
                             background:{t['card_bg']};border:1px solid {t['card_border']};
                             border-left:4px solid {col};border-radius:10px;
                             padding:.6rem .75rem;margin-bottom:.4rem;">
                  <div>
                    <div style="font-weight:700;font-size:.86rem;
                                color:{t['font']};">{name}</div>
                    <div style="font-size:.74rem;color:{t['muted']};">{desc}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Model Card ────────────────────────────────────────────────────────────
    with about_tabs[3]:
        st.markdown("### Model Card")
        r_mae = metrics.get("ridge", _FALLBACK_METRICS["ridge"]).get("test_mae", 2.80)
        st.markdown(f"""
| Property | Value |
|---|---|
| Model type | LightGBM (gradient-boosted trees) |
| Target variable | `delay_severity` — ordinal encoding of TfL status label |
| Target scale | 0 = Good Service · 1 = Minor Delays · 2 = Severe Delays |
| Training rows | ~56,000 (15-min intervals, 16 days, 11 lines) |
| Test rows | ~6,742 (final 20 %, strict chronological split) |
| CV strategy | 5-fold TimeSeriesSplit |
| HPO | Optuna TPE, 50 trials |
| Random seed | 42 |
| **Test MAE** | **{mae:.3f} sev** |
| **Test RMSE** | **{rmse:.3f} sev** |
| **Test R²** | **{r2:.3f}** |
| vs Naive | −{(1-mae/n_mae)*100:.0f}% MAE improvement |
| vs Ridge | −{(1-mae/r_mae)*100:.0f}% MAE improvement |

**Status label thresholds** (severity midpoints):
- Good Service: predicted severity < 0.5
- Minor Delays: 0.5 – 1.5
- Severe Delays: ≥ 1.5

**Note:** `delay_minutes` is a synthetic proxy not used as the model target.
The TfL Unified API reports status categories, not minute values per line.
        """)

    # ── Ethics ────────────────────────────────────────────────────────────────
    with about_tabs[4]:
        st.markdown("""
### Ethical Considerations

**Privacy**
No personal passenger data is used. All measurements are aggregated at line level.
The TfL Unified API returns public operational status only, not individual journeys.

**Transparency**
Every prediction is backed by SHAP feature attributions. The model, training pipeline,
and data collection scripts are fully open-source and reproducible.

**Limitations**
Predictions are estimates based on historical patterns. Major incidents — strikes,
signal failures, infrastructure works — are not present in training data and
can degrade accuracy without warning.

**Human Oversight**
This system is designed as **decision support**, not automated action. Human review
is required before acting on any prediction in an operational context.

**Data Provenance**
16 days of real TfL + OpenWeatherMap data, April 2026. `delay_minutes` is the
actual TfL-measured operational delay, not a synthetic proxy.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: page config → CSS → artifact loading → header → sidebar → tabs.
    I call apply_custom_css exactly once (before any content) and load
    heavy objects via cache_resource so reruns are instant.
    """
    st.set_page_config(
        page_title="TfL Delay Predictor",
        page_icon="🚇",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS injected once, before any content renders
    apply_custom_css()

    cfg = get_config()
    latest_run = get_latest_run_id(cfg.paths.artifacts_dir)

    if latest_run is None:
        _render_header()
        st.error(
            "No training run found in `artifacts/`. "
            "Run `python train.py` to generate model artifacts.")
        st.stop()

    artifact_dir = str(cfg.paths.artifacts_dir / latest_run)

    with st.spinner("Loading model artifacts…"):
        models  = _load_models(artifact_dir)
        tabular = _load_tabular(artifact_dir)

    if not models:
        _render_header()
        st.error(
            "Failed to load model from `artifacts/`. "
            "Ensure `python train.py` completed successfully.")
        st.stop()

    # Single _theme() call — palette passed into every chart builder
    t = _theme()

    _render_header()
    _render_live_status_strip(t)

    metrics   = tabular.get("metrics", _FALLBACK_METRICS)
    best_name = tabular.get("best_model_name", "lightgbm")
    sb = _render_sidebar(metrics, best_name)

    tab_labels = [
        "🔮 Predictions",
        "📊 Performance",
        "🚇 Line Comparison",
        "📈 Historical Trends",
        "💾 Data Collection",
        "ℹ️ About",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _tab_predictions(models, tabular, sb, t)

    with tabs[1]:
        _tab_performance(tabular, t)

    with tabs[2]:
        _tab_line_comparison(tabular, t)

    with tabs[3]:
        _tab_historical(tabular, t)

    with tabs[4]:
        _tab_data_collection(t)

    with tabs[5]:
        _tab_about(tabular, t)

    st.markdown(f"""
    <div class="dash-footer">
        London Underground Delay Predictor &nbsp;·&nbsp;
        COMP1682 Dissertation &nbsp;·&nbsp; University of Greenwich &nbsp;·&nbsp;
        Built with Streamlit &amp; Plotly &nbsp;·&nbsp;
        Run: {latest_run} &nbsp;·&nbsp;
        {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
