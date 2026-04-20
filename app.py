"""
London Underground Delay Predictor — Streamlit Dashboard
=========================================================
COMP1682 Final Year Dissertation · University of Greenwich · 2026
Author: Marwan Amla  |  Supervisor: Tuan Vuong

I designed this dashboard to present my LightGBM delay-prediction model in a
production-quality interface. Every design decision — from the TfL colour palette
to the theme-safe Plotly figures — is documented in the dissertation write-up.

Run:  streamlit run app.py  (from project root, with .venv activated)
"""

from __future__ import annotations

import sys
import json
import logging
import re
import subprocess
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
from data_collection import LINE_CROWDING_WEIGHT as CROWDING_WEIGHTS
from events import VENUES, generate_event_calendar
from utils import get_latest_run_id
from app.map_data import LINE_PATHS, INTERCHANGE_STATIONS

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


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a float to a closed interval."""
    return max(lo, min(hi, v))


PEAK_HOURS: List[Tuple[int, int]] = [(7, 9), (17, 19)]

# 24 h × 12 readings/h = 288 five-minute intervals — used to cap test-set
# time-series charts to the most recent 24 hours of data.
_RECENT_TEST_SAMPLES: int = 288

# Computed once at import; survives all Streamlit reruns.
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(ROOT),
        ).decode().strip()
    except Exception:
        return "unknown"

_BUILD_COMMIT: str = _git_commit()


def _build_info() -> str:
    """Return a one-line build stamp for the footer."""
    try:
        import lightgbm as _lgb
        lgb_ver = _lgb.__version__
    except Exception:
        lgb_ver = "?"
    pv = sys.version_info
    return (
        f"Commit: {_BUILD_COMMIT} &nbsp;·&nbsp; "
        f"Python {pv.major}.{pv.minor}.{pv.micro} &nbsp;·&nbsp; "
        f"LightGBM {lgb_ver}"
    )


def _default_crowding(line: str, hour: int) -> float:
    """Default sidebar crowding value based on line baseline and peak/off-peak."""
    base = CROWDING_WEIGHTS.get(line, 0.05)
    if any(start <= hour < end for start, end in PEAK_HOURS):
        return _clamp(base * 1.6, 0.0, 1.0)
    return _clamp(base * 0.9, 0.0, 1.0)

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

    FRAGILITY NOTE: selectors below target Streamlit's internal data-testid
    attributes (e.g. "stSidebar", "stMetricLabel", "stExpander").  These are
    implementation details — not a public API — and Streamlit may rename or
    remove them without notice.  If the styling breaks after a Streamlit
    upgrade, audit the DOM with browser DevTools and update the selectors here.
    The broad rule `section[data-testid="stSidebar"] * { color:... !important }`
    is the most likely casualty; replace with narrower selectors if it causes
    conflicts.
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
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border:1px solid rgba(255,255,255,0.16);
        border-radius:10px;
        background:rgba(255,255,255,0.08);
        box-shadow:0 8px 24px rgba(0,20,58,0.16);
        overflow:hidden;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details {
        background:rgba(255,255,255,0.08) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        padding:0.15rem 0.2rem;
        background:rgba(255,255,255,0.08) !important;
        transition:background .18s ease, box-shadow .18s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        background:rgba(255,255,255,0.12) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] {
        background:rgba(255,255,255,0.08) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] summary {
        background:rgba(255,255,255,0.08) !important;
        box-shadow:inset 0 -1px 0 rgba(255,255,255,0.10);
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] summary:hover {
        background:rgba(255,255,255,0.12) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
        background-color:transparent !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > div {
        background:rgba(255,255,255,0.08) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        color:#DCE8FF !important;
        font-size:0.87rem !important;
        font-weight:700 !important;
        letter-spacing:.02em !important;
        text-transform:none !important;
    }
    section[data-testid="stSidebar"] .st-key-refresh_data button {
        border-radius:10px !important;
        min-height:2.7rem !important;
        background:rgba(255,255,255,0.08) !important;
        color:#DCE8FF !important;
        border:1px solid rgba(255,255,255,0.16) !important;
        box-shadow:0 8px 24px rgba(0,20,58,0.16) !important;
    }
    section[data-testid="stSidebar"] .st-key-refresh_data button:hover {
        transform:translateY(-2px);
        background:rgba(255,255,255,0.12) !important;
        box-shadow:0 12px 30px rgba(0,20,58,0.22) !important;
    }
    section[data-testid="stSidebar"] .st-key-refresh_data button p {
        color:#DCE8FF !important;
    }

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

    /* ── Risk count cards (Risk Map tab) ──────────────────────────────── */
    .risk-count-card {
        background:var(--rcc-bg); border:1px solid var(--rcc-border);
        border-left:5px solid var(--rcc-accent);
        border-radius:10px; padding:0.9rem 1.1rem; font-family:'Inter',sans-serif;
    }
    .risk-count-card .rcc-label {
        font-size:0.72rem; color:var(--rcc-sub);
        text-transform:uppercase; letter-spacing:.06em;
    }
    .risk-count-card .rcc-value {
        font-size:2rem; font-weight:800; color:var(--rcc-accent);
    }

    /* ── Risk detail cards (Risk Map tab) ─────────────────────────────── */
    .risk-detail-card {
        background:var(--rdc-bg); border:1px solid var(--rdc-accent);
        border-left:5px solid var(--rdc-line);
        border-radius:12px; padding:0.85rem 1rem;
        font-family:'Inter',sans-serif; margin-bottom:0.5rem;
    }
    .risk-detail-card .rdc-title { font-weight:700; font-size:0.9rem; color:var(--rdc-txt); }
    .risk-detail-card .rdc-row   { display:flex; justify-content:space-between; margin-top:0.4rem; }
    .risk-detail-card .rdc-risk  { font-size:1.4rem; font-weight:800; color:var(--rdc-accent); }
    .risk-detail-card .rdc-sev   { font-size:0.8rem; color:var(--rdc-sub); align-self:flex-end; }
    .risk-detail-card .rdc-meta  { font-size:0.75rem; color:var(--rdc-sub); margin-top:0.3rem; }
    """
    # base64 encodes the CSS *text* so it can be embedded safely inside a JS
    # string literal — no font file is embedded here; 'Inter' loads via the
    # system font stack fallback chain defined in the font-family rule above.
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
        # joblib unpickling resolves class names against __main__; register here so
        # NaiveBaselineModel is found when the pickle was saved from train.py.
        sys.modules["__main__"].NaiveBaselineModel = train.NaiveBaselineModel  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug("Could not register NaiveBaselineModel in __main__: %s", exc)

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

    fi_p = path / "feature_importance.csv"
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


@st.cache_data(show_spinner=False, ttl=300)
def _load_run_metadata(artifact_dir: str) -> Dict:
    """Load run-level metadata for the About tab from saved artifacts."""
    path = Path(artifact_dir)
    meta: Dict = {}

    data_info_p = path / "data_info.txt"
    if data_info_p.exists():
        try:
            txt = data_info_p.read_text(encoding="utf-8", errors="ignore")
            mode_m = re.search(r"Data mode:\s*(.+)", txt)
            shape_m = re.search(r"Shape:\s*\((\d+),\s*(\d+)\)", txt)
            range_m = re.search(r"Date range:\s*(.+?)\s+to\s+(.+)", txt)
            if mode_m:
                meta["data_mode"] = mode_m.group(1).strip()
            if shape_m:
                meta["data_rows"] = int(shape_m.group(1))
                meta["data_cols"] = int(shape_m.group(2))
            if range_m:
                meta["date_start"] = pd.to_datetime(range_m.group(1).strip())
                meta["date_end"] = pd.to_datetime(range_m.group(2).strip())
        except Exception as exc:
            logger.warning("Could not parse data_info.txt: %s", exc)

    model_info_p = path / "model_info.json"
    if model_info_p.exists():
        try:
            with open(model_info_p, encoding="utf-8") as f:
                meta.update(json.load(f))
        except Exception as exc:
            logger.warning("Could not parse model_info.json: %s", exc)

    run_log_p = path / "run.log"
    if run_log_p.exists():
        try:
            log_txt = run_log_p.read_text(encoding="utf-8", errors="ignore")
            train_m = re.search(r"Training set:\s*\((\d+),\s*(\d+)\)", log_txt)
            test_m = re.search(r"Test set:\s*\((\d+),\s*(\d+)\)", log_txt)
            time_m = re.search(r"Total time:\s*([^\r\n]+)", log_txt)
            if train_m:
                meta["train_rows"] = int(train_m.group(1))
                meta["feature_count"] = int(train_m.group(2))
            if test_m:
                meta["test_rows"] = int(test_m.group(1))
            if time_m:
                meta["train_duration"] = time_m.group(1).strip()
        except Exception as exc:
            logger.warning("Could not parse run.log: %s", exc)

    return meta


@st.cache_data(show_spinner=False, ttl=300)
def _load_analysis_outputs(analysis_dir: str) -> Dict:
    """Load walk-forward backtest, drift, and ablation outputs from analysis/outputs/."""
    out: Dict = {}
    base = Path(analysis_dir)

    wf_p = base / "walk_forward_backtest.json"
    if wf_p.exists():
        try:
            with open(wf_p) as f:
                out["walk_forward"] = json.load(f)
        except Exception as exc:
            logger.warning("Could not load walk_forward_backtest.json: %s", exc)

    drift_p = base / "drift_results.json"
    if drift_p.exists():
        try:
            with open(drift_p) as f:
                out["drift"] = json.load(f)
        except Exception as exc:
            logger.warning("Could not load drift_results.json: %s", exc)

    abl_json_p = base / "ablation_results.json"
    if abl_json_p.exists():
        try:
            with open(abl_json_p) as f:
                out["ablation"] = json.load(f)
        except Exception as exc:
            logger.warning("Could not load ablation_results.json: %s", exc)

    abl_png_p = base / "ablation_results.png"
    if abl_png_p.exists():
        out["ablation_png"] = str(abl_png_p)

    for key, filename in [
        ("equity",           "equity_results.json"),
        ("arima",            "arima_results.json"),
        ("ensemble",         "ensemble_results.json"),
        ("shap_selection",   "shap_feature_selection.json"),
    ]:
        p = base / filename
        if p.exists():
            try:
                with open(p) as f:
                    out[key] = json.load(f)
            except Exception as exc:
                logger.warning("Could not load %s: %s", filename, exc)

    return out


@st.cache_data(show_spinner=False, ttl=300)
def _compute_cv_mae_breakdown(artifact_dir: str) -> Optional[Dict]:
    """Return fold-level CV MAE breakdown.

    Loads cv_fold_scores.json written by train.py — zero model loading cost.
    Returns None for runs predating that file; re-train to generate it.
    """
    path = Path(artifact_dir)
    saved = path / "cv_fold_scores.json"
    if saved.exists():
        try:
            with open(saved) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load cv_fold_scores.json: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_future_predictor(artifact_dir: str):
    """Load the inference predictor once per artifact directory."""
    from future_prediction import FutureDelayPredictor

    art = Path(artifact_dir)
    return FutureDelayPredictor(
        str(art / "best_model.pkl"),
        str(art / "feature_metadata.pkl"),
    )


def _status_from_severity(severity: float) -> Tuple[str, str]:
    """Map predicted severity to the corresponding service-status label."""
    if severity < STATUS_GOOD_MAX:
        return "#00B140", "Good Service"
    if severity < STATUS_MINOR_MAX:
        return "#FFD300", "Minor Delays"
    return "#DC241F", "Severe Delays"


def _predict_scenario(
    artifact_dir: str,
    line: str, hour: int, day_of_week: int,
    temp_c: float, precip_mm: float, humidity: int,
    crowding_index: Optional[float] = None,
) -> Optional[float]:
    """
    Call FutureDelayPredictor for a single scenario point.
    I wrap this in try/except so the dashboard never crashes if the predictor
    is unavailable (e.g., feature_metadata missing from an old run).
    Returns None on any failure so callers can fall back to test-data averages.
    """
    try:
        predictor = _get_future_predictor(artifact_dir)
        result = _predict_with_predictor(
            predictor,
            line=line,
            target_datetime=_resolve_point_datetime(hour, day_of_week),
            temp_c=temp_c, precip_mm=precip_mm, humidity=humidity,
            crowding_index=crowding_index,
        )
        return None if result is None else float(result["prediction"])
    except Exception as exc:
        logger.warning(
            "Scenario prediction failed for %s at %02d:00 on weekday %s: %s",
            line, hour, day_of_week, exc,
        )
        return None


def _resolve_scenario_date(
    day_of_week: int,
    reference_dt: Optional[datetime] = None,
) -> datetime.date:
    """Return one fixed calendar date for the selected weekday."""
    ref = reference_dt or datetime.now()
    days_ahead = (day_of_week - ref.weekday()) % 7
    return (ref + timedelta(days=days_ahead)).date()


def _resolve_point_datetime(
    hour: int,
    day_of_week: int,
    reference_dt: Optional[datetime] = None,
) -> datetime:
    """Resolve the next occurrence of the selected weekday/hour for point predictions."""
    ref = reference_dt or datetime.now()
    base_date = _resolve_scenario_date(day_of_week, ref)
    candidate = datetime.combine(base_date, datetime.min.time()).replace(hour=hour)
    if candidate <= ref and day_of_week == ref.weekday():
        candidate += timedelta(days=7)
    return candidate


def _day_hour_datetime(
    forecast_date: datetime.date,
    hour: int,
) -> datetime:
    """Build a deterministic datetime for one hour on the selected forecast date."""
    return datetime.combine(forecast_date, datetime.min.time()).replace(hour=hour)


def _predict_with_predictor(
    predictor,
    line: str,
    target_datetime: datetime,
    temp_c: float,
    precip_mm: float,
    humidity: int,
    crowding_index: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Predict a single scenario using one predictor instance.

    Uses the public engineer_features wrapper then predict_from_features so the
    dashboard does not call private underscore methods on the predictor.
    """
    try:
        features = predictor.engineer_features(
            line=line,
            target_datetime=target_datetime,
            weather_forecast={
                "temperature": temp_c,
                "precipitation": precip_mm,
                "humidity": humidity,
            },
            recent_delays=None,
        )
        if crowding_index is not None:
            crowding_val = _clamp(float(crowding_index), 0.0, 1.0)
            features.at[0, "crowding_index"] = crowding_val
            peak_flag = float(features.at[0, "peak_time"]) if "peak_time" in features.columns else 0.0
            if "crowding_x_peak" in features.columns:
                features.at[0, "crowding_x_peak"] = crowding_val * peak_flag
        result = predictor.predict_from_features(features, line)
        return {**result, "hour": float(target_datetime.hour)}
    except Exception as exc:
        logger.warning(
            "Predictor failed for %s at %s: %s",
            line,
            target_datetime.isoformat(),
            exc,
        )
        return None


def _event_note(line: str, hour: int, day_of_week: int) -> Optional[str]:
    """Return a short event badge for the selected scenario when relevant."""
    try:
        target = _resolve_point_datetime(hour, day_of_week)
        cal = generate_event_calendar(target.date(), target.date())
        candidates: List[Tuple[float, str]] = []
        for _, row in cal.iterrows():
            venue = VENUES.get(row["venue"])
            if venue is None or line not in venue.lines:
                continue
            ev_start = pd.Timestamp(row["date"]).to_pydatetime()
            ev_end = ev_start + timedelta(hours=venue.impact_hours)
            if ev_start <= target <= ev_end:
                level = "high" if venue.capacity >= 60000 else "moderate"
                text = (
                    f"{row['name']} {ev_start.strftime('%H:%M')} - "
                    f"{level} crowding expected"
                )
                candidates.append((venue.capacity, text))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _gauge(severity: float, t: Dict) -> go.Figure:
    """
    Semi-circle gauge on a 0–2 severity scale (model target: delay_severity).
    Colour zones: green 0–0.5 (Good), yellow 0.5–1.5 (Minor), red 1.5–2 (Severe).
    """
    colour, label = _status_from_severity(severity)
    val = round(max(0.0, min(severity, 2.5)), 2)
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
        annotations=[
            dict(
                x=0.5, y=0.02, xref="paper", yref="paper", showarrow=False,
                text=f"Approx. {_fmt_min(severity * SEV_TO_MIN)}",
                font=dict(size=13, color=t["muted"]),
            )
        ],
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
    ldf = test_preds[test_preds["line"] == line].sort_values("timestamp").tail(_RECENT_TEST_SAMPLES)
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


def _scenario_forecast_chart(
    hours: List[int],
    values_a: List[float],
    ci_lo_a: List[float],
    ci_hi_a: List[float],
    line_name: str,
    t: Dict,
    values_b: Optional[List[float]] = None,
    ci_lo_b: Optional[List[float]] = None,
    ci_hi_b: Optional[List[float]] = None,
    label_a: str = "Scenario A",
    label_b: str = "Scenario B",
) -> go.Figure:
    """24-hour scenario chart with prediction-interval shading."""
    lc_a = LINE_COLOURS.get(line_name, "#003B6F")
    r, g, b = int(lc_a[1:3], 16), int(lc_a[3:5], 16), int(lc_a[5:7], 16)
    lc_b = "#E86B00"
    rb, gb, bb = int(lc_b[1:3], 16), int(lc_b[3:5], 16), int(lc_b[5:7], 16)

    fig = go.Figure()
    for s, e in PEAK_HOURS:
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(227,32,23,0.07)", line_width=0)

    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=ci_hi_a + ci_lo_a[::-1],
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.12)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name=f"{label_a} 95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=values_a, mode="lines",
        line=dict(color=lc_a, width=3), name=label_a,
        hovertemplate="%{x:02d}:00 -> %{y:.2f} sev<extra></extra>",
    ))

    if values_b is not None and ci_lo_b is not None and ci_hi_b is not None:
        fig.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=ci_hi_b + ci_lo_b[::-1],
            fill="toself", fillcolor=f"rgba({rb},{gb},{bb},0.11)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name=f"{label_b} 95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=values_b, mode="lines",
            line=dict(color=lc_b, width=2.5, dash="dash"), name=label_b,
            hovertemplate="%{x:02d}:00 -> %{y:.2f} sev<extra></extra>",
        ))

    fig = _fig_base(fig, t, f"24-Hour Scenario Forecast - {line_name} Line", height=360)
    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title="Hour of Day",
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            tickfont=dict(color=t["muted"]),
        ),
        yaxis=dict(title="Predicted Severity (0-2)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _feature_correlation_chart(
    test_preds: pd.DataFrame,
    feat_imp: Optional[pd.DataFrame],
    t: Dict,
) -> Optional[go.Figure]:
    """Correlation heatmap using the most relevant numeric columns available."""
    df = test_preds.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["peak_time"] = (
        df["hour"].between(7, 8) | df["hour"].between(17, 18)
    ).astype(int)
    df["abs_error"] = (df["actual"] - df["pred_best"]).abs()

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in {"Unnamed: 0"}
    ]
    if feat_imp is not None and "feature" in feat_imp.columns:
        preferred = [c for c in feat_imp["feature"].tolist() if c in numeric_cols]
    else:
        preferred = []
    fallback = [c for c in numeric_cols if c not in preferred]
    cols = (preferred + fallback)[:15]
    if len(cols) < 4:
        return None

    corr = df[cols].corr(method="pearson").fillna(0.0)
    plain = {
        "actual": "Actual severity",
        "pred_best": "Predicted severity",
        "abs_error": "Absolute error",
        "hour": "Hour",
        "day_of_week": "Day of week",
        "month": "Month",
        "is_weekend": "Weekend",
        "peak_time": "Peak hours",
    }
    labels = [plain.get(c, c.replace("_", " ").title()) for c in cols]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        reversescale=True,
        colorbar=dict(title="r", tickfont=dict(color=t["font"])),
        hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=t["paper"], plot_bgcolor=t["paper"],
        font=dict(color=t["font"]),
        margin=dict(l=180, r=20, t=48, b=90), height=520,
        title=dict(text="Feature Correlation Matrix", x=0.01,
                   font=dict(size=14, color=t["font"])),
        xaxis=dict(tickangle=-40, tickfont=dict(color=t["muted"])),
        yaxis=dict(tickfont=dict(color=t["font"])),
    )
    return fig


def _model_comparison_chart(metrics: Dict, t: Dict) -> go.Figure:
    """Grouped bar chart: MAE / RMSE / R² across Naive · Ridge · XGBoost · LightGBM."""
    models = [k for k in ("naive", "ridge", "xgboost", "best") if k in metrics]
    labels = {"naive": "Naive", "ridge": "Ridge", "xgboost": "XGBoost", "best": "LightGBM"}
    colors = {"naive": "#9CA3AF", "ridge": "#0098D4", "xgboost": "#FF7733", "best": "#003B6F"}

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
    except Exception as exc:
        logger.warning("TfL status fetch failed: %s", exc)
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

    def _badge(line: str, status: str) -> str:
        col = _dot_col(status)  # resolve once — used four times below
        return (
            f'<span style="display:inline-flex;align-items:center;gap:.3rem;'
            f'background:{col}1A;border:1px solid {col}66;'
            f'border-radius:20px;padding:.18rem .55rem;font-size:.72rem;font-weight:700;'
            f'color:{col};white-space:nowrap;margin:.15rem;">'
            f'<span style="width:7px;height:7px;border-radius:50%;'
            f'background:{col};display:inline-block;flex-shrink:0;"></span>'
            f'{line}</span>'
        )

    badges = "".join(
        _badge(line, status)
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
          <div id="tfl-live-clock" style="font-size:1.95rem;font-weight:900;color:#FFFFFF;line-height:1;">
            {clock_str}</div>
          <div id="tfl-live-date" style="font-size:.73rem;color:rgba(255,255,255,.65);margin-top:.1rem;">
            {date_str}</div>
          <div style="font-size:.62rem;color:rgba(255,255,255,.45);margin-top:.08rem;">
            London Time</div>
          <div id="tfl-live-refresh" style="font-size:.6rem;color:rgba(255,255,255,.38);margin-top:.08rem;">
            Auto-refresh every 60s</div>
        </div>
      </div>
    </div>
    <hr class="tfl-underline">
    """, unsafe_allow_html=True)


def _enable_live_clock() -> None:
    """Keep the header clock live via a 1-second timer.
    No page reload — data freshness is handled by cache TTLs and the
    manual Refresh Data button in the sidebar.
    Falls back to a local requestAnimationFrame loop when the iframe is
    sandboxed and cannot write to window.parent (e.g. Streamlit Cloud).
    """
    import streamlit.components.v1 as components

    components.html(
        """<script>
        (function() {
            function formatClock(now) {
                return new Intl.DateTimeFormat('en-GB', {
                    timeZone: 'Europe/London', hour: '2-digit', minute: '2-digit', hour12: false
                }).format(now);
            }
            function formatDate(now) {
                return new Intl.DateTimeFormat('en-GB', {
                    timeZone: 'Europe/London', weekday: 'long', day: '2-digit', month: 'long', year: 'numeric'
                }).format(now);
            }
            function applyToDoc(doc) {
                const now = new Date();
                const clockEl = doc.getElementById('tfl-live-clock');
                const dateEl  = doc.getElementById('tfl-live-date');
                const subEl   = doc.getElementById('tfl-live-refresh');
                if (clockEl) clockEl.textContent = formatClock(now);
                if (dateEl)  dateEl.textContent  = formatDate(now);
                if (subEl)   subEl.textContent   = 'Live clock \u00b7 use Refresh Data to reload';
            }

            // Primary path: update parent document via setInterval.
            try {
                const parentWin = window.parent;
                applyToDoc(parentWin.document);
                if (!parentWin.__tflClockInterval) {
                    parentWin.__tflClockInterval = parentWin.setInterval(
                        function() { applyToDoc(parentWin.document); }, 1000
                    );
                }
            } catch (e) {
                // Sandboxed iframe: fall back to a local RAF-paced loop that
                // updates elements in this iframe's own document.
                var _lastTick = 0;
                function rafTick(ts) {
                    if (ts - _lastTick >= 1000) {
                        applyToDoc(document);
                        _lastTick = ts;
                    }
                    window.requestAnimationFrame(rafTick);
                }
                window.requestAnimationFrame(rafTick);
            }
        })();
        </script>""",
        height=0,
        scrolling=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def _scenario_key(prefix: str, field: str) -> str:
    return f"{prefix}_{field}"


def _sync_weather_preset(prefix: str) -> None:
    """Keep numeric weather inputs aligned with the selected preset label."""
    weather = st.session_state[_scenario_key(prefix, "weather")]
    st.session_state[_scenario_key(prefix, "precip")] = float(WEATHER_PRECIP[weather])
    st.session_state[_scenario_key(prefix, "humidity")] = WEATHER_HUMIDITY[weather]


def _ensure_weather_state(prefix: str, default_weather: str) -> None:
    """Initialize per-scenario weather state once without clobbering user edits."""
    weather_key = _scenario_key(prefix, "weather")
    precip_key = _scenario_key(prefix, "precip")
    humidity_key = _scenario_key(prefix, "humidity")

    if weather_key not in st.session_state:
        st.session_state[weather_key] = default_weather
    if precip_key not in st.session_state:
        st.session_state[precip_key] = float(WEATHER_PRECIP[st.session_state[weather_key]])
    if humidity_key not in st.session_state:
        st.session_state[humidity_key] = WEATHER_HUMIDITY[st.session_state[weather_key]]


def _ensure_compare_defaults(
    weather: str,
    temp_c: float,
    crowding_index: float,
) -> None:
    """Seed Scenario B with a contrasting weather preset on first use."""
    # Pick the opposite end of the weather spectrum so B is visually distinct
    # from A by default, making the comparison immediately meaningful.
    _CONTRAST = {
        "Clear": "Heavy Rain",
        "Light Rain": "Wind",
        "Heavy Rain": "Clear",
        "Wind": "Clear",
        "Fog": "Clear",
    }
    contrast_weather = _CONTRAST.get(weather, "Heavy Rain")
    defaults = {
        _scenario_key("scenario_b", "weather"): contrast_weather,
        _scenario_key("scenario_b", "temp_c"): max(0, temp_c - 8),
        _scenario_key("scenario_b", "crowding"): crowding_index,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    _ensure_weather_state("scenario_b", st.session_state[_scenario_key("scenario_b", "weather")])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar(metrics: Dict, best_name: str) -> Dict:
    """Build sidebar controls and return the current scenario selections."""
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
        value=False,
        key="network_mode",
        help="Show all 11 lines simultaneously instead of a single line",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<span style='font-size:.7rem;font-weight:700;letter-spacing:.08em;"
        "color:#A8C4F0;text-transform:uppercase;'>Scenario</span>",
        unsafe_allow_html=True,
    )

    sel_line = st.sidebar.selectbox(
        "Tube Line",
        LINE_NAMES,
        index=LINE_NAMES.index("Central"),
        disabled=network_mode,
        help="Choose the line to predict",
    )
    lc = LINE_COLOURS.get(sel_line, "#003B6F")
    st.sidebar.markdown(
        f'<div style="width:100%;height:3px;background:{lc};'
        f'border-radius:2px;margin:-6px 0 6px;"></div>',
        unsafe_allow_html=True,
    )

    hour = st.sidebar.slider("Hour of Day", 0, 23, datetime.now().hour)
    is_peak = any(s <= hour < e for s, e in PEAK_HOURS)
    st.sidebar.caption("🔴 Peak hour" if is_peak else "🟢 Off-peak")

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = st.sidebar.selectbox("Day of Week", day_names, index=datetime.now().weekday())
    dow = day_names.index(day_name)
    crowding_default = _default_crowding(sel_line, hour)

    _ensure_weather_state("scenario_a", WEATHER_OPTIONS[0])
    weather = st.sidebar.selectbox(
        "Weather",
        WEATHER_OPTIONS,
        index=WEATHER_OPTIONS.index(st.session_state[_scenario_key("scenario_a", "weather")]),
        key=_scenario_key("scenario_a", "weather"),
        on_change=_sync_weather_preset,
        args=("scenario_a",),
        format_func=lambda w: f"{WEATHER_ICONS[w]} {w}",
    )

    temp_key = _scenario_key("scenario_a", "temp_c")
    crowd_key = _scenario_key("scenario_a", "crowding")
    if temp_key not in st.session_state:
        st.session_state[temp_key] = 15
    if crowd_key not in st.session_state:
        st.session_state[crowd_key] = float(crowding_default)

    temp_c = st.sidebar.slider("Temperature (°C)", -5, 35, key=temp_key)
    crowding_index = st.sidebar.slider(
        "Crowding Index",
        0.0,
        1.0,
        step=0.01,
        key=crowd_key,
        help="Manual crowding override used in live scenario prediction.",
    )

    with st.sidebar.expander("Advanced Weather"):
        precip_mm = st.slider(
            "Precipitation (mm)",
            0.0,
            30.0,
            step=0.5,
            key=_scenario_key("scenario_a", "precip"),
        )
        humidity = st.slider(
            "Humidity (%)",
            20,
            100,
            key=_scenario_key("scenario_a", "humidity"),
        )

    st.sidebar.caption(
        f"Typical peak crowding weight for {sel_line}: "
        f"{CROWDING_WEIGHTS.get(sel_line, 0.05):.2f}"
    )

    _ensure_compare_defaults(weather, temp_c, crowding_index)
    compare_weather = st.session_state[_scenario_key("scenario_b", "weather")]
    compare_temp_c = float(st.session_state[_scenario_key("scenario_b", "temp_c")])
    compare_precip_mm = float(st.session_state[_scenario_key("scenario_b", "precip")])
    compare_humidity = int(st.session_state[_scenario_key("scenario_b", "humidity")])
    compare_crowding = float(st.session_state[_scenario_key("scenario_b", "crowding")])

    with st.sidebar.expander(
        "Scenario B",
        expanded=st.session_state.get("compare_enabled", False),
    ):
        compare_enabled = st.toggle(
            "Enable comparison",
            value=False,
            key="compare_enabled",
            help="Overlay a second what-if scenario on the 24-hour forecast.",
        )
        if compare_enabled:
            compare_weather = st.selectbox(
                "Scenario B Weather",
                WEATHER_OPTIONS,
                index=WEATHER_OPTIONS.index(st.session_state[_scenario_key("scenario_b", "weather")]),
                key=_scenario_key("scenario_b", "weather"),
                on_change=_sync_weather_preset,
                args=("scenario_b",),
                format_func=lambda w: f"{WEATHER_ICONS[w]} {w}",
            )
            compare_temp_c = st.slider(
                "Scenario B Temperature (°C)",
                -5,
                35,
                key=_scenario_key("scenario_b", "temp_c"),
            )
            compare_precip_mm = st.slider(
                "Scenario B Precipitation (mm)",
                0.0,
                30.0,
                step=0.5,
                key=_scenario_key("scenario_b", "precip"),
            )
            compare_humidity = st.slider(
                "Scenario B Humidity (%)",
                20,
                100,
                key=_scenario_key("scenario_b", "humidity"),
            )
            compare_crowding = st.slider(
                "Scenario B Crowding",
                0.0,
                1.0,
                step=0.01,
                key=_scenario_key("scenario_b", "crowding"),
            )
        else:
            compare_enabled = False

    st.sidebar.markdown("---")

    if st.sidebar.button(
        "🔄 Refresh Data",
        key="refresh_data",
        help="Clear all caches and reload from disk",
    ):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    m = metrics.get("best", {})
    mae = m.get("test_mae", 2.41)
    r2 = m.get("test_r2", 0.749)
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

    return dict(
        line=sel_line,
        hour=hour,
        dow=dow,
        weather=weather,
        temp_c=temp_c,
        precip_mm=precip_mm,
        humidity=humidity,
        network_mode=network_mode,
        crowding_index=crowding_index,
        compare_enabled=compare_enabled,
        compare_weather=compare_weather,
        compare_temp_c=compare_temp_c,
        compare_precip_mm=compare_precip_mm,
        compare_humidity=compare_humidity,
        compare_crowding=compare_crowding,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _cached_predict(
    artifact_dir: str,
    line: str,
    target_datetime_iso: str,
    temp_c: float,
    precip_mm: float,
    humidity: int,
    crowding_index: Optional[float],
) -> Optional[Dict[str, float]]:
    """Cache-friendly wrapper: takes only hashable args so cache_data can key on them."""
    predictor = _get_future_predictor(artifact_dir)
    if predictor is None:
        return None
    return _predict_with_predictor(
        predictor, line, datetime.fromisoformat(target_datetime_iso),
        temp_c, precip_mm, humidity, crowding_index,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _predict_24h_scenario(
    artifact_dir: str,
    line: str,
    forecast_date_iso: str,
    temp_c: float,
    precip_mm: float,
    humidity: int,
    crowding_index: Optional[float] = None,
) -> Dict[str, object]:
    """Return deterministic hourly predictions for one fixed forecast date."""
    try:
        predictor = _get_future_predictor(artifact_dir)
        forecast_date = datetime.fromisoformat(forecast_date_iso).date()
        preds: List[Optional[float]] = []
        ci_los: List[Optional[float]] = []
        ci_his: List[Optional[float]] = []
        failed_hours: List[int] = []
        for h in range(24):
            res = _predict_with_predictor(
                predictor,
                line=line,
                target_datetime=_day_hour_datetime(forecast_date, h),
                temp_c=temp_c,
                precip_mm=precip_mm,
                humidity=humidity,
                crowding_index=crowding_index,
            )
            if res is None:
                preds.append(None)
                ci_los.append(None)
                ci_his.append(None)
                failed_hours.append(h)
            else:
                preds.append(float(res["prediction"]))
                ci_los.append(float(res["ci_lo"]))
                ci_his.append(float(res["ci_hi"]))
        return {
            "pred": preds,
            "ci_lo": ci_los,
            "ci_hi": ci_his,
            "failed_hours": failed_hours,
            "forecast_date_iso": forecast_date.isoformat(),
        }
    except Exception as exc:
        logger.warning(
            "24-hour scenario forecast failed for %s on %s: %s",
            line,
            forecast_date_iso,
            exc,
        )
        return {
            "pred": [None] * 24,
            "ci_lo": [None] * 24,
            "ci_hi": [None] * 24,
            "failed_hours": list(range(24)),
            "forecast_date_iso": forecast_date_iso,
        }


def _tab_predictions(models: Dict, tabular: Dict, sb: Dict, t: Dict, artifact_dir: str) -> None:
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
    crowding_index = sb["crowding_index"]

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
                val = _predict_scenario(
                    artifact_dir,
                    ln,
                    hour,
                    dow,
                    temp_c,
                    precip_mm,
                    humidity,
                    crowding_index,
                )
                all_preds[ln] = val if val is not None else float(
                    test_preds[test_preds["line"] == ln][model_col].mean()
                    if not test_preds[test_preds["line"] == ln].empty else 0.0)

        # Overall network KPI (crowding-weighted)
        total_w   = sum(CROWDING_WEIGHTS.values())
        net_delay = (sum(all_preds.get(ln, 0) * CROWDING_WEIGHTS.get(ln, 0.05)
                         for ln in LINE_NAMES) / total_w)
        net_col, net_lbl = _status_from_severity(net_delay)
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
            sc, sl = _status_from_severity(delay)
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
    mae = float(np.abs(line_df["actual"] - line_df[model_col]).mean())

    with st.spinner("Computing prediction…"):
        live_pred = _predict_scenario(
            artifact_dir,
            line,
            hour,
            dow,
            temp_c,
            precip_mm,
            humidity,
            crowding_index,
        )

    pred_val = live_pred if live_pred is not None else float(line_df[model_col].mean())
    sc, sl   = _status_from_severity(pred_val)

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
    event_note = _event_note(line, hour, dow)

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
        # Severity scale legend — one-liner so any first-time user knows the units
        st.markdown(
            f"<div style='text-align:center;font-size:.75rem;color:{t['muted']};margin-top:-.4rem;'>"
            "<span style='color:#00B140;font-weight:700;'>0</span> Good Service &nbsp;·&nbsp; "
            "<span style='color:#FFD300;font-weight:700;'>1</span> Minor Delays &nbsp;·&nbsp; "
            "<span style='color:#DC241F;font-weight:700;'>2</span> Severe Delays"
            "</div>",
            unsafe_allow_html=True,
        )

    with f_col:
        st.markdown("**What's driving this prediction?**")
        st.caption("Bars show how much each factor shifts the predicted severity away from a typical seasonal baseline.")

        # Compute counterfactual contributions so bars reflect the current scenario.
        # 1. Baseline = seasonal typical weather, model-estimated crowding (pure temporal signal).
        # 2. Delta-weather = baseline vs current weather (isolates weather contribution).
        # 3. Delta-crowding = baseline vs current crowding (isolates crowding contribution).
        _SEASONAL_WEATHER: Dict[int, Dict] = {
            1: {"temp_c": 7.0,  "precip_mm": 2.2, "humidity": 80},
            2: {"temp_c": 7.0,  "precip_mm": 1.6, "humidity": 77},
            3: {"temp_c": 9.0,  "precip_mm": 1.7, "humidity": 72},
            4: {"temp_c": 11.0, "precip_mm": 1.8, "humidity": 68},
            5: {"temp_c": 15.0, "precip_mm": 2.0, "humidity": 66},
            6: {"temp_c": 18.0, "precip_mm": 1.8, "humidity": 65},
            7: {"temp_c": 20.0, "precip_mm": 1.8, "humidity": 64},
            8: {"temp_c": 20.0, "precip_mm": 2.0, "humidity": 66},
            9: {"temp_c": 17.0, "precip_mm": 2.0, "humidity": 70},
            10: {"temp_c": 14.0, "precip_mm": 2.7, "humidity": 75},
            11: {"temp_c": 10.0, "precip_mm": 2.4, "humidity": 80},
            12: {"temp_c": 7.0,  "precip_mm": 2.2, "humidity": 82},
        }
        target_dt = _resolve_point_datetime(hour, dow)
        seas = _SEASONAL_WEATHER[target_dt.month]
        target_dt_iso = target_dt.isoformat()
        top3_data: List[Tuple[str, float, str]] = []
        if live_pred is not None:
            # Zero-crowding + seasonal weather = pure temporal signal.
            # Using crowding=0.0 (not None) ensures _estimate_crowding() never
            # runs, so the baseline is always the same reference point.
            _r_base = _cached_predict(
                artifact_dir, line, target_dt_iso,
                seas["temp_c"], seas["precip_mm"], int(seas["humidity"]),
                crowding_index=0.0,
            )
            # Weather contribution: user weather vs seasonal weather, both at zero crowding.
            _r_weather = _cached_predict(
                artifact_dir, line, target_dt_iso,
                temp_c, precip_mm, humidity,
                crowding_index=0.0,
            )
            # Crowding contribution: user crowding vs zero crowding, seasonal weather.
            _r_crowd = _cached_predict(
                artifact_dir, line, target_dt_iso,
                seas["temp_c"], seas["precip_mm"], int(seas["humidity"]),
                crowding_index=crowding_index,
            )
            if _r_base is not None:
                _base_val      = float(_r_base["prediction"])
                _weather_delta = abs(float(_r_weather["prediction"]) - _base_val) if _r_weather else 0.0
                _crowd_delta   = abs(float(_r_crowd["prediction"])  - _base_val) if _r_crowd  else 0.0
                top3_data = [
                    ("Time of day & history", _base_val,     "#0098D4"),
                    ("Weather conditions",    _weather_delta, "#E86B00"),
                    ("Crowding level",        _crowd_delta,   "#00782A"),
                ]
                # Flag when both contributions are near zero after proper isolation.
                if _weather_delta < 0.01 and _crowd_delta < 0.01:
                    st.caption(
                        "Weather and crowding are contributing very little — "
                        "the prediction is dominated by time-of-day and recent "
                        "delay history. Try Heavy Rain or a high Crowding Index "
                        "to see their impact."
                    )

        if not top3_data:
            # Fallback: extract real importances from the loaded model.
            best_model = models.get("best_model")
            if best_model is not None:
                try:
                    final_est = best_model.steps[-1][1]
                    fi_arr = final_est.feature_importances_
                    fn_arr = best_model[:-1].get_feature_names_out()
                    _PLAIN = {
                        "num__hour": "Time of day", "num__hour_sin": "Time of day",
                        "num__hour_cos": "Time of day", "num__peak_time": "Time of day",
                        "num__day_of_week": "Day of week",
                        "num__rolling_mean_delay_3": "Recent delay history",
                        "num__rolling_mean_delay_12": "Recent delay history",
                        "num__rolling_std_delay_12": "Recent delay history",
                        "num__lag_delay_1": "Recent delay history",
                        "num__lag_delay_3": "Recent delay history",
                        "num__recent_disruption_rate": "Recent delay history",
                        "num__temp_c": "Weather conditions",
                        "num__precipitation_mm": "Weather conditions",
                        "num__humidity": "Weather conditions",
                        "num__temp_delta_1h": "Weather conditions",
                        "num__precipitation_x_temp": "Weather conditions",
                        "num__crowding_index": "Crowding level",
                        "num__crowding_x_peak": "Crowding level",
                        "num__network_avg_delay": "Network conditions",
                        "num__network_delay_volatility": "Network conditions",
                    }
                    group_imp: Dict[str, float] = {}
                    for fname, fimp in zip(fn_arr, fi_arr):
                        group = _PLAIN.get(fname, fname.replace("num__","").replace("_"," ").title())
                        group_imp[group] = group_imp.get(group, 0.0) + float(fimp)
                    top3_sorted = sorted(group_imp.items(), key=lambda x: -x[1])[:3]
                    _cols = ["#0098D4", "#E86B00", "#00782A"]
                    top3_data = [(n, v, _cols[i]) for i, (n, v) in enumerate(top3_sorted)]
                except Exception as exc:
                    logger.warning("Feature importance fallback failed: %s", exc)

        if not top3_data:
            top3_data = [
                ("Time of day",        0.40, "#0098D4"),
                ("Recent delay",       0.35, "#00782A"),
                ("Weather conditions", 0.20, "#E86B00"),
            ]

        max_v = max(v for _, v, _ in top3_data) or 1.0
        for name, val, col in top3_data:
            pct     = val / max_v * 100
            # Show the value as a "+X.XX sev" delta for the first bar (baseline)
            # and as "±X.XX sev" for the weather/crowding contribution bars.
            val_label = f"{val:.2f} sev"
            st.markdown(f"""
            <div style="margin-bottom:.5rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:.18rem;">
                <span style="font-size:.82rem;font-weight:600;
                             color:{t['font']};">{name}</span>
                <span style="font-size:.75rem;color:{t['muted']};">{val_label}</span>
              </div>
              <div style="background:{t['grid']};border-radius:4px;height:7px;">
                <div style="background:{col};width:{pct:.0f}%;height:7px;
                            border-radius:4px;transition:width .4s;"></div>
              </div>
            </div>""", unsafe_allow_html=True)


    # ── 24-hour day forecast sparkline ───────────────────────────────────────
    st.markdown(f"### 24-Hour Forecast — {line} Line")
    forecast_date = _resolve_scenario_date(dow)
    forecast_date_iso = forecast_date.isoformat()
    with st.spinner("Computing scenario forecast…"):
        live_24h = _predict_24h_scenario(
            artifact_dir, line, forecast_date_iso,
            temp_c, precip_mm, humidity, crowding_index,
        )

    used_live = sum(1 for v in live_24h["pred"] if v is not None)
    day_data: List[float] = []
    ci_lo_data: List[float] = []
    ci_hi_data: List[float] = []
    for h, live_val in enumerate(live_24h["pred"]):
        if live_val is not None:
            day_data.append(float(live_val))
            ci_lo_data.append(float(live_24h["ci_lo"][h]))
            ci_hi_data.append(float(live_24h["ci_hi"][h]))
        else:
            fb = test_preds[
                (test_preds["line"] == line) &
                (test_preds["timestamp"].dt.hour == h) &
                (test_preds["timestamp"].dt.dayofweek == dow)
            ][model_col].mean()
            fb_val = float(fb) if pd.notna(fb) else 0.0
            day_data.append(fb_val)
            ci_lo_data.append(max(0.0, fb_val - mae))
            ci_hi_data.append(fb_val + mae)

    compare_day_data: Optional[List[float]] = None
    compare_lo: Optional[List[float]] = None
    compare_hi: Optional[List[float]] = None
    if sb["compare_enabled"]:
        with st.spinner("Computing Scenario B forecast…"):
            live_24h_b = _predict_24h_scenario(
                artifact_dir, line, forecast_date_iso,
                sb["compare_temp_c"], sb["compare_precip_mm"],
                sb["compare_humidity"], sb["compare_crowding"],
            )
        compare_day_data = []
        compare_lo = []
        compare_hi = []
        for h, live_val in enumerate(live_24h_b["pred"]):
            if live_val is not None:
                compare_day_data.append(float(live_val))
                compare_lo.append(float(live_24h_b["ci_lo"][h]))
                compare_hi.append(float(live_24h_b["ci_hi"][h]))
            else:
                fb = test_preds[
                    (test_preds["line"] == line) &
                    (test_preds["timestamp"].dt.hour == h) &
                    (test_preds["timestamp"].dt.dayofweek == dow)
                ][model_col].mean()
                fb_val = float(fb) if pd.notna(fb) else 0.0
                compare_day_data.append(fb_val)
                compare_lo.append(max(0.0, fb_val - mae))
                compare_hi.append(fb_val + mae)

    spark = _scenario_forecast_chart(
        list(range(24)), day_data, ci_lo_data, ci_hi_data, line, t,
        values_b=compare_day_data, ci_lo_b=compare_lo, ci_hi_b=compare_hi,
        label_a="Scenario A", label_b="Scenario B",
    )
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

    if sb["compare_enabled"] and compare_day_data is not None:
        ca, cb = st.columns(2)
        with ca:
            st.metric(
                "Scenario A Peak Severity",
                f"{max(day_data):.2f} sev",
                delta=f"≈ {_fmt_min(max(day_data) * SEV_TO_MIN)}",
            )
        with cb:
            st.metric(
                "Scenario B Peak Severity",
                f"{max(compare_day_data):.2f} sev",
                delta=f"≈ {_fmt_min(max(compare_day_data) * SEV_TO_MIN)}",
            )
    st.caption(
        f"Scenario inputs: {WEATHER_ICONS.get(sb['weather'], '')} {sb['weather']} "
        f"· {temp_c}°C · {precip_mm} mm · crowding {crowding_index:.2f}."
    )
    if sb["compare_enabled"] and compare_day_data is not None:
        st.caption(
            f"Scenario B: {WEATHER_ICONS.get(sb['compare_weather'], '')} {sb['compare_weather']} "
            f"· {sb['compare_temp_c']}°C · {sb['compare_precip_mm']} mm "
            f"· crowding {sb['compare_crowding']:.2f}."
        )

    export_df = pd.DataFrame({
        "hour": list(range(24)),
        "scenario_a_pred_sev": day_data,
        "scenario_a_ci_lo_sev": ci_lo_data,
        "scenario_a_ci_hi_sev": ci_hi_data,
        "scenario_a_minutes_est": [v * SEV_TO_MIN for v in day_data],
    })
    if compare_day_data is not None and compare_lo is not None and compare_hi is not None:
        export_df["scenario_b_pred_sev"] = compare_day_data
        export_df["scenario_b_ci_lo_sev"] = compare_lo
        export_df["scenario_b_ci_hi_sev"] = compare_hi
        export_df["scenario_b_minutes_est"] = [v * SEV_TO_MIN for v in compare_day_data]
    _pred_csv_bytes = export_df.to_csv(index=False).encode()
    _pred_mb = len(_pred_csv_bytes) / 1_048_576
    if _pred_mb > 10:
        st.warning(f"CSV is {_pred_mb:.1f} MB — download may be slow.")
    st.download_button(
        label=f"Download 24-hour forecast CSV ({_pred_mb:.1f} MB)",
        data=_pred_csv_bytes,
        file_name=f"{line.lower().replace(' & ', '_').replace(' ', '_')}_24h_forecast.csv",
        mime="text/csv",
    )

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
# EVALUATION CHART BUILDERS (walk-forward · drift · ablation)
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_chart(wf: Dict, t: Dict) -> go.Figure:
    """Grouped bar chart of per-fold MAE + improvement % for walk-forward backtest."""
    folds    = wf["folds"]
    summary  = wf["summary"]
    labels   = [f"Fold {r['fold']}" for r in folds]
    lgbm_mae = [r["mae"]       for r in folds]
    naive_mae = [r["naive_mae"] for r in folds]
    imps     = [r["improvement_pct"] for r in folds]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("MAE per Fold vs Naive", "Improvement over Naive (%)"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        name="LightGBM", x=labels, y=lgbm_mae,
        marker_color="#003B6F", offsetgroup="A",
        hovertemplate="<b>%{x}</b><br>LightGBM MAE: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="Naive", x=labels, y=naive_mae,
        marker_color="#DC241F", opacity=0.75, offsetgroup="B",
        hovertemplate="<b>%{x}</b><br>Naive MAE: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    imp_colours = ["#00B140" if v >= 0 else "#DC241F" for v in imps]
    fig.add_trace(go.Bar(
        name="Improvement %", x=labels, y=imps,
        marker_color=imp_colours, showlegend=False,
        hovertemplate="<b>%{x}</b><br>%{y:+.1f}% vs Naive<extra></extra>",
        text=[f"{v:+.1f}%" for v in imps],
        textposition="outside",
        textfont=dict(color=t["font"], size=11),
    ), row=1, col=2)
    fig.add_hline(y=0, line_width=1, line_color=t["grid"], row=1, col=2)

    title = (
        f"Walk-Forward Backtesting — {summary['n_folds']} folds  "
        f"|  Mean MAE: {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}  "
        f"|  Mean R²: {summary['r2_mean']:.4f}  "
        f"|  Mean improvement: {summary['improvement_pct_mean']:+.1f}%"
    )
    fig = _fig_base(fig, t, title, height=400)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return fig


def _drift_chart(drift: Dict, t: Dict) -> go.Figure:
    """Line chart of per-window MAE with trend line for drift analysis."""
    windows = drift["windows"]
    trend   = drift["trend"]
    labels  = [w["window"] for w in windows]
    mae_b   = [w["mae_best"]  for w in windows]
    mae_n   = [w["mae_naive"] for w in windows]
    x       = list(range(len(labels)))

    z        = np.polyfit(x, mae_b, 1)
    trend_y  = [float(np.poly1d(z)(xi)) for xi in x]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=mae_b, mode="lines+markers", name="LightGBM MAE",
        line=dict(color="#003B6F", width=2.5),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>MAE: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=mae_n, mode="lines+markers", name="Naive MAE",
        line=dict(color="#DC241F", width=1.8, dash="dash"),
        marker=dict(size=5), opacity=0.75,
        hovertemplate="<b>%{x}</b><br>Naive MAE: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=trend_y, mode="lines",
        name=f"Trend (slope={z[0]:+.4f})",
        line=dict(color=t["muted"], width=1.5, dash="dot"),
        hoverinfo="skip",
    ))

    drift_tag = "DRIFT DETECTED" if trend["drift_detected"] else "No drift"
    title = (
        f"Model MAE Over Time  |  "
        f"Spearman ρ={trend['spearman_rho']:.3f}  p={trend['p_value']:.4f}  "
        f"|  {drift_tag}"
    )
    fig = _fig_base(fig, t, title, height=360)
    fig.update_layout(
        xaxis_title=f"Time window ({trend['window_type']})",
        yaxis_title="MAE (severity units)",
        xaxis=dict(tickangle=-30),
    )
    return fig


def _ablation_plotly_chart(ablation: Dict, t: Dict) -> go.Figure:
    """Horizontal bar chart of MAE delta per feature group from ablation study."""
    records  = ablation["records"]
    full_mae = ablation["full_mae"]
    df_abl   = pd.DataFrame(records).sort_values("delta", ascending=True)

    bar_colours = ["#DC241F" if d > 0 else "#0098D4" for d in df_abl["delta"]]
    fig = go.Figure(go.Bar(
        x=df_abl["delta"],
        y=df_abl["group"],
        orientation="h",
        marker_color=bar_colours,
        text=[f"{'+'if d>=0 else ''}{d:.3f}" for d in df_abl["delta"]],
        textposition="outside",
        textfont=dict(color=t["font"], size=11),
        customdata=list(zip(df_abl["mae"], df_abl["pct_change"], df_abl["n_removed"])),
        hovertemplate=(
            "<b>Remove %{y}</b><br>"
            "MAE delta: %{x:+.4f}<br>"
            "Ablated MAE: %{customdata[0]:.4f}<br>"
            "% change: %{customdata[1]:+.1f}%<br>"
            "Features removed: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_width=1.5, line_color=t["muted"])
    fig = _fig_base(
        fig, t,
        f"Ablation Study — Feature Group Contribution  |  Full model MAE: {full_mae:.4f}",
        height=350,
    )
    fig.update_layout(
        xaxis_title="MAE delta vs full model (red = group helped; blue = group hurt)",
        yaxis_title="",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PERFORMANCE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def _tab_performance(tabular: Dict, t: Dict) -> None:
    """KPI row · model comparison · feature importance · error analysis · per-line table."""
    metrics    = tabular.get("metrics", {})
    test_preds = tabular.get("test_predictions")
    feat_imp   = tabular.get("feature_importance")
    comp_df    = tabular.get("model_comparison")

    best  = metrics.get("best",  {})
    naive = metrics.get("naive", {})
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
    st.markdown("### Feature Correlation Matrix")
    if test_preds is not None:
        corr_fig = _feature_correlation_chart(test_preds, feat_imp, t)
        if corr_fig is not None:
            st.plotly_chart(corr_fig, width='stretch', key="perf_feature_correlation")
            st.caption("Pearson correlations across the most relevant numeric columns available in the evaluation set.")
        else:
            st.info("Not enough numeric feature columns were available to build a correlation matrix.")

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

    # ── Load outputs from analysis/ scripts ───────────────────────────────────
    analysis = _load_analysis_outputs(str(ROOT / "analysis" / "outputs"))

    # ── Walk-forward backtesting ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Walk-Forward Backtesting")
    st.caption(
        "The model is retrained from scratch on all data *before* each evaluation window "
        "and tested on the next unseen window — proving generalisation across multiple future "
        "periods rather than just the final 20%."
    )
    if "walk_forward" in analysis:
        wf = analysis["walk_forward"]
        summary = wf["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean MAE", f"{summary['mae_mean']:.4f}",
                  delta=f"±{summary['mae_std']:.4f} std")
        c2.metric("MAE Range",
                  f"{summary['mae_min']:.4f} – {summary['mae_max']:.4f}")
        c3.metric("Mean R²", f"{summary['r2_mean']:.4f}")
        c4.metric("Mean improvement vs Naive",
                  f"{summary['improvement_pct_mean']:+.1f}%",
                  delta_color="inverse" if summary["improvement_pct_mean"] < 0 else "normal")
        st.plotly_chart(
            _walk_forward_chart(wf, t),
            width='stretch', key="perf_walk_forward",
        )
        with st.expander("Per-fold detail", expanded=False):
            fold_rows = [
                {
                    "Fold":        r["fold"],
                    "Test window": f"{r['test_start'][:10]} → {r['test_end'][:10]}",
                    "Train rows":  r["train_rows"],
                    "Test rows":   r["test_rows"],
                    "MAE":         r["mae"],
                    "RMSE":        r["rmse"],
                    "R²":          r["r2"],
                    "vs Naive":    f"{r['improvement_pct']:+.1f}%",
                }
                for r in wf["folds"]
            ]
            st.dataframe(pd.DataFrame(fold_rows), hide_index=True, width='stretch')
    else:
        st.info(
            "No walk-forward backtest results found. "
            "Generate them with: `python analysis/walk_forward_backtest.py`"
        )

    # ── Drift analysis ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Drift Analysis")
    st.caption(
        "MAE is grouped into time windows over the held-out test set. "
        "A rising trend indicates distributional drift away from the training distribution."
    )
    if "drift" in analysis:
        drift = analysis["drift"]
        trend = drift["trend"]
        verdict_colour = "#DC241F" if trend["drift_detected"] else "#00B140"
        verdict_label  = "Drift detected" if trend["drift_detected"] else "No drift"
        st.markdown(
            f'<div style="border-left:4px solid {verdict_colour};'
            f'padding:0.5rem 0.9rem;border-radius:6px;margin-bottom:0.8rem;">'
            f'<strong style="color:{verdict_colour};">{verdict_label}</strong> — '
            f'{trend["interpretation"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Spearman ρ", f"{trend['spearman_rho']:+.4f}")
        d2.metric("p-value", f"{trend['p_value']:.4f}")
        d3.metric("Relative MAE increase",
                  f"{trend['relative_increase']*100:+.1f}%")
        st.plotly_chart(
            _drift_chart(drift, t),
            width='stretch', key="perf_drift",
        )
    else:
        st.info(
            "No drift analysis results found. "
            "Generate them with: `python analysis/drift_analysis.py`"
        )

    # ── Ablation study ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Ablation Study — Feature Group Contributions")
    st.caption(
        "Each feature group is removed in turn and the model is retrained. "
        "A positive MAE delta means that group was helpful; negative means removing it "
        "slightly improved performance (redundant or noisy features)."
    )
    if "ablation" in analysis:
        abl = analysis["ablation"]
        a1, a2, a3 = st.columns(3)
        a1.metric("Full model MAE", f"{abl['full_mae']:.4f}")
        a2.metric("Naive baseline MAE", f"{abl['naive_mae']:.4f}")
        gain = (1 - abl["full_mae"] / abl["naive_mae"]) * 100
        a3.metric("Full model improvement", f"{gain:.1f}%", delta_color="inverse")
        st.plotly_chart(
            _ablation_plotly_chart(abl, t),
            width='stretch', key="perf_ablation",
        )
        with st.expander("Ablation detail table", expanded=False):
            abl_rows = [
                {
                    "Feature group":   r["group"],
                    "MAE w/o group":   r["mae"],
                    "Delta vs full":   f"{'+'if r['delta']>=0 else''}{r['delta']:.4f}",
                    "% change":        f"{'+'if r['pct_change']>=0 else''}{r['pct_change']:.1f}%",
                    "Features removed": r["n_removed"],
                }
                for r in sorted(abl["records"], key=lambda x: x["delta"], reverse=True)
            ]
            st.dataframe(pd.DataFrame(abl_rows), hide_index=True, width='stretch')
    elif "ablation_png" in analysis:
        st.image(analysis["ablation_png"],
                 caption="Ablation study — MAE delta per feature group (run analysis/ablation_study.py to generate interactive chart)")
    else:
        st.info(
            "No ablation results found. "
            "Generate them with: `python analysis/ablation_study.py`"
        )

    # ── ARIMA vs LightGBM comparison ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ARIMA vs LightGBM — Per-Line Model Comparison")
    st.caption(
        "A univariate SARIMA(1,1,1)(1,1,1,96) model — using only the delay "
        "time series with no exogenous features — is compared against LightGBM "
        "per line. Lines where ARIMA wins indicate that autocorrelation alone "
        "captures most of the signal; additional features add noise at the "
        "current data volume."
    )
    if "arima" in analysis:
        arima_data = analysis["arima"]
        arima_records = arima_data.get("records", [])
        if arima_records:
            arima_df = pd.DataFrame(arima_records).dropna(subset=["lgbm_mae"])
            arima_wins  = arima_data.get("arima_wins_lines", [])
            lgbm_wins   = arima_data.get("lgbm_wins_lines", [])

            a1, a2 = st.columns(2)
            a1.metric("Lines where ARIMA wins", len(arima_wins))
            a2.metric("Lines where LightGBM wins", len(lgbm_wins))

            fig_arima = go.Figure()
            fig_arima.add_trace(go.Bar(
                name="ARIMA/SARIMA",
                x=arima_df["line"],
                y=arima_df["arima_mae"],
                marker_color="#ff7f0e",
            ))
            fig_arima.add_trace(go.Bar(
                name="LightGBM",
                x=arima_df["line"],
                y=arima_df["lgbm_mae"],
                marker_color="#1f77b4",
            ))
            fig_arima = _fig_base(fig_arima, t, "ARIMA vs LightGBM MAE per Line", height=380)
            fig_arima.update_layout(
                barmode="group",
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(gridcolor=t["grid"]),
                yaxis=dict(title="MAE (severity units)", gridcolor=t["grid"]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_arima, width='stretch', key="perf_arima_comparison")

            with st.expander("ARIMA comparison detail", expanded=False):
                tbl = arima_df[["line", "arima_mae", "lgbm_mae", "winner", "model_used"]].copy()
                tbl.columns = ["Line", "ARIMA MAE", "LightGBM MAE", "Winner", "ARIMA model"]
                st.dataframe(tbl.style.apply(
                    lambda row: ["background-color: rgba(255,127,14,0.15)" if row["Winner"] == "ARIMA"
                                 else "background-color: rgba(31,119,180,0.15)"] * len(row), axis=1
                ), hide_index=True, width='stretch')

            st.caption(
                "**Interpretation:** ARIMA outperforms LightGBM on lines with stable, "
                "autocorrelated delay patterns. LightGBM's advantage on high-variance "
                "lines (e.g. Victoria, Metropolitan) comes from its ability to handle "
                "occasional large spikes using cross-line network features. "
                "See `analysis/line_adaptive_ensemble.py` for the combined model."
            )
    else:
        st.info(
            "No ARIMA comparison results found. "
            "Generate them with: `python analysis/arima_baseline.py`"
        )

    # ── Line-adaptive ensemble results ────────────────────────────────────────
    if "ensemble" in analysis:
        ens = analysis["ensemble"]
        st.markdown("#### Line-Adaptive Ensemble")
        st.caption(
            "Combines ARIMA (for stable lines) and LightGBM (for high-variance lines). "
            "Winner selected on validation set — test set never seen during selection."
        )
        e1, e2, e3 = st.columns(3)
        overall = ens.get("overall_mae", {})
        e1.metric("Ensemble test MAE", f"{overall.get('ensemble', 0):.4f}",
                  delta=f"{ens.get('ensemble_vs_lgbm_pct', 0):+.1f}% vs LightGBM",
                  delta_color="inverse")
        e2.metric("LightGBM only", f"{overall.get('lgbm', 0):.4f}")
        e3.metric("ARIMA only",    f"{overall.get('arima', 0):.4f}")

        sel = ens.get("selection", {})
        if sel:
            arima_lines = [l for l, m in sel.items() if m == "ARIMA"]
            lgbm_lines  = [l for l, m in sel.items() if m == "LightGBM"]
            st.caption(
                f"**ARIMA routes:** {', '.join(arima_lines) or 'none'}   |   "
                f"**LightGBM routes:** {', '.join(lgbm_lines) or 'none'}"
            )

    # ── Feature selection (overfitting analysis) ──────────────────────────────
    st.markdown("---")
    st.markdown("### SHAP Feature Selection — Overfitting Analysis")
    st.caption(
        "Retrains LightGBM with decreasing feature counts (ranked by SHAP importance) "
        "to test whether fewer features narrows the train/val overfitting gap."
    )
    if "shap_selection" in analysis:
        sel_data = analysis["shap_selection"]
        sel_results = sel_data.get("results", [])
        if sel_results:
            labels      = [r["label"]       for r in sel_results]
            train_maes  = [r["train_mae"]   for r in sel_results]
            val_maes    = [r["cv_val_mae"]  for r in sel_results]
            test_maes   = [r["test_mae"]    for r in sel_results]

            fig_sel = go.Figure()
            fig_sel.add_trace(go.Bar(name="Train MAE",  x=labels, y=train_maes,  marker_color="#1f77b4"))
            fig_sel.add_trace(go.Bar(name="CV-Val MAE", x=labels, y=val_maes,    marker_color="#ff7f0e"))
            fig_sel.add_trace(go.Bar(name="Test MAE",   x=labels, y=test_maes,   marker_color="#2ca02c"))
            fig_sel = _fig_base(fig_sel, t, "Train / CV-Val / Test MAE by Feature Count", height=370)
            fig_sel.update_layout(
                barmode="group",
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(gridcolor=t["grid"]),
                yaxis=dict(title="MAE (severity units)", gridcolor=t["grid"]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sel, width='stretch', key="perf_shap_selection")

            conclusion = sel_data.get("conclusion", "")
            if conclusion:
                st.info(conclusion)

            with st.expander("Feature selection detail", expanded=False):
                tbl_rows = [
                    {
                        "Model":          r["label"],
                        "Features":       r["n_features"],
                        "Train MAE":      r["train_mae"],
                        "CV-Val MAE":     r["cv_val_mae"],
                        "Test MAE":       r["test_mae"],
                        "Overfit gap":    r["overfit_gap"],
                    }
                    for r in sel_results
                ]
                st.dataframe(pd.DataFrame(tbl_rows), hide_index=True, width='stretch')
    else:
        st.info(
            "No feature selection results found. "
            "Generate them with: `python analysis/shap_feature_selection.py`"
        )

    # ── Equity analysis ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Equity Analysis — Prediction Error vs Community Deprivation")
    st.caption(
        "Do lines serving more deprived communities have higher prediction error? "
        "Each line is assigned a catchment-area IMD score (English Indices of "
        "Deprivation 2019, weighted by number of stations per borough). "
        "Spearman rank correlation tests whether error and deprivation are related."
    )
    if "equity" in analysis:
        eq = analysis["equity"]
        eq_records = eq.get("records", [])
        rho   = eq.get("spearman_rho", 0)
        p_val = eq.get("p_value", 1)
        sig   = eq.get("significant", False)

        eq1, eq2, eq3 = st.columns(3)
        eq1.metric("Spearman ρ", f"{rho:+.4f}")
        eq2.metric("p-value", f"{p_val:.4f}")
        eq3.metric("Significant (p < 0.05)", "Yes" if sig else "No",
                   delta_color="off")

        verdict_colour = "#DC241F" if (sig and rho > 0.3) else "#00B140"
        st.markdown(
            f'<div style="border-left:4px solid {verdict_colour}; '
            f'padding:0.5rem 0.9rem; border-radius:6px; margin-bottom:0.8rem;">'
            f'{eq.get("interpretation", "")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if eq_records:
            eq_df = pd.DataFrame(eq_records)

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_df["imd_score"], y=eq_df["mae"],
                mode="markers+text",
                text=eq_df["line"],
                textposition="top right",
                marker=dict(
                    size=12,
                    color=eq_df["mae"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="MAE", thickness=12),
                    line=dict(color="white", width=1),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "IMD score: %{x:.1f}<br>"
                    "MAE: %{y:.4f}<extra></extra>"
                ),
            ))

            # Trend line
            if len(eq_df) >= 3:
                z = np.polyfit(eq_df["imd_score"], eq_df["mae"], 1)
                x_rng = np.linspace(eq_df["imd_score"].min(), eq_df["imd_score"].max(), 50)
                fig_eq.add_trace(go.Scatter(
                    x=x_rng, y=np.polyval(z, x_rng),
                    mode="lines",
                    name="Trend",
                    line=dict(color="rgba(0,0,0,0.3)", dash="dash", width=1.5),
                    showlegend=True,
                ))

            fig_eq = _fig_base(
                fig_eq, t,
                f"MAE vs Community Deprivation (Spearman ρ = {rho:+.3f}, p = {p_val:.3f})",
                height=420,
            )
            fig_eq.update_layout(
                margin=dict(l=20, r=20, t=60, b=20),
                xaxis=dict(
                    title="Catchment-area IMD score (% most deprived LSOAs, higher = more deprived)",
                    gridcolor=t["grid"],
                ),
                yaxis=dict(title="Test MAE (severity units)", gridcolor=t["grid"]),
                showlegend=True,
            )
            st.plotly_chart(fig_eq, width='stretch', key="perf_equity_scatter")

            with st.expander("IMD scores by line", expanded=False):
                tbl_eq = eq_df[["line", "imd_score", "mae", "complexity"]].copy()
                tbl_eq.columns = ["Line", "IMD score", "MAE", "Route complexity"]
                st.dataframe(tbl_eq.style.background_gradient(
                    subset=["IMD score"], cmap="RdYlGn_r"
                ), hide_index=True, width='stretch')

            st.caption(
                f"IMD source: English Indices of Deprivation 2019 (MHCLG). "
                f"Scores weighted by number of stations in each borough. "
                f"n = {eq.get('n_lines', len(eq_df))} lines."
            )
    else:
        st.info(
            "No equity analysis results found. "
            "Generate them with: `python analysis/equity_analysis.py`"
        )


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
    _par_clrs = [LINE_COLOURS.get(ln, "#003B6F") for ln in fdf["Line"]]
    _n = max(len(fdf) - 1, 1)
    _par_scale = [[i / _n, c] for i, c in enumerate(_par_clrs)]
    # Plotly requires at least two colorscale stops (at 0 and 1).
    if len(_par_scale) == 1:
        _par_scale = [[0.0, _par_clrs[0]], [1.0, _par_clrs[0]]]
    fig_par = go.Figure(go.Parcoords(
        line=dict(
            color=list(range(len(fdf))),
            colorscale=_par_scale,
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
    _csv_bytes = df.to_csv(index=False).encode()
    _size_mb = len(_csv_bytes) / 1_048_576
    if _size_mb > 10:
        st.warning(
            f"This export is {_size_mb:.1f} MB ({len(df):,} rows). "
            "Downloading may be slow on a poor connection."
        )
    st.download_button(
        label=f"Download tfl_merged.csv ({_size_mb:.1f} MB)",
        data=_csv_bytes,
        file_name=f"tfl_merged_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — RISK MAP
# ─────────────────────────────────────────────────────────────────────────────

_RISK_THRESHOLDS: Dict[str, float] = {"Low": STATUS_GOOD_MAX, "Medium": STATUS_MINOR_MAX}
_RISK_COLOURS:    Dict[str, str]   = {
    "Low":    "#00782A",
    "Medium": "#E86B00",
    "High":   "#D41515",
}
_RISK_BG: Dict[str, str] = {
    "Low":    "#e8f5e9",
    "Medium": "#fff3e0",
    "High":   "#fdecea",
}


def _schematic_risk_map(risk_lookup: Dict[str, str], t: Dict) -> go.Figure:
    """
    Draw the London Underground schematic (Beck-style) with each line segment
    coloured by its risk band (Low=green / Medium=amber / High=red).

    Coordinates come from app/map_data.LINE_PATHS (x 0-100, y 0-70 grid).
    The map is drawn as a Plotly Scatter figure — no external mapping library needed.
    """
    fig = go.Figure()

    # ── Line traces ──────────────────────────────────────────────────────────
    for line_name, path in LINE_PATHS.items():
        risk   = risk_lookup.get(line_name, "Low")
        colour = _RISK_COLOURS[risk]
        xs     = [p[0] for p in path]
        ys     = [p[1] for p in path]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=colour, width=4),
            name=line_name,
            hovertemplate=f"<b>{line_name}</b><br>Risk: {risk}<extra></extra>",
            showlegend=False,
        ))

    # ── Interchange station dots ─────────────────────────────────────────────
    sx = [s[0] for s in INTERCHANGE_STATIONS]
    sy = [s[1] for s in INTERCHANGE_STATIONS]
    sl = [s[2] for s in INTERCHANGE_STATIONS]
    dot_colour = "#FFFFFF" if t["dark"] else "#333333"
    fig.add_trace(go.Scatter(
        x=sx, y=sy,
        mode="markers+text",
        marker=dict(size=7, color=dot_colour,
                    line=dict(color=t["paper"], width=1.5)),
        text=sl,
        textposition="top center",
        textfont=dict(size=8, color=t["muted"]),
        hoverinfo="text",
        hovertext=sl,
        showlegend=False,
        name="",
    ))

    # ── Risk-level legend entries ────────────────────────────────────────────
    for level, col in _RISK_COLOURS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=col, width=4),
            name=f"{level} Risk",
            showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor=t["paper"],
        plot_bgcolor=t["plot"],
        font=dict(color=t["font"],
                  family="'Inter','Segoe UI',sans-serif"),
        xaxis=dict(visible=False, range=[-2, 104]),
        yaxis=dict(visible=False, range=[-2, 73], scaleanchor="x", scaleratio=1),
        margin=dict(l=8, r=8, t=36, b=8),
        height=520,
        legend=dict(
            orientation="h", x=0.02, y=1.04,
            bgcolor="rgba(0,0,0,0.12)" if t["dark"] else "rgba(255,255,255,0.85)",
            bordercolor=t["grid"], borderwidth=1,
            font=dict(size=11, color=t["font"]),
        ),
        title=dict(
            text="Network Risk — colour shows predicted delay risk per line",
            font=dict(size=13, color=t["font"]), x=0.01,
        ),
    )
    return fig


def _risk_level(severity: float) -> str:
    """Map a predicted severity value (0–2 scale) to a risk band label."""
    if severity < _RISK_THRESHOLDS["Low"]:
        return "Low"
    if severity < _RISK_THRESHOLDS["Medium"]:
        return "Medium"
    return "High"


def _tab_risk(tabular: Dict, t: Dict) -> None:
    """
    Per-line risk visualisation derived from the held-out test set.

    Risk bands (severity 0–2 scale):
      Low    — median predicted severity < 0.5  (Good Service expected)
      Medium — 0.5 – 1.5  (Minor Delays expected)
      High   — ≥ 1.5  (Severe Delays expected)

    Shows a KPI summary row, a colour-coded horizontal bar chart, and
    per-line detail cards with median/P95 severity and test-set MAE.
    """
    test_preds = tabular.get("test_predictions")
    if test_preds is None or test_preds.empty:
        st.warning(
            "No prediction data available. "
            "Run `python train.py` then `python explain.py` first."
        )
        return

    st.markdown("### Network Risk Map")
    st.caption(
        "Risk level per line is derived from the **median predicted severity** "
        "on the held-out 20 % test set (strict chronological split). "
        "Severity is the real TfL label — 0 = Good Service · 1 = Minor Delays · 2 = Severe Delays. "
        f"Thresholds: **Low** < {STATUS_GOOD_MAX} · **Medium** {STATUS_GOOD_MAX}–{STATUS_MINOR_MAX} · **High** ≥ {STATUS_MINOR_MAX}."
    )

    # ── Per-line statistics ───────────────────────────────────────────────────
    records = []
    for line, grp in test_preds.groupby("line"):
        median_pred = float(grp["pred_best"].median())
        p95_pred    = float(grp["pred_best"].quantile(0.95))
        mae         = float((grp["actual"] - grp["pred_best"]).abs().mean())
        records.append({
            "line":        line,
            "median_pred": median_pred,
            "p95_pred":    p95_pred,
            "mae":         mae,
            "risk":        _risk_level(median_pred),
        })

    risk_df = (
        pd.DataFrame(records)
        .sort_values("median_pred", ascending=False)
        .reset_index(drop=True)
    )

    # ── KPI summary row ───────────────────────────────────────────────────────
    counts = risk_df["risk"].value_counts()
    k1, k2, k3 = st.columns(3)
    bg  = t["card_bg"]
    bdr = t["card_border"]
    txt = t["font"]
    sub = t["muted"]

    for col, level in zip([k1, k2, k3], ["High", "Medium", "Low"]):
        n      = int(counts.get(level, 0))
        colour = _RISK_COLOURS[level]
        col.markdown(
            f'<div class="risk-count-card" style="--rcc-accent:{colour};'
            f'--rcc-bg:{bg};--rcc-border:{bdr};--rcc-sub:{sub};">'
            f'<div class="rcc-label">{level} Risk Lines</div>'
            f'<div class="rcc-value">{n}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Horizontal bar chart ─────────────────────────────────────────────────
    bar_colours = [_RISK_COLOURS[r] for r in risk_df["risk"]]
    fig = go.Figure(go.Bar(
        x=risk_df["median_pred"],
        y=risk_df["line"],
        orientation="h",
        marker_color=bar_colours,
        text=[f"{v:.2f}" for v in risk_df["median_pred"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Median predicted severity: %{x:.2f}<br>"
            "95th percentile: %{customdata[0]:.2f}<br>"
            "Test MAE: %{customdata[1]:.3f}<extra></extra>"
        ),
        customdata=risk_df[["p95_pred", "mae"]].values,
    ))
    fig.add_vline(
        x=_RISK_THRESHOLDS["Low"], line_dash="dot", line_color="#00782A",
        annotation_text="Low / Medium", annotation_position="top right",
    )
    fig.add_vline(
        x=_RISK_THRESHOLDS["Medium"], line_dash="dot", line_color="#D41515",
        annotation_text="Medium / High", annotation_position="top right",
    )
    fig = _fig_base(fig, t, "Median Predicted Severity by Line (test set)", height=440)
    fig.update_layout(
        xaxis_title="Median Predicted Severity (0 = Good · 1 = Minor · 2 = Severe)",
        yaxis_title="",
        margin=dict(l=20, r=90, t=50, b=20),
    )
    st.plotly_chart(fig, width='stretch', key="risk_bar_chart")

    # ── Schematic tube map coloured by risk ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### Network Schematic — Risk Heatmap")
    risk_lookup = dict(zip(risk_df["line"], risk_df["risk"]))
    st.plotly_chart(
        _schematic_risk_map(risk_lookup, t),
        width='stretch',
        config={"displayModeBar": False},
        key="risk_schematic_map",
    )
    st.caption(
        f"Each line is coloured by its risk band: "
        f"🟢 Low (median sev < {STATUS_GOOD_MAX}) · "
        f"🟠 Medium ({STATUS_GOOD_MAX}–{STATUS_MINOR_MAX}) · "
        f"🔴 High (≥ {STATUS_MINOR_MAX}). "
        f"Station dots mark key interchanges."
    )
    st.markdown("---")

    # ── Per-line risk cards ───────────────────────────────────────────────────
    st.markdown("#### Line-by-Line Risk Detail")
    cols_per_row = 3
    rows = [
        risk_df.iloc[i : i + cols_per_row]
        for i in range(0, len(risk_df), cols_per_row)
    ]
    for row_slice in rows:
        cols = st.columns(cols_per_row)
        for col, (_, r) in zip(cols, row_slice.iterrows()):
            colour  = _RISK_COLOURS[r["risk"]]
            card_bg = _RISK_BG[r["risk"]] if not t["dark"] else t["card_bg"]
            lc      = LINE_COLOURS.get(r["line"], "#003B6F")
            col.markdown(
                f'<div class="risk-detail-card" style="--rdc-accent:{colour};'
                f'--rdc-bg:{card_bg};--rdc-line:{lc};--rdc-txt:{txt};--rdc-sub:{sub};">'
                f'<div class="rdc-title">{r["line"]}</div>'
                f'<div class="rdc-row">'
                f'<span class="rdc-risk">{r["risk"]}</span>'
                f'<span class="rdc-sev">med {r["median_pred"]:.2f} sev</span>'
                f'</div>'
                f'<div class="rdc-meta">p95 {r["p95_pred"]:.2f} &nbsp;·&nbsp; MAE {r["mae"]:.3f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.caption(
        "Risk is computed from the best model's predictions on the held-out 20 % "
        "test set (chronological split). This is a retrospective measure for "
        "evaluation purposes, not a real-time operational alert."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────

def _tab_about(tabular: Dict, t: Dict) -> None:
    """
    Visual project showcase: architecture diagram · key stats grid · tech stack
    badges · model card · ethics. I avoided a plain wall of text so the tab
    reads as a professional portfolio piece for the dissertation demo.
    """
    metrics   = tabular.get("metrics", {})
    best_name = tabular.get("best_model_name", "lightgbm").upper()
    best_m    = metrics.get("best", {})
    mae   = best_m.get("test_mae")
    rmse  = best_m.get("test_rmse")
    r2    = best_m.get("test_r2")
    n_mae = metrics.get("naive", {}).get("test_mae")
    _f = lambda v, fmt=".3f", sfx="": (format(v, fmt) + sfx) if v is not None else "N/A"

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
                             ("Test MAE", _f(mae, ".3f", " sev")),
                             ("R²", _f(r2, ".3f")),
                             ("Lines", "11"),
                             ("Features", "40"),
                             ("vs Naive", f"−{(1-mae/n_mae)*100:.0f}%" if mae is not None and n_mae else "N/A")])}
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
                Commit: {_BUILD_COMMIT}<br>Random seed: 42<br>
                Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}<br>LightGBM {__import__('lightgbm').__version__}</div>
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
        r_mae = metrics.get("ridge", {}).get("test_mae", 2.80)
        test_mae_ci = best_m.get("test_mae_ci")
        test_rmse_ci = best_m.get("test_rmse_ci")
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
        st.markdown(f"""
**Additional evaluation detail**
- Test MAE 95% CI: {f"{test_mae_ci[0]:.3f}-{test_mae_ci[1]:.3f} sev" if test_mae_ci else "Unavailable"}
- Test RMSE 95% CI: {f"{test_rmse_ci[0]:.3f}-{test_rmse_ci[1]:.3f} sev" if test_rmse_ci else "Unavailable"}
- CV fold-level MAE: not persisted in the current artifact

**Feature groups**
- Temporal: 12
- Weather: 5
- Historical / lag / rolling: 6
- Network / crowding: 6
- Event / seasonal: 3
- Line / topology / service pattern: 8

**Limitations**
- `delay_minutes` is synthetic and only used for UI interpretation.
- Predictions are line-level, not station-level.
- Signal faults, engineering works, staffing issues, and unplanned incidents are only partially observed.
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
synthetic proxy used for interpretation only; the model target is `delay_severity`.
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

    # If the user re-trained (artifact_dir changed), stale scenario_a/b session
    # state from the old run would produce mismatched predictions. Clear it.
    if st.session_state.get("_artifact_dir") != artifact_dir:
        for key in list(st.session_state.keys()):
            if key.startswith("scenario_"):
                del st.session_state[key]
        st.session_state["_artifact_dir"] = artifact_dir

    with st.spinner("Loading model artifacts…"):
        models  = _load_models(artifact_dir)
        tabular = _load_tabular(artifact_dir)

    if not models:
        _render_header()
        st.error(
            "Failed to load model from `artifacts/`. "
            "Ensure `python train.py` completed successfully.")
        st.stop()

    if "metrics" not in tabular:
        _render_header()
        st.error(
            "`all_metrics.json` not found in `artifacts/`. "
            "Re-run `python train.py` to regenerate metrics.")
        st.stop()

    # Single _theme() call — palette passed into every chart builder
    t = _theme()

    _render_header()
    _enable_live_clock()   # clock only — no auto-reload
    _render_live_status_strip(t)

    metrics   = tabular["metrics"]
    best_name = tabular.get("best_model_name", "lightgbm")
    sb = _render_sidebar(metrics, best_name)

    tab_labels = [
        "🔮 Predictions",
        "📊 Performance",
        "🚇 Line Comparison",
        "📈 Historical Trends",
        "💾 Data Collection",
        "⚠️ Risk Map",
        "ℹ️ About",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _tab_predictions(models, tabular, sb, t, artifact_dir)

    with tabs[1]:
        _tab_performance(tabular, t)

    with tabs[2]:
        _tab_line_comparison(tabular, t)

    with tabs[3]:
        _tab_historical(tabular, t)

    with tabs[4]:
        _tab_data_collection(t)

    with tabs[5]:
        _tab_risk(tabular, t)

    with tabs[6]:
        _tab_about(tabular, t)

    st.markdown(
        f'<div class="dash-footer">'
        f'London Underground Delay Predictor &nbsp;·&nbsp; '
        f'COMP1682 Dissertation &nbsp;·&nbsp; University of Greenwich &nbsp;·&nbsp; '
        f'Built with Streamlit &amp; Plotly &nbsp;·&nbsp; '
        f'Run: {latest_run} &nbsp;·&nbsp; '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;·&nbsp; '
        f'{_build_info()}'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
