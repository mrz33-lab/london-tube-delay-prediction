"""
Plotly chart builders for the TfL dashboard.

Each function returns a go.Figure ready for st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict

from app.constants import LINE_COLOURS, STATUS_COLOURS


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


def delay_to_status(delay_minutes: float) -> str:
    """Map delay minutes to TfL service status."""
    if delay_minutes < 2:
        return "Good Service"
    elif delay_minutes < 5:
        return "Minor Delays"
    elif delay_minutes < 10:
        return "Moderate Delays"
    else:
        return "Severe Delays"


def create_gauge_chart(delay_minutes: float, dark: bool = False) -> go.Figure:
    """Dial gauge showing predicted delay severity with TfL colour thresholds."""
    label = delay_to_status(delay_minutes)
    colour = STATUS_COLOURS[label]

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
