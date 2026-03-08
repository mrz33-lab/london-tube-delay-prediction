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
    )
    return fig


def create_forecast_chart(
    predictions: pd.DataFrame,
    selected_line: str,
    model_col: str,
    dark: bool = False,
) -> go.Figure:
    """24-hour forecast chart with Â±1 MAE confidence band."""
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
        name="Â±1 MAE band",
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

    fig = _plotly_layout(fig, title=f"Delay Forecast â€” {selected_line} Line", dark=dark)
    fig.update_layout(
        height=360,
        hovermode="x unified",
        xaxis=dict(title="Time", tickformat="%H:%M"),
        yaxis=dict(title="Delay (minutes)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_line_heatmap(predictions: pd.DataFrame, model_col: str, dark: bool = False) -> go.Figure:
    """Hour-of-day Ã— tube-line heatmap of average predicted delay."""
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
    """Grouped bar chart comparing MAE, RMSE, RÂ² across models."""
    model_names = [k for k in ("naive", "ridge", "best") if k in metrics]
    display_names = {"naive": "Naive Baseline", "ridge": "Ridge Regression", "best": "Best Model"}
    bar_colours   = {"naive": "#6c757d",         "ridge": "#0098D4",           "best": "#003688"}

    mae_vals  = [metrics[m].get("test_mae",  0) for m in model_names]
    rmse_vals = [metrics[m].get("test_rmse", 0) for m in model_names]
    r2_vals   = [metrics[m].get("test_r2",   0) for m in model_names]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Mean Absolute Error (â†“ better)", "RMSE (â†“ better)", "RÂ² Score (â†‘ better)"),
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
                             hovertemplate=f"RÂ²: %{{y:.4f}}<extra></extra>",
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


def create_confusion_matrix_chart(predictions: pd.DataFrame, model_col: str, dark: bool = False) -> go.Figure:
    """Confusion matrix comparing true vs predicted TfL service statuses."""
    from sklearn.metrics import confusion_matrix

    labels = ["Good Service", "Minor Delays", "Moderate Delays", "Severe Delays"]
    
    y_true = predictions["actual"].apply(delay_to_status)
    y_pred = predictions[model_col].apply(delay_to_status)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate row percentages
    cm_pct = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(len(labels)):
        if row_sums[i] > 0:
            cm_pct[i] = (cm[i] / row_sums[i]) * 100
            
    # Prepare text annotations
    text_annotations = []
    for i in range(len(labels)):
        row_text = []
        for j in range(len(labels)):
            count = cm[i][j]
            pct = cm_pct[i][j]
            row_text.append(f"{count}<br>({pct:.1f}%)")
        text_annotations.append(row_text)

    paper_bg = "#0d1117" if dark else "#ffffff"
    font_col = "#e6edf3" if dark else "#1a1a2e"
    
    # Use a custom TfL-branded Blues scale to match dashboard styling
    colorscale = [[0.0, paper_bg], [1.0, "#003688"]]

    fig = go.Figure(data=go.Heatmap(
        z=cm_pct,
        x=labels,
        y=labels,
        text=text_annotations,
        texttemplate="%{text}",
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title="% of Actual"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>"
    ))
    
    fig = _plotly_layout(fig, title="Service Status Confusion Matrix", dark=dark)
    
    fig.update_layout(
        xaxis_title="Predicted Status",
        yaxis_title="Actual Status",
        yaxis=dict(autorange="reversed"),
        height=450,
        margin=dict(l=120, r=20, t=60, b=60),
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
