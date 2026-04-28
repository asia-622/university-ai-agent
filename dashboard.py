"""
dashboard.py — All Plotly chart builders used in the Streamlit UI.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
BG      = "rgba(0,0,0,0)"      # transparent
PAPER   = "rgba(17,24,39,0)"   # transparent (dark card bg handled by CSS)
FONT    = dict(family="IBM Plex Sans, sans-serif", color="#e2e8f0")


def _base_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=FONT,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
    )
    base.update(kwargs)
    return base


# ── 1. Marks comparison bar chart ─────────────────────────────────────────────
def marks_bar_chart(meta: dict) -> go.Figure:
    df = meta["df"]
    scols = meta["subject_cols"]
    if not scols:
        return _empty_fig("No subject columns detected")

    avgs = df[scols].mean().reset_index()
    avgs.columns = ["Subject", "Average Marks"]

    fig = px.bar(
        avgs, x="Subject", y="Average Marks",
        color="Subject", color_discrete_sequence=PALETTE,
        text_auto=".1f",
        title="Average Marks per Subject",
    )
    fig.update_traces(marker_line_width=0, textfont_size=11)
    fig.update_layout(**_base_layout(
        xaxis=dict(showgrid=False, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", color="#94a3b8"),
    ))
    return fig


# ── 2. Department distribution pie ────────────────────────────────────────────
def department_pie(meta: dict) -> go.Figure:
    df = meta["df"]
    dept_col = meta.get("dept_col")
    if not dept_col:
        return _empty_fig("No department column detected")

    counts = df[dept_col].value_counts().reset_index()
    counts.columns = ["Department", "Count"]

    fig = px.pie(
        counts, names="Department", values="Count",
        color_discrete_sequence=PALETTE,
        title="Students per Department",
        hole=0.4,
    )
    fig.update_traces(textposition="outside", textinfo="label+percent")
    fig.update_layout(**_base_layout())
    return fig


# ── 3. Attendance histogram ────────────────────────────────────────────────────
def attendance_histogram(meta: dict) -> go.Figure:
    df = meta["df"]
    attend_col = meta.get("attend_col")
    if not attend_col:
        return _empty_fig("No attendance column detected")

    fig = px.histogram(
        df, x=attend_col, nbins=20,
        color_discrete_sequence=[PALETTE[2]],
        title="Attendance Distribution",
        labels={attend_col: "Attendance (%)"},
    )
    fig.add_vline(x=75, line_dash="dash", line_color="#f59e0b",
                  annotation_text="75% threshold", annotation_font_color="#f59e0b")
    fig.update_layout(**_base_layout(
        xaxis=dict(showgrid=False, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", color="#94a3b8"),
        bargap=0.05,
    ))
    return fig


# ── 4. Department avg marks bar ───────────────────────────────────────────────
def dept_marks_bar(meta: dict) -> go.Figure:
    df = meta["df"]
    dept_col = meta.get("dept_col")
    scols = meta["subject_cols"]
    if not dept_col or not scols:
        return _empty_fig("Need department + subject columns")

    dept_avg = df.groupby(dept_col)[scols].mean().mean(axis=1).reset_index()
    dept_avg.columns = ["Department", "Average Marks"]
    dept_avg = dept_avg.sort_values("Average Marks", ascending=False)

    fig = px.bar(
        dept_avg, x="Department", y="Average Marks",
        color="Average Marks", color_continuous_scale="Teal",
        text_auto=".1f",
        title="Average Marks by Department",
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(**_base_layout(
        xaxis=dict(showgrid=False, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", color="#94a3b8"),
        coloraxis_showscale=False,
    ))
    return fig


# ── 5. Grade distribution donut ───────────────────────────────────────────────
def grade_distribution(meta: dict) -> go.Figure:
    df = meta["df"]
    if "Grade" not in df.columns:
        return _empty_fig("No Grade column (need subject marks)")

    grade_order = ["A+", "A", "B", "C", "D", "F"]
    counts = df["Grade"].value_counts().reindex(grade_order).dropna().reset_index()
    counts.columns = ["Grade", "Count"]

    grade_colors = {
        "A+": "#22d3ee", "A": "#34d399", "B": "#a3e635",
        "C": "#fbbf24", "D": "#fb923c", "F": "#f87171",
    }
    colors = [grade_colors.get(g, "#94a3b8") for g in counts["Grade"]]

    fig = px.pie(
        counts, names="Grade", values="Count",
        color_discrete_sequence=colors,
        title="Grade Distribution",
        hole=0.5,
    )
    fig.update_traces(textposition="outside", textinfo="label+percent")
    fig.update_layout(**_base_layout())
    return fig


# ── 6. Student subject bar (single student) ───────────────────────────────────
def student_subject_bar(row: pd.Series, subject_cols: list[str], name: str) -> go.Figure:
    scores = [float(row[sc]) for sc in subject_cols if sc in row.index]
    fig = px.bar(
        x=subject_cols, y=scores,
        color=subject_cols, color_discrete_sequence=PALETTE,
        title=f"Subject-wise Marks — {name}",
        labels={"x": "Subject", "y": "Marks"},
        text_auto=".1f",
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(**_base_layout(
        xaxis=dict(showgrid=False, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", color="#94a3b8"),
        showlegend=False,
    ))
    return fig


# ── 7. Radar chart for student comparison ─────────────────────────────────────
def comparison_radar(students_df: pd.DataFrame, subject_cols: list[str]) -> go.Figure:
    fig = go.Figure()
    for _, row in students_df.iterrows():
        name = str(row.get("__name__", "Student"))
        values = [float(row.get(sc, 0)) for sc in subject_cols]
        values.append(values[0])   # close polygon
        theta = subject_cols + [subject_cols[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=theta, fill="toself",
            name=name, opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(30,41,59,0.6)",
            radialaxis=dict(visible=True, color="#94a3b8", gridcolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(color="#94a3b8"),
        ),
        title="Student Comparison (Radar)",
        **_base_layout(),
    )
    return fig


# ── 8. Comparison grouped bar ─────────────────────────────────────────────────
def comparison_bar(students_df: pd.DataFrame, subject_cols: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, (_, row) in enumerate(students_df.iterrows()):
        name = str(row.get("__name__", f"Student {i+1}"))
        values = [float(row.get(sc, 0)) for sc in subject_cols]
        fig.add_trace(go.Bar(
            name=name, x=subject_cols, y=values,
            marker_color=PALETTE[i % len(PALETTE)],
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Student Marks Comparison",
        **_base_layout(
            xaxis=dict(showgrid=False, color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", color="#94a3b8"),
        ),
    )
    return fig


# ── 9. Subject top students bar ───────────────────────────────────────────────
def subject_top_students(df: pd.DataFrame, subject: str, name_col: str, n: int = 10) -> go.Figure:
    if name_col not in df.columns or subject not in df.columns:
        return _empty_fig("Missing name or subject column")
    top = df[[name_col, subject]].dropna().nlargest(n, subject)
    fig = px.bar(
        top, x=name_col, y=subject,
        color=subject, color_continuous_scale="Viridis",
        text_auto=".1f",
        title=f"Top {n} Students — {subject}",
        labels={name_col: "Student", subject: "Score"},
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(**_base_layout(
        xaxis=dict(showgrid=False, color="#94a3b8", tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", color="#94a3b8"),
        coloraxis_showscale=False,
    ))
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────
def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#94a3b8"))
    fig.update_layout(**_base_layout())
    return fig
