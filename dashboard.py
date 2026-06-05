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


# ── 4. [REMOVED] dept_marks_bar — replaced by dept_subject_analysis ──────────
# Function removed as per new feature requirement.
# Use dept_subject_analysis() instead for department-wise subject breakdown.


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


# ══════════════════════════════════════════════════════════════════════════════
# ── 10. NEW: Department-wise Subject Analysis (replaces dept_marks_bar) ───────
# ══════════════════════════════════════════════════════════════════════════════

def _detect_semester_from_columns(subject_cols: list[str]) -> dict[str, list[str]]:
    """
    Try to group subject columns by semester from their names.
    Patterns detected (case-insensitive):
      - Sem1_Math / Math_Sem1 / Math_S1 / S1_Math / Semester1_Math
      - Math1, Physics2  (trailing digit after known keywords)
    Returns dict: { "Semester 1": [...cols], "Semester 2": [...cols], ... }
    If no pattern found, returns { "All Subjects": [...all cols] }
    """
    import re
    sem_map: dict[str, list[str]] = {}

    patterns = [
        r"sem(?:ester)?[\s_\-]?(\d+)",   # sem1, semester_2, sem-3
        r"s(\d+)[\s_\-]",                 # S1_, S2_
        r"[\s_\-]s(\d+)$",               # _S1 at end
        r"[\s_\-]sem(\d+)",              # _sem2
    ]

    unmatched = []
    for col in subject_cols:
        col_lower = col.lower()
        matched = False
        for pat in patterns:
            m = re.search(pat, col_lower)
            if m:
                sem_num = int(m.group(1))
                key = f"Semester {sem_num}"
                sem_map.setdefault(key, []).append(col)
                matched = True
                break
        if not matched:
            unmatched.append(col)

    # If nothing matched at all, return single group
    if not sem_map:
        return {"All Subjects": subject_cols}

    # Unmatched go into "Other"
    if unmatched:
        sem_map["Other"] = unmatched

    return dict(sorted(sem_map.items()))


def _get_dept_sem_subjects(
    df: pd.DataFrame,
    dept_col: str,
    dept_name: str,
    sem_col: str,
    subject_cols: list[str],
) -> dict[str, list[str]]:
    """
    For dept+semester data: find which subjects are NEW in each semester
    (appear in sem N but NOT in sem N-1 = first time taught).
    Returns { "Semester 1": [cols], "Semester 2": [cols], ... }
    """
    dept_df = df[df[dept_col] == dept_name]
    semesters = sorted(dept_df[sem_col].dropna().unique())
    sem_groups = {}
    FILL_THRESH = 0.5  # subject must be filled in >=50% of rows to count

    for sem_val in semesters:
        label = f"Semester {int(sem_val)}"
        curr_rows = dept_df[dept_df[sem_col] == sem_val]
        n_curr = max(len(curr_rows), 1)

        # Which subject cols are filled in this semester
        curr_filled = set(
            col for col in subject_cols
            if col in curr_rows.columns
            and curr_rows[col].dropna().shape[0] / n_curr >= FILL_THRESH
        )

        # Remove carry-over from previous semester
        prev_sem = sem_val - 1
        if prev_sem >= 1:
            prev_rows = dept_df[dept_df[sem_col] == prev_sem]
            n_prev = max(len(prev_rows), 1)
            prev_filled = set(
                col for col in curr_filled
                if col in prev_rows.columns
                and prev_rows[col].dropna().shape[0] / n_prev >= FILL_THRESH
            )
            new_this_sem = sorted(curr_filled - prev_filled)
        else:
            new_this_sem = sorted(curr_filled)

        if new_this_sem:
            sem_groups[label] = new_this_sem

    return sem_groups if sem_groups else {}


def _get_dept_subjects(
    df: pd.DataFrame,
    dept_col: str,
    dept_name: str,
    subject_cols: list[str],
    sem_col: str | None,
) -> dict[str, list[str]]:
    """
    Returns { semester_label: [subject_cols] } for the chosen department.
    Uses sem_col if available (best accuracy), else parses column names.
    """
    if sem_col and sem_col in df.columns:
        return _get_dept_sem_subjects(df, dept_col, dept_name, sem_col, subject_cols)
    return _detect_semester_from_columns(subject_cols)



def dept_subject_bar(
    df: pd.DataFrame,
    dept_col: str,
    dept_name: str,
    subject_cols: list[str],
    sem_col: str | None = None,
) -> list[go.Figure]:
    """
    Returns a list of Bar figures — one per semester (or one for all subjects).
    Each figure shows average marks per subject for the selected department.
    """
    dept_df = df[df[dept_col] == dept_name]
    if dept_df.empty:
        return [_empty_fig(f"No data for department: {dept_name}")]

    sem_groups = _get_dept_subjects(df, dept_col, dept_name, subject_cols, sem_col)
    figs = []

    for sem_label, scols in sem_groups.items():
        # If sem_col exists, filter rows too
        if sem_col and sem_col in df.columns and sem_label != "All Subjects":
            sem_val = sem_label.replace("Semester ", "").strip()
            rows = dept_df[
                dept_df[sem_col].astype(str).str.strip() == sem_val
            ]
            if rows.empty:
                rows = dept_df  # fallback
        else:
            rows = dept_df

        valid_scols = [c for c in scols if c in rows.columns]
        if not valid_scols:
            continue

        avgs = rows[valid_scols].mean().reset_index()
        avgs.columns = ["Subject", "Average Marks"]
        avgs = avgs.dropna()

        fig = px.bar(
            avgs, x="Subject", y="Average Marks",
            color="Subject", color_discrete_sequence=PALETTE,
            text_auto=".1f",
            title=f"{dept_name} — {sem_label} (Bar)",
            labels={"Subject": "Subject", "Average Marks": "Avg Marks"},
        )
        fig.update_traces(marker_line_width=0, textfont_size=11)
        fig.update_layout(**_base_layout(
            xaxis=dict(showgrid=False, color="#94a3b8", tickangle=-20),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", color="#94a3b8"),
            showlegend=False,
        ))
        figs.append(fig)

    return figs if figs else [_empty_fig(f"No subject data for {dept_name}")]


def dept_subject_pie(
    df: pd.DataFrame,
    dept_col: str,
    dept_name: str,
    subject_cols: list[str],
    sem_col: str | None = None,
) -> list[go.Figure]:
    """
    Returns a list of Pie/Donut figures — one per semester.
    Each pie shows the share of average marks per subject (relative performance).
    """
    dept_df = df[df[dept_col] == dept_name]
    if dept_df.empty:
        return [_empty_fig(f"No data for department: {dept_name}")]

    sem_groups = _get_dept_subjects(df, dept_col, dept_name, subject_cols, sem_col)
    figs = []

    for sem_label, scols in sem_groups.items():
        if sem_col and sem_col in df.columns and sem_label != "All Subjects":
            sem_val = sem_label.replace("Semester ", "").strip()
            rows = dept_df[
                dept_df[sem_col].astype(str).str.strip() == sem_val
            ]
            if rows.empty:
                rows = dept_df
        else:
            rows = dept_df

        valid_scols = [c for c in scols if c in rows.columns]
        if not valid_scols:
            continue

        avgs = rows[valid_scols].mean().dropna()
        if avgs.empty:
            continue

        pie_df = avgs.reset_index()
        pie_df.columns = ["Subject", "Average Marks"]

        fig = px.pie(
            pie_df, names="Subject", values="Average Marks",
            color_discrete_sequence=PALETTE,
            title=f"{dept_name} — {sem_label} (Pie)",
            hole=0.4,
        )
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(**_base_layout())
        figs.append(fig)

    return figs if figs else [_empty_fig(f"No subject data for {dept_name}")]


def dept_subject_analysis(
    df: pd.DataFrame,
    dept_col: str,
    dept_name: str,
    subject_cols: list[str],
    sem_col: str | None = None,
) -> list[tuple[str, go.Figure, go.Figure]]:
    """
    Master function — returns list of (semester_label, bar_fig, pie_fig).
    Call this from Streamlit to render both chart types per semester.

    Usage in app.py / ui.py:
        results = dept_subject_analysis(df, dept_col, selected_dept, subject_cols, sem_col)
        for sem_label, bar_fig, pie_fig in results:
            st.subheader(sem_label)
            col1, col2 = st.columns(2)
            col1.plotly_chart(bar_fig, use_container_width=True)
            col2.plotly_chart(pie_fig, use_container_width=True)
    """
    dept_df = df[df[dept_col] == dept_name]
    if dept_df.empty:
        empty = _empty_fig(f"No data for {dept_name}")
        return [("No Data", empty, empty)]

    sem_groups = _get_dept_subjects(df, dept_col, dept_name, subject_cols, sem_col)
    results = []

    for sem_label, scols in sem_groups.items():
        # ✅ Row filter — only this dept + this semester
        if sem_col and sem_col in df.columns and sem_label != "All Subjects":
            sem_val = sem_label.replace("Semester ", "").strip()
            rows = dept_df[dept_df[sem_col].astype(str).str.strip() == sem_val]
            if rows.empty:
                rows = dept_df
        else:
            rows = dept_df

        _non_meta = {'Roll_No','Student_Name','Department','Year','Semester',
                     'Attendance','Average','Grade'}
        valid_scols = [c for c in scols if c in rows.columns
                       and c not in _non_meta
                       and pd.api.types.is_numeric_dtype(rows[c])]
        if not valid_scols:
            continue

        # Drop subjects with 0 or null average (not taught this semester)
        avgs = rows[valid_scols].mean().dropna()
        avgs = avgs[avgs > 0]
        if avgs.empty:
            continue
        avgs = avgs.reset_index()
        avgs.columns = ["Subject", "Average Marks"]
        # Clean display names: remove dept prefix (e.g. CS_English → English)
        avgs["Subject"] = avgs["Subject"].apply(
            lambda x: x.split("_", 1)[1].replace("_", " ") if "_" in x else x
        )

        # ── Bar ──
        bar_fig = px.bar(
            avgs, x="Subject", y="Average Marks",
            color="Subject", color_discrete_sequence=PALETTE,
            text_auto=".1f",
            title=f"{dept_name} · {sem_label} — Subject Avg (Bar)",
        )
        bar_fig.update_traces(marker_line_width=0, textfont_size=11)
        bar_fig.update_layout(**_base_layout(
            xaxis=dict(showgrid=False, color="#94a3b8", tickangle=-20),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", color="#94a3b8"),
            showlegend=False,
        ))

        # ── Pie ──
        pie_fig = px.pie(
            avgs, names="Subject", values="Average Marks",
            color_discrete_sequence=PALETTE,
            title=f"{dept_name} · {sem_label} — Subject Share (Pie)",
            hole=0.4,
        )
        pie_fig.update_traces(textposition="outside", textinfo="label+percent")
        pie_fig.update_layout(**_base_layout())

        results.append((sem_label, bar_fig, pie_fig))

    return results if results else [("No Data", _empty_fig("No subjects found"), _empty_fig("No subjects found"))]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#94a3b8"))
    fig.update_layout(**_base_layout())
    return fig
