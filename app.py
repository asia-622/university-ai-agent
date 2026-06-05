"""
app.py — University AI Analytics Agent
Streamlit UI with 7 sections:
  Home | Upload & Analyze | Dashboard | Subject Analysis |
  Student Search | Comparison | AI Agent Chat
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="University AI Analytics Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark academic theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0f172a;
    --surface:   #1e293b;
    --surface2:  #334155;
    --border:    #475569;
    --text:      #e2e8f0;
    --muted:     #94a3b8;
    --accent:    #38bdf8;
    --accent2:   #818cf8;
    --success:   #34d399;
    --warning:   #fbbf24;
    --danger:    #f87171;
}

html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
[data-testid="stMetricValue"]  { color: var(--accent) !important; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.1em; }

/* Headers */
h1 { color: var(--accent) !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2 { color: var(--text) !important; font-weight: 600 !important; }
h3 { color: var(--accent2) !important; font-weight: 500 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #0f172a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding: 0.5rem 1.5rem !important;
    transition: transform 0.1s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.35) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* DataFrames */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* Chat messages */
.chat-bubble-user {
    background: linear-gradient(135deg, var(--accent2), #6366f1);
    color: #fff;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.4rem 0 0.4rem 15%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-bubble-ai {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.4rem 15% 0.4rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-label { font-size: 0.7rem; color: var(--muted); margin-bottom: 0.2rem; text-transform: uppercase; letter-spacing: 0.1em; }

/* Stat cards */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}

/* Tags */
.tag {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    color: var(--accent);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    margin: 0.1rem;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'IBM Plex Sans', sans-serif !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }

/* Expanders */
.streamlit-expanderHeader { background: var(--surface) !important; color: var(--text) !important; border-radius: 8px !important; }
.streamlit-expanderContent { background: var(--surface) !important; border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Module imports
# ─────────────────────────────────────────────────────────────────────────────
from file_handler import load_file
from data_preprocessing import preprocess, get_student_row, get_student_subjects
from rag_engine import RAGEngine, build_chunks
from chatbot import UniversityAgent
from model import train_model, predict_batch
import dashboard as dash
import plotly.graph_objects as go
from tools import (
    get_dataset_summary, get_department_stats,
    get_top_students, get_attendance_analysis, get_subject_analysis,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load API key from Streamlit secrets (hidden from UI)
# ─────────────────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# ✅ CACHED CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_marks_bar(df: pd.DataFrame, subject_cols: tuple) -> object:
    return dash.marks_bar_chart({"df": df, "subject_cols": list(subject_cols)})

@st.cache_data(show_spinner=False)
def _cached_dept_pie(df: pd.DataFrame, dept_col: str) -> object:
    return dash.department_pie({"df": df, "dept_col": dept_col})

@st.cache_data(show_spinner=False)
def _cached_attendance_hist(df: pd.DataFrame, attend_col: str) -> object:
    return dash.attendance_histogram({"df": df, "attend_col": attend_col})

@st.cache_data(show_spinner=False)
def _cached_grade_dist(df: pd.DataFrame) -> object:
    return dash.grade_distribution({"df": df})

@st.cache_data(show_spinner=False)
def _cached_subject_top(df: pd.DataFrame, subject: str, name_col: str) -> object:
    return dash.subject_top_students(df, subject, name_col, n=10)

@st.cache_data(show_spinner=False)
def _cached_box_plot(df: pd.DataFrame, subject_cols: tuple) -> object:
    import plotly.express as px
    scols = list(subject_cols)
    fig = px.box(
        df[scols].melt(var_name="Subject", value_name="Score"),
        x="Subject", y="Score",
        color="Subject",
        title="Score Distribution (Box Plot)",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#e2e8f0"),
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=False,
        xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8"),
    )
    return fig

@st.cache_data(show_spinner=False)
def _cached_student_bar(row_dict: dict, subject_cols: tuple, name: str) -> object:
    row = pd.Series(row_dict)
    return dash.student_subject_bar(row, list(subject_cols), name)

@st.cache_data(show_spinner=False)
def _cached_comparison_bar(comp_dict: dict, subject_cols: tuple) -> object:
    comp_df = pd.DataFrame(comp_dict)
    return dash.comparison_bar(comp_df, list(subject_cols))

@st.cache_data(show_spinner=False)
def _cached_comparison_radar(comp_dict: dict, subject_cols: tuple) -> object:
    comp_df = pd.DataFrame(comp_dict)
    return dash.comparison_radar(comp_df, list(subject_cols))

@st.cache_data(show_spinner=False)
def _cached_dept_stats(df: pd.DataFrame, dept_col: str, subject_cols: tuple, attend_col) -> dict:
    """Cache department summary stats used in the dashboard table."""
    meta_lite = {
        "df": df,
        "dept_col": dept_col,
        "subject_cols": list(subject_cols),
        "attend_col": attend_col,
        "name_col": None,
        "roll_col": None,
        "year_col": None,
        "n_students": len(df),
        "n_departments": df[dept_col].nunique() if dept_col else 0,
        "has_attendance": attend_col is not None,
    }
    from tools import get_department_stats
    return get_department_stats(meta_lite)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "meta": None,
        "agent": None,
        "rag": None,
        "ml_model": None,
        "chat_history": [],
        "api_key": _get_api_key(),
        "rag_built": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 University Agent")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📂 Upload & Analyze", "📊 Dashboard",
         "📚 Subject Analysis", "🔍 Student Search",
         "⚖️ Comparison", "🤖 AI Agent Chat"],
        label_visibility="collapsed",
    )

    if st.session_state["meta"]:
        meta = st.session_state["meta"]
        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(f"- **Rows:** {meta['n_students']:,}")
        st.markdown(f"- **Departments:** {meta['n_departments']}")
        st.markdown(f"- **Subjects:** {len(meta['subject_cols'])}")
        if meta["has_attendance"]:
            st.markdown("- ✅ Attendance column")
        if st.session_state["rag_built"]:
            st.markdown("- ✅ RAG index built")
        if st.session_state["ml_model"]:
            m = st.session_state["ml_model"]
            st.markdown(f"- ✅ ML model R²={m['metrics']['r2']}")

    st.markdown("---")
    st.caption("University AI Analytics Agent v1.0")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _require_data():
    if st.session_state["meta"] is None:
        st.warning("⚠️ Please upload a dataset first on the **Upload & Analyze** page.")
        st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🎓 University AI Analytics Agent")
    st.markdown("### Intelligent academic data analysis powered by LLM + RAG + Tools")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stat-card'>
        <h3>🧠 AI Agent</h3>
        <p style='color:#94a3b8;font-size:0.9rem'>
        GPT-4 powered agent with tool calling, RAG retrieval, and conversation memory.
        </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
        <h3>📡 RAG Engine</h3>
        <p style='color:#94a3b8;font-size:0.9rem'>
        FAISS vector index over your dataset for semantic search and context retrieval.
        </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-card'>
        <h3>🔧 8 Tools</h3>
        <p style='color:#94a3b8;font-size:0.9rem'>
        Dataset summary, department stats, top students, attendance analysis, ML prediction & more.
        </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    steps = [
        ("1", "📂 Upload & Analyze", "Upload CSV / Excel / JSON dataset"),
        ("2", "📊 Dashboard", "Explore auto-generated charts"),
        ("3", "🔍 Student Search", "Search and view student profiles"),
        ("4", "🤖 AI Agent Chat", "Ask questions in natural language"),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class='stat-card' style='text-align:center'>
            <div style='font-size:2rem;font-weight:700;color:#38bdf8'>{num}</div>
            <div style='font-weight:600;margin:0.4rem 0'>{title}</div>
            <div style='color:#94a3b8;font-size:0.8rem'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💬 Sample Questions")
    samples = [
        "Which department has the highest average marks?",
        "Show me students with attendance below 75%",
        "Who are the top 5 students overall?",
        "Predict performance for student John",
        "What is the average score in Mathematics?",
        "Compare department-wise attendance",
    ]
    cols = st.columns(3)
    for i, q in enumerate(samples):
        with cols[i % 3]:
            st.markdown(f"<span class='tag'>💬 {q}</span>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Upload & Analyze
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📂 Upload & Analyze":
    st.markdown("# 📂 Upload & Analyze Dataset")
    st.markdown("Supports CSV, Excel (.xlsx/.xls), and JSON. Files up to 200 MB+.")

    uploaded = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )

    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_last_file_id") != file_id:
            st.session_state["_last_file_id"] = file_id

            with st.spinner("📥 Loading file…"):
                df_raw = load_file(uploaded)

            if df_raw is not None:
                with st.spinner("⚙️ Preprocessing…"):
                    meta = preprocess(df_raw)
                st.session_state["meta"] = meta

                with st.spinner("🧠 Building RAG index…"):
                    rag = RAGEngine(api_key=st.session_state.get("api_key"))
                    chunks = build_chunks(meta)
                    rag.build(chunks)
                    st.session_state["rag"] = rag
                    st.session_state["rag_built"] = True

                with st.spinner("📈 Training ML model…"):
                    ml = train_model(meta)
                    st.session_state["ml_model"] = ml

                agent = UniversityAgent(api_key=st.session_state.get("api_key"))
                agent.attach_data(meta, rag, ml)
                st.session_state["agent"] = agent

                # ✅ Cache clear karein naye dataset ke liye
                _cached_marks_bar.clear()
                _cached_dept_pie.clear()
                _cached_attendance_hist.clear()
                _cached_grade_dist.clear()
                _cached_subject_top.clear()
                _cached_box_plot.clear()
                _cached_dept_stats.clear()

        meta = st.session_state.get("meta")
        if meta:
            ml = st.session_state.get("ml_model")

            st.success(f"✅ Dataset loaded! {meta['n_students']:,} students, {len(meta['subject_cols'])} subjects detected.")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{meta['n_students']:,}")
            col2.metric("Columns", len(meta["df"].columns))
            col3.metric("Departments", meta["n_departments"])
            col4.metric("Subjects", len(meta["subject_cols"]))

            st.markdown("---")

            with st.expander("🔍 Detected Column Mapping", expanded=True):
                cols_info = {
                    "Student Name": meta.get("name_col") or "Not detected",
                    "Department": meta.get("dept_col") or "Not detected",
                    "Attendance": meta.get("attend_col") or "Not detected",
                    "Roll No": meta.get("roll_col") or "Not detected",
                    "Year/Semester": meta.get("year_col") or "Not detected",
                    "Subject Columns": ", ".join(meta["subject_cols"]) or "None",
                }
                for k, v in cols_info.items():
                    color = "#34d399" if "Not detected" not in str(v) and "None" not in str(v) else "#f87171"
                    st.markdown(f"**{k}:** <span style='color:{color}'>{v}</span>", unsafe_allow_html=True)

            dept_subject_map = meta.get("dept_subject_map", {})
            if dept_subject_map:
                with st.expander("🗂️ Department → Subjects Mapping", expanded=False):
                    for dept, subjects in dept_subject_map.items():
                        st.markdown(f"**{dept}** ({len(subjects)} subjects): "
                                    f"<span style='color:#94a3b8'>{', '.join(subjects)}</span>",
                                    unsafe_allow_html=True)

            st.markdown("### 👀 Data Preview (first 20 rows)")
            st.dataframe(meta["df"].head(20), use_container_width=True)

            if st.session_state["rag_built"]:
                st.info(f"🧠 RAG index ready")

            if ml:
                metrics = ml["metrics"]
                st.success(f"📈 ML model trained — R²: {metrics['r2']}  MAE: {metrics['mae']}  "
                           f"Target: `{ml['target_col']}`  Features: {ml['feature_cols']}")

            st.markdown("---")
            st.markdown("### ⬇️ Download")
            c1, c2 = st.columns(2)
            with c1:
                csv_buf = meta["df"].to_csv(index=False).encode()
                st.download_button("📥 Download Cleaned CSV", csv_buf,
                                   file_name="cleaned_data.csv", mime="text/csv")
            with c2:
                xls_buf = io.BytesIO()
                with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
                    meta["df"].to_excel(writer, index=False, sheet_name="Cleaned")
                st.download_button("📥 Download Excel", xls_buf.getvalue(),
                                   file_name="cleaned_data.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    _require_data()
    meta = st.session_state["meta"]
    st.markdown("# 📊 Analytics Dashboard")

    summary = get_dataset_summary(meta)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Students", f"{summary['total_students']:,}")
    c2.metric("🏛️ Departments", summary["n_departments"])
    c3.metric("📚 Subjects", len(summary["subject_columns"]))
    if "class_average" in summary:
        c4.metric("📈 Class Avg", f"{summary['class_average']:.1f}")
    if "average_attendance" in summary:
        c5.metric("✅ Avg Attendance", f"{summary['average_attendance']:.1f}%")

    st.markdown("---")

    df = meta["df"]
    scols = tuple(meta["subject_cols"])
    dept_col = meta.get("dept_col")
    attend_col = meta.get("attend_col")

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_cached_marks_bar(df, scols), use_container_width=True)
    with col2:
        if dept_col:
            st.plotly_chart(_cached_dept_pie(df, dept_col), use_container_width=True)

    # Charts row 2
    col3, col4 = st.columns(2)
    with col3:
        if attend_col:
            st.plotly_chart(_cached_attendance_hist(df, attend_col), use_container_width=True)
    with col4:
        st.plotly_chart(_cached_grade_dist(df), use_container_width=True)

    # Department summary table
    if dept_col:
        st.markdown("---")
        st.markdown("### 🏛️ Department Summary Table")
        dept_stats = _cached_dept_stats(df, dept_col, scols, attend_col)
        if "departments" in dept_stats:
            rows = []
            for dept, info in dept_stats["departments"].items():
                row = {"Department": dept, "Count": info["count"]}
                if "avg_marks" in info: row["Avg Marks"] = info["avg_marks"]
                if "avg_attendance" in info: row["Avg Attendance"] = info["avg_attendance"]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).sort_values("Count", ascending=False),
                         use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Subject Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📚 Subject Analysis":
    _require_data()
    meta = st.session_state["meta"]
    st.markdown("# 📚 Subject-wise Analysis")

    scols = meta["subject_cols"]
    if not scols:
        st.error("❌ No subject/marks columns detected in this dataset.")
        st.stop()

    subject = st.selectbox("Select Subject", scols)

    df = meta["df"]
    col = df[subject]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average", f"{col.mean():.2f}")
    c2.metric("Highest", f"{col.max():.2f}")
    c3.metric("Lowest", f"{col.min():.2f}")
    c4.metric("Std Dev", f"{col.std():.2f}")

    st.markdown("---")

    name_col = meta.get("name_col")
    if name_col:
        st.plotly_chart(
            _cached_subject_top(df, subject, name_col),
            use_container_width=True,
        )

    st.markdown("### 📊 All Subjects Summary")
    subj_data = get_subject_analysis(meta)
    if "subjects" in subj_data:
        rows = [
            {
                "Subject": s,
                "Average": info["average"],
                "Max": info["max"],
                "Min": info["min"],
                "Std Dev": info["std"],
            }
            for s, info in subj_data["subjects"].items()
        ]
        st.dataframe(pd.DataFrame(rows).sort_values("Average", ascending=False),
                     use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(_cached_marks_bar(df, tuple(scols)), use_container_width=True)
    with col_r:
        st.plotly_chart(_cached_box_plot(df, tuple(scols)), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Student Search
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Student Search":
    _require_data()
    meta = st.session_state["meta"]
    st.markdown("# 🔍 Student Search & Profile")

    name_col = meta.get("name_col")
    if name_col is None:
        st.error("❌ No student name column detected.")
        st.stop()

    query = st.text_input("🔎 Enter student name", placeholder="e.g. Alice, John…")

    if query:
        rows = get_student_row(meta, query)
        if rows.empty:
            st.warning(f"No students found matching **'{query}'**")
        else:
            st.success(f"Found **{len(rows)}** student(s)")
            for idx in range(min(len(rows), 5)):
                row = rows.iloc[idx]
                name = str(row.get(name_col, "?"))
                st.markdown(f"### 👤 {name}")

                info_cols = st.columns(4)
                i = 0
                for col_name in [meta.get("dept_col"), meta.get("roll_col"),
                                  meta.get("year_col"), meta.get("attend_col")]:
                    if col_name and col_name in row.index:
                        val = row[col_name]
                        label = col_name.replace("_", " ").title()
                        info_cols[i % 4].metric(label, f"{val}")
                        i += 1

                if "Average" in row.index:
                    info_cols[i % 4].metric("Average", f"{row['Average']:.2f}")
                    i += 1
                if "Grade" in row.index:
                    info_cols[i % 4].metric("Grade", str(row["Grade"]))

                student_scols = get_student_subjects(meta, row)
                if student_scols:
                    dept_name = str(row.get(meta.get("dept_col"), "")) if meta.get("dept_col") else ""
                    st.markdown(
                        f"<span style='color:#94a3b8;font-size:0.8rem'>"
                        f"📚 Showing {len(student_scols)} subjects for {dept_name} department</span>",
                        unsafe_allow_html=True,
                    )
                    row_dict = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else str(v))
                                for k, v in row.items() if not str(k).startswith("_")}
                    fig = _cached_student_bar(row_dict, tuple(student_scols), name)
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Full Record"):
                    record = {k: v for k, v in row.items()
                              if not str(k).startswith("_")}
                    st.json({k: (float(v) if isinstance(v, (np.floating, np.integer)) else str(v))
                             for k, v in record.items()})

                ml = st.session_state.get("ml_model")
                if ml:
                    try:
                        row_df = pd.DataFrame([row])
                        pred = predict_batch(ml, row_df).iloc[0]
                        st.info(f"🤖 Predicted **{ml['target_col']}**: **{pred:.2f}**  "
                                f"*(ML model — R²={ml['metrics']['r2']})*")
                    except Exception:
                        pass

                st.markdown("---")

    if query and not get_student_row(meta, query).empty:
        result_df = get_student_row(meta, query)
        csv_bytes = result_df.to_csv(index=False).encode()
        st.download_button("📥 Download Results", csv_bytes,
                           file_name=f"search_{query}.csv", mime="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Comparison  ✅ FIXED — cross-department support
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Comparison":
    _require_data()
    meta = st.session_state["meta"]
    st.markdown("# ⚖️ Student Comparison")

    name_col   = meta.get("name_col")
    scols      = meta["subject_cols"]
    dept_col   = meta.get("dept_col")
    attend_col = meta.get("attend_col")

    if name_col is None:
        st.error("❌ No student name column detected.")
        st.stop()
    if not scols:
        st.error("❌ No subject columns detected.")
        st.stop()

    all_names = meta["df"][name_col].dropna().astype(str).unique().tolist()
    selected = st.multiselect(
        "Select 2–5 students to compare",
        options=all_names,
        max_selections=5,
    )

    if len(selected) < 2:
        st.info("Please select at least 2 students.")
    else:
        rows = []
        for name in selected:
            match = get_student_row(meta, name)
            if not match.empty:
                r = match.iloc[0].copy()
                r["__name__"] = name
                rows.append(r)

        if len(rows) < 2:
            st.warning("Could not find data for the selected students.")
        else:
            comp_df = pd.DataFrame(rows).reset_index(drop=True)

            # ── Detect departments of selected students ──────────────────────
            student_depts = []
            for row in rows:
                d = str(row.get(dept_col, "Unknown")) if dept_col else "Unknown"
                student_depts.append(d)

            same_dept = len(set(student_depts)) == 1

            # ════════════════════════════════════════════════════════════════
            # CASE A: Same department → standard shared-subject comparison
            # ════════════════════════════════════════════════════════════════
            if same_dept:
                compare_scols = get_student_subjects(meta, rows[0])
                comp_dict     = comp_df.to_dict(orient="list")

                st.plotly_chart(
                    _cached_comparison_bar(comp_dict, tuple(compare_scols)),
                    use_container_width=True,
                )

                if len(compare_scols) >= 3:
                    st.plotly_chart(
                        _cached_comparison_radar(comp_dict, tuple(compare_scols)),
                        use_container_width=True,
                    )

            # ════════════════════════════════════════════════════════════════
            # CASE B: Different departments → per-student charts + overall bar
            # ════════════════════════════════════════════════════════════════
            else:
                st.info(
                    "ℹ️ Selected students are from **different departments** — "
                    "showing individual subject charts and an overall comparison."
                )

                # ── Per-student subject bar charts ───────────────────────────
                st.markdown("### 📊 Individual Subject Performance")
                for row in rows:
                    s_name       = str(row.get("__name__", "Student"))
                    s_dept       = str(row.get(dept_col, "")) if dept_col else ""
                    student_scols = get_student_subjects(meta, row)

                    if student_scols:
                        st.markdown(
                            f"**👤 {s_name}** "
                            f"<span style='color:#94a3b8;font-size:0.85rem'>— {s_dept}</span>",
                            unsafe_allow_html=True,
                        )
                        row_dict = {
                            k: (float(v) if isinstance(v, (np.floating, np.integer)) else str(v))
                            for k, v in row.items()
                            if not str(k).startswith("_")
                        }
                        fig = _cached_student_bar(row_dict, tuple(student_scols), s_name)
                        st.plotly_chart(fig, use_container_width=True)

                # ── Overall Average + Attendance comparison ──────────────────
                st.markdown("### ⚖️ Overall Comparison")

                compare_metrics = []
                if "Average" in comp_df.columns:
                    compare_metrics.append("Average")
                if attend_col and attend_col in comp_df.columns:
                    compare_metrics.append(attend_col)

                if compare_metrics:
                    overall_fig = go.Figure()
                    palette     = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171"]

                    for idx, metric in enumerate(compare_metrics):
                        vals  = [float(r.get(metric, 0)) for r in rows]
                        names = [str(r.get("__name__", "?")) for r in rows]
                        overall_fig.add_trace(go.Bar(
                            name=metric,
                            x=names,
                            y=vals,
                            marker_color=palette[idx % len(palette)],
                            text=[f"{v:.1f}" for v in vals],
                            textposition="outside",
                        ))

                    overall_fig.update_layout(
                        barmode      = "group",
                        title        = "Overall Average & Attendance Comparison",
                        paper_bgcolor= "rgba(0,0,0,0)",
                        plot_bgcolor = "rgba(0,0,0,0)",
                        font         = dict(family="IBM Plex Sans, sans-serif", color="#e2e8f0"),
                        margin       = dict(l=30, r=30, t=50, b=30),
                        legend       = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
                        xaxis        = dict(color="#94a3b8", showgrid=False),
                        yaxis        = dict(color="#94a3b8",
                                            gridcolor="rgba(148,163,184,0.15)",
                                            range=[0, 110]),
                    )
                    st.plotly_chart(overall_fig, use_container_width=True)

                # ── Grade badges ─────────────────────────────────────────────
                if "Grade" in comp_df.columns:
                    st.markdown("### 🏅 Grade Summary")
                    grade_cols = st.columns(len(rows))
                    grade_color = {
                        "A+": "#22d3ee", "A": "#34d399", "B": "#a3e635",
                        "C": "#fbbf24", "D": "#fb923c", "F": "#f87171",
                    }
                    for gc, row in zip(grade_cols, rows):
                        s_name = str(row.get("__name__", "Student"))
                        s_dept = str(row.get(dept_col, "")) if dept_col else ""
                        grade  = str(row.get("Grade", "N/A"))
                        avg    = row.get("Average", 0)
                        color  = grade_color.get(grade, "#94a3b8")
                        gc.markdown(
                            f"<div style='background:var(--surface);border:1px solid {color};"
                            f"border-radius:12px;padding:1rem;text-align:center'>"
                            f"<div style='font-size:2rem;font-weight:700;color:{color}'>{grade}</div>"
                            f"<div style='font-weight:600;margin:0.3rem 0'>{s_name}</div>"
                            f"<div style='color:#94a3b8;font-size:0.8rem'>{s_dept}</div>"
                            f"<div style='color:#94a3b8;font-size:0.8rem'>Avg: {float(avg):.2f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── Comparison table — always show ───────────────────────────────
            st.markdown("### 📋 Comparison Table")

            # Build display columns: name + each student's own subjects union + Average + Attendance
            all_student_scols: list[str] = []
            for row in rows:
                for sc in get_student_subjects(meta, row):
                    if sc not in all_student_scols:
                        all_student_scols.append(sc)

            display_cols = ["__name__"] + all_student_scols
            if "Average"  in comp_df.columns: display_cols.append("Average")
            if attend_col and attend_col in comp_df.columns:
                display_cols.append(attend_col)
            if "Grade"    in comp_df.columns: display_cols.append("Grade")
            if dept_col   and dept_col in comp_df.columns:
                display_cols.insert(1, dept_col)

            tbl = comp_df[[c for c in display_cols if c in comp_df.columns]].copy()
            tbl = tbl.rename(columns={"__name__": "Student"})
            st.dataframe(tbl, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: AI Agent Chat
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Agent Chat":
    st.markdown("# 🤖 AI Agent Chat")
    st.markdown("Ask anything about the dataset. The agent uses **RAG + Tools + Memory**.")

    if st.session_state["meta"] is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    if st.session_state.get("agent"):
        agent: UniversityAgent = st.session_state["agent"]
        if st.session_state.get("api_key") and agent.client is None:
            try:
                from groq import Groq
                agent.client = Groq(api_key=st.session_state["api_key"])
            except Exception:
                pass
    else:
        agent = UniversityAgent(api_key=st.session_state.get("api_key"))
        agent.attach_data(
            st.session_state["meta"],
            st.session_state.get("rag"),
            st.session_state.get("ml_model"),
        )
        st.session_state["agent"] = agent

    has_key     = bool(st.session_state.get("api_key"))
    status_color = "#34d399" if has_key else "#fbbf24"
    status_text  = (
        "🟢 Groq LLaMA + RAG + Tools Active" if has_key
        else "🟡 RAG+Tools only (add Groq API key in Streamlit secrets)"
    )
    st.markdown(
        f"<div style='background:rgba(30,41,59,0.8);border:1px solid #334155;"
        f"border-radius:8px;padding:0.6rem 1rem;margin-bottom:1rem;"
        f"color:{status_color};font-size:0.85rem'>{status_text}</div>",
        unsafe_allow_html=True,
    )

    chat_container = st.container()
    with chat_container:
        for role, content in st.session_state["chat_history"]:
            if role == "user":
                st.markdown(
                    f"<div class='chat-label'>You</div>"
                    f"<div class='chat-bubble-user'>{content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-label'>🤖 Agent</div>"
                    f"<div class='chat-bubble-ai'>{content}</div>",
                    unsafe_allow_html=True,
                )

    # ── Suggestion buttons — click karo, seedha jawab aayega ─────────────────
    suggestions = [
        "Summarise the dataset",
        "Which department has highest marks?",
        "Show students with low attendance",
        "Who are the top 5 students?",
        "Analyse subject performance",
    ]
    sug_cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        if sug_cols[i].button(sug, key=f"sug_{i}"):
            st.session_state["chat_history"].append(("user", sug))
            with st.spinner("🤖 Thinking…"):
                reply = agent.chat(sug)
            st.session_state["chat_history"].append(("assistant", reply))
            st.rerun()

    # ── Manual input form ─────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Message",
            placeholder="Ask about students, departments, subjects, attendance…",
            height=80,
            label_visibility="collapsed",
        )
        c_send, c_clear = st.columns([3, 1])
        send  = c_send.form_submit_button("📨 Send",  use_container_width=True)
        clear = c_clear.form_submit_button("🗑 Clear", use_container_width=True)

    if clear:
        st.session_state["chat_history"] = []
        agent.reset()
        st.rerun()

    if send and user_input.strip():
        st.session_state["chat_history"].append(("user", user_input.strip()))
        with st.spinner("🤖 Thinking…"):
            reply = agent.chat(user_input.strip())
        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()

    if st.session_state["chat_history"]:
        chat_text = "\n\n".join(
            f"{'User' if r == 'user' else 'Agent'}: {c}"
            for r, c in st.session_state["chat_history"]
        )
        st.download_button(
            "📥 Export Chat",
            chat_text.encode(),
            file_name="chat_history.txt",
            mime="text/plain",
        )
