"""
utils.py — Shared utilities, logging, and constants
"""
import logging
import os
import re
import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("university_agent")


# ── Column-name aliases ───────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "student_name": [
        "name", "student", "student_name", "studentname", "full_name",
        "fullname", "sname", "student name", "s_name",
    ],
    "department": [
        "dept", "department", "faculty", "branch", "stream",
        "program", "course", "division", "dep",
    ],
    "attendance": [
        "attendance", "attend", "attendance_pct", "attendance_%",
        "attendance_percent", "att", "present", "presence",
    ],
    "roll_no": [
        "roll", "roll_no", "rollno", "roll_number", "id",
        "student_id", "sid", "reg_no", "regno", "registration",
    ],
    "year": [
        "year", "yr", "semester", "sem", "batch", "grade",
        "class", "level", "section",
    ],
    "gender": ["gender", "sex", "g"],
    "cgpa": ["cgpa", "gpa", "grade_point", "gradepoint"],
}

SUBJECT_BLACKLIST = {
    "name", "student", "roll", "id", "department", "dept",
    "attendance", "year", "semester", "gender", "cgpa", "gpa",
    "branch", "section", "batch", "class", "stream", "program",
    "grade", "level", "status", "result", "pass", "fail",
    "total", "average", "avg", "percentage", "percent", "remarks",
    "division", "rank", "serial", "no", "sr", "s.no", "index",
    "faculty", "course",
}


# ── Column normaliser ─────────────────────────────────────────────────────────
def normalise_col(col: str) -> str:
    """Lower-case, strip, replace spaces/hyphens with underscores."""
    return re.sub(r"[\s\-]+", "_", str(col).strip().lower())


def detect_column(df: pd.DataFrame, field: str) -> str | None:
    """Return the first DataFrame column that matches *field*'s alias list."""
    aliases = COLUMN_ALIASES.get(field, [field])
    norm_map = {normalise_col(c): c for c in df.columns}
    for alias in aliases:
        if normalise_col(alias) in norm_map:
            return norm_map[normalise_col(alias)]
    return None


# ── Subject columns ───────────────────────────────────────────────────────────
def detect_subject_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric columns that are likely to be subject/marks columns.
    Heuristic: numeric dtype + not in SUBJECT_BLACKLIST + values in [0, 150].
    """
    subject_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        norm = normalise_col(col)
        if norm in SUBJECT_BLACKLIST:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        if series.min() >= 0 and series.max() <= 150:
            subject_cols.append(col)
    return subject_cols


# ── Numeric helpers ───────────────────────────────────────────────────────────
def safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean()) if not series.dropna().empty else 0.0


def safe_pct(part: float, total: float) -> float:
    return round((part / total) * 100, 2) if total else 0.0


# ── Text helpers ──────────────────────────────────────────────────────────────
def truncate(text: str, max_chars: int = 2000) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


def df_to_text(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert a (possibly large) DataFrame to a compact text representation."""
    sample = df.head(max_rows)
    return sample.to_csv(index=False)


# ── Environment ───────────────────────────────────────────────────────────────
def get_openai_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")

def get_groq_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
