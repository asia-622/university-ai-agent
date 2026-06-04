"""
data_preprocessing.py — Schema-agnostic data cleaning & feature extraction.
Auto-detects student name, department, attendance, marks, and subject columns.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from utils import (
    COLUMN_ALIASES,
    detect_column,
    detect_subject_columns,
    normalise_col,
)

logger = logging.getLogger("university_agent.preprocessing")


# ── Public API ────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> dict:
    """
    Clean df and extract a metadata dictionary used by all other modules.

    Returns
    -------
    dict with keys:
        df                   – cleaned DataFrame
        name_col             – str | None
        dept_col             – str | None
        attend_col           – str | None
        roll_col             – str | None
        year_col             – str | None
        subject_cols         – list[str]   (ALL subjects across all depts)
        dept_subject_map     – dict[str, list[str]]  (dept -> its subjects)
        numeric_cols         – list[str]
        n_students           – int
        n_departments        – int
        departments          – list[str]
        has_attendance       – bool
        has_subjects         – bool
    """
    df = df.copy()
    df = _coerce_numeric(df)

    meta: dict = {}

    # Detect semantic columns BEFORE filling missing
    # (so we can use original NaN pattern to find dept->subject mapping)
    meta["name_col"]   = detect_column(df, "student_name")
    meta["dept_col"]   = detect_column(df, "department")
    meta["attend_col"] = detect_column(df, "attendance")
    meta["roll_col"]   = detect_column(df, "roll_no")
    meta["year_col"]   = detect_column(df, "year")
    meta["subject_cols"] = detect_subject_columns(df)

    # Build department -> subjects mapping BEFORE filling NaNs
    meta["dept_subject_map"] = _build_dept_subject_map(df, meta)

    # Now fill missing values
    df = _fill_missing(df)

    meta["numeric_cols"]  = list(df.select_dtypes(include=[np.number]).columns)
    meta["df"]            = df
    meta["n_students"]    = len(df)

    # Department stats
    if meta["dept_col"]:
        depts = df[meta["dept_col"]].dropna().unique().tolist()
        meta["departments"]   = [str(d) for d in depts]
        meta["n_departments"] = len(depts)
    else:
        meta["departments"]   = []
        meta["n_departments"] = 0

    meta["has_attendance"] = meta["attend_col"] is not None
    meta["has_subjects"]   = len(meta["subject_cols"]) > 0

    _add_derived_columns(df, meta)

    logger.info(
        "Preprocessing done — students=%d  depts=%d  subjects=%d",
        meta["n_students"], meta["n_departments"], len(meta["subject_cols"]),
    )
    return meta


# ── Department → Subjects mapping ─────────────────────────────────────────────
def _build_dept_subject_map(df: pd.DataFrame, meta: dict) -> dict[str, list[str]]:
    """
    For each department, find which subject columns have actual (non-NaN)
    values. This must be called BEFORE _fill_missing so NaN pattern is intact.

    Strategy:
      - Group rows by department
      - For each subject column, check if the majority of students in that
        department have a real (non-NaN) value
      - If >50% students in a dept have a real value → that subject belongs
        to that department
    """
    dept_col  = meta.get("dept_col")
    scols     = meta.get("subject_cols", [])

    if not dept_col or not scols or dept_col not in df.columns:
        return {}

    dept_subject_map: dict[str, list[str]] = {}
    threshold = 0.5   # >50% students must have a real score

    for dept in df[dept_col].dropna().unique():
        dept_df   = df[df[dept_col] == dept]
        dept_subjects = []
        for s in scols:
            if s in dept_df.columns:
                non_null_ratio = dept_df[s].notna().mean()
                if non_null_ratio > threshold:
                    dept_subjects.append(s)
        if dept_subjects:
            dept_subject_map[str(dept)] = dept_subjects

    return dept_subject_map


# ── Internal helpers ──────────────────────────────────────────────────────────
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Try to convert object columns that look numeric."""
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.7:          # >70 % parseable → numeric
            df[col] = converted
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")
    return df


def _add_derived_columns(df: pd.DataFrame, meta: dict) -> None:
    """Add Total, Average, and Grade columns when subject cols exist."""
    scols = meta["subject_cols"]
    if not scols:
        return

    if "Total" not in df.columns:
        df["Total"] = df[scols].sum(axis=1)
        meta["numeric_cols"].append("Total")

    if "Average" not in df.columns:
        df["Average"] = df[scols].mean(axis=1).round(2)
        meta["numeric_cols"].append("Average")

    if "Grade" not in df.columns:
        df["Grade"] = df["Average"].apply(_grade)

    meta["df"] = df


def _grade(avg: float) -> str:
    if avg >= 90:  return "A+"
    if avg >= 80:  return "A"
    if avg >= 70:  return "B"
    if avg >= 60:  return "C"
    if avg >= 50:  return "D"
    return "F"


# ── Convenience re-export for other modules ───────────────────────────────────
def get_student_row(meta: dict, name: str) -> pd.DataFrame:
    """Return rows matching *name* (case-insensitive partial match)."""
    df, name_col = meta["df"], meta["name_col"]
    if name_col is None:
        return pd.DataFrame()
    mask = df[name_col].astype(str).str.lower().str.contains(
        re.escape(name.lower()), na=False
    )
    return df[mask].reset_index(drop=True)


def get_student_subjects(meta: dict, row: pd.Series) -> list[str]:
    """
    Return only the subjects relevant to this student's department.
    Falls back to all subject_cols if mapping unavailable.
    """
    dept_col         = meta.get("dept_col")
    dept_subject_map = meta.get("dept_subject_map", {})
    all_scols        = meta.get("subject_cols", [])

    if dept_col and dept_col in row.index:
        dept = str(row[dept_col])
        if dept in dept_subject_map and dept_subject_map[dept]:
            return dept_subject_map[dept]

    # Fallback — return all subjects
    return all_scols
