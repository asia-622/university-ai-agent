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
        df            – cleaned DataFrame
        name_col      – str | None
        dept_col      – str | None
        attend_col    – str | None
        roll_col      – str | None
        year_col      – str | None
        subject_cols  – list[str]
        numeric_cols  – list[str]
        n_students    – int
        n_departments – int
        departments   – list[str]
        has_attendance – bool
        has_subjects   – bool
    """
    df = df.copy()
    df = _coerce_numeric(df)
    df = _fill_missing(df)

    meta: dict = {}

    # Detect semantic columns
    for field in ("student_name", "department", "attendance", "roll_no", "year"):
        meta[f"{field.replace('student_', '')}_col" if field != "student_name"
             else "name_col"] = detect_column(df, field)
    # fix naming
    meta["name_col"]   = detect_column(df, "student_name")
    meta["dept_col"]   = detect_column(df, "department")
    meta["attend_col"] = detect_column(df, "attendance")
    meta["roll_col"]   = detect_column(df, "roll_no")
    meta["year_col"]   = detect_column(df, "year")

    meta["subject_cols"]  = detect_subject_columns(df)
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
