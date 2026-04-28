"""
tools.py — Agent tools (function calling).
Each tool receives the shared *meta* dict and returns a JSON-serialisable result.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from data_preprocessing import get_student_row
from utils import safe_mean

logger = logging.getLogger("university_agent.tools")


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_summary",
            "description": "Returns a high-level summary of the uploaded dataset.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_total_students",
            "description": "Returns the total number of students in the dataset.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_stats",
            "description": "Returns statistics for each department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Optional: specific department name to filter.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_students",
            "description": "Returns the top N students by average marks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of top students (default 5)."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_student",
            "description": "Search for a student by name and return their full profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Student name or partial name."}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subject_analysis",
            "description": "Returns average scores and top students per subject.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attendance_analysis",
            "description": "Analyses attendance distribution and low-attendance students.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Attendance % below which a student is flagged (default 75).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_student_performance",
            "description": "Predicts performance category for a student using a trained ML model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Student name."}
                },
                "required": ["name"],
            },
        },
    },
]


# ── Tool dispatcher ───────────────────────────────────────────────────────────
def call_tool(tool_name: str, arguments: dict, meta: dict, model=None) -> str:
    """
    Dispatch a tool call and return a JSON string result.
    *meta*  – preprocessed dataset metadata
    *model* – optional trained ML model (from model.py)
    """
    try:
        if tool_name == "get_dataset_summary":
            return json.dumps(get_dataset_summary(meta))
        elif tool_name == "get_total_students":
            return json.dumps(get_total_students(meta))
        elif tool_name == "get_department_stats":
            return json.dumps(get_department_stats(meta, arguments.get("department")))
        elif tool_name == "get_top_students":
            return json.dumps(get_top_students(meta, int(arguments.get("n", 5))))
        elif tool_name == "search_student":
            return json.dumps(search_student(meta, arguments.get("name", "")))
        elif tool_name == "get_subject_analysis":
            return json.dumps(get_subject_analysis(meta))
        elif tool_name == "get_attendance_analysis":
            return json.dumps(get_attendance_analysis(meta, float(arguments.get("threshold", 75))))
        elif tool_name == "predict_student_performance":
            return json.dumps(predict_student_performance(meta, arguments.get("name", ""), model))
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        return json.dumps({"error": str(exc)})


# ── Individual tool implementations ──────────────────────────────────────────
def get_dataset_summary(meta: dict) -> dict:
    df = meta["df"]
    result = {
        "total_students": meta["n_students"],
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "n_departments": meta["n_departments"],
        "departments": meta["departments"],
        "subject_columns": meta["subject_cols"],
        "has_attendance": meta["has_attendance"],
        "has_subjects": meta["has_subjects"],
    }
    if meta["has_subjects"] and meta["subject_cols"]:
        result["average_marks"] = round(float(df[meta["subject_cols"]].mean().mean()), 2)
    if meta["has_attendance"] and meta["attend_col"]:
        result["average_attendance"] = round(float(df[meta["attend_col"]].mean()), 2)
    if "Average" in df.columns:
        result["class_average"] = round(float(df["Average"].mean()), 2)
    return result


def get_total_students(meta: dict) -> dict:
    return {"total_students": meta["n_students"]}


def get_department_stats(meta: dict, department: str | None = None) -> dict:
    df = meta["df"]
    dept_col = meta.get("dept_col")
    if not dept_col:
        return {"error": "No department column detected."}

    grouped = df.groupby(dept_col)
    stats = {}
    for dept, grp in grouped:
        if department and str(department).lower() not in str(dept).lower():
            continue
        entry: dict[str, Any] = {"count": len(grp)}
        if meta["subject_cols"]:
            entry["avg_marks"] = round(float(grp[meta["subject_cols"]].mean().mean()), 2)
        if meta["attend_col"]:
            entry["avg_attendance"] = round(float(grp[meta["attend_col"]].mean()), 2)
        if "Grade" in grp.columns:
            entry["grade_dist"] = grp["Grade"].value_counts().to_dict()
        stats[str(dept)] = entry

    if not stats:
        return {"error": f"Department '{department}' not found.", "available": meta["departments"]}
    return {"departments": stats}


def get_top_students(meta: dict, n: int = 5) -> dict:
    df = meta["df"]
    sort_col = "Average" if "Average" in df.columns else (
        meta["subject_cols"][0] if meta["subject_cols"] else None
    )
    if sort_col is None:
        return {"error": "No numeric marks columns found."}

    top = df.nlargest(n, sort_col)
    name_col = meta.get("name_col") or df.columns[0]
    result = []
    for _, row in top.iterrows():
        entry = {
            "name": str(row.get(name_col, "?")),
            sort_col: round(float(row[sort_col]), 2),
        }
        if meta.get("dept_col"):
            entry["department"] = str(row.get(meta["dept_col"], "?"))
        if meta.get("attend_col"):
            entry["attendance"] = round(float(row.get(meta["attend_col"], 0)), 2)
        result.append(entry)
    return {"top_students": result}


def search_student(meta: dict, name: str) -> dict:
    if not name:
        return {"error": "No name provided."}
    rows = get_student_row(meta, name)
    if rows.empty:
        return {"error": f"No student found matching '{name}'."}
    records = rows.head(3).to_dict(orient="records")
    # clean numpy types
    clean = []
    for r in records:
        clean.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                      for k, v in r.items()})
    return {"students": clean, "count": len(rows)}


def get_subject_analysis(meta: dict) -> dict:
    df = meta["df"]
    scols = meta["subject_cols"]
    if not scols:
        return {"error": "No subject/marks columns detected."}

    analysis = {}
    name_col = meta.get("name_col")
    for sc in scols:
        col_data = df[sc].dropna()
        top_idx = col_data.nlargest(3).index
        top_students = []
        if name_col:
            for i in top_idx:
                top_students.append({
                    "name": str(df.at[i, name_col]),
                    "score": round(float(df.at[i, sc]), 2),
                })
        analysis[sc] = {
            "average": round(float(col_data.mean()), 2),
            "max": round(float(col_data.max()), 2),
            "min": round(float(col_data.min()), 2),
            "std": round(float(col_data.std()), 2),
            "top_3": top_students,
        }
    return {"subjects": analysis}


def get_attendance_analysis(meta: dict, threshold: float = 75.0) -> dict:
    df = meta["df"]
    attend_col = meta.get("attend_col")
    if not attend_col:
        return {"error": "No attendance column detected."}

    att = df[attend_col].dropna()
    low = df[att < threshold]
    result = {
        "average_attendance": round(float(att.mean()), 2),
        "max_attendance": round(float(att.max()), 2),
        "min_attendance": round(float(att.min()), 2),
        "students_below_threshold": len(low),
        "threshold_used": threshold,
    }
    # distribution bands
    result["distribution"] = {
        "excellent (>=90%)": int((att >= 90).sum()),
        "good (75-90%)": int(((att >= 75) & (att < 90)).sum()),
        "low (<75%)": int((att < 75).sum()),
    }
    if meta.get("dept_col") and not low.empty:
        result["low_attendance_by_dept"] = (
            low[meta["dept_col"]].value_counts().head(5).to_dict()
        )
    return result


def predict_student_performance(meta: dict, name: str, model=None) -> dict:
    if not name:
        return {"error": "No name provided."}
    rows = get_student_row(meta, name)
    if rows.empty:
        return {"error": f"Student '{name}' not found."}
    row = rows.iloc[0]

    if model is None:
        # Rule-based prediction
        avg = float(row.get("Average", 0))
        att = float(row.get(meta.get("attend_col", ""), 0)) if meta.get("attend_col") else 75
        if avg >= 80 and att >= 80:
            pred = "Excellent"
        elif avg >= 65 and att >= 70:
            pred = "Good"
        elif avg >= 50:
            pred = "Average"
        else:
            pred = "At Risk"
        return {
            "student": str(row.get(meta.get("name_col", row.index[0]), "?")),
            "predicted_performance": pred,
            "average_marks": round(avg, 2),
            "method": "rule-based",
        }

    # ML model
    try:
        features = model.get("feature_cols", [])
        if not features:
            return {"error": "Model has no feature columns."}
        X = row[features].values.reshape(1, -1)
        pred_val = model["estimator"].predict(X)[0]
        return {
            "student": str(row.get(meta.get("name_col", row.index[0]), "?")),
            "predicted_value": round(float(pred_val), 2),
            "method": "ML regression",
        }
    except Exception as exc:
        return {"error": str(exc)}
