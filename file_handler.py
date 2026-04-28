"""
file_handler.py — Robust file ingestion for CSV / Excel / JSON
Supports large files (200 MB+) via chunked reading.
"""
from __future__ import annotations

import io
import json
import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger("university_agent.file_handler")

CHUNK_SIZE = 50_000   # rows per chunk for large CSVs


# ── Public entry-point ────────────────────────────────────────────────────────
def load_file(uploaded_file) -> pd.DataFrame | None:
    """
    Accept a Streamlit UploadedFile object and return a DataFrame.
    Returns None on unrecoverable error (displays st.error automatically).
    """
    if uploaded_file is None:
        return None

    name: str = uploaded_file.name.lower()
    raw: bytes = uploaded_file.read()

    logger.info("Loading file: %s  (%d bytes)", uploaded_file.name, len(raw))

    try:
        if name.endswith(".csv"):
            df = _load_csv(raw)
        elif name.endswith((".xlsx", ".xls")):
            df = _load_excel(raw, name)
        elif name.endswith(".json"):
            df = _load_json(raw)
        else:
            st.error("❌ Unsupported file type. Please upload CSV, Excel, or JSON.")
            return None
    except Exception as exc:
        logger.exception("File loading failed: %s", exc)
        st.error(f"❌ Failed to parse file: {exc}")
        return None

    if df is None or df.empty:
        st.error("❌ The uploaded file appears to be empty.")
        return None

    df = _clean_headers(df)
    _debug_info(df, uploaded_file.name)
    return df


# ── Format-specific loaders ───────────────────────────────────────────────────
def _load_csv(raw: bytes) -> pd.DataFrame:
    size_mb = len(raw) / (1024 ** 2)
    if size_mb > 50:                     # large file → chunk
        logger.info("Large CSV (%.1f MB) — reading in chunks", size_mb)
        chunks = pd.read_csv(
            io.BytesIO(raw),
            chunksize=CHUNK_SIZE,
            encoding="utf-8",
            on_bad_lines="skip",
            low_memory=False,
        )
        return pd.concat(list(chunks), ignore_index=True)
    # try utf-8, fall back to latin-1
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=enc,
                on_bad_lines="skip",
                low_memory=False,
            )
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot detect encoding for CSV file.")


def _load_excel(raw: bytes, name: str) -> pd.DataFrame:
    engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
    xf = pd.ExcelFile(io.BytesIO(raw), engine=engine)
    if len(xf.sheet_names) == 1:
        return xf.parse(xf.sheet_names[0])
    # multiple sheets → concatenate or pick largest
    dfs = []
    for sheet in xf.sheet_names:
        try:
            dfs.append(xf.parse(sheet))
        except Exception:
            pass
    if not dfs:
        raise ValueError("No readable sheets found in Excel file.")
    return max(dfs, key=lambda d: len(d))   # largest sheet


def _load_json(raw: bytes) -> pd.DataFrame:
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        # try records / columns orientations
        for orient in ("records", "columns", "index", "values"):
            try:
                return pd.read_json(io.BytesIO(raw), orient=orient)
            except Exception:
                continue
    raise ValueError("Cannot interpret JSON structure as a table.")


# ── Header normalisation ──────────────────────────────────────────────────────
def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]          # drop duplicate cols
    df = df.dropna(how="all").dropna(axis=1, how="all")  # drop blank rows/cols
    return df.reset_index(drop=True)


# ── Debug ─────────────────────────────────────────────────────────────────────
def _debug_info(df: pd.DataFrame, filename: str) -> None:
    logger.info("Loaded  '%s'  shape=%s", filename, df.shape)
    logger.info("dtypes:\n%s", df.dtypes.to_string())
    print(f"\n[DEBUG] File: {filename}")
    print(f"[DEBUG] Shape: {df.shape}")
    print(f"[DEBUG] dtypes:\n{df.dtypes}\n")
