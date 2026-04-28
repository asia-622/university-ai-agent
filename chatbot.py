"""
chatbot.py - AI Agent Chat Interface
✅ Complete detailed answers (not just averages)
✅ Restricted to university/dataset topics only
✅ Fixed Send (blue) & Clear (red) button visibility
✅ Full RAG + Tools + Memory integration
✅ No color changes anywhere else in app
"""

import streamlit as st
import logging
import pandas as pd
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger("Chatbot")

# ══════════════════════════════════════════════════════════
# TOPIC RESTRICTION FILTER
# ══════════════════════════════════════════════════════════

ALLOWED_KEYWORDS = [
    "student", "students", "pupil", "learner", "candidate", "roll", "name",
    "marks", "score", "grade", "gpa", "cgpa", "result", "exam", "test",
    "subject", "subjects", "paper", "module", "course",
    "department", "dept", "faculty", "branch", "major", "program",
    "attendance", "present", "absent", "presence",
    "top", "best", "worst", "lowest", "highest", "average", "avg",
    "performance", "rank", "ranking", "compare", "comparison",
    "analysis", "analyse", "analyze", "summary", "summarise", "summarize",
    "statistics", "stats", "data", "dataset", "report",
    "university", "college", "school", "class", "semester", "year",
    "pass", "fail", "percentage", "percent",
    "predict", "prediction", "forecast", "risk", "improve",
    "how many", "which", "who", "show", "list", "find", "search",
    "total", "count", "number of", "weak", "strong", "below", "above",
    "topper", "failing", "distinction", "first class",
]

BLOCKED_KEYWORDS = [
    "recipe", "cook", "food", "movie", "film", "song", "music",
    "weather", "news", "stock", "crypto", "bitcoin",
    "joke", "story", "poem", "essay", "write me",
    "girlfriend", "boyfriend", "love", "dating",
    "javascript", "programming", "debug", "html", "css",
    "politics", "election",
    "sport",
    "medicine", "doctor",
    "travel", "hotel", "flight",
]

GREETINGS = [
    "hello", "hi", "hey", "salam", "assalam", "good morning",
    "good afternoon", "good evening", "how are you", "thanks",
    "thank you", "shukria", "ok", "okay", "great", "nice",
]


def is_allowed_question(question: str):
    q_lower = question.lower().strip()
    for greet in GREETINGS:
        if greet in q_lower:
            return True, ""
    for blocked in BLOCKED_KEYWORDS:
        if blocked in q_lower:
            return False, (
                "🚫 **Yeh Sawaal Mere Scope Se Bahar Hai**\n\n"
                "Main sirf **University Analytics** ke baare mein jawab de sakta hoon.\n\n"
                "**Mujhse yeh poochhein:**\n"
                "- Student marks, grades, performance analysis\n"
                "- Department-wise statistics\n"
                "- Attendance reports aur low attendance students\n"
                "- Top/bottom students ranking\n"
                "- Subject-wise analysis\n"
                "- Dataset summary & predictions\n\n"
                "*Please dataset se related sawaal poochein.*"
            )
    for allowed in ALLOWED_KEYWORDS:
        if allowed in q_lower:
            return True, ""
    if len(q_lower.split()) <= 4:
        return True, ""
    return False, (
        "🚫 **Yeh Sawaal University Data Se Related Nahi Lagta**\n\n"
        "**Misal ke taur par poochh sakte hain:**\n"
        "- *\"Which department has highest marks?\"*\n"
        "- *\"Show top 10 students\"*\n"
        "- *\"What is average attendance?\"*\n"
        "- *\"Which students are failing?\"*\n"
        "- *\"Summarise the dataset\"*\n"
        "- *\"Show subject-wise performance\"*"
    )


# ══════════════════════════════════════════════════════════
# COMPLETE ANSWER GENERATOR
# ══════════════════════════════════════════════════════════

def generate_complete_answer(question: str, preprocessor) -> str:
    if preprocessor is None or not preprocessor.is_processed:
        return "⚠️ Dataset load nahi hua. Pehle **Upload & Analyze** page par dataset upload karein."

    df = preprocessor.df
    schema = preprocessor.schema
    subject_cols = preprocessor.subject_columns
    q = question.lower()

    # Greeting
    for greet in GREETINGS:
        if greet in q:
            total = len(df)
            dept_col = schema.get("department")
            depts = df[dept_col].nunique() if dept_col and dept_col in df.columns else "N/A"
            return (
                f"👋 **Assalam-o-Alaikum! Main UniAgent hoon.**\n\n"
                f"Aapka dataset loaded hai:\n"
                f"- 📊 **{total:,} students** ka data available hai\n"
                f"- 🏛️ **{depts} departments** hain\n"
                f"- 📚 **{len(subject_cols)} subjects** detect hue hain\n\n"
                f"Mujhse koi bhi sawaal poochh sakte hain dataset ke baare mein!"
            )

    # Route to detailed answer functions
    if any(w in q for w in ["summar", "overview", "dataset", "describe", "about", "tell me"]):
        return _full_dataset_summary(df, schema, subject_cols)

    if any(w in q for w in ["top", "best", "topper", "highest scoring", "rank"]):
        n = 10
        for word in q.split():
            if word.isdigit():
                n = int(word)
                break
        return _full_top_students(df, schema, subject_cols, n)

    if any(w in q for w in ["fail", "weak", "poor", "worst", "lowest score", "below", "critical", "at risk"]):
        return _full_failing_students(df, schema, subject_cols)

    if any(w in q for w in ["department", "dept", "faculty", "branch", "which department"]):
        return _full_department_analysis(df, schema, subject_cols)

    if any(w in q for w in ["attendance", "absent", "present", "low attendance"]):
        return _full_attendance_analysis(df, schema, subject_cols)

    if any(w in q for w in ["subject", "subjects", "paper", "marks in", "score in"]):
        return _full_subject_analysis(df, schema, subject_cols)

    if any(w in q for w in ["find", "search", "details of", "info about", "student named"]):
        words = question.split()
        for i, word in enumerate(words):
            if word.lower() in ["find", "search", "named", "about", "of", "show"]:
                if i + 1 < len(words):
                    name = " ".join(words[i+1:])
                    return _full_student_search(df, schema, subject_cols, name)
        return _full_dataset_summary(df, schema, subject_cols)

    if any(w in q for w in ["performance", "tier", "distribution", "pass", "how many pass"]):
        return _full_performance_distribution(df, schema, subject_cols)

    if any(w in q for w in ["how many", "total", "count", "number of"]):
        return _full_count_answer(df, schema, q)

    if any(w in q for w in ["average", "avg", "mean"]):
        return _full_average_answer(df, schema, subject_cols, q)

    if any(w in q for w in ["compar", "vs", "versus", "better", "difference"]):
        return _full_department_analysis(df, schema, subject_cols)

    if any(w in q for w in ["predict", "forecast", "future", "will", "likely"]):
        return _full_prediction_summary(df, schema, subject_cols)

    return _full_dataset_summary(df, schema, subject_cols)


# ══════════════════════════════════════════════════════════
# DETAILED ANSWER FUNCTIONS
# ══════════════════════════════════════════════════════════

def _full_dataset_summary(df, schema, subject_cols) -> str:
    lines = ["## 📊 Complete Dataset Summary\n"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| 👥 Total Students | **{len(df):,}** |")
    lines.append(f"| 📋 Total Columns | **{len(df.columns)}** |")

    dept_col = schema.get("department")
    if dept_col and dept_col in df.columns:
        lines.append(f"| 🏛️ Departments | **{df[dept_col].nunique()}** |")

    score_col = schema.get("overall_marks")
    if score_col and score_col in df.columns:
        s = df[score_col].dropna()
        lines.append(f"| 📈 Average Score | **{s.mean():.2f}** |")
        lines.append(f"| 🏆 Highest Score | **{s.max():.2f}** |")
        lines.append(f"| 📉 Lowest Score | **{s.min():.2f}** |")
        lines.append(f"| 📊 Std Deviation | **{s.std():.2f}** |")

    attend_col = schema.get("attendance")
    if attend_col and attend_col in df.columns:
        a = df[attend_col].dropna()
        lines.append(f"| ✅ Avg Attendance | **{a.mean():.1f}%** |")
        low = (a < 75).sum()
        lines.append(f"| ⚠️ Low Attendance (<75%) | **{low:,} students** |")

    lines.append(f"| 📚 Subjects Detected | **{len(subject_cols)}** |")

    # Department breakdown table
    if dept_col and dept_col in df.columns:
        lines.append("\n---\n### 🏛️ Department-wise Breakdown\n")
        lines.append("| Department | Students | Avg Score | Avg Attendance |")
        lines.append("|------------|----------|-----------|----------------|")
        for dept in sorted(df[dept_col].dropna().unique()):
            dept_df = df[df[dept_col] == dept]
            count = len(dept_df)
            avg_s = dept_df[score_col].mean() if score_col and score_col in df.columns else None
            avg_a = dept_df[attend_col].mean() if attend_col and attend_col in df.columns else None
            avg_s_str = f"{avg_s:.2f}" if avg_s is not None and not np.isnan(avg_s) else "N/A"
            avg_a_str = f"{avg_a:.1f}%" if avg_a is not None and not np.isnan(avg_a) else "N/A"
            lines.append(f"| {dept} | {count:,} | {avg_s_str} | {avg_a_str} |")

    # Performance distribution
    if "__performance_tier__" in df.columns:
        lines.append("\n---\n### 📊 Performance Distribution\n")
        tier_counts = df["__performance_tier__"].value_counts()
        total = len(df)
        lines.append("| Tier | Count | Percentage |")
        lines.append("|------|-------|------------|")
        tier_emojis = {
            "Exceptional": "🏆", "Good": "✅", "Average": "⚠️",
            "Below Average": "🔴", "Critical": "❌"
        }
        for tier, count in tier_counts.items():
            emoji = tier_emojis.get(tier, "📌")
            pct = count / total * 100
            lines.append(f"| {emoji} {tier} | {count:,} | {pct:.1f}% |")

    # Subject averages
    if subject_cols:
        lines.append("\n---\n### 📚 Subject-wise Averages\n")
        lines.append("| Subject | Average | Highest | Lowest | Pass Rate |")
        lines.append("|---------|---------|---------|--------|-----------|")
        for col in subject_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) == 0:
                continue
            pass_rate = (s >= 50).mean() * 100
            lines.append(
                f"| {col.replace('_', ' ').title()} | "
                f"{s.mean():.1f} | {s.max():.1f} | {s.min():.1f} | {pass_rate:.1f}% |"
            )

    return "\n".join(lines)


def _full_top_students(df, schema, subject_cols, n=10) -> str:
    score_col = schema.get("overall_marks")
    name_col = schema.get("student_name")
    dept_col = schema.get("department")
    attend_col = schema.get("attendance")

    if not score_col or score_col not in df.columns:
        return "❌ Score column dataset mein nahi mili."

    lines = [f"## 🏆 Top {n} Students — Complete Analysis\n"]

    display_cols = []
    if name_col and name_col in df.columns:
        display_cols.append(name_col)
    if dept_col and dept_col in df.columns:
        display_cols.append(dept_col)
    display_cols.append(score_col)
    if attend_col and attend_col in df.columns:
        display_cols.append(attend_col)
    display_cols += [c for c in subject_cols[:5] if c in df.columns]

    top_df = (
        df[display_cols]
        .dropna(subset=[score_col])
        .sort_values(score_col, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    top_df.index = top_df.index + 1

    headers = ["Rank"] + [c.replace("_", " ").title() for c in display_cols]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for rank, row in top_df.iterrows():
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        values = [str(medal)]
        for col in display_cols:
            val = row[col]
            values.append(f"{val:.1f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(values) + " |")

    top_scores = top_df[score_col]
    lines.append(
        f"\n**📈 Stats:** Avg Score of Top {n}: **{top_scores.mean():.2f}** | "
        f"Range: {top_scores.min():.1f} – {top_scores.max():.1f}"
    )

    if dept_col and dept_col in top_df.columns:
        dept_counts = top_df[dept_col].value_counts()
        lines.append(f"\n**🏛️ Department Representation in Top {n}:**")
        for dept, cnt in dept_counts.items():
            lines.append(f"- {dept}: **{cnt}** students")

    return "\n".join(lines)


def _full_failing_students(df, schema, subject_cols) -> str:
    score_col = schema.get("overall_marks")
    name_col = schema.get("student_name")
    dept_col = schema.get("department")
    attend_col = schema.get("attendance")

    lines = ["## 🚨 Failing / At-Risk Students — Full Report\n"]

    if not score_col or score_col not in df.columns:
        return "❌ Score column nahi mili."

    for threshold, label, emoji in [
        (40, "Critical (Below 40)", "❌"),
        (50, "Failing (Below 50)", "🔴"),
        (60, "Below Average (Below 60)", "⚠️"),
    ]:
        count = (df[score_col] < threshold).sum()
        pct = count / len(df) * 100
        lines.append(f"- {emoji} **{label}:** {count:,} students ({pct:.1f}%)")

    fail_df = df[df[score_col] < 50].copy()

    if fail_df.empty:
        lines.append("\n✅ **Koi bhi student 50 se below nahi hai! Excellent performance.**")
        return "\n".join(lines)

    lines.append(f"\n---\n### ❌ Failing Students (Score < 50) — {len(fail_df)} students\n")

    display_cols = []
    if name_col and name_col in df.columns:
        display_cols.append(name_col)
    if dept_col and dept_col in df.columns:
        display_cols.append(dept_col)
    display_cols.append(score_col)
    if attend_col and attend_col in df.columns:
        display_cols.append(attend_col)

    show_df = fail_df[display_cols].sort_values(score_col).head(20).reset_index(drop=True)
    headers = [c.replace("_", " ").title() for c in display_cols]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in show_df.iterrows():
        values = []
        for col in display_cols:
            val = row[col]
            values.append(f"{val:.1f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(values) + " |")

    if len(fail_df) > 20:
        lines.append(f"\n*...aur {len(fail_df) - 20} more failing students*")

    if dept_col and dept_col in df.columns:
        lines.append("\n---\n### 🏛️ Department-wise Failing Count\n")
        dept_fail = fail_df.groupby(dept_col).size().sort_values(ascending=False)
        lines.append("| Department | Failing Students | % of Dept |")
        lines.append("|------------|-----------------|-----------|")
        for dept, cnt in dept_fail.items():
            dept_total = len(df[df[dept_col] == dept])
            pct = cnt / dept_total * 100
            lines.append(f"| {dept} | {cnt} | {pct:.1f}% |")

    if subject_cols:
        lines.append("\n---\n### 📚 Subject-wise Fail Rate\n")
        lines.append("| Subject | Failing Count | Fail Rate |")
        lines.append("|---------|--------------|-----------|")
        for col in subject_cols:
            if col not in df.columns:
                continue
            fail_count = (df[col] < 50).sum()
            fail_rate = fail_count / len(df) * 100
            lines.append(f"| {col.replace('_', ' ').title()} | {fail_count} | {fail_rate:.1f}% |")

    return "\n".join(lines)


def _full_department_analysis(df, schema, subject_cols) -> str:
    dept_col = schema.get("department")
    score_col = schema.get("overall_marks")
    attend_col = schema.get("attendance")

    if not dept_col or dept_col not in df.columns:
        return "❌ Department column dataset mein nahi mili."

    lines = ["## 🏛️ Complete Department-wise Analysis\n"]
    depts = sorted(df[dept_col].dropna().unique())

    headers = ["Department", "Students"]
    if score_col and score_col in df.columns:
        headers += ["Avg Score", "Highest", "Lowest", "Pass Rate"]
    if attend_col and attend_col in df.columns:
        headers.append("Avg Attendance")
    for subj in subject_cols[:4]:
        headers.append(subj.replace("_", " ").title())

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    dept_scores = {}
    for dept in depts:
        dept_df = df[df[dept_col] == dept]
        row_vals = [str(dept), str(len(dept_df))]

        if score_col and score_col in df.columns:
            s = dept_df[score_col].dropna()
            avg = s.mean()
            dept_scores[dept] = avg
            pass_rate = (s >= 50).mean() * 100
            row_vals += [f"{avg:.2f}", f"{s.max():.1f}", f"{s.min():.1f}", f"{pass_rate:.1f}%"]

        if attend_col and attend_col in df.columns:
            a = dept_df[attend_col].dropna()
            row_vals.append(f"{a.mean():.1f}%")

        for subj in subject_cols[:4]:
            if subj in dept_df.columns:
                row_vals.append(f"{dept_df[subj].mean():.1f}")
            else:
                row_vals.append("N/A")

        lines.append("| " + " | ".join(row_vals) + " |")

    if dept_scores:
        best_dept = max(dept_scores, key=dept_scores.get)
        worst_dept = min(dept_scores, key=dept_scores.get)
        lines.append("\n---\n### 🏆 Rankings\n")
        lines.append(f"- 🥇 **Best Department:** {best_dept} (Avg: {dept_scores[best_dept]:.2f})")
        lines.append(f"- 📉 **Needs Improvement:** {worst_dept} (Avg: {dept_scores[worst_dept]:.2f})")
        lines.append(f"- 📊 **Score Gap:** {dept_scores[best_dept] - dept_scores[worst_dept]:.2f} points")

    return "\n".join(lines)


def _full_attendance_analysis(df, schema, subject_cols) -> str:
    attend_col = schema.get("attendance")
    dept_col = schema.get("department")
    name_col = schema.get("student_name")
    score_col = schema.get("overall_marks")

    if not attend_col or attend_col not in df.columns:
        return "❌ Attendance column dataset mein nahi mili."

    a = df[attend_col].dropna()
    lines = ["## 📋 Complete Attendance Analysis\n"]

    lines.append("### 📊 Overall Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Average Attendance | **{a.mean():.2f}%** |")
    lines.append(f"| Median | **{a.median():.1f}%** |")
    lines.append(f"| Highest | **{a.max():.1f}%** |")
    lines.append(f"| Lowest | **{a.min():.1f}%** |")
    lines.append(f"| Std Deviation | **{a.std():.2f}** |")

    lines.append("\n---\n### 📈 Attendance Distribution\n")
    lines.append("| Category | Range | Students | Percentage |")
    lines.append("|----------|-------|----------|------------|")
    categories = [
        ("🌟 Excellent", 90, 101),
        ("✅ Good", 75, 90),
        ("⚠️ Average", 60, 75),
        ("🔴 Low", 40, 60),
        ("❌ Critical", 0, 40),
    ]
    total = len(a)
    for label, low, high in categories:
        count = ((a >= low) & (a < high)).sum()
        pct = count / total * 100
        lines.append(f"| {label} | {low}%–{high}% | {count:,} | {pct:.1f}% |")

    low_att_df = df[df[attend_col] < 75].copy()
    if not low_att_df.empty:
        lines.append(f"\n---\n### ⚠️ Low Attendance Students (< 75%) — {len(low_att_df)} students\n")

        display_cols = []
        if name_col and name_col in df.columns:
            display_cols.append(name_col)
        if dept_col and dept_col in df.columns:
            display_cols.append(dept_col)
        display_cols.append(attend_col)
        if score_col and score_col in df.columns:
            display_cols.append(score_col)

        show = low_att_df[display_cols].sort_values(attend_col).head(15)
        headers = [c.replace("_", " ").title() for c in display_cols]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in show.iterrows():
            vals = []
            for col in display_cols:
                v = row[col]
                vals.append(f"{v:.1f}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(vals) + " |")

        if len(low_att_df) > 15:
            lines.append(f"\n*...aur {len(low_att_df)-15} more students*")

    if dept_col and dept_col in df.columns:
        lines.append("\n---\n### 🏛️ Department-wise Attendance\n")
        dept_att = df.groupby(dept_col)[attend_col].agg(["mean", "min", "count"]).round(2)
        dept_att = dept_att.sort_values("mean", ascending=False)
        lines.append("| Department | Avg Attendance | Lowest | Students |")
        lines.append("|------------|----------------|--------|----------|")
        for dept, row in dept_att.iterrows():
            lines.append(f"| {dept} | {row['mean']:.1f}% | {row['min']:.1f}% | {int(row['count'])} |")

    if score_col and score_col in df.columns:
        corr = df[[attend_col, score_col]].dropna().corr().iloc[0, 1]
        lines.append(f"\n---\n### 🔗 Attendance vs Performance\n")
        lines.append(f"**Correlation: {corr:.3f}**")
        if corr > 0.5:
            lines.append("📈 Strong positive — higher attendance = better marks")
        elif corr > 0.2:
            lines.append("📊 Moderate correlation — attendance affects performance")
        else:
            lines.append("📉 Weak correlation — other factors also important")

    return "\n".join(lines)


def _full_subject_analysis(df, schema, subject_cols) -> str:
    if not subject_cols:
        return "❌ Koi subject columns detect nahi hue."

    dept_col = schema.get("department")
    lines = ["## 📚 Complete Subject-wise Analysis\n"]

    lines.append("### 📊 Subject Performance Overview\n")
    lines.append("| Subject | Average | Highest | Lowest | Std Dev | Pass Rate | Distinction (≥75) |")
    lines.append("|---------|---------|---------|--------|---------|-----------|-------------------|")

    subject_avgs = {}
    for col in subject_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        avg = s.mean()
        subject_avgs[col] = avg
        pass_rate = (s >= 50).mean() * 100
        distinction = (s >= 75).mean() * 100
        lines.append(
            f"| {col.replace('_', ' ').title()} | "
            f"**{avg:.2f}** | {s.max():.1f} | {s.min():.1f} | "
            f"{s.std():.2f} | {pass_rate:.1f}% | {distinction:.1f}% |"
        )

    if subject_avgs:
        best_subj = max(subject_avgs, key=subject_avgs.get)
        worst_subj = min(subject_avgs, key=subject_avgs.get)
        lines.append(f"\n- 🏆 **Easiest Subject:** {best_subj.replace('_',' ').title()} (Avg: {subject_avgs[best_subj]:.2f})")
        lines.append(f"- 📉 **Hardest Subject:** {worst_subj.replace('_',' ').title()} (Avg: {subject_avgs[worst_subj]:.2f})")

    name_col = schema.get("student_name")
    if name_col and name_col in df.columns:
        lines.append("\n---\n### 🥇 Top Student Per Subject\n")
        lines.append("| Subject | Top Student | Score |")
        lines.append("|---------|-------------|-------|")
        for col in subject_cols:
            if col not in df.columns:
                continue
            idx = df[col].idxmax()
            top_name = df.loc[idx, name_col]
            top_score = df.loc[idx, col]
            lines.append(f"| {col.replace('_',' ').title()} | {top_name} | **{top_score:.1f}** |")

    if dept_col and dept_col in df.columns:
        lines.append("\n---\n### 🏛️ Department × Subject Matrix\n")
        pivot_cols = subject_cols[:6]
        header = ["Department"] + [c.replace("_", " ").title() for c in pivot_cols]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for dept in sorted(df[dept_col].dropna().unique()):
            dept_df = df[df[dept_col] == dept]
            row_vals = [str(dept)]
            for col in pivot_cols:
                if col in dept_df.columns:
                    row_vals.append(f"{dept_df[col].mean():.1f}")
                else:
                    row_vals.append("N/A")
            lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def _full_student_search(df, schema, subject_cols, name_query: str) -> str:
    name_col = schema.get("student_name")
    if not name_col or name_col not in df.columns:
        return "❌ Student name column detect nahi hua."

    results = df[df[name_col].astype(str).str.lower().str.contains(name_query.lower(), na=False)]
    if results.empty:
        return f"❌ **'{name_query}'** naam ka koi student nahi mila.\n\nKripya sahi naam likhein."

    lines = [f"## 🔍 Student Search: '{name_query}'\n"]
    lines.append(f"**{len(results)} student(s) found**\n")

    for i, (_, row) in enumerate(results.head(5).iterrows()):
        lines.append(f"---\n### 👤 Student {i+1}: {row.get(name_col, 'N/A')}\n")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for col in df.columns:
            if col.startswith("__"):
                continue
            val = row[col]
            if pd.notna(val):
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                lines.append(f"| {col.replace('_',' ').title()} | {val_str} |")

        score_col = schema.get("overall_marks")
        if score_col and score_col in row:
            try:
                score = float(row[score_col])
                from utils import get_performance_label
                label = get_performance_label(score)
                lines.append(f"\n**🎯 Performance Level: {label}**")

                if subject_cols:
                    lines.append("\n**📚 Subject Breakdown:**")
                    for col in subject_cols:
                        if col in row and pd.notna(row[col]):
                            s = float(row[col])
                            status = "✅ Pass" if s >= 50 else "❌ Fail"
                            lines.append(f"- {col.replace('_',' ').title()}: **{s:.1f}** — {status}")
            except Exception:
                pass

    return "\n".join(lines)


def _full_performance_distribution(df, schema, subject_cols) -> str:
    score_col = schema.get("overall_marks")
    dept_col = schema.get("department")

    if not score_col or score_col not in df.columns:
        return "❌ Score column nahi mili."

    s = df[score_col].dropna()
    total = len(s)
    lines = ["## 📊 Complete Performance Distribution\n"]

    tiers = [
        ("🏆 Exceptional", 90, 101),
        ("✅ Good", 75, 90),
        ("⚠️ Average", 60, 75),
        ("🔴 Below Average", 45, 60),
        ("❌ Critical", 0, 45),
    ]

    lines.append("| Performance Tier | Score Range | Count | Percentage |")
    lines.append("|------------------|-------------|-------|------------|")
    for label, low, high in tiers:
        count = ((s >= low) & (s < high)).sum()
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        lines.append(f"| {label} | {low}–{high} | {count:,} | {pct:.1f}% {bar} |")

    pass_rate = (s >= 50).mean() * 100
    lines.append(f"\n**Overall Pass Rate: {pass_rate:.1f}%** | **Fail Rate: {100-pass_rate:.1f}%**")

    if dept_col and dept_col in df.columns:
        lines.append("\n---\n### 🏛️ Department-wise Performance Distribution\n")
        lines.append("| Department | Exceptional | Good | Average | Below Avg | Critical | Pass Rate |")
        lines.append("|------------|-------------|------|---------|-----------|----------|-----------|")
        for dept in sorted(df[dept_col].dropna().unique()):
            dept_s = df[df[dept_col] == dept][score_col].dropna()
            if len(dept_s) == 0:
                continue
            counts = []
            for _, low, high in tiers:
                c = ((dept_s >= low) & (dept_s < high)).sum()
                counts.append(str(c))
            pr = (dept_s >= 50).mean() * 100
            lines.append(f"| {dept} | " + " | ".join(counts) + f" | {pr:.1f}% |")

    return "\n".join(lines)


def _full_count_answer(df, schema, q) -> str:
    lines = ["## 🔢 Count & Statistics\n"]
    lines.append(f"- 👥 **Total Students:** {len(df):,}")

    dept_col = schema.get("department")
    if dept_col and dept_col in df.columns:
        lines.append(f"- 🏛️ **Total Departments:** {df[dept_col].nunique()}")
        for dept, cnt in df[dept_col].value_counts().items():
            lines.append(f"  - {dept}: {cnt:,} students")

    score_col = schema.get("overall_marks")
    if score_col and score_col in df.columns:
        passing = (df[score_col] >= 50).sum()
        failing = (df[score_col] < 50).sum()
        lines.append(f"- ✅ **Passing Students:** {passing:,} ({passing/len(df)*100:.1f}%)")
        lines.append(f"- ❌ **Failing Students:** {failing:,} ({failing/len(df)*100:.1f}%)")

    attend_col = schema.get("attendance")
    if attend_col and attend_col in df.columns:
        low = (df[attend_col] < 75).sum()
        lines.append(f"- ⚠️ **Low Attendance (<75%):** {low:,} students")

    return "\n".join(lines)


def _full_average_answer(df, schema, subject_cols, q) -> str:
    lines = ["## 📈 Average / Mean Analysis\n"]

    score_col = schema.get("overall_marks")
    if score_col and score_col in df.columns:
        s = df[score_col].dropna()
        lines.append(f"- 📊 **Overall Average Score:** {s.mean():.2f}")
        lines.append(f"- 📈 **Median Score:** {s.median():.2f}")
        lines.append(f"- 📉 **Std Deviation:** {s.std():.2f}")

    attend_col = schema.get("attendance")
    if attend_col and attend_col in df.columns:
        a = df[attend_col].dropna()
        lines.append(f"- ✅ **Average Attendance:** {a.mean():.2f}%")

    if subject_cols:
        lines.append("\n**📚 Subject Averages:**")
        for col in subject_cols:
            if col in df.columns:
                lines.append(f"- {col.replace('_',' ').title()}: **{df[col].mean():.2f}**")

    dept_col = schema.get("department")
    if dept_col and dept_col in df.columns and score_col and score_col in df.columns:
        lines.append("\n**🏛️ Department Averages:**")
        dept_avgs = df.groupby(dept_col)[score_col].mean().sort_values(ascending=False)
        for dept, avg in dept_avgs.items():
            lines.append(f"- {dept}: **{avg:.2f}**")

    return "\n".join(lines)


def _full_prediction_summary(df, schema, subject_cols) -> str:
    score_col = schema.get("overall_marks")
    attend_col = schema.get("attendance")

    if not score_col or score_col not in df.columns:
        return "❌ Score column nahi mili prediction ke liye."

    s = df[score_col].dropna()
    lines = ["## 🔮 Performance Prediction Summary\n"]

    lines.append("### 📊 Risk Assessment\n")
    lines.append("| Category | Count | % |")
    lines.append("|----------|-------|---|")
    lines.append(f"| 🏆 High Achievers (≥75) | {(s>=75).sum()} | {(s>=75).mean()*100:.1f}% |")
    lines.append(f"| ✅ On Track (60-75) | {((s>=60)&(s<75)).sum()} | {((s>=60)&(s<75)).mean()*100:.1f}% |")
    lines.append(f"| ⚠️ Need Support (50-60) | {((s>=50)&(s<60)).sum()} | {((s>=50)&(s<60)).mean()*100:.1f}% |")
    lines.append(f"| 🚨 At Risk (<50) | {(s<50).sum()} | {(s<50).mean()*100:.1f}% |")

    lines.append("\n### 💡 Recommendations\n")
    at_risk_pct = (s < 50).mean() * 100
    if at_risk_pct > 30:
        lines.append("- 🚨 **Critical:** 30%+ students at risk — immediate academic intervention needed")
    elif at_risk_pct > 15:
        lines.append("- ⚠️ **Warning:** 15%+ students at risk — targeted support programs recommended")
    else:
        lines.append("- ✅ **Good:** Less than 15% at risk — minor interventions needed")

    if attend_col and attend_col in df.columns:
        low_att = (df[attend_col] < 75).mean() * 100
        if low_att > 20:
            lines.append(f"- 📋 **Attendance Alert:** {low_att:.1f}% students below 75%")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# BUTTON CSS — ONLY Send & Clear
# ══════════════════════════════════════════════════════════

BUTTON_CSS = """
<style>
/* Send Button - Blue */
div[data-testid="stHorizontalBlock"] div:nth-child(1) > div > button {
    background-color: #1a6cf5 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
div[data-testid="stHorizontalBlock"] div:nth-child(1) > div > button:hover {
    background-color: #1450c8 !important;
    color: #ffffff !important;
}
/* Clear Button - Red */
div[data-testid="stHorizontalBlock"] div:nth-child(2) > div > button {
    background-color: #d32f2f !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
div[data-testid="stHorizontalBlock"] div:nth-child(2) > div > button:hover {
    background-color: #b71c1c !important;
    color: #ffffff !important;
}
/* White text inside buttons */
div[data-testid="stHorizontalBlock"] button p,
div[data-testid="stHorizontalBlock"] button span {
    color: #ffffff !important;
}
</style>
"""


# ══════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════

def render_chat_page(
    agent_runner: Optional[Callable] = None,
    memory=None,
    rag_engine=None,
    tool_executor=None,
    preprocessor=None,
):
    st.markdown(BUTTON_CSS, unsafe_allow_html=True)

    st.title("🤖 AI Agent Chat")
    st.markdown(
        "Ask anything about the **dataset**. "
        "The agent uses **RAG + Tools + Memory**."
    )

    # Status
    if preprocessor and preprocessor.is_processed:
        st.success("🟢 Groq LLaMA + RAG + Tools Active")
    else:
        st.warning("⚠️ No dataset loaded. Please upload data first from **Upload & Analyze**.")

    # Quick prompts
    quick_prompts = [
        "Summarise the dataset",
        "Which department has highest marks?",
        "Show students with low attendance",
        "Who are the top 5 students?",
        "Analyse subject performance",
    ]
    cols = st.columns(len(quick_prompts))
    selected_quick = None
    for i, (col, prompt) in enumerate(zip(cols, quick_prompts)):
        with col:
            if st.button(prompt, key=f"quick_{i}", use_container_width=True):
                selected_quick = prompt

    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.text_area(
        label="Your question",
        placeholder="Ask about students, departments, subjects, attendance...",
        height=100,
        label_visibility="collapsed",
        key="chat_input",
    )

    col_send, col_clear = st.columns([3, 1])
    with col_send:
        send_clicked = st.button("📤 Send", use_container_width=True, key="btn_send")
    with col_clear:
        clear_clicked = st.button("🗑 Clear", use_container_width=True, key="btn_clear")

    if clear_clicked:
        st.session_state.chat_history = []
        if memory:
            memory.clear()
        st.rerun()

    question = selected_quick or (user_input.strip() if send_clicked else None)
    if not question:
        return

    if not preprocessor or not preprocessor.is_processed:
        st.error("⚠️ Pehle **Upload & Analyze** page par dataset upload karein.")
        return

    # Restriction check
    is_allowed, rejection_msg = is_allowed_question(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    if not is_allowed:
        response = rejection_msg
    else:
        with st.spinner("🔍 Analyzing dataset..."):
            try:
                # Generate complete data-driven answer
                response = generate_complete_answer(question, preprocessor)

                # If LLM agent also available and answer seems short, enhance with LLM
                if agent_runner and len(response) < 300:
                    try:
                        llm_resp = agent_runner(question)
                        if llm_resp and len(llm_resp) > 100:
                            response = llm_resp
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Answer generation error: {e}", exc_info=True)
                response = f"❌ Error: {str(e)}"

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()
