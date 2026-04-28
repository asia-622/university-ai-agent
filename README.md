# 🎓 University AI Analytics Agent

A production-grade **AI Agent** web application built with Python and Streamlit, deployable on Hugging Face Spaces.

## Architecture

```
AI Agent = LLM (GPT-4) + RAG (FAISS) + Tools (8 functions) + Memory (buffer) + API (OpenAI) + UI (Streamlit)
```

## Features

| Module | Technology | Description |
|--------|-----------|-------------|
| LLM | GPT-4o-mini | Natural language understanding & generation |
| RAG | FAISS + OpenAI Embeddings (TF-IDF fallback) | Semantic retrieval over dataset |
| Tools | Function calling (8 tools) | Data queries, analysis, ML prediction |
| Memory | Buffer memory (last 20 messages) | Context-aware conversation |
| API | OpenAI API | Chat completions + embeddings |
| UI | Streamlit | 7-page responsive web interface |

## File Structure

```
├── app.py                  # Main Streamlit application (7 pages)
├── file_handler.py         # CSV/Excel/JSON ingestion (200MB+ support)
├── data_preprocessing.py   # Schema-agnostic auto-detection
├── model.py                # Linear Regression ML model
├── dashboard.py            # All Plotly chart builders
├── rag_engine.py           # FAISS/TF-IDF RAG engine
├── tools.py                # 8 agent tools (function calling)
├── memory.py               # Conversation buffer memory
├── chatbot.py              # Core agent loop (LLM + RAG + Tools)
├── utils.py                # Shared utilities & column detection
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── sample_university_data.csv  # Example dataset
```

## Quickstart

```bash
# 1. Clone / download the project
git clone <repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export OPENAI_API_KEY=sk-your-key-here
# OR create a .env file from .env.example

# 4. Run
streamlit run app.py
```

## Hugging Face Spaces Deployment

1. Create a new Space (SDK: Streamlit)
2. Upload all files
3. Add `OPENAI_API_KEY` as a Secret in Space settings
4. The app auto-starts

## Tools Available

| Tool | Description |
|------|-------------|
| `get_dataset_summary()` | Full dataset overview |
| `get_total_students()` | Count of students |
| `get_department_stats()` | Per-department analytics |
| `get_top_students(n)` | Top N students by marks |
| `search_student(name)` | Full student profile |
| `get_subject_analysis()` | Per-subject statistics |
| `get_attendance_analysis()` | Attendance patterns |
| `predict_student_performance(name)` | ML-based prediction |

## Supported File Formats

- **CSV** — up to 200MB+, auto-encoding detection
- **Excel** — .xlsx and .xls, multi-sheet support
- **JSON** — records/columns/index orientations

## Auto-detected Columns

The system automatically detects:
- Student name, roll number, department, year
- Attendance percentage
- Subject/marks columns (any numeric 0–150)
- Derived: Total, Average, Grade

## Sample Questions

- *"Which department has the highest average marks?"*
- *"Show students with attendance below 75%"*
- *"Who are the top 5 students in Mathematics?"*
- *"Predict performance for Alice Smith"*
- *"Compare Computer Science and Engineering departments"*
