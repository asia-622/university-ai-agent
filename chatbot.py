"""
chatbot.py — Core AI Agent loop.
Combines: OpenAI LLM + RAG context + Tool calling + Conversation memory.
"""
from __future__ import annotations

import json
import logging
from typing import Generator

from memory import ConversationMemory
from rag_engine import RAGEngine
from tools import TOOL_SCHEMAS, call_tool
from utils import get_groq_key, truncate

logger = logging.getLogger("university_agent.chatbot")

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

SYSTEM_PROMPT = """You are a University AI Analytics Agent — an expert data analyst and academic advisor with access to a university dataset.

You have the following capabilities:
1. TOOLS: Call any of the registered tools to query the dataset directly.
2. RAG: You will receive relevant data snippets retrieved from the dataset.
3. MEMORY: You remember the conversation context.

Guidelines:
- Always use tools when the user asks for data-driven answers.
- Combine RAG context + tool results to give comprehensive answers.
- If data is missing, say so clearly.
- Format numbers neatly. Use bullet points for lists.
- Be concise but thorough. Avoid hallucinating data.
- When asked to compare or analyse, use multiple tools if needed.
"""


class UniversityAgent:
    def __init__(self, api_key: str | None = None):
        key = api_key or get_groq_key()
        self.client = Groq(api_key=key) if (key and GROQ_OK) else None
        self.memory = ConversationMemory(max_messages=20)
        self.rag: RAGEngine | None = None
        self.meta: dict | None = None
        self.model = None
        self._setup_memory()

    def _setup_memory(self) -> None:
        self.memory.set_system(SYSTEM_PROMPT)

    # ── Public API ────────────────────────────────────────────────────────────
    def attach_data(self, meta: dict, rag: RAGEngine, ml_model=None) -> None:
        self.meta = meta
        self.rag = rag
        self.model = ml_model

    def chat(self, user_message: str) -> str:
        """Send a user message; return agent response string."""
        if not self.client:
            return self._no_llm_response(user_message)

        # 1. Retrieve RAG context
        rag_context = ""
        if self.rag and self.rag.is_ready:
            rag_context = self.rag.format_context(user_message, top_k=6)

        # 2. Augment message with context
        augmented = user_message
        if rag_context and rag_context != "No relevant data found.":
            augmented = (
                f"{user_message}\n\n"
                f"[Retrieved dataset context]\n{truncate(rag_context, 1500)}"
            )

        self.memory.add_user(augmented)

        try:
            response = self._agent_loop(augmented)
        except Exception as exc:
            logger.exception("Agent loop error: %s", exc)
            response = f"⚠️ Error: {exc}"

        self.memory.add_assistant(response)
        return response

    # ── Agent loop (tool calling) ─────────────────────────────────────────────
    def _agent_loop(self, user_msg: str, max_rounds: int = 5) -> str:
        messages = self.memory.get_messages()

        for round_n in range(max_rounds):
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOL_SCHEMAS if self.meta else None,
                tool_choice="auto" if self.meta else None,
                temperature=0.3,
                max_tokens=1200,
            )
            choice = resp.choices[0]

            # No tool call → final answer
            if choice.finish_reason != "tool_calls":
                return choice.message.content or ""

            # Process tool calls
            tool_calls = choice.message.tool_calls
            messages.append(choice.message)  # append assistant turn with tool_calls

            for tc in tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                logger.info("Tool call: %s(%s)", fn_name, args)

                result = call_tool(fn_name, args, self.meta, self.model)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Fallback: ask model to summarise with available context
        messages.append({"role": "user", "content": "Please provide your best answer based on the tool results above."})
        final = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1200,
        )
        return final.choices[0].message.content or ""

    # ── Fallback when no OpenAI key ───────────────────────────────────────────
    def _no_llm_response(self, query: str) -> str:
        if self.meta is None:
            return "⚠️ Please upload a dataset first, then enter your Groq API key in the sidebar."

        q = query.lower()

        # Quick keyword routing to tools
        if any(w in q for w in ["summary", "overview", "about"]):
            r = json.loads(call_tool("get_dataset_summary", {}, self.meta))
        elif any(w in q for w in ["total", "how many", "count"]):
            r = json.loads(call_tool("get_total_students", {}, self.meta))
        elif any(w in q for w in ["department", "dept", "branch"]):
            r = json.loads(call_tool("get_department_stats", {}, self.meta))
        elif any(w in q for w in ["top", "best", "highest", "topper"]):
            r = json.loads(call_tool("get_top_students", {}, self.meta))
        elif any(w in q for w in ["attendance", "absent", "present"]):
            r = json.loads(call_tool("get_attendance_analysis", {}, self.meta))
        elif any(w in q for w in ["subject", "marks", "score"]):
            r = json.loads(call_tool("get_subject_analysis", {}, self.meta))
        else:
            # Try RAG
            if self.rag and self.rag.is_ready:
                ctx = self.rag.format_context(query, top_k=5)
                return f"📊 **Relevant data from dataset:**\n\n{ctx}\n\n*(Add a Groq API key in the sidebar for AI-powered responses.)*"
            return "🔑 Please add your Groq API key in the sidebar for intelligent responses."

        return f"```json\n{json.dumps(r, indent=2)}\n```\n*(Add a Groq API key for natural language answers.)*"

    # ── Helpers ───────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.memory.clear()
        self._setup_memory()

    @property
    def history(self) -> list[dict]:
        return [m for m in self.memory.get_messages() if m["role"] != "system"]
