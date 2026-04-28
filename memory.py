"""
memory.py — Conversation buffer memory for the AI agent.
Keeps the last N turns and exposes a formatted prompt block.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ROLE = Literal["user", "assistant", "system"]
DEFAULT_BUFFER = 12   # last N messages to keep in context


@dataclass
class Message:
    role: ROLE
    content: str


@dataclass
class ConversationMemory:
    max_messages: int = DEFAULT_BUFFER
    _history: list[Message] = field(default_factory=list)

    # ── Write ────────────────────────────────────────────────────────────────
    def add(self, role: ROLE, content: str) -> None:
        self._history.append(Message(role=role, content=content))
        # trim to buffer
        if len(self._history) > self.max_messages:
            # always keep system message at index 0 if present
            if self._history[0].role == "system":
                self._history = [self._history[0]] + self._history[-(self.max_messages - 1):]
            else:
                self._history = self._history[-self.max_messages:]

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def set_system(self, content: str) -> None:
        if self._history and self._history[0].role == "system":
            self._history[0] = Message(role="system", content=content)
        else:
            self._history.insert(0, Message(role="system", content=content))

    # ── Read ─────────────────────────────────────────────────────────────────
    def get_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._history]

    def get_recent(self, n: int = 6) -> list[dict]:
        recent = [m for m in self._history if m.role != "system"][-n:]
        sys_msg = [m for m in self._history if m.role == "system"]
        return [{"role": m.role, "content": m.content} for m in (sys_msg + recent)]

    def format_for_prompt(self) -> str:
        lines = []
        for m in self._history:
            if m.role == "system":
                continue
            tag = "User" if m.role == "user" else "Assistant"
            lines.append(f"{tag}: {m.content}")
        return "\n".join(lines[-10:])   # last 10 turns as plain text

    def clear(self) -> None:
        # preserve system message
        sys = [m for m in self._history if m.role == "system"]
        self._history = sys

    def __len__(self) -> int:
        return len(self._history)
