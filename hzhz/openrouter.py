"""Бэкенд OpenRouter (OpenAI-совместимый chat completions API).

Позволяет использовать агента с ключом `sk-or-...` от https://openrouter.ai
и любой моделью с https://openrouter.ai/models (Gemini, Claude, GPT, DeepSeek …).

Стриминг через SSE, инструменты — в формате OpenAI function calling.
Зависимость: httpx (идёт в комплекте с google-genai, но указана явно).
"""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Сокращения моделей Gemini → имена на OpenRouter
_MODEL_ALIASES = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "claude-opus": "anthropic/claude-opus-4.1",
    "gpt": "openai/gpt-5",
    "deepseek": "deepseek/deepseek-chat",
}


class ApiError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


def normalize_model(name: str) -> str:
    """Привести имя модели к формату OpenRouter (vendor/model)."""
    name = (name or "").strip()
    if "/" in name:
        return name
    return _MODEL_ALIASES.get(name, f"google/{name}")


def to_openai_tools(tool_schemas: list[dict]) -> list[dict]:
    """Наши TOOL_SCHEMAS → формат OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in tool_schemas
    ]


def stream_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
) -> Iterator[tuple]:
    """Стримить ответ модели.

    События:
      ("text", str)                    — дельта текста;
      ("done", assistant_msg, calls)   — финал: сообщение для истории и
                                         список вызовов [(id, name, args_dict), ...].
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/FsbAidol/hzhz",
        "X-Title": "hzhz",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": normalize_model(model),
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    text_parts: list[str] = []
    tool_acc: dict[int, dict] = {}

    with httpx.Client(timeout=httpx.Timeout(30.0, read=600.0)) as client:
        with client.stream("POST", OPENROUTER_URL, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = resp.read().decode(errors="replace")
                raise ApiError(resp.status_code, body)
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise ApiError(200, json.dumps(chunk["error"], ensure_ascii=False))
                delta = ((chunk.get("choices") or [{}])[0]).get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield ("text", content)
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index") or 0)
                    acc = tool_acc.setdefault(idx, {"id": "", "name": "", "args": []})
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc["name"] = fn["name"]
                    if fn.get("arguments"):
                        acc["args"].append(fn["arguments"])

    calls: list[tuple[str, str, dict]] = []
    for idx in sorted(tool_acc):
        acc = tool_acc[idx]
        raw = "".join(acc["args"])
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            args = {"_raw": raw}
        if not isinstance(args, dict):
            args = {"value": args}
        calls.append((acc["id"] or f"call_{idx}", acc["name"] or "unknown", args))

    assistant_msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if calls:
        assistant_msg["tool_calls"] = [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
            for cid, name, args in calls
        ]
    yield ("done", assistant_msg, calls)
