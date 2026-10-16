"""Бэкенд OpenRouter (OpenAI-совместимый chat completions API).

Позволяет использовать агента с ключом `sk-or-...` от https://openrouter.ai
и любой моделью с https://openrouter.ai/models (Gemini, Claude, GPT, DeepSeek …).

Стриминг через SSE, инструменты — в формате OpenAI function calling.
Транспорт: httpx, а при блокировке Cloudflare по TLS-отпечатку (403
"Access denied by security policy") автоматически переключается на curl.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Iterator

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


def _build_payload(model: str, messages: list[dict], tools: list[dict] | None, temperature: float) -> dict:
    payload: dict[str, Any] = {
        "model": normalize_model(model),
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def _parse_sse(lines: Iterator[str]) -> Iterator[tuple]:
    """Разобрать поток SSE в события ("text"/"done")."""
    text_parts: list[str] = []
    tool_acc: dict[int, dict] = {}

    for line in lines:
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


def _lines_httpx(api_key: str, payload: dict) -> Iterator[str]:
    """Строки SSE через httpx (быстрый путь)."""
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/FsbAidol/hzhz",
        "X-Title": "hzhz",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=httpx.Timeout(30.0, read=600.0)) as client:
        with client.stream("POST", OPENROUTER_URL, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = resp.read().decode(errors="replace")
                raise ApiError(resp.status_code, body)
            yield from resp.iter_lines()


def _lines_curl(api_key: str, payload: dict) -> Iterator[str]:
    """Строки SSE через curl (проходит Cloudflare, который режет TLS-отпечатки)."""
    cmd = [
        "curl", "-sS", "-N", "--no-buffer", "--fail-with-body", "--max-time", "600",
        "-X", "POST", OPENROUTER_URL,
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-H", "HTTP-Referer: https://github.com/FsbAidol/hzhz",
        "-H", "X-Title: hzhz",
        "--data-binary", "@-",
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    body_captured: list[str] = []
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False))
        proc.stdin.close()
        for line in proc.stdout:
            body_captured.append(line)
            yield line
    finally:
        proc.kill()
        proc.wait()
    stderr = (proc.stderr.read() if proc.stderr else "") or ""
    # --fail-with-body: при HTTP-ошибке (403 и т.п.) curl вернёт код 22, тело — в stdout
    if proc.returncode not in (0, -9):
        raise ApiError(proc.returncode, (stderr or "".join(body_captured))[:500])


def stream_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
) -> Iterator[tuple]:
    """Стримить ответ модели. Авто-фолбэк httpx → curl при блокировке Cloudflare.

    События:
      ("text", str)                    — дельта текста;
      ("done", assistant_msg, calls)   — финал: сообщение для истории и
                                         список вызовов [(id, name, args_dict), ...].
    """
    payload = _build_payload(model, messages, tools, temperature)

    yielded = False
    try:
        for event in _parse_sse(_lines_httpx(api_key, payload)):
            yielded = True
            yield event
        return
    except Exception:
        if yielded:
            raise  # частично ответ уже показан — не повторяем через curl

    # Cloudflare заблокировал httpx по TLS-отпечатку (403 security policy) —
    # повторяем через curl, чей отпечаток проходит.
    yield from _parse_sse(_lines_curl(api_key, payload))
