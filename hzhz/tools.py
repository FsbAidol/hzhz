"""Инструменты агента: схемы для Gemini Function Calling + их реализация.

Реализация не зависит от SDK — агент сам диспатчит вызовы (для подтверждений
опасных действий пользователем), поэтому здесь обычные Python-функции.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Опасные паттерны: даже в режиме auto-approve требуют подтверждения
# ---------------------------------------------------------------------------
DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(p)
    for p in [
        r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+",          # rm -rf и т.п.
        r"\brm\s+.*\*",                                    # rm с шаблонами
        r"\bmkfs(\.[a-z0-9]+)?\b",
        r"\bdd\s+.*\bof=",
        r">\s*/dev/(sd|nvme|hd|loop|mem)",
        r"\b(shutdown|reboot|halt|poweroff)\b",
        r"\bmkswap\b",
        r"\b(chmod|chown)\s+(-R\s+)?\S+\s+/(home|root|etc|usr|var)?\s*$",
        r"curl\s[^|]*\|\s*(sudo\s+)?(ba)?sh",
        r"wget\s[^|]*\|\s*(sudo\s+)?(ba)?sh",
        r"\b(pacman|yay|paru)\s+(-\w+\s+)*(-R|--remove)\b",
        r"\b(pacman|yay|paru)\s+(-\w+\s+)*-Syu?\b",        # системное обновление
        r"\bparted\b|\bfdisk\b|\bgdisk\b|\bcfdisk\b",
        r"\bshred\b|\bwipe\b",
        r"\bkill\s+-9\s+-1\b|\bkillall\b|\bpkill\s",
        r"\bsystemctl\s+(stop|disable|mask|isolate)\b",
        r"\btruncate\b",
        r"\bpasswd\b|\bchpasswd\b|\buserdel\b|\busermod\b",
        r"\bhistory\s+-c\b",
        r":\(\)\s*{.*}.*;",                                 # fork-bomb
        r"\bdrop\s+table\b|\bdrop\s+database\b",
    ]
]

MAX_OUTPUT = 8000          # символов на поток вывода
MAX_MATCHES = 200
MAX_READ_LINES = 2000


def is_dangerous(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_PATTERNS)


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n... [обрезано {len(text) - limit} символов] ...\n"
        + text[-half:]
    )


# ---------------------------------------------------------------------------
# Реализации инструментов
# ---------------------------------------------------------------------------

def tool_run_command(args: dict, *, workspace: Path, timeout_default: int = 120) -> dict:
    command = str(args.get("command", "")).strip()
    if not command:
        return {"status": "error", "error": "пустая команда"}
    timeout = int(args.get("timeout") or timeout_default)
    timeout = max(1, min(timeout, 3600))
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            start_new_session=True,
            env={**os.environ, "PAGER": "cat", "SYSTEMD_PAGER": "cat", "GIT_PAGER": "cat"},
        )
        elapsed = round(time.monotonic() - started, 2)
        out = _truncate(proc.stdout or "")
        err = _truncate(proc.stderr or "")
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "stdout": out,
            "stderr": err,
            "duration_sec": elapsed,
        }
    except subprocess.TimeoutExpired as e:
        elapsed = round(time.monotonic() - started, 2)
        return {
            "status": "error",
            "error": f"таймаут {timeout} с",
            "stdout": _truncate((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")),
            "stderr": _truncate((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")),
            "duration_sec": elapsed,
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def tool_read_file(args: dict, **_: Any) -> dict:
    path = Path(args.get("path", "")).expanduser()
    if not path.is_file():
        return {"status": "error", "error": f"файл не найден: {path}"}
    try:
        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or MAX_READ_LINES)
        limit = max(1, min(limit, MAX_READ_LINES))
        lines = path.read_text(errors="replace").splitlines()
        chunk = lines[offset - 1 : offset - 1 + limit]
        numbered = "\n".join(f"{offset + i}→ {ln}" for i, ln in enumerate(chunk))
        return {
            "status": "ok",
            "path": str(path),
            "total_lines": len(lines),
            "shown_lines": len(chunk),
            "content": _truncate(numbered, 24000),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def tool_write_file(args: dict, **_: Any) -> dict:
    path = Path(args.get("path", "")).expanduser()
    content = args.get("content", "")
    if content is None:
        return {"status": "error", "error": "нет content"}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {"status": "ok", "path": str(path), "bytes": len(content.encode())}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def tool_edit_file(args: dict, **_: Any) -> dict:
    path = Path(args.get("path", "")).expanduser()
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    if not path.is_file():
        return {"status": "error", "error": f"файл не найден: {path}"}
    if not old_text:
        return {"status": "error", "error": "пустой old_text"}
    try:
        text = path.read_text(errors="replace")
        count = text.count(old_text)
        if count == 0:
            return {"status": "error", "error": "old_text не найден в файле"}
        if count > 1:
            return {
                "status": "error",
                "error": f"old_text встречается {count} раз(а) — сделай фрагмент уникальным",
            }
        path.write_text(text.replace(old_text, new_text, 1))
        return {"status": "ok", "path": str(path), "replaced": 1}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def tool_list_dir(args: dict, **_: Any) -> dict:
    path = Path(args.get("path") or ".").expanduser()
    if not path.exists():
        return {"status": "error", "error": f"нет такого пути: {path}"}
    if path.is_file():
        return {"status": "ok", "path": str(path), "entries": [f"[файл] {path.name} ({path.stat().st_size} B)"]}
    try:
        entries = []
        for p in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if p.is_symlink():
                kind = "ссылка"
            elif p.is_dir():
                kind = "каталог"
            else:
                kind = "файл"
            size = p.stat().st_size if p.is_file() else ""
            entries.append(f"[{kind}] {p.name}{f' ({size} B)' if size != '' else ''}")
            if len(entries) >= MAX_MATCHES:
                entries.append("... [список обрезан]")
                break
        return {"status": "ok", "path": str(path.resolve()), "entries": entries}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def tool_search_files(args: dict, **_: Any) -> dict:
    pattern = str(args.get("pattern", ""))
    path = Path(args.get("path") or ".").expanduser()
    if not pattern:
        return {"status": "error", "error": "пустой pattern"}
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"status": "error", "error": f"плохое регулярное выражение: {e}"}
    glob = str(args.get("file_glob") or "*")
    matches: list[str] = []
    scanned = 0
    try:
        root = path if path.is_dir() else path.parent
        for p in root.rglob(glob):
            if not p.is_file():
                continue
            if any(part in {".git", "node_modules", ".venv", "venv", "__pycache__", ".cache", "target", "dist", "build"} for part in p.parts):
                continue
            scanned += 1
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{p}:{i}: {line.strip()[:300]}")
                        if len(matches) >= MAX_MATCHES:
                            return {
                                "status": "ok",
                                "truncated": True,
                                "scanned_files": scanned,
                                "matches": matches + ["... [результаты обрезаны]"],
                            }
            except (OSError, UnicodeError):
                continue
        return {"status": "ok", "scanned_files": scanned, "matches": matches or ["совпадений нет"]}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def tool_current_time(args: dict, **_: Any) -> dict:
    tz_name = args.get("timezone") or os.environ.get("HZZ_TZ") or "Europe/Paris"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
        tz_name = "UTC"
    now = datetime.now(tz)
    return {
        "status": "ok",
        "timezone": tz_name,
        "iso": now.isoformat(timespec="seconds"),
        "human": now.strftime("%A, %d %B %Y, %H:%M:%S %Z"),
    }


def tool_ask_user(args: dict, *, ask: Callable[[str], str] | None = None, **_: Any) -> dict:
    question = str(args.get("question", "")).strip()
    if ask is None:
        return {"status": "error", "error": "интерактивный ввод недоступен"}
    answer = ask(question)
    return {"status": "ok", "answer": answer}


# ---------------------------------------------------------------------------
# Схемы инструментов (Function Declarations для Gemini)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "run_command",
        "description": (
            "Выполнить shell-команду в рабочей директории пользователя и вернуть stdout/stderr/код возврата. "
            "Используй для любых действий в системе: установка пакетов, git, сборка, запуск скриптов и т.д. "
            "Деструктивные команды требуют подтверждения пользователя."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "команда для выполнения (bash)"},
                "timeout": {"type": "integer", "description": "таймаут в секундах (по умолчанию 120)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Прочитать текстовый файл (показывает номера строк). Используй перед любыми правками.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "путь к файлу"},
                "offset": {"type": "integer", "description": "первая строка (с 1)"},
                "limit": {"type": "integer", "description": "максимум строк (до 2000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Создать или полностью перезаписать файл указанным содержимым. Родительские каталоги создаются автоматически.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "путь к файлу"},
                "content": {"type": "string", "description": "новое содержимое файла"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Точечная правка файла: заменить ровно одно вхождение old_text на new_text. "
            "Предпочитай этот инструмент полной перезаписи файла."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "путь к файлу"},
                "old_text": {"type": "string", "description": "точный фрагмент для замены (должен встречаться один раз)"},
                "new_text": {"type": "string", "description": "заменяющий фрагмент"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "list_dir",
        "description": "Показать содержимое каталога (тип, имя, размер).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "каталог (по умолчанию текущий)"},
            },
        },
    },
    {
        "name": "search_files",
        "description": "Поиск по регулярному выражению в файлах (рекурсивно, как grep). Игнорирует .git, node_modules и т.п.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "регулярное выражение"},
                "path": {"type": "string", "description": "каталог или файл для поиска"},
                "file_glob": {"type": "string", "description": "фильтр имён файлов, например '*.py'"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "current_time",
        "description": "Текущие дата и время с учётом часового пояса (по умолчанию Europe/Paris).",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA-имя пояса, например 'Europe/Moscow'"},
            },
        },
    },
    {
        "name": "ask_user",
        "description": "Задать пользователю уточняющий вопрос и получить ответ. Используй, только когда без ответа никак.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "вопрос пользователю"},
            },
            "required": ["question"],
        },
    },
]

# Инструменты, которые можно выполнять без подтверждения (только чтение)
AUTO_TOOLS = {"read_file", "list_dir", "search_files", "current_time", "ask_user"}


@dataclass
class ToolBox:
    """Диспетчер инструментов агента."""

    workspace: Path
    ask: Callable[[str], str] | None = None
    default_timeout: int = 120

    def dispatch(self, name: str, args: dict) -> dict:
        try:
            if name == "run_command":
                return tool_run_command(args, workspace=self.workspace, timeout_default=self.default_timeout)
            if name == "read_file":
                return tool_read_file(args)
            if name == "write_file":
                return tool_write_file(args)
            if name == "edit_file":
                return tool_edit_file(args)
            if name == "list_dir":
                return tool_list_dir(args)
            if name == "search_files":
                return tool_search_files(args)
            if name == "current_time":
                return tool_current_time(args)
            if name == "ask_user":
                return tool_ask_user(args, ask=self.ask)
            return {"status": "error", "error": f"неизвестный инструмент: {name}"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def summary(name: str, args: dict) -> str:
        """Короткое описание вызова для показа пользователю."""
        try:
            if name == "run_command":
                return f"$ {args.get('command', '')}"
            if name == "write_file":
                return f"записать {args.get('path')} ({len(str(args.get('content', '')))} символов)"
            if name == "edit_file":
                return f"правка {args.get('path')}"
            if name == "read_file":
                return f"прочитать {args.get('path')}"
            if name == "list_dir":
                return f"ls {args.get('path') or '.'}"
            if name == "search_files":
                return f"поиск /{args.get('pattern')}/ в {args.get('path') or '.'}"
            if name == "current_time":
                return "текущее время"
            if name == "ask_user":
                return f"вопрос: {str(args.get('question', ''))[:120]}"
            return json.dumps(args, ensure_ascii=False)[:200]
        except Exception:  # noqa: BLE001
            return name

    @staticmethod
    def is_dangerous(name: str, args: dict) -> bool:
        if name == "run_command":
            return is_dangerous(str(args.get("command", "")))
        return False
