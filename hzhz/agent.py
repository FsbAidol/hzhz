"""Агент: диалог с Gemini + ручной цикл вызова инструментов.

Схемы инструментов передаются как Function Declarations (без callables),
поэтому SDK сам отключает automatic function calling, и мы обрабатываем
вызовы вручную — это нужно для подтверждений и безопасности.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from google import genai
from google.genai import types

from .config import Config
from .openrouter import ApiError, normalize_model, stream_chat, to_openai_tools
from .prompts import SYSTEM_PROMPT
from .tools import AUTO_TOOLS, TOOL_SCHEMAS, ToolBox

MAX_TOOL_ROUNDS = 25  # максимум итераций «модель → инструменты → модель» за один ход


class UI(Protocol):
    """Интерфейс, который реализует CLI (или тесты)."""

    def on_text_delta(self, text: str) -> None: ...

    def on_tool_start(self, name: str, summary: str, dangerous: bool) -> None: ...

    def on_tool_result(self, name: str, status: str, preview: str) -> None: ...

    def on_info(self, message: str) -> None: ...

    def confirm(self, name: str, summary: str, dangerous: bool) -> str:
        """Вернуть: 'yes' | 'no' | 'always'."""
        ...

    def ask_text(self, question: str) -> str:
        """Спросить текст у пользователя (инструмент ask_user)."""
        ...

    def ask_password(self, prompt: str) -> str:
        """Скрыто ввести пароль опасных команд."""
        ...


class NullUI:
    def on_text_delta(self, text: str) -> None:
        print(text, end="", flush=True)

    def on_tool_start(self, name: str, summary: str, dangerous: bool) -> None:
        pass

    def on_tool_result(self, name: str, status: str, preview: str) -> None:
        pass

    def on_info(self, message: str) -> None:
        print(f"\n[{message}]")

    def confirm(self, name: str, summary: str, dangerous: bool) -> str:
        return "no"

    def ask_text(self, question: str) -> str:
        return ""

    def ask_password(self, prompt: str) -> str:
        return ""


class AgentError(RuntimeError):
    pass


def _fc_schemas() -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name=s["name"],
            description=s["description"],
            parameters=s["parameters"],
        )
        for s in TOOL_SCHEMAS
    ]


def _preview(result: dict, limit: int = 1500) -> str:
    try:
        text = json.dumps(result, ensure_ascii=False, indent=1)
    except Exception:  # noqa: BLE001
        text = str(result)
    return text if len(text) <= limit else text[:limit] + " …"


class Agent:
    def __init__(self, config: Config, ui: UI | None = None, danger_enabled: bool = False):
        self.config = config
        self.ui = ui or NullUI()
        api_key = config.resolve_api_key()
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        provider = (getattr(config, "provider", "") or "").lower()
        if not provider:
            if api_key.startswith("sk-or-") or (not api_key and or_key):
                provider = "openrouter"
            else:
                provider = "gemini"
        if provider not in ("gemini", "openrouter"):
            raise AgentError(f"неизвестный провайдер: {provider} (доступны: gemini, openrouter)")
        self.provider = provider
        if provider == "openrouter":
            if api_key and not api_key.startswith("sk-or-"):
                api_key = ""  # ключ Gemini не подходит для OpenRouter
            api_key = api_key or or_key
            if not api_key:
                raise AgentError(
                    "Не найден API-ключ OpenRouter. Задайте OPENROUTER_API_KEY=sk-or-... "
                    "или выполните /key sk-or-... в REPL. Ключ: https://openrouter.ai/keys"
                )
            self._or_key = api_key
            self._or_model = normalize_model(config.model)
            self._or_messages: list[dict] | None = None
            self.client = None
            self.chat = None
            self._history: list[dict] = []
        else:
            if not api_key:
                raise AgentError(
                    "Не найден API-ключ Gemini. Задайте GEMINI_API_KEY=... или выполните /key в REPL."
                )
            self.client = genai.Client(api_key=api_key)
            self.chat = None
            self._history: list[dict] = []
        self.workspace = Path(config.workspace or Path.cwd()).expanduser().resolve()
        self.auto_approve = config.auto_approve
        self.web_search = config.web_search
        # Разблокировка опасных команд: флаг --danger (или /danger).
        # Пароль (/password) разблокирует одно действие; флаг — всю сессию.
        self.danger_enabled = danger_enabled
        self._always_ok: set[str] = set()
        self._always_danger_ok = False
        self.chat = None
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Чат
    # ------------------------------------------------------------------
    def _build_chat(self) -> None:
        tools: list[types.ToolUnion] = [types.Tool(function_declarations=_fc_schemas())]
        if self.web_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
            temperature=self.config.temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self.chat = self.client.chats.create(
            model=self.config.model,
            config=cfg,
            history=self._history or None,
        )

    def clear(self) -> None:
        self._history = []
        self.chat = None
        if self.provider == "openrouter":
            self._or_messages = None

    def send(self, user_text: str) -> None:
        """Один пользовательский ход: стриминг текста + цикл инструментов."""
        if self.provider == "openrouter":
            return self._send_openrouter(user_text)
        if self.chat is None:
            self._build_chat()

        message: Any = user_text
        for _round in range(MAX_TOOL_ROUNDS):
            text_parts: list[str] = []
            calls: list[types.FunctionCall] = []

            stream = self.chat.send_message_stream(message)
            for chunk in stream:
                candidate = (chunk.candidates or [None])[0]
                if candidate is None or candidate.content is None:
                    continue
                for part in candidate.content.parts or []:
                    if part is None:
                        continue
                    if getattr(part, "thought", False):
                        continue  # скрытые «мысли» не показываем
                    if part.text:
                        text_parts.append(part.text)
                        self.ui.on_text_delta(part.text)
                    if part.function_call:
                        calls.append(part.function_call)

            if not calls:
                if text_parts:
                    self.ui.on_text_delta("\n")
                return

            # Выполняем все вызовы инструментов этого раунда
            responses: list[types.Part] = []
            for call in calls:
                name = call.name or ""
                args = dict(call.args or {})
                result = self._execute_tool(name, args)
                responses.append(
                    types.Part.from_function_response(name=name, response=result)
                )
            message = responses

        self.ui.on_info(f"достигнут лимит в {MAX_TOOL_ROUNDS} раундов инструментов")

    def _send_openrouter(self, user_text: str) -> None:
        """Ход через OpenRouter (OpenAI-совместимый API)."""
        if self._or_messages is None:
            self._or_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._or_messages.append({"role": "user", "content": user_text})
        tools = to_openai_tools(TOOL_SCHEMAS)

        for _round in range(MAX_TOOL_ROUNDS):
            assistant_msg: dict | None = None
            calls: list[tuple] = []
            try:
                for event in stream_chat(
                    api_key=self._or_key,
                    model=self._or_model,
                    messages=self._or_messages,
                    tools=tools,
                    temperature=self.config.temperature,
                ):
                    if event[0] == "text":
                        self.ui.on_text_delta(event[1])
                    else:
                        _, assistant_msg, calls = event
            except ApiError as exc:
                raise AgentError(friendly_error(exc)) from exc

            if not calls:
                if assistant_msg is not None:
                    self._or_messages.append(assistant_msg)
                self.ui.on_text_delta("\n")
                return

            if assistant_msg is not None:
                self._or_messages.append(assistant_msg)
            for call_id, name, args in calls:
                result = self._execute_tool(name, args)
                self._or_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        self.ui.on_info(f"достигнут лимит в {MAX_TOOL_ROUNDS} раундов инструментов")

    # ------------------------------------------------------------------
    # Инструменты
    # ------------------------------------------------------------------
    def _execute_tool(self, name: str, args: dict) -> dict:
        summary = ToolBox.summary(name, args)
        dangerous = ToolBox.is_dangerous(name, args)

        # Логика допуска:
        #  - read-only (AUTO_TOOLS) — без вопроса;
        #  - ОПАСНЫЕ — заблокированы по умолчанию: нужны флаг --danger (/danger)
        #    ИЛИ верный пароль опасных команд (/password). Обычное y — недостаточно.
        #  - остальные (запись/команды) — спрашиваем, пока пользователь не разрешил.
        if dangerous:
            allowed, deny_reason = self._danger_gate(name, summary)
            if not allowed:
                self.ui.on_tool_result(name, "denied", deny_reason)
                return {"status": "error", "error": deny_reason}
        elif name in AUTO_TOOLS:
            allowed = True
        elif name in self._always_ok or self.auto_approve:
            allowed = True
            self.ui.on_tool_start(name, summary, False)
        else:
            self.ui.on_tool_start(name, summary, False)
            answer = self.ui.confirm(name, summary, False)
            allowed = answer in ("yes", "always")
            if answer == "always":
                self._always_ok.add(name)

        if not allowed:
            self.ui.on_tool_result(name, "denied", "пользователь запретил выполнение")
            return {"status": "error", "error": "пользователь запретил выполнение этого действия"}

        box = ToolBox(
            workspace=self.workspace,
            ask=self.ui.ask_text,
            default_timeout=self.config.command_timeout,
        )
        result = box.dispatch(name, args)
        status = str(result.get("status", "ok"))
        self.ui.on_tool_result(name, status, _preview(result))
        return result

    def _danger_gate(self, name: str, summary: str) -> tuple[bool, str]:
        """Пропуск опасного действия: только пароль или флаг --danger.

        Возвращает (allowed, причина_отказа).
        """
        self.ui.on_tool_start(name, summary, True)

        # Способ 1: флаг --danger / /danger — разблокирована вся сессия.
        if self.danger_enabled:
            if self.auto_approve or self._always_danger_ok:
                return True, ""
            answer = self.ui.confirm(name, summary, True)
            if answer == "always":
                self._always_danger_ok = True
            return answer in ("yes", "always"), "пользователь запретил выполнение"

        # Способ 2: пароль опасных команд — разблокирует это действие.
        if self.config.has_danger_password():
            password = self.ui.ask_password("Пароль опасных команд: ")
            if not password:
                return False, (
                    "ОПАСНАЯ КОМАНДА ЗАБЛОКИРОВАНА: ввод пароля отменён. "
                    "Действие не выполнено. Пользователь может разрешить опасные команды "
                    "флагом --danger (или командой /danger) либо повторить и ввести пароль."
                )
            if self.config.verify_danger_password(password):
                return True, ""
            return False, (
                "ОПАСНАЯ КОМАНДА ЗАБЛОКИРОВАНА: неверный пароль. "
                "Действие не выполнено. Не пытайся обойти защиту."
            )

        return False, (
            "ОПАСНАЯ КОМАНДА ЗАБЛОКИРОВАНА: опасные команды запрещены без разблокировки. "
            "Пользователь должен либо запустить hzhz с флагом --danger (или выполнить /danger), "
            "либо задать пароль опасных команд командой /password — после чего повторить действие. "
            "Не пытайся обойти защиту."
        )


def friendly_error(exc: Exception) -> str:
    """Человекочитаемая подсказка для типичных ошибок API."""
    text = str(exc)
    low = text.lower()
    if "api key" in low or "401" in low or "unauthenticated" in low or "permission" in low and "key" in low:
        return (
            "Ошибка авторизации: API-ключ не принят. Проверьте GEMINI_API_KEY "
            "или создайте ключ на https://aistudio.google.com/apikey"
        )
    if "429" in low or "quota" in low or "resource has been exhausted" in low:
        return "Превышена квота API (429). Подождите или смените модель/ключ."
    if "402" in low or "insufficient" in low or "credit" in low or "too many tokens" in low:
        return (
            "Недостаточно кредитов OpenRouter (402). Пополните баланс на https://openrouter.ai/credits "
            "или выберите бесплатную модель (/model)."
        )
    if "503" in low or "500" in low or "unavailable" in low or "deadline" in low:
        return "Сервис Gemini временно недоступен. Повторите попытку чуть позже."
    if "not found" in low and "model" in low:
        return f"Модель не найдена: {text}. Проверьте имя модели (/model) — например, gemini-2.5-flash."
    if any(s in low for s in ("tls", "ssl", "connection", "network", "resolve", "timed out", "eof")):
        return (
            "Сетевая ошибка при обращении к API Gemini. Проверьте подключение к интернету "
            "и доступность generativelanguage.googleapis.com (прокси/файрвол/VPN)."
        )
    return f"Ошибка API: {text}"
