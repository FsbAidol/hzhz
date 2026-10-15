"""CLI hzhz: интерактивный REPL и одноразовый режим (-p)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

from . import __version__
from .agent import Agent, AgentError, friendly_error
from .config import DEFAULT_MODEL, Config, load_config
from .tools import TOOL_SCHEMAS

THEME = Theme(
    {
        "hzhz": "bold cyan",
        "tool": "bold yellow",
        "danger": "bold red",
        "ok": "green",
        "denied": "red",
        "info": "dim",
    }
)

BANNER = """\
[cyan bold]hzhz[/] — терминальный агент на Google Gemini • CachyOS/Linux
/help — команды • /quit — выход • модели: gemini-2.5-flash, gemini-2.5-pro
"""

HELP = """\
[cyan]/help[/]     — эта справка
[cyan]/clear[/]    — очистить историю диалога
[cyan]/model[/]    — показать или сменить модель (/model gemini-2.5-pro)
[cyan]/auto[/]     — вкл/выкл авто-подтверждение инструментов (опасные всё равно спрашиваем)
[cyan]/search[/]   — вкл/выкл Google Search (веб-поиск)
[cyan]/tools[/]    — список инструментов агента
[cyan]/key[/]      — сохранить API-ключ (в ~/.config/hzhz/config.json, chmod 600)
[cyan]/save[/]     — сохранить диалог в markdown (/save chat.md)
[cyan]/quit[/]     — выход

Обычный текст = сообщение агенту. Инструменты (команды, файлы) спрашивают
подтверждение: [ok]y[/] — да, [ok]n[/] — нет, [ok]a[/] — да и больше не спрашивать.
"""


class ConsoleUI:
    def __init__(self, console: Console):
        self.console = console
        self._in_tool = False

    # --- потоковый текст модели ---
    def on_text_delta(self, text: str) -> None:
        self.console.print(text, end="", highlight=False, markup=False)

    # --- инструменты ---
    def on_tool_start(self, name: str, summary: str, dangerous: bool) -> None:
        self._in_tool = True
        style = "danger" if dangerous else "tool"
        label = "⚠ ОПАСНО" if dangerous else "инструмент"
        self.console.print()
        self.console.print(
            Panel(summary, title=f"[{style}]{label}: {name}[/]", border_style=style, expand=False)
        )

    def on_tool_result(self, name: str, status: str, preview: str) -> None:
        self._in_tool = False
        style = {"ok": "ok", "error": "denied", "denied": "denied"}.get(status, "info")
        self.console.print(f"[{style}]{name}: {status}[/]")

    def on_info(self, message: str) -> None:
        self.console.print(f"\n[info]— {message} —[/]")

    def confirm(self, name: str, summary: str, dangerous: bool) -> str:
        if dangerous:
            self.console.print("[danger]Команда помечена как потенциально разрушительная.[/]")
        try:
            answer = Prompt.ask(
                "Выполнить?",
                choices=["y", "n", "a", "q"],
                default="y" if dangerous else "n",
                show_choices=True,
            )
        except (KeyboardInterrupt, EOFError):
            self.console.print()
            return "no"
        if answer == "q":
            self.console.print("[info]Выход по запросу пользователя[/]")
            raise SystemExit(0)
        return {"y": "yes", "n": "no", "a": "always"}[answer]

    def ask_text(self, question: str) -> str:
        self.console.print(Panel(question, title="[tool]агент спрашивает[/]", border_style="tool", expand=False))
        try:
            return Prompt.ask("Ответ")
        except (KeyboardInterrupt, EOFError):
            self.console.print()
            return ""


def _print_error(console: Console, exc: Exception) -> None:
    msg = friendly_error(exc) if not isinstance(exc, AgentError) else str(exc)
    console.print(f"[danger]✗ {msg}[/]")


def _repl(agent: Agent, console: Console, cfg: Config) -> int:
    console.print(BANNER)
    if not agent.auto_approve:
        console.print("[info]Подтверждение действий включено ( /auto — авто-режим )[/]\n")
    transcript: list[str] = []

    while True:
        try:
            line = Prompt.ask("\n[bold hzhz]вы ▸[/]", default="")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[info]пока![/]")
            return 0
        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            cmd = cmd.lower()
            rest = rest.strip()
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[info]пока![/]")
                return 0
            if cmd == "/help":
                console.print(HELP)
            elif cmd == "/clear":
                agent.clear()
                transcript.clear()
                console.print("[ok]История очищена[/]")
            elif cmd == "/model":
                if rest:
                    agent.config.model = rest
                    agent.clear()
                    console.print(f"[ok]Модель: {rest}[/]")
                else:
                    console.print(f"Текущая модель: [cyan]{agent.config.model}[/]")
            elif cmd == "/auto":
                agent.auto_approve = not agent.auto_approve
                state = "вкл" if agent.auto_approve else "выкл"
                console.print(f"[ok]Авто-подтверждение: {state}[/]")
            elif cmd == "/search":
                agent.web_search = not agent.web_search
                agent.clear()
                state = "вкл" if agent.web_search else "выкл"
                console.print(f"[ok]Google Search: {state}[/]")
            elif cmd == "/tools":
                for s in TOOL_SCHEMAS:
                    console.print(f"[tool]{s['name']}[/] — {s['description']}")
            elif cmd == "/key":
                if not rest:
                    console.print("[info]Использование: /key AIza...[/]")
                else:
                    cfg.api_key = rest
                    path = cfg.save()
                    console.print(f"[ok]Ключ сохранён в {path} (права 600). Перезапустите hzhz.[/]")
            elif cmd == "/save":
                path = Path(rest or "hzhz-chat.md")
                path.write_text("\n\n".join(transcript) + "\n")
                console.print(f"[ok]Сохранено в {path}[/]")
            else:
                console.print(f"[info]Неизвестная команда: {cmd}. /help[/]")
            continue

        # --- обычное сообщение ---
        transcript.append(f"**вы:** {line}")
        console.print("[hzhz]hzhz ▸[/] ", end="")
        try:
            agent.send(line)
        except SystemExit:
            raise
        except KeyboardInterrupt:
            console.print("\n[info]прервано[/]")
            continue
        except Exception as exc:  # noqa: BLE001
            _print_error(console, exc)
            continue
        transcript.append("**hzhz:** *(ответ выше)*")


def _oneshot(agent: Agent, console: Console, prompt: str) -> int:
    try:
        agent.send(prompt)
        return 0
    except KeyboardInterrupt:
        console.print("\n[info]прервано[/]")
        return 130
    except Exception as exc:  # noqa: BLE001
        _print_error(console, exc)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hzhz",
        description="hzhz — терминальный ИИ-агент на Google Gemini для Linux (CachyOS)",
    )
    parser.add_argument("prompt", nargs="*", help="сообщение (или запустите без аргументов для REPL)")
    parser.add_argument("-p", "--oneshot", action="store_true", help="одноразовый режим: ответить и выйти")
    parser.add_argument("-m", "--model", help=f"модель (по умолчанию {DEFAULT_MODEL})")
    parser.add_argument("-w", "--workspace", help="рабочая директория для команд (по умолчанию — текущая)")
    parser.add_argument("-y", "--yes", action="store_true", help="авто-подтверждать инструменты (опасные — всё равно спрашивать)")
    parser.add_argument("--search", action="store_true", help="включить Google Search (веб-поиск)")
    parser.add_argument("--key", help="API-ключ Gemini (или GEMINI_API_KEY)")
    parser.add_argument("--no-color", action="store_true", help="без цвета")
    parser.add_argument("--version", action="version", version=f"hzhz {__version__}")
    args = parser.parse_args(argv)

    console = Console(no_color=args.no_color, theme=THEME)
    cfg = load_config()
    if args.key:
        cfg.api_key = args.key
    if args.model:
        cfg.model = args.model
    if args.workspace:
        cfg.workspace = args.workspace
    if args.yes:
        cfg.auto_approve = True
    if args.search:
        cfg.web_search = True

    try:
        agent = Agent(cfg, ui=ConsoleUI(console))
    except AgentError as exc:
        console.print(f"[danger]✗ {exc}[/]")
        console.print(
            "\n[info]Получите ключ: https://aistudio.google.com/apikey — затем:\n"
            "  export GEMINI_API_KEY='ваш_ключ'\n"
            "  или: hzhz --key ваш_ключ  (сохранится в ~/.config/hzhz/config.json)[/]"
        )
        return 1

    text = " ".join(args.prompt).strip()
    if args.oneshot or text:
        if not text:
            parser.error("в режиме -p нужно передать текст запроса")
        if not cfg.auto_approve:
            console.print("[info]Одноразовый режим без --yes: инструменты с записью/командами будут запрещены.[/]")
        return _oneshot(agent, console, text)
    return _repl(agent, console, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
