"""Конфигурация hzhz: ~/.config/hzhz/config.json + переменные окружения."""
from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gemini-2.5-flash"

CONFIG_DIR = Path(os.environ.get("HZZ_CONFIG_DIR") or Path.home() / ".config" / "hzhz")
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class Config:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    workspace: str = ""
    auto_approve: bool = False
    web_search: bool = False
    command_timeout: int = 120
    temperature: float = 0.7
    extra: dict = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        return (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )

    def save(self) -> Path:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        try:
            CONFIG_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        except OSError:
            pass
        return CONFIG_PATH


def load_config() -> Config:
    data: dict = {}
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
    known = {f for f in Config.__dataclass_fields__ if f != "extra"}
    kwargs = {k: v for k, v in data.items() if k in known}
    extra = {k: v for k, v in data.items() if k not in known}
    return Config(**kwargs, extra=extra)
