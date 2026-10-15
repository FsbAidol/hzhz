"""Конфигурация hzhz: ~/.config/hzhz/config.json + переменные окружения."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gemini-2.5-flash"
_PBKDF2_ITERATIONS = 100_000

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
    danger_password_salt: str = ""
    danger_password_hash: str = ""
    extra: dict = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        return (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )

    # --- пароль опасных команд (хранится только как PBKDF2-хеш) ---
    def has_danger_password(self) -> bool:
        return bool(self.danger_password_hash)

    def set_danger_password(self, password: str) -> None:
        salt = secrets.token_hex(16)
        self.danger_password_salt = salt
        self.danger_password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS
        ).hex()

    def clear_danger_password(self) -> None:
        self.danger_password_salt = ""
        self.danger_password_hash = ""

    def verify_danger_password(self, password: str) -> bool:
        if not password or not self.has_danger_password():
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), self.danger_password_salt.encode(), _PBKDF2_ITERATIONS
        ).hex()
        return secrets.compare_digest(candidate, self.danger_password_hash)

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
