#!/usr/bin/env bash
# Установщик hzhz для CachyOS / Arch Linux (и любого Linux с Python >= 3.10).
# Создаёт виртуальное окружение в ~/.local/share/hzhz/venv и команду `hzhz` в ~/.local/bin.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/hzhz"
VENV_DIR="$APP_DIR/venv"
BIN_DIR="$HOME/.local/bin"

echo "==> hzhz: установка из $REPO_DIR"

# --- Python ---
PYTHON="$(command -v python3 || true)"
if [[ -z "$PYTHON" ]]; then
    echo "✗ python3 не найден. Установите: sudo pacman -S python" >&2
    exit 1
fi
"$PYTHON" - <<'EOF' || { echo "✗ нужен Python >= 3.10"; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

# --- venv ---
if ! "$PYTHON" -c "import venv" 2>/dev/null; then
    echo "✗ модуль venv недоступен. На Arch/CachyOS: sudo pacman -S python" >&2
    echo "  на Debian/Ubuntu: sudo apt install python3-venv" >&2
    exit 1
fi

echo "==> создаю виртуальное окружение: $VENV_DIR"
mkdir -p "$APP_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> устанавливаю пакет hzhz и зависимости"
pip install --quiet --upgrade pip
pip install --quiet "$REPO_DIR"

# --- команда в PATH ---
mkdir -p "$BIN_DIR"
ln -sf "$VENV_DIR/bin/hzhz" "$BIN_DIR/hzhz"
echo "==> команда установлена: $BIN_DIR/hzhz"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo "⚠ $BIN_DIR нет в PATH. Добавьте в ~/.bashrc или ~/.zshrc:"
        echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
        ;;
esac

# --- API-ключ ---
if [[ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]]; then
    echo
    echo "⚠ переменная GEMINI_API_KEY не задана."
    echo "  1) Получите ключ: https://aistudio.google.com/apikey"
    echo "  2) Добавьте в ~/.bashrc / ~/.zshrc / config.fish:"
    echo "       export GEMINI_API_KEY='ваш_ключ'"
    echo "  — или сохраните через hzhz:  hzhz --key ваш_ключ"
fi

echo
echo "✓ Установка завершена. Запуск:  hzhz"
