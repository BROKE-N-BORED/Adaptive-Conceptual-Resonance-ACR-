#!/usr/bin/env bash
# ACR installer: private .venv with PySide6, launcher, acr command.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Installing ACR..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is needed and was not found."
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet PySide6-Essentials

mkdir -p "$HOME/.local/share/applications" "$HOME/.local/bin"

sed "s|__APP_DIR__|$DIR|g" ACR.desktop \
  > "$HOME/.local/share/applications/ACR.desktop"

if [ -d "$HOME/Desktop" ]; then
  cp "$HOME/.local/share/applications/ACR.desktop" \
    "$HOME/Desktop/ACR.desktop"
  chmod +x "$HOME/Desktop/ACR.desktop"
  gio set "$HOME/Desktop/ACR.desktop" metadata::trusted true \
    2>/dev/null || true
fi

{
  echo '#!/usr/bin/env bash'
  echo "exec \"$DIR/.venv/bin/python\" \"$DIR/acr_app.py\" \"\$@\""
} > "$HOME/.local/bin/acr"
chmod +x "$HOME/.local/bin/acr"

echo "ACR installed."
echo "Start it from the ACR icon, or run: acr"
echo "If 'acr' is not found, log out and back in once."

if [ "${ACR_GUI_INSTALL:-}" = "1" ]; then
  read -rp "Done. Press Enter to close."
fi
