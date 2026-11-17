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

echo "Setting up the app folder..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

echo "Updating pip..."
.venv/bin/pip install --quiet --upgrade pip

# v0.11 (A9): the PySide6 install shows its own progress (no
# --quiet): it is the long step, and a silent minute looks like
# a hung install.
echo "Downloading PySide6 (this takes a while)..."
.venv/bin/pip install PySide6-Essentials

# v0.11 (A9): the smoke check. The app starts headless, drives
# 16 theme steps through the real window, and prints OK. A bad
# venv or a missing Qt library fails here, with a plain message,
# instead of on the first icon click. The private instance name
# keeps the check from touching a running ACR.
echo "Checking the app starts..."
if ! ACR_INSTANCE_NAME=acr-smoke-test \
     QT_QPA_PLATFORM=offscreen \
     .venv/bin/python acr_app.py --smoke-test; then
  echo "The app did not start, so the install is not finished. See the message above."
  exit 1
fi

echo "Writing the launcher..."
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
  echo "exec \"$DIR/.venv/bin/python\" \"$DIR/acr_launch.py\" \"\$@\""
} > "$HOME/.local/bin/acr"
chmod +x "$HOME/.local/bin/acr"

echo "ACR installed."
echo "Start it from the ACR icon, or run: acr"
echo "If 'acr' is not found, log out and back in once."

if [ "${ACR_GUI_INSTALL:-}" = "1" ]; then
  read -rp "Done. Press Enter to close."
fi
