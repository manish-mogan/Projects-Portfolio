#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv ]]; then
  echo ".venv already exists"
  exit 0
fi

echo "Creating .venv with python3..."
python3 -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Safety: Panel is required for the Austin dashboard notebooks
python -m pip install panel

echo "Done. Interpreter: $(python -c 'import sys; print(sys.executable)')"
