#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR=".venv"
if [[ -n "${CODESPACES:-}" ]]; then
  VENV_DIR="${HOME}/.venvs/projects-portfolio"
fi
VENV_DIR="${VENV_DIR_OVERRIDE:-$VENV_DIR}"

PROFILE_DEFAULT="full"
if [[ -n "${CODESPACES:-}" ]]; then
  PROFILE_DEFAULT="base"
fi
REQUIREMENTS_PROFILE="${REQUIREMENTS_PROFILE:-$PROFILE_DEFAULT}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-}"

if [[ "$VENV_DIR" != /* ]]; then
  VENV_DIR="$(pwd)/$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "Virtualenv already exists: $VENV_DIR"
  exit 0
fi

echo "Creating virtualenv with python3: $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

if [[ -n "$REQUIREMENTS_FILE" ]]; then
  python -m pip install -r "$REQUIREMENTS_FILE"
else
  case "$REQUIREMENTS_PROFILE" in
    base)
      python -m pip install -r requirements-base.txt
      ;;
    geo)
      python -m pip install -r requirements-base.txt
      python -m pip install -r requirements-geo.txt
      ;;
    ml)
      python -m pip install -r requirements-base.txt
      python -m pip install -r requirements-ml.txt
      ;;
    full)
      python -m pip install -r requirements.txt
      ;;
    *)
      echo "Unknown REQUIREMENTS_PROFILE: $REQUIREMENTS_PROFILE" >&2
      echo "Valid values: base | geo | ml | full" >&2
      exit 2
      ;;
  esac
fi

# Safety: Panel is required for the Austin dashboard notebooks
python -m pip install panel

echo "Done. Interpreter: $(python -c 'import sys; print(sys.executable)')"
