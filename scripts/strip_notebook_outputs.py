#!/usr/bin/env python3
"""Strip Jupyter notebook outputs to keep the repo fast.

By default this keeps visible outputs (useful for sharing), while removing
large widget state blobs that can bloat the repo.

Optional modes:
- Strip outputs + execution counts (for a lightweight repo or clean diffs)

Usage:
    python scripts/strip_notebook_outputs.py
    python scripts/strip_notebook_outputs.py path1.ipynb path2.ipynb
    python scripts/strip_notebook_outputs.py --strip-outputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import argparse


def _iter_notebooks(paths: list[str]) -> list[Path]:
    if paths:
        notebooks: list[Path] = []
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                notebooks.extend(sorted(p.rglob("*.ipynb")))
            else:
                notebooks.append(p)
        return notebooks

    # Default: portfolio notebooks live under code/
    return sorted(Path("code").rglob("*.ipynb"))


def strip_notebook(path: Path, *, strip_outputs: bool) -> bool:
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    changed = False

    metadata = nb.get("metadata")
    if isinstance(metadata, dict):
        # Commonly huge when interactive widgets are used.
        for key in ("widgets", "widget_state", "widget"):
            if key in metadata:
                metadata.pop(key, None)
                changed = True

    cells = nb.get("cells", [])
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue

            if cell.get("cell_type") != "code":
                continue

            if strip_outputs and cell.get("outputs"):
                cell["outputs"] = []
                changed = True

            if strip_outputs and cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

            if strip_outputs:
                cell_meta = cell.get("metadata")
                if isinstance(cell_meta, dict) and "execution" in cell_meta:
                    cell_meta.pop("execution", None)
                    changed = True

    if not changed:
        return False

    # Keep JSON stable and readable; avoid ASCII escaping.
    path.write_text(
        json.dumps(nb, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Notebook paths or directories. If omitted, uses code/**/*.ipynb",
    )
    parser.add_argument(
        "--strip-outputs",
        action="store_true",
        help="Also remove cell outputs and execution counts.",
    )
    args = parser.parse_args()

    notebooks = _iter_notebooks(args.paths)
    if not notebooks:
        print("No notebooks found.")
        return 0

    updated = 0
    missing = 0
    for nb_path in notebooks:
        if not nb_path.exists():
            missing += 1
            continue
        try:
            if strip_notebook(nb_path, strip_outputs=args.strip_outputs):
                updated += 1
        except json.JSONDecodeError as e:
            print(f"ERROR: {nb_path}: invalid JSON ({e})", file=sys.stderr)
            return 2

    print(f"Notebooks updated: {updated}")
    if missing:
        print(f"Notebooks missing: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
