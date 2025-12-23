#!/usr/bin/env python3
"""Strip Jupyter notebook outputs to keep the repo fast.

- Removes cell outputs and execution counts
- Removes large widget state blobs from notebook metadata

Usage:
  python scripts/strip_notebook_outputs.py           # strips code/**/*.ipynb
  python scripts/strip_notebook_outputs.py path1.ipynb path2.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _iter_notebooks(args: list[str]) -> list[Path]:
    if args:
        notebooks: list[Path] = []
        for arg in args:
            p = Path(arg)
            if p.is_dir():
                notebooks.extend(sorted(p.rglob("*.ipynb")))
            else:
                notebooks.append(p)
        return notebooks

    return sorted(Path("code").rglob("*.ipynb"))


def strip_notebook(path: Path) -> bool:
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

            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True

            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

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
    notebooks = _iter_notebooks(sys.argv[1:])
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
            if strip_notebook(nb_path):
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
