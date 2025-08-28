#!/usr/bin/env python3
"""
make_requirements.py
--------------------
Recursively scans every *.py under the current directory *except*
those inside dot-folders (.git/, .venv/, .mypy_cache/, …).  Collects
imported top-level modules, maps them to installed PyPI distributions,
and writes requirements.txt (overwrites if it exists).

Python ≥ 3.10 required.
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path
from collections import defaultdict
from importlib import metadata

# ────────────────────────────────────────────────────────────────────
HERE        = Path.cwd()Jupyter
SCRIPT_PATH = Path(__file__).resolve()
REQ_PATH    = HERE / "requirements.txt"

# ────────────────────────────────────────────────────────────────────
def modules_in_file(py_file: Path) -> set[str]:
    """Return the set of top-level modules imported in *py_file*."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.partition(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.partition(".")[0])
    return mods

# ────────────────────────────────────────────────────────────────────
# 1. collect imports
# ────────────────────────────────────────────────────────────────────
all_imports: set[str] = set()

for py in HERE.rglob("*.py"):
    if py.resolve() == SCRIPT_PATH or not py.is_file():
        continue
    # ── skip any path containing a dot-folder ──────────────────────
    rel_parts = py.relative_to(HERE).parts
    if any(part.startswith(".") for part in rel_parts[:-1]):   # exclude filename itself
        continue
    all_imports |= modules_in_file(py)

# ────────────────────────────────────────────────────────────────────
# 2. stdlib filter
# ────────────────────────────────────────────────────────────────────
stdlib       = sys.stdlib_module_names
third_party  = {m for m in all_imports if m not in stdlib and m != "__future__"}

# ────────────────────────────────────────────────────────────────────
# 3. module → distribution(s)
# ────────────────────────────────────────────────────────────────────
pkg_dists = metadata.packages_distributions()
needed: dict[str, set[str]] = defaultdict(set)

for mod in sorted(third_party):
    dists = pkg_dists.get(mod)
    if dists:
        for dist in dists:
            needed[dist].add(mod)
    else:
        needed[mod].add(mod)          # fallback: assume same name as package

# ────────────────────────────────────────────────────────────────────
# 4. write requirements.txt
# ────────────────────────────────────────────────────────────────────
with REQ_PATH.open("w", encoding="utf-8") as fp:
    for dist in sorted(needed):
        try:
            ver = metadata.version(dist)
            fp.write(f"{dist}=={ver}\n")
        except metadata.PackageNotFoundError:
            fp.write(f"{dist}\n")

print(f"✓ requirements.txt written ({len(needed)} packages)")
