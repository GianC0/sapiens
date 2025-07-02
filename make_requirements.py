#!/usr/bin/env python3
"""
make_requirements.py
--------------------
Scans all .py files in the current directory, collects imported
modules, maps them to their installed PyPI distributions and writes
requirements.txt (overwrites if it exists).

Python ≥ 3.10 required (for importlib.metadata).
"""

from __future__ import annotations
import ast
import os
import sys
from pathlib import Path
from collections import defaultdict
from importlib import metadata

HERE = Path.cwd()
REQ_PATH = HERE / "requirements.txt"

# ----------------------------------------------------------------------
# 1. gather top-level modules imported in every .py file
# ----------------------------------------------------------------------
def modules_in_file(py_file: Path) -> set[str]:
    """Return a set of top-level module names imported in one file."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                for n in names:
                    mods.add(n.name.partition(".")[0])
            case ast.ImportFrom(module=mod, level=0) if mod:
                mods.add(mod.partition(".")[0])
    return mods


imports: set[str] = set()
for file in HERE.iterdir():
    if file.suffix == ".py" and file.name != Path(__file__).name:
        imports |= modules_in_file(file)

# ----------------------------------------------------------------------
# 2. separate stdlib vs third-party
# ----------------------------------------------------------------------
stdlib = sys.stdlib_module_names  # Python 3.10+
third_party = {m for m in imports if m not in stdlib}

# ----------------------------------------------------------------------
# 3. map module → distribution(s)
# ----------------------------------------------------------------------
pkg_distributions = metadata.packages_distributions()
need: dict[str, set[str]] = defaultdict(set)

for mod in third_party:
    dists = pkg_distributions.get(mod)
    if dists:                      # normal case
        for d in dists:
            need[d].add(mod)
    else:
        # fallback: module not found in metadata; keep the name itself
        need[mod].add(mod)

# ----------------------------------------------------------------------
# 4. write requirements.txt
# ----------------------------------------------------------------------
with REQ_PATH.open("w", encoding="utf-8") as fp:
    for dist in sorted(need):
        try:
            version = metadata.version(dist)
            fp.write(f"{dist}=={version}\n")
        except metadata.PackageNotFoundError:
            # unknown distribution – write plain name
            fp.write(f"{dist}\n")

print(f"✓ requirements.txt written with {len(need)} package(s)")
