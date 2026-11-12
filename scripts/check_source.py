"""Dependency-free static syntax/merge-conflict check; not a type checker."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ("core", "parsers", "extractors", "profiling", "verification", "benchmark", "tests", "scripts"):
    for path in (ROOT / folder).glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in ROOT.glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("Static syntax checks passed")
