"""Walk a project, route files to the right chunker, return all chunks.

Default include rules tuned for InThink-shaped repos. `register_project.py`
will eventually let projects override these.
"""
from pathlib import Path
from typing import Iterable

from mip.rag.chunkers import Chunk
from mip.rag.chunkers.markdown import chunk_markdown
from mip.rag.chunkers.sql import chunk_sql
from mip.rag.chunkers.ts import chunk_typescript

DEFAULT_INCLUDE_DIRS = ("server/src", "web/src", "scripts")
# Excluded from the doc set on purpose:
#   CHANGELOG.md       — history, not knowledge; floods retrieval with dated bullets
#   architecture.md    — high-level prose that ranks high on generic code queries but
#                        gives the model nothing concrete to copy
#   LEARNINGS.md /     — removed from the InThink repo on 2026-05-10
#   TECHDEBT.md
DEFAULT_INCLUDE_FILES = (
    "CLAUDE.md",
    "claude-workflows.md",
    "DEPLOY.md",
    "README.md",
)
DEFAULT_EXCLUDE_DIRS = {"node_modules", ".git", "dist", "build", "data", ".venv", "__pycache__"}
DEFAULT_EXTENSIONS = {".ts", ".tsx", ".md", ".sql"}


def iter_files(repo_root: Path) -> Iterable[Path]:
    for name in DEFAULT_INCLUDE_FILES:
        p = repo_root / name
        if p.exists() and p.is_file():
            yield p
    for sub in DEFAULT_INCLUDE_DIRS:
        d = repo_root / sub
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if any(part in DEFAULT_EXCLUDE_DIRS for part in p.parts):
                continue
            if p.suffix in DEFAULT_EXTENSIONS:
                yield p


def chunk_file(repo_root: Path, file: Path) -> list[Chunk]:
    rel = file.relative_to(repo_root)
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file.read_text(encoding="utf-8", errors="replace")

    if file.suffix in {".ts", ".tsx"}:
        return chunk_typescript(rel, text)
    if file.suffix == ".sql":
        return chunk_sql(rel, text)
    if file.suffix == ".md":
        return chunk_markdown(rel, text)
    return []
