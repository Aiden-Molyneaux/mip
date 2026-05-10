"""TypeScript / TSX chunker.

Phase 2 ships a fixed-line-window fallback so the pipeline runs end-to-end on
day one. Phase 5 (or earlier if retrieval quality is poor) replaces the body of
`chunk_typescript` with a tree-sitter-typescript walk that emits one chunk per
top-level function / class / exported const, with `symbol` populated.

The fallback uses a 60-line window with 10-line overlap. Files <= 60 lines are
returned as a single chunk so small modules (most of InThink's services) stay
intact.
"""
from pathlib import Path

from mip.rag.chunkers import Chunk

_WINDOW = 60
_OVERLAP = 10


def _lang_for(path: Path) -> str:
    return "tsx" if path.suffix == ".tsx" else "ts"


def _module_for(rel_path: Path) -> str:
    parts = rel_path.parts
    if "modules" in parts:
        i = parts.index("modules")
        if i + 1 < len(parts):
            return parts[i + 1]
    if "components" in parts:
        return "components"
    if "pages" in parts:
        return "pages"
    if "lib" in parts:
        return "lib"
    return ""


def chunk_typescript(rel_path: Path, text: str) -> list[Chunk]:
    lines = text.splitlines()
    rel_str = str(rel_path).replace("\\", "/")
    lang = _lang_for(rel_path)
    module = _module_for(rel_path)

    if len(lines) <= _WINDOW:
        return [
            Chunk(
                text=text,
                metadata={
                    "path": rel_str,
                    "kind": "code",
                    "lang": lang,
                    "module": module,
                    "line_start": 1,
                    "line_end": len(lines),
                },
            )
        ]

    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        end = min(i + _WINDOW, len(lines))
        body = "\n".join(lines[i:end])
        chunks.append(
            Chunk(
                text=body,
                metadata={
                    "path": rel_str,
                    "kind": "code",
                    "lang": lang,
                    "module": module,
                    "line_start": i + 1,
                    "line_end": end,
                },
            )
        )
        if end >= len(lines):
            break
        i = end - _OVERLAP
    return chunks
