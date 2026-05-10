"""TypeScript / TSX chunker.

Strategy:

  1. Extract the file's import block (consecutive `import ...` lines plus
     interleaved comments and blank lines at the top of the file) and prepend
     it to every chunk from that file. Even a mid-file slice now demonstrates
     the project's import patterns — `.js` suffixes, which paths use `node:` vs
     bare names, whether the module uses `import type`, etc. This is the single
     biggest lever for getting the model to follow InThink's `.js`-suffix rule:
     it has to see the pattern by example, not just be told.

  2. Extract top-level export symbols via regex (`export function|class|const|
     type|interface|enum`) and store as a `symbols` metadata array. Future
     retrieval can exact-match queries mentioning those symbols to surface the
     right file regardless of embedding score.

  3. Fall back to the same 60-line / 10-line-overlap window as before for the
     chunk body. Full tree-sitter symbol-aware chunking is Tier 2-F; the
     imports-header + symbol extraction here gets most of the benefit without
     the native-compilation install pain on Windows.
"""
import re
from pathlib import Path

from mip.rag.chunkers import Chunk

_WINDOW = 60
_OVERLAP = 10

# Top-level export patterns. Anchored to start of line to skip nested decls.
_EXPORT_PATTERNS = (
    re.compile(r"^export\s+(?:async\s+)?function\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+(?:default\s+)?class\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+const\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+type\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+interface\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+enum\s+(\w+)", re.MULTILINE),
)


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


def _extract_imports_block(lines: list[str]) -> tuple[str, int]:
    """Walk from the top until we hit the first non-import, non-blank, non-comment
    line. Return the joined imports block and its exclusive end line index.
    Blank lines and `//` comments interleaved with imports are kept inside the block.
    """
    imports_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import "):
            imports_end = i + 1
        elif stripped == "" or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        else:
            break
    if imports_end == 0:
        return "", 0
    return "\n".join(lines[:imports_end]), imports_end


def _extract_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for pat in _EXPORT_PATTERNS:
        symbols.extend(pat.findall(text))
    # Preserve order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def chunk_typescript(rel_path: Path, text: str) -> list[Chunk]:
    lines = text.splitlines()
    rel_str = str(rel_path).replace("\\", "/")
    lang = _lang_for(rel_path)
    module = _module_for(rel_path)
    symbols = _extract_symbols(text)
    imports_block, _imports_end = _extract_imports_block(lines)

    base_metadata = {
        "path": rel_str,
        "kind": "code",
        "lang": lang,
        "module": module,
        "symbols": symbols,
    }

    if len(lines) <= _WINDOW:
        return [
            Chunk(
                text=text,
                metadata={**base_metadata, "line_start": 1, "line_end": len(lines)},
            )
        ]

    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        end = min(i + _WINDOW, len(lines))
        body = "\n".join(lines[i:end])
        if i > 0 and imports_block:
            # Prepend imports so the chunk demonstrates the file's import patterns
            # (notably the `.js` suffix). The comment marker tells the model where
            # the original chunk's lines pick up.
            chunk_text = imports_block + f"\n\n// ...lines 1-{i} omitted...\n\n" + body
        else:
            chunk_text = body
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={**base_metadata, "line_start": i + 1, "line_end": end},
            )
        )
        if end >= len(lines):
            break
        i = end - _OVERLAP
    return chunks
