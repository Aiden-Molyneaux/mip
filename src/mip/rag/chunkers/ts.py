"""TypeScript / TSX chunker — symbol-aware.

A file is split into chunks by walking top-level `export` declarations and
emitting one chunk per export. Each chunk contains:

  1. The file's import block (so `.js` suffixes and library origins are
     always visible).
  2. The file's pre-export preamble (types, helpers, regex constants, etc.
     between the imports and the first export — necessary context for any
     export to make sense).
  3. The export's body — from its `export` line up to the line before the
     next top-level `export`, or end of file.

This is the unit that retrieval should be returning: a complete demonstration
of one symbol in its file's idiom. Pure tree-sitter would be more correct in
edge cases (template literals, multi-line strings with braces) but requires
native compilation that's painful on Windows + Python 3.14. The column-0
`export ` detector works for InThink's consistently-formatted code.

Files with no `export` keyword (rare — barrel files use `export *` which the
detector catches; entry-point scripts get whole-file fallback) are returned as
a single chunk.
"""
import re
from pathlib import Path

from mip.rag.chunkers import Chunk

# Top-level export patterns (for the `symbols` metadata array — separate from
# the chunk-splitting logic, which only needs to find the `export` keyword).
_EXPORT_PATTERNS = (
    re.compile(r"^export\s+(?:async\s+)?function\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+(?:default\s+)?class\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+const\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+(?:async\s+)?function\*?\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+type\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+interface\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+enum\s+(\w+)", re.MULTILINE),
)

_PRIMARY_SYMBOL_PATTERNS = (
    re.compile(r"^export\s+(?:async\s+)?function\s+(\w+)"),
    re.compile(r"^export\s+(?:default\s+)?class\s+(\w+)"),
    re.compile(r"^export\s+const\s+(\w+)"),
    re.compile(r"^export\s+type\s+(\w+)"),
    re.compile(r"^export\s+interface\s+(\w+)"),
    re.compile(r"^export\s+enum\s+(\w+)"),
    re.compile(r"^export\s+default\s+(\w+)"),
    re.compile(r"^export\s+\*\s+from\s+['\"]([^'\"]+)"),
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


def _extract_all_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for pat in _EXPORT_PATTERNS:
        symbols.extend(pat.findall(text))
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _primary_symbol_of(text: str) -> str:
    for line in text.splitlines():
        if not line.startswith("export "):
            continue
        for pat in _PRIMARY_SYMBOL_PATTERNS:
            m = pat.match(line)
            if m:
                return m.group(1)
    return ""


def chunk_typescript(rel_path: Path, text: str) -> list[Chunk]:
    lines = text.splitlines()
    rel_str = str(rel_path).replace("\\", "/")
    lang = _lang_for(rel_path)
    module = _module_for(rel_path)
    filename = Path(rel_str).name
    symbols_all = _extract_all_symbols(text)
    imports_block, imports_end = _extract_imports_block(lines)

    base_metadata = {
        "path": rel_str,
        "filename": filename,
        "kind": "code",
        "lang": lang,
        "module": module,
        "symbols": symbols_all,
    }

    # Top-level export starts (column 0, must start with "export ").
    export_starts: list[int] = [
        i for i in range(imports_end, len(lines)) if lines[i].startswith("export ")
    ]

    # No exports → return the whole file as one chunk (entry-point scripts,
    # side-effect-only modules). Keeps the pipeline simple.
    if not export_starts:
        return [
            Chunk(
                text=text,
                metadata={**base_metadata, "symbol": "", "line_start": 1, "line_end": len(lines)},
            )
        ]

    # Preamble: everything between the imports and the first export
    # (types, regex constants, internal helpers). Attached to every chunk
    # because exports often reference these types.
    preamble_lines = lines[imports_end : export_starts[0]]
    preamble = "\n".join(preamble_lines).strip()

    chunks: list[Chunk] = []
    for idx, start in enumerate(export_starts):
        end_excl = export_starts[idx + 1] if idx + 1 < len(export_starts) else len(lines)
        # Trim trailing blank lines from the export body.
        end = end_excl - 1
        while end > start and lines[end].strip() == "":
            end -= 1
        export_body = "\n".join(lines[start : end + 1])
        primary = _primary_symbol_of(export_body)

        parts: list[str] = []
        if imports_block:
            parts.append(imports_block)
        if preamble:
            parts.append(preamble)
        parts.append(export_body)
        chunk_text = "\n\n".join(parts)

        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    **base_metadata,
                    "symbol": primary,
                    "line_start": start + 1,
                    "line_end": end + 1,
                },
            )
        )
    return chunks
