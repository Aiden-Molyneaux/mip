"""SQL chunker for SQLite migrations.

Splits per top-level statement (CREATE TABLE / ALTER TABLE / CREATE INDEX / ...).
The migration number is parsed from the filename prefix and preserved in metadata
so retrieval can cite the migration that introduced a table.
"""
import re
from pathlib import Path

from mip.rag.chunkers import Chunk

# Top-level DDL/DML statements, anchored to start of line, captured up to the first `;`.
_STATEMENT_RE = re.compile(
    r"^(?:CREATE|ALTER|DROP|INSERT|UPDATE|DELETE)\b.*?;",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_MIGRATION_NUM_RE = re.compile(r"^(\d{3,})_")


def chunk_sql(rel_path: Path, text: str) -> list[Chunk]:
    migration_num: int | None = None
    m = _MIGRATION_NUM_RE.match(rel_path.name)
    if m:
        migration_num = int(m.group(1))

    rel_str = str(rel_path).replace("\\", "/")
    chunks: list[Chunk] = []
    for match in _STATEMENT_RE.finditer(text):
        stmt = match.group(0).strip()
        head = stmt.split("(", 1)[0].strip()
        line_start = text[: match.start()].count("\n") + 1
        line_end = line_start + stmt.count("\n")
        chunks.append(
            Chunk(
                text=stmt,
                metadata={
                    "path": rel_str,
                    "kind": "schema",
                    "lang": "sql",
                    "migration": migration_num,
                    "statement_head": head[:120],
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )
        )

    if not chunks:
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    "path": rel_str,
                    "kind": "schema",
                    "lang": "sql",
                    "migration": migration_num,
                    "line_start": 1,
                    "line_end": len(text.splitlines()),
                },
            )
        )
    return chunks
