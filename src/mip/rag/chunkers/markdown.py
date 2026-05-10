"""Heading-aware markdown chunker.

Splits on `## ` so each H2 section is a self-contained chunk. Files with no
`## ` are returned as a single chunk.
"""
from pathlib import Path

from mip.rag.chunkers import Chunk


def chunk_markdown(rel_path: Path, text: str) -> list[Chunk]:
    rel_str = str(rel_path).replace("\\", "/")
    filename = Path(rel_str).name

    if "\n## " not in "\n" + text:
        return [
            Chunk(
                text=text,
                metadata={
                    "path": rel_str,
                    "filename": filename,
                    "kind": "doc",
                    "lang": "md",
                    "section": "",
                    "line_start": 1,
                    "line_end": len(text.splitlines()),
                },
            )
        ]

    chunks: list[Chunk] = []
    lines = text.splitlines()
    current_heading: str = ""
    current_lines: list[str] = []
    section_start = 1

    def flush(end_line: int) -> None:
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append(
                Chunk(
                    text=body,
                    metadata={
                        "path": rel_str,
                        "filename": filename,
                        "kind": "doc",
                        "lang": "md",
                        "section": current_heading,
                        "line_start": section_start,
                        "line_end": end_line,
                    },
                )
            )

    for i, line in enumerate(lines, start=1):
        if line.startswith("## "):
            flush(i - 1)
            current_heading = line[3:].strip()
            current_lines = [line]
            section_start = i
        else:
            current_lines.append(line)
    flush(len(lines))
    return chunks
