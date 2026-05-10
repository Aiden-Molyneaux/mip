"""Initial full index of a project into Qdrant.

Usage:
    uv run python scripts/bootstrap.py --repo C:/personal/InThink --collection inthink
"""
import argparse
import asyncio
import hashlib
from pathlib import Path

from mip.inference.ollama_client import OllamaClient
from mip.rag.chunkers import Chunk
from mip.rag.embed import embed_texts
from mip.rag.ingest import chunk_file, iter_files
from mip.rag.qdrant_store import ensure_collection, get_client, upsert_chunks

# nomic-embed-text v1.5 with Ollama's default 2048-token context fits ~7k chars.
# Cap at 6000 for safety headroom. Oversized chunks get split before embedding.
MAX_CHUNK_CHARS = 6000


def chunk_id(path: str, line_start: int, line_end: int, sub: int = 0) -> str:
    """Stable UUID-shaped ID. `sub` distinguishes chunks split from the same line range."""
    h = hashlib.sha1(f"{path}:{line_start}-{line_end}:{sub}".encode()).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _smart_split(text: str, max_chars: int) -> list[str]:
    """Split text into pieces no larger than max_chars. Prefers paragraph boundaries
    (blank-line separators); falls back to hard character windows for paragraphs that
    are themselves too long.
    """
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    buf = ""
    for piece in text.split("\n\n"):
        candidate = (buf + "\n\n" + piece) if buf else piece
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            parts.append(buf)
            buf = ""
        if len(piece) <= max_chars:
            buf = piece
        else:
            for i in range(0, len(piece), max_chars):
                parts.append(piece[i:i + max_chars])
    if buf:
        parts.append(buf)
    return parts


def _split_oversized(chunks: list[Chunk], max_chars: int) -> list[Chunk]:
    out: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= max_chars:
            out.append(c)
            continue
        for i, part in enumerate(_smart_split(c.text, max_chars)):
            out.append(Chunk(text=part, metadata={**c.metadata, "chunk_index": i}))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    repo: Path = args.repo.resolve()
    if not repo.is_dir():
        raise SystemExit(f"--repo {repo} is not a directory")

    qdrant = get_client()
    ensure_collection(qdrant, args.collection)

    ollama = OllamaClient()
    try:
        files = list(iter_files(repo))
        print(f"[bootstrap] {len(files)} files to chunk")

        raw_chunks = []
        for f in files:
            for c in chunk_file(repo, f):
                raw_chunks.append(c)

        non_empty = [c for c in raw_chunks if c.text.strip()]
        skipped = len(raw_chunks) - len(non_empty)
        all_chunks = _split_oversized(non_empty, MAX_CHUNK_CHARS)
        oversized = len(all_chunks) - len(non_empty)
        msg = f"[bootstrap] {len(all_chunks)} chunks total"
        if skipped:
            msg += f" ({skipped} empty skipped)"
        if oversized:
            msg += f" ({oversized} sub-chunks from splitting >{MAX_CHUNK_CHARS}-char inputs)"
        print(msg)

        for batch_start in range(0, len(all_chunks), args.batch):
            batch = all_chunks[batch_start : batch_start + args.batch]
            try:
                vectors = await embed_texts(ollama, [c.text for c in batch])
            except Exception as e:
                # Surface which chunks were in the failing batch so we can identify the bad input.
                paths_in_batch = sorted({c.metadata.get("path", "?") for c in batch})
                lengths = [len(c.text) for c in batch]
                print(
                    f"[bootstrap] FAILED on batch {batch_start}-{batch_start + len(batch) - 1}: {e}\n"
                    f"  files in batch: {paths_in_batch}\n"
                    f"  chunk lengths: min={min(lengths)} max={max(lengths)} avg={sum(lengths) // len(lengths)}"
                )
                raise
            points = []
            for c, vec in zip(batch, vectors):
                payload = {**c.metadata, "text": c.text}
                pid = chunk_id(
                    c.metadata.get("path", "?"),
                    c.metadata.get("line_start", 0),
                    c.metadata.get("line_end", 0),
                    sub=c.metadata.get("chunk_index", 0),
                )
                points.append((payload, vec, pid))
            upsert_chunks(qdrant, args.collection, points)
            print(
                f"[bootstrap] embedded+upserted {batch_start + len(batch)}/{len(all_chunks)}"
            )

        print("[bootstrap] done.")
    finally:
        await ollama.aclose()


if __name__ == "__main__":
    asyncio.run(main())
