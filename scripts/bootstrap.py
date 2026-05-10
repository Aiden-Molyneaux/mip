"""Initial full index of a project into Qdrant.

Usage:
    uv run python scripts/bootstrap.py --repo C:/personal/InThink --collection inthink
"""
import argparse
import asyncio
import hashlib
from pathlib import Path

from mip.inference.ollama_client import OllamaClient
from mip.rag.embed import embed_texts
from mip.rag.ingest import chunk_file, iter_files
from mip.rag.qdrant_store import ensure_collection, get_client, upsert_chunks


def chunk_id(path: str, line_start: int, line_end: int) -> str:
    """Stable UUID-shaped ID derived from path + line range. Re-running upsert overwrites."""
    h = hashlib.sha1(f"{path}:{line_start}-{line_end}".encode()).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


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

        all_chunks = [c for c in raw_chunks if c.text.strip()]
        skipped = len(raw_chunks) - len(all_chunks)
        print(
            f"[bootstrap] {len(all_chunks)} chunks total"
            + (f" ({skipped} empty/whitespace skipped)" if skipped else "")
        )

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
