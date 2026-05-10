"""Retrieve and assemble RAG context for a query."""
from mip.inference.ollama_client import OllamaClient
from mip.rag.embed import embed_texts
from mip.rag.qdrant_store import get_client, search

# Rough chars→tokens ratio for English code+docs. Conservative on the high side.
_CHARS_PER_TOKEN = 3.5


async def retrieve(
    ollama: OllamaClient,
    collection: str,
    query: str,
    top_k: int = 10,
    token_budget: int = 6000,
) -> list[dict]:
    vectors = await embed_texts(ollama, [query])
    if not vectors:
        return []
    qdrant = get_client()
    hits = search(qdrant, collection, vectors[0], limit=top_k)

    selected: list[dict] = []
    used_chars = 0
    char_budget = int(token_budget * _CHARS_PER_TOKEN)
    for h in hits:
        payload = h["payload"]
        text = payload.get("text", "")
        if not text:
            continue
        if used_chars + len(text) > char_budget:
            continue
        selected.append({"score": h["score"], **payload})
        used_chars += len(text)
    return selected


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for c in chunks:
        head = f"[{c.get('path', '?')}"
        if c.get("section"):
            head += f" §{c['section']}"
        elif c.get("symbol"):
            head += f" :{c['symbol']}"
        elif c.get("line_start"):
            head += f" L{c['line_start']}-{c.get('line_end', '?')}"
        head += "]"
        parts.append(f"{head}\n{c.get('text', '')}")
    return "\n\n---\n\n".join(parts)
