"""Retrieve and assemble RAG context for a query.

The retrieval strategy is intent-aware and quota-based:

  1. Classify the user's query as code-pattern / convention / balanced via a
     keyword heuristic (no LLM call).
  2. Run separate filtered searches per chunk kind (code / doc / schema), each
     pulling its quota's share of the budget.
  3. Within each kind, cap at MAX_PER_FILE chunks so one big file can't
     monopolize the context.
  4. Fill in priority order (code → doc → schema for code-pattern queries;
     doc → code → schema for convention queries) until the token budget runs out.

The reason for this shape: pure vector similarity heavily favored prose chunks
from CLAUDE.md and architecture docs because they contain the same surface
vocabulary as user questions ("module", "server", "routes"). The model then
saw no real code to copy from. Quotas force a code floor regardless of how the
embedding similarity stacks up.
"""
import re

from mip.inference.ollama_client import OllamaClient
from mip.rag.embed import embed_texts
from mip.rag.qdrant_store import get_client, search

# Rough chars→tokens for English code+docs.
_CHARS_PER_TOKEN = 3.5
# Hard cap on chunks-per-file in the final assembly. Prevents one CLAUDE.md
# section or one giant TS file from dominating.
MAX_PER_FILE = 2

_CODE_INTENT_RE = re.compile(
    r"\b(how (do|should|would) i|show me|write|create|implement|add|build|"
    r"generate|make( a| an)?|set up|wire up|hook up|register|extend)\b",
    re.IGNORECASE,
)
_CONVENTION_INTENT_RE = re.compile(
    r"\b(convention|rule|should i|best practice|when (do|should)|why do we|"
    r"what'?s the pattern|how does the project)\b",
    re.IGNORECASE,
)
# File-extension or symbol mentions push toward code intent even when the verb
# is ambiguous (e.g. "what does service.ts do here").
_FILE_OR_SYMBOL_RE = re.compile(
    r"\b\w+\.(ts|tsx|sql|md)\b|`[A-Za-z_]\w+`|\b[a-z]+[A-Z]\w+\b"
)


def intent_quota(query: str) -> dict[str, int]:
    """Classify the query and return per-kind quotas summing to ~10.
    Returns a dict with keys "code", "doc", "schema".
    """
    if _CODE_INTENT_RE.search(query) or _FILE_OR_SYMBOL_RE.search(query):
        return {"code": 7, "doc": 1, "schema": 2}
    if _CONVENTION_INTENT_RE.search(query):
        return {"doc": 5, "code": 3, "schema": 0}
    return {"code": 5, "doc": 3, "schema": 2}


def _kind_priority(quota: dict[str, int]) -> list[str]:
    """Fill order: highest-quota kind first so it gets the most token budget."""
    return sorted(quota.keys(), key=lambda k: -quota[k])


async def retrieve(
    ollama: OllamaClient,
    collection: str,
    query: str,
    top_k: int = 10,  # kept for backward-compat; no longer the primary control
    token_budget: int = 6000,
) -> list[dict]:
    quota = intent_quota(query)

    vectors = await embed_texts(ollama, [query])
    if not vectors:
        return []
    qdrant = get_client()

    char_budget = int(token_budget * _CHARS_PER_TOKEN)
    used_chars = 0
    final: list[dict] = []
    seen_ids: set[str] = set()

    for kind in _kind_priority(quota):
        n = quota[kind]
        if n <= 0:
            continue
        # Over-fetch so per-file cap + token cap have headroom to skip.
        hits = search(qdrant, collection, vectors[0], limit=n * 4, filter_kind=kind)
        per_file: dict[str, int] = {}
        taken = 0
        for h in hits:
            if taken >= n:
                break
            payload = h["payload"]
            path = payload.get("path", "?")
            if per_file.get(path, 0) >= MAX_PER_FILE:
                continue
            text = payload.get("text", "")
            if not text:
                continue
            # Dedupe across kinds in case a chunk matches multiple filters.
            ident = f"{path}:{payload.get('line_start', 0)}-{payload.get('line_end', 0)}:{payload.get('chunk_index', 0)}"
            if ident in seen_ids:
                continue
            if used_chars + len(text) > char_budget:
                continue
            final.append({"score": h["score"], "kind_quota": kind, **payload})
            seen_ids.add(ident)
            per_file[path] = per_file.get(path, 0) + 1
            used_chars += len(text)
            taken += 1
    return final


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for c in chunks:
        head_bits = [c.get("path", "?")]
        if c.get("section"):
            head_bits.append(f"§{c['section']}")
        elif c.get("symbol"):
            head_bits.append(f":{c['symbol']}")
        elif c.get("line_start"):
            head_bits.append(f"L{c['line_start']}-{c.get('line_end', '?')}")
        head = "[" + " ".join(head_bits) + "]"
        parts.append(f"{head}\n{c.get('text', '')}")
    return "\n\n---\n\n".join(parts)
