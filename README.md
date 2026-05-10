# Mip

Local coding assistant. OpenAI-compatible API over Ollama, with project-aware RAG against your own repos.

Architecture and phased plan: `~/.claude/plans/intake-that-prompt-ask-purrfect-jellyfish.md`.

## Quick start (Phase 1: API only, no RAG)

Prereqs on the host machine:
- Ollama installed and running on `:11434`, with `qwen2.5-coder:14b` pulled.
- Python 3.11+ and `uv`.

```bash
uv sync
cp .env.example .env   # edit MIP_BEARER_TOKEN
uv run uvicorn mip.api:app --host 0.0.0.0 --port 8088
```

Smoke test:
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8088/v1/models
```

## Phase 2 (RAG)

Stand up Qdrant:
```bash
docker compose -f docker/qdrant-compose.yml up -d
```

Pull embeddings model:
```bash
ollama pull nomic-embed-text
```

Index a project:
```bash
uv run python scripts/bootstrap.py --repo C:/personal/InThink --collection inthink
```

End-to-end smoke:
```bash
uv run python scripts/smoke_test.py --collection inthink --query "how do I add a new server module?"
```

## Phases

- **Phase 1 (here):** OpenAI-compatible proxy to Ollama, bearer-token auth.
- **Phase 2 (here):** RAG (Qdrant + nomic-embed-text + chunkers).
- **Phase 3:** Auto-sync via `git fetch` polling. Stub files in place.
- **Phase 4:** Multi-project. Stub in place.
- **Phase 5:** Reranker if quality demands it.
