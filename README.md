# Mip

Mip is a small server that gives a coding agent an OpenAI-compatible endpoint backed by a model running in Ollama on your own machine. Before each request reaches the model, Mip searches an index of your project and adds the relevant code and docs to the system prompt, so the answers follow the project's conventions instead of the model's habits.

## Why I built it

I wanted a coding assistant that ran on my own GPU and that I could point OpenCode at over Tailscale. A local model can write TypeScript, but it doesn't know my project. In InThink, the app I built Mip for, every relative import ends in `.js`, shipped migrations are never edited, and each server module splits into `routes.ts` and `service.ts`. Mip puts those rules, and real examples of them, in front of the model on every request.

It is a single-user tool and it is still tuned for that one repo. The system prompt (`prompts/system_inthink.md`) and the files the indexer reads are InThink's. To use it on another project, write your own prompt and change the include rules in `src/mip/rag/ingest.py`.

## How it works

`POST /v1/chat/completions` takes an OpenAI chat request, streaming or not, and forwards it to Ollama's `/api/chat`, translating the formats in both directions. Before forwarding, Mip:

1. Embeds the last user message with `nomic-embed-text`.
2. Sorts the question into code, convention or balanced with a keyword check, and gives each a quota of code, doc and schema chunks. Without quotas, prose from `CLAUDE.md` outranked real code on most questions and the model had nothing to copy.
3. Searches Qdrant per chunk kind, at most two chunks per file, within about 4,000 tokens. If the question names a file such as `routes.ts`, that file's chunks go in first.
4. Merges the project prompt and the chunks into the request's existing system message. Qwen ignored a second system message stacked on top of the client's.

If Qdrant is down or the collection is missing, Mip logs it and the chat carries on without context.

The indexer splits TypeScript into one chunk per top-level export, each carrying the file's imports and preamble so the model always sees the import style. Markdown splits at `##` headings, and SQL migrations split per statement with the migration number kept.

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/chat/completions` | Bearer | Chat, with retrieved context |
| `GET /v1/models` | None | Lists the primary and fallback models; clients call it to validate the URL |
| `GET /admin/preview_rag?q=...` | Bearer | Shows exactly what retrieval would add for `q` |
| `GET /admin/status` | None | Config snapshot |
| `GET /health` | None | Liveness |

CORS is open to every origin. Keep Mip on a private network.

## Requirements

- Ollama on `:11434` with `qwen2.5-coder:14b` pulled
- Python 3.11+ and `uv`
- Docker, for Qdrant (phase 2)

## Setup

Phase 1 is the API alone. Without Qdrant, chat still works, just with no retrieved context. Run these from the repo root, since settings load from `.env` in the working directory:

```bash
uv sync
cp .env.example .env   # edit MIP_BEARER_TOKEN
uv run uvicorn mip.api:app --host 0.0.0.0 --port 8088
curl -H "Authorization: Bearer <token>" http://localhost:8088/v1/models
```

Phase 2 adds retrieval. Start Qdrant, pull the embedding model, and index a repo:

```bash
docker compose -f docker/qdrant-compose.yml up -d
ollama pull nomic-embed-text
uv run python scripts/bootstrap.py --repo /path/to/your/repo --collection myproject
```

Set `MIP_DEFAULT_COLLECTION=myproject` in `.env` and restart the server. To test retrieval and a completion end to end:

```bash
uv run python scripts/smoke_test.py --collection myproject --query "how do I add a new server module?"
```

The indexer reads `CLAUDE.md`, `claude-workflows.md`, `DEPLOY.md` and `README.md` at the repo root, plus `.ts`, `.tsx`, `.md` and `.sql` files under `server/src`, `web/src` and `scripts`. Nothing re-indexes automatically yet. Re-run `bootstrap.py` with `--reset` after the code changes so stale chunks don't linger.

Any OpenAI-compatible client can use Mip: set its base URL to `http://<host>:8088/v1` and its API key to your bearer token.

## Configuration

All settings live in `.env` (see `.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `MIP_BEARER_TOKEN` | `change-me` | Required for chat and the RAG preview |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `MIP_PRIMARY_MODEL` | `qwen2.5-coder:14b` | Used when a request names no model |
| `MIP_FALLBACK_MODEL` | `qwen2.5-coder:7b` | Only listed by `/v1/models` |
| `MIP_EMBED_MODEL` | `nomic-embed-text` | Collections are created with 768-dimension vectors |
| `QDRANT_URL` | `http://localhost:6333` | |
| `MIP_DEFAULT_COLLECTION` | `inthink` | The collection chat requests search |

`MIP_HOST`, `MIP_PORT`, `MIP_STATE_DB` and `MIP_LOG_LEVEL` are read but not used yet. The host and port come from the `uvicorn` flags.

## Status

Phases 1 (the API) and 2 (retrieval) work. Phase 3 (re-index on `git fetch`), phase 4 (several projects, each with its own prompt and collection) and phase 5 (a reranker, if quality needs it) exist only as stubs in `src/mip/sync/`, `src/mip/state/` and `scripts/register_project.py`.

## Licence

MIT
