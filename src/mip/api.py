"""FastAPI app exposing an OpenAI-compatible surface backed by Ollama.

Each /v1/chat/completions request is augmented with project-aware RAG: the user's
last message is embedded, top-k chunks are retrieved from MIP_DEFAULT_COLLECTION,
and a system message containing the project rules + retrieved context is prepended
before forwarding to Ollama. RAG failures fall through silently (logged) so chat
keeps working when Qdrant is down or the collection is missing.

Endpoints:
  POST /v1/chat/completions   OpenAI chat-completions, streaming or non-streaming. Auth required.
  GET  /v1/models             OpenAI model list. No auth — clients use this for URL validation.
  GET  /admin/status          Liveness + config snapshot.
  GET  /health                Process-level liveness.

CORS is wide open (allow_origins=["*"]) because Mip is reachable only via the host's
firewall + Tailscale ACL. Tighten if exposure widens.
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from mip.config import settings
from mip.inference.ollama_client import OllamaClient
from mip.inference.prompt import build_messages, load_system_prompt
from mip.rag.retrieve import format_context, intent_quota, retrieve

# Uvicorn only configures its own loggers; without basicConfig, our logger.info()
# calls vanish into a handler-less root. Configure once at import.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mip.api")

# Resolve the system prompt path once, relative to the repo root (../../../prompts/...).
_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "system_inthink.md"
_SYSTEM_PROMPT_CACHE: str | None = None

_ollama: OllamaClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ollama
    _ollama = OllamaClient()
    yield
    if _ollama is not None:
        await _ollama.aclose()


app = FastAPI(title="Mip", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client() -> OllamaClient:
    if _ollama is None:
        raise RuntimeError("Ollama client not initialized — startup did not complete")
    return _ollama


def require_token(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {settings().mip_bearer_token}":
        raise HTTPException(status_code=401, detail="invalid bearer token")


def _ollama_chunk_to_openai(chunk: dict, model: str, completion_id: str) -> dict:
    delta: dict = {}
    finish: str | None = None
    msg = chunk.get("message") or {}
    if "content" in msg:
        delta["content"] = msg["content"]
    if chunk.get("done"):
        finish = "stop"
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _ollama_full_to_openai(resp: dict, model: str, completion_id: str) -> dict:
    msg = resp.get("message") or {}
    prompt_tokens = resp.get("prompt_eval_count", 0)
    completion_tokens = resp.get("eval_count", 0)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _options_from_openai(body: dict) -> dict:
    options: dict = {}
    if "temperature" in body:
        options["temperature"] = body["temperature"]
    if "top_p" in body:
        options["top_p"] = body["top_p"]
    if "max_tokens" in body:
        options["num_predict"] = body["max_tokens"]
    if "stop" in body and body["stop"] is not None:
        options["stop"] = body["stop"]
    return options


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI lets messages[].content be a string OR a list of content parts
    ([{"type":"text","text":"..."}, {"type":"image_url",...}, ...]). Ollama's
    /api/chat only accepts strings, so flatten the list form to joined text.
    Non-text parts (images, etc.) are dropped — qwen2.5-coder is text-only.
    """
    normalized: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
            content = "\n".join(parts)
        normalized.append({**m, "content": content if content is not None else ""})
    return normalized


def _system_prompt() -> str:
    """Read the project system prompt from disk once, then cache."""
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is None:
        try:
            _SYSTEM_PROMPT_CACHE = load_system_prompt(_SYSTEM_PROMPT_PATH)
        except FileNotFoundError:
            logger.warning("system prompt not found at %s — RAG context only", _SYSTEM_PROMPT_PATH)
            _SYSTEM_PROMPT_CACHE = ""
    return _SYSTEM_PROMPT_CACHE


def _last_user_query(messages: list[dict]) -> str:
    """Walk back through messages to find the last non-empty user message.
    That string is the embedding query for retrieval."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


async def _augment_with_rag(
    ollama: OllamaClient,
    messages: list[dict],
    collection: str,
) -> list[dict]:
    """Merge the InThink system prompt + retrieved chunks INTO the existing
    system message (or prepend if there isn't one). Two consecutive system
    messages caused Qwen to ignore the InThink rules entirely — a single
    merged system message keeps everything in one place.

    Failures (Qdrant down, missing collection, embed errors) are logged and
    the original messages returned — never break the chat over RAG issues.
    """
    query = _last_user_query(messages)
    if not query:
        logger.info("rag: skip — no non-empty user message in request")
        return messages
    try:
        chunks = await retrieve(ollama, collection, query, top_k=10, token_budget=4000)
    except Exception as e:
        logger.warning(
            "rag: retrieval failed for query=%r collection=%r: %s — proceeding without context",
            query[:80], collection, e,
        )
        return messages
    if not chunks:
        logger.info(
            "rag: 0 chunks returned for query=%r collection=%r — proceeding without context",
            query[:80], collection,
        )
        return messages

    rag_context = format_context(chunks)
    base_prompt = _system_prompt()
    rag_block = (
        "\n\n# Retrieved project context\n\n"
        "These snippets were pulled from the project repository because they may be relevant. "
        "Prefer reusing existing utilities and conventions over inventing new ones. "
        "If a snippet contradicts your prior knowledge, trust the snippet.\n\n"
        + rag_context
    )
    our_content = base_prompt + rag_block

    out: list[dict] = []
    merged = False
    for m in messages:
        if not merged and m.get("role") == "system":
            existing = m.get("content", "") or ""
            out.append({
                "role": "system",
                "content": our_content + "\n\n---\n\n" + existing if existing else our_content,
            })
            merged = True
        else:
            out.append(m)
    if not merged:
        out.insert(0, {"role": "system", "content": our_content})

    quota = intent_quota(query)
    kind_counts = {"code": 0, "doc": 0, "schema": 0}
    for c in chunks:
        k = c.get("kind_quota") or c.get("kind", "?")
        if k in kind_counts:
            kind_counts[k] += 1
    top = [(c.get("path", "?"), c.get("kind_quota") or c.get("kind"), round(c.get("score", 0), 3)) for c in chunks[:5]]
    logger.info(
        "rag: %d chunks for query=%r quota=%s got=%s merged=%s top=%s",
        len(chunks), query[:80], quota, kind_counts, merged, top,
    )
    return out


@app.post("/v1/chat/completions", dependencies=[Depends(require_token)])
async def chat_completions(req: Request):
    body = await req.json()
    model = body.get("model") or settings().mip_primary_model
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")
    stream = bool(body.get("stream", False))
    options = _options_from_openai(body)
    messages = _normalize_messages(messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    client = _client()
    messages = await _augment_with_rag(client, messages, settings().mip_default_collection)

    # Diagnostic: what are we actually sending to Ollama? Helps distinguish
    # "content didn't make it" from "model ignored it".
    sizes = [(m.get("role"), len(m.get("content", "") or "")) for m in messages]
    logger.info("ollama: sending %d messages, sizes=%s", len(messages), sizes)

    if stream:
        async def event_stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in client.chat_stream(model=model, messages=messages, options=options):
                    openai_chunk = _ollama_chunk_to_openai(chunk, model, completion_id)
                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
            except httpx.HTTPStatusError as e:
                err = {"error": {"message": f"ollama upstream {e.response.status_code}: {e.response.text[:400]}", "type": "upstream_error"}}
                yield f"data: {json.dumps(err)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        resp = await client.chat(model=model, messages=messages, options=options)
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"ollama upstream {e.response.status_code}: {e.response.text[:400]}",
        ) from e
    return JSONResponse(_ollama_full_to_openai(resp, model, completion_id))


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    s = settings()
    advertised = [s.mip_primary_model, s.mip_fallback_model]
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": int(time.time()), "owned_by": "mip"}
                for m in advertised
            ],
        }
    )


@app.get("/admin/status")
async def status() -> JSONResponse:
    s = settings()
    return JSONResponse(
        {
            "ok": True,
            "ollama_url": s.ollama_base_url,
            "primary_model": s.mip_primary_model,
            "fallback_model": s.mip_fallback_model,
            "embed_model": s.mip_embed_model,
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})
