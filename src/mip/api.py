"""FastAPI app exposing an OpenAI-compatible surface backed by Ollama.

Endpoints:
  POST /v1/chat/completions   OpenAI chat-completions, streaming or non-streaming. Auth required.
  GET  /v1/models             OpenAI model list. No auth — clients use this for URL validation.
  GET  /admin/status          Liveness + config snapshot.
  GET  /health                Process-level liveness.

Auth accepts either:
  Authorization: Bearer <MIP_BEARER_TOKEN>
  Authorization: Basic base64("<anything>:<MIP_BEARER_TOKEN>")
The Basic form lets clients (e.g. OpenCode) that expose Username/Password fields
authenticate by putting the token in the password field.
"""
import base64
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mip.config import settings
from mip.inference.ollama_client import OllamaClient

_ollama: OllamaClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ollama
    _ollama = OllamaClient()
    yield
    if _ollama is not None:
        await _ollama.aclose()


app = FastAPI(title="Mip", lifespan=lifespan)


def _client() -> OllamaClient:
    if _ollama is None:
        raise RuntimeError("Ollama client not initialized — startup did not complete")
    return _ollama


def require_token(authorization: str | None = Header(default=None)) -> None:
    if authorization is None:
        raise HTTPException(status_code=401, detail="missing Authorization header")
    token = settings().mip_bearer_token

    if authorization == f"Bearer {token}":
        return

    if authorization.startswith("Basic "):
        try:
            decoded = base64.b64decode(authorization[len("Basic "):]).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=401, detail=f"invalid Basic credentials: {e}") from e
        _user, sep, password = decoded.partition(":")
        if sep and password == token:
            return

    raise HTTPException(status_code=401, detail="invalid credentials")


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
    return options


@app.post("/v1/chat/completions", dependencies=[Depends(require_token)])
async def chat_completions(req: Request):
    body = await req.json()
    model = body.get("model") or settings().mip_primary_model
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")
    stream = bool(body.get("stream", False))
    options = _options_from_openai(body)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    client = _client()

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
