"""OpenAI-compatible chat client for hosted providers.

Works with anything that speaks `POST /chat/completions` in OpenAI format:
Groq, OpenRouter, Gemini's OpenAI-compat endpoint, OpenAI itself, etc.

Mip receives OpenAI-format requests from OpenCode and (after RAG augmentation)
forwards them mostly verbatim to the upstream — no format translation, just
the body's `messages` get a richer system message and the bearer auth header
flips from Mip's to the provider's.

Embeddings are NOT handled here — `OllamaClient` continues to provide them
locally via `nomic-embed-text`. Hosted providers either don't offer embeddings
on their free tier or charge per token; the local path stays.
"""
import json
from typing import AsyncIterator

import httpx


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=30.0),
        )

    async def chat(self, body: dict) -> dict:
        payload = {**body, "stream": False}
        r = await self._client.post("/chat/completions", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"upstream {r.status_code}: {r.text[:600]}")
        return r.json()

    async def chat_stream(self, body: dict) -> AsyncIterator[dict]:
        payload = {**body, "stream": True}
        async with self._client.stream("POST", "/chat/completions", json=payload) as r:
            if r.status_code >= 400:
                await r.aread()
                raise RuntimeError(f"upstream {r.status_code}: {r.text[:600]}")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    async def aclose(self) -> None:
        await self._client.aclose()
