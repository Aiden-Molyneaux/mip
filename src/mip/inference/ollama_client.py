"""Thin async wrapper around Ollama's /api/chat and /api/embed."""
import json
from typing import AsyncIterator

import httpx

from mip.config import settings


class OllamaClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings().ollama_base_url,
            timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0),
        )

    async def chat(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
    ) -> dict:
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options
        r = await self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        return r.json()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
    ) -> AsyncIterator[dict]:
        payload: dict = {"model": model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options
        async with self._client.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code >= 400:
                # Streaming responses don't load the body until requested. Pull it now
                # so callers can read e.response.text in their HTTPStatusError handler.
                await r.aread()
                r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                yield json.loads(line)

    async def embed(self, model: str, input_texts: list[str]) -> list[list[float]]:
        r = await self._client.post("/api/embed", json={"model": model, "input": input_texts})
        if r.status_code >= 400:
            # Include Ollama's actual error body so callers can see which input was rejected
            # instead of just "400 Bad Request".
            raise RuntimeError(
                f"ollama /api/embed {r.status_code}: {r.text[:600]}"
            )
        return r.json()["embeddings"]

    async def list_models(self) -> list[dict]:
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    async def aclose(self) -> None:
        await self._client.aclose()
