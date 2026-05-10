"""Embed text via Ollama using the configured embed model."""
from mip.config import settings
from mip.inference.ollama_client import OllamaClient


async def embed_texts(client: OllamaClient, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return await client.embed(model=settings().mip_embed_model, input_texts=texts)
