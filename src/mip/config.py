from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mip_host: str = "0.0.0.0"
    mip_port: int = 8088
    mip_bearer_token: str = "change-me"

    # Chat backend selection.
    #   "openai_compat" — forward to any OpenAI-format provider (Groq, OpenRouter,
    #                     Gemini OpenAI endpoint, OpenAI). Uses mip_chat_base_url
    #                     and mip_chat_api_key.
    #   "ollama"        — local Ollama at ollama_base_url. Used for both chat and
    #                     embeddings; kept around for offline/private use.
    mip_chat_backend: Literal["openai_compat", "ollama"] = "openai_compat"
    # Default URL is Groq. Switch to any OpenAI-compatible endpoint by changing this:
    #   OpenRouter: https://openrouter.ai/api/v1
    #   Gemini:     https://generativelanguage.googleapis.com/v1beta/openai
    #   OpenAI:     https://api.openai.com/v1
    mip_chat_base_url: str = "https://api.groq.com/openai/v1"
    mip_chat_api_key: str = ""

    # Local Ollama — always used for embeddings; also used for chat when
    # mip_chat_backend == "ollama".
    ollama_base_url: str = "http://localhost:11434"
    mip_embed_model: str = "nomic-embed-text"

    # Default model the client sees. Should match the chat backend's naming:
    #   Groq examples:   llama-3.3-70b-versatile, qwen-2.5-coder-32b
    #   Ollama examples: qwen2.5-coder:14b
    mip_primary_model: str = "llama-3.3-70b-versatile"
    mip_fallback_model: str = "llama-3.1-8b-instant"

    qdrant_url: str = "http://localhost:6333"
    mip_default_collection: str = "inthink"

    mip_state_db: str = "./data/state.db"
    mip_log_level: str = "INFO"


@lru_cache
def settings() -> Settings:
    return Settings()
