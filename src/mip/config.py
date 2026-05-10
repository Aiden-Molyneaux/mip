from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mip_host: str = "0.0.0.0"
    mip_port: int = 8088
    mip_bearer_token: str = "change-me"

    ollama_base_url: str = "http://localhost:11434"
    mip_primary_model: str = "qwen2.5-coder:14b"
    mip_fallback_model: str = "qwen2.5-coder:7b"
    mip_embed_model: str = "nomic-embed-text"

    qdrant_url: str = "http://localhost:6333"
    mip_default_collection: str = "inthink"

    mip_state_db: str = "./data/state.db"
    mip_log_level: str = "INFO"


@lru_cache
def settings() -> Settings:
    return Settings()
