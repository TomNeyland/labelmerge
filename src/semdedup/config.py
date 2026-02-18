from __future__ import annotations

from pydantic_settings import BaseSettings


class SemDedupConfig(BaseSettings):
    """Configuration for semdedup, loaded from environment variables."""

    # Embedding
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None
    embedding_batch_size: int = 512

    # Dedup
    similarity_threshold: float = 0.85
    max_component_size: int = 100
    overflow_threshold_bump: float = 0.05

    # Cache
    cache_enabled: bool = True
    cache_dir: str = "~/.cache/semdedup"

    # Naming (optional)
    naming_model: str = "gpt-4o-mini"
    naming_temperature: float = 0.0

    model_config = {"env_prefix": "SEMDEDUP_"}
