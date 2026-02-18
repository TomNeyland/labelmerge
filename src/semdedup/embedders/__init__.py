from __future__ import annotations

from semdedup.embedders.openai import OpenAIEmbedder
from semdedup.embedders.precomputed import PrecomputedEmbedder
from semdedup.embedders.protocol import EmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbedder",
    "PrecomputedEmbedder",
]
