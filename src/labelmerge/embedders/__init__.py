from __future__ import annotations

from labelmerge.embedders.openai import OpenAIEmbedder
from labelmerge.embedders.precomputed import PrecomputedEmbedder
from labelmerge.embedders.protocol import EmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbedder",
    "PrecomputedEmbedder",
]
