from __future__ import annotations

from semdedup.core import SemDedup
from semdedup.embedders.openai import OpenAIEmbedder
from semdedup.embedders.precomputed import PrecomputedEmbedder
from semdedup.embedders.protocol import EmbeddingProvider
from semdedup.models import Group, Member, Result

__all__ = [
    "EmbeddingProvider",
    "Group",
    "Member",
    "OpenAIEmbedder",
    "PrecomputedEmbedder",
    "Result",
    "SemDedup",
]
