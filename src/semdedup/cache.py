from __future__ import annotations

import hashlib
import os

import diskcache


class EmbeddingCache:
    """Content-addressed embedding cache. Never re-embed the same text."""

    def __init__(self, cache_dir: str = "~/.cache/semdedup") -> None:
        self._cache: diskcache.Cache = diskcache.Cache(os.path.expanduser(cache_dir))

    def key(self, text: str, model: str) -> str:
        """Generate cache key from text and model name."""
        return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding, or None if not cached."""
        result = self._cache.get(self.key(text, model))  # type: ignore[reportUnknownMemberType]
        return result  # type: ignore[reportReturnType]

    def put(self, text: str, model: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        self._cache[self.key(text, model)] = embedding

    def get_batch(self, texts: list[str], model: str) -> tuple[list[str], dict[str, list[float]]]:
        """Split texts into uncached (need API call) and cached (have embeddings).

        Returns:
            Tuple of (uncached_texts, cached_dict) where cached_dict maps
            text -> embedding for texts that were found in cache.
        """
        cached: dict[str, list[float]] = {}
        uncached: list[str] = []
        for text in texts:
            emb = self.get(text, model)
            if emb is not None:
                cached[text] = emb
            else:
                uncached.append(text)
        return uncached, cached

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "size": len(self._cache),  # type: ignore[reportArgumentType]
            "volume": self._cache.volume(),  # type: ignore[reportUnknownMemberType]
        }

    def close(self) -> None:
        """Close the cache."""
        self._cache.close()
