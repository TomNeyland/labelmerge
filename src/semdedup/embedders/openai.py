from __future__ import annotations

import openai

from semdedup.cache import EmbeddingCache


class OpenAIEmbedder:
    """Embedding provider using the OpenAI API.

    Supports batching and caching. Async-first.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 512,
        cache: EmbeddingCache | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._cache = cache
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via OpenAI API, using cache when available."""
        if self._cache is not None:
            uncached, cached = self._cache.get_batch(texts, self.model)
        else:
            uncached = texts
            cached = {}

        new_embeddings: dict[str, list[float]] = {}
        for i in range(0, len(uncached), self.batch_size):
            batch = uncached[i : i + self.batch_size]
            if self.dimensions is not None:
                response = await self._client.embeddings.create(
                    input=batch, model=self.model, dimensions=self.dimensions
                )
            else:
                response = await self._client.embeddings.create(input=batch, model=self.model)
            for item, text in zip(response.data, batch, strict=True):
                new_embeddings[text] = item.embedding
                if self._cache is not None:
                    self._cache.put(text, self.model, item.embedding)

        result: list[list[float]] = []
        for text in texts:
            if text in cached:
                result.append(cached[text])
            else:
                result.append(new_embeddings[text])
        return result
