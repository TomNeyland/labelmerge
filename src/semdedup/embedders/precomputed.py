from __future__ import annotations

import numpy as np


class PrecomputedEmbedder:
    """Wraps a numpy array of precomputed embeddings.

    No API calls — just returns rows from the provided matrix.
    Texts are matched by index position.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings
        self._index = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return the next len(texts) rows from the precomputed matrix."""
        start = self._index
        end = start + len(texts)
        batch: np.ndarray = self._embeddings[start:end]
        self._index = end
        return batch.tolist()
