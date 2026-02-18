from __future__ import annotations

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Embedding provider using sentence-transformers (local models, no API).

    Requires: pip install labelmerge[local]
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model = model
        self._st = SentenceTransformer(model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using a local sentence-transformer model."""
        embeddings = self._st.encode(texts, convert_to_numpy=True)  # type: ignore[reportUnknownMemberType]
        return embeddings.tolist()  # type: ignore[reportUnknownMemberType]
