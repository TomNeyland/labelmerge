from __future__ import annotations

import litellm


class LiteLLMEmbedder:
    """Embedding provider using LiteLLM (supports any provider litellm supports).

    Requires: pip install labelmerge[litellm]
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via LiteLLM."""
        response = await litellm.aembedding(model=self.model, input=texts)  # type: ignore[reportUnknownMemberType]
        return [item["embedding"] for item in response.data]  # type: ignore[reportUnknownVariableType,reportUnknownMemberType]
