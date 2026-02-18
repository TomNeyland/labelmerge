from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from semdedup._stop_words import strip_stop_words
from semdedup.cache import EmbeddingCache
from semdedup.components import find_groups
from semdedup.embedders.openai import OpenAIEmbedder
from semdedup.embedders.protocol import EmbeddingProvider
from semdedup.models import Group, Member, Result
from semdedup.naming import name_groups


class SemDedup:
    """Semantic deduplication: embed text, find near-duplicates, name groups."""

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        threshold: float = 0.85,
        max_component_size: int = 100,
        overflow_threshold_bump: float = 0.05,
        stop_words: list[str] | None = None,
        cache_dir: str = "~/.cache/semdedup",
        cache_enabled: bool = True,
    ) -> None:
        self.threshold = threshold
        self.max_component_size = max_component_size
        self.overflow_threshold_bump = overflow_threshold_bump
        self.stop_words = stop_words if stop_words is not None else []

        self._cache: EmbeddingCache | None = EmbeddingCache(cache_dir) if cache_enabled else None

        if embedder is not None:
            self._embedder = embedder
        else:
            self._embedder = OpenAIEmbedder(cache=self._cache)

        self._model_name: str = self._embedder.model  # type: ignore[attr-defined]

    async def dedupe(self, texts: list[str]) -> Result:
        """Deduplicate a list of text strings.

        Returns a Result with groups and singletons.
        """
        # Deduplicate input and count occurrences
        counts = Counter(texts)
        unique_texts = list(counts.keys())

        # Prepare texts for embedding (strip stop words if configured)
        if self.stop_words:
            embed_texts = [strip_stop_words(t, self.stop_words) for t in unique_texts]
        else:
            embed_texts = unique_texts

        # Embed
        raw_embeddings = await self._embedder.embed(embed_texts)
        embeddings = np.array(raw_embeddings, dtype=np.float64)

        return self._build_result(unique_texts, counts, embeddings)

    def dedupe_precomputed(self, texts: list[str], embeddings: np.ndarray) -> Result:
        """Deduplicate using precomputed embeddings. No API calls."""
        counts = Counter(texts)
        unique_texts = list(counts.keys())

        # If texts has duplicates, we need to select the unique embeddings
        if len(unique_texts) != len(texts):
            seen: dict[str, int] = {}
            unique_indices: list[int] = []
            for i, t in enumerate(texts):
                if t not in seen:
                    seen[t] = i
                    unique_indices.append(i)
            embeddings = embeddings[unique_indices]

        return self._build_result(unique_texts, counts, embeddings)

    async def dedupe_file(
        self,
        path: str | Path,
        path_expr: str | None = None,
        column: str | None = None,
    ) -> Result:
        """Deduplicate texts from a file.

        Supports JSON, JSONL, CSV, and plain text.
        """
        from semdedup.io.readers import read_csv, read_json, read_jsonl, read_text

        file_path = Path(path)
        suffix = file_path.suffix.lower()

        if suffix == ".json" and path_expr is not None:
            texts = read_json(file_path, path_expr)
        elif suffix == ".jsonl" and path_expr is not None:
            texts = read_jsonl(file_path, path_expr)
        elif suffix == ".csv" and column is not None:
            texts = read_csv(file_path, column)
        else:
            texts = read_text(file_path)

        return await self.dedupe(texts)

    async def name_groups(
        self, result: Result, model: str = "gpt-4o-mini", temperature: float = 0.0
    ) -> Result:
        """Name all groups in a result using an LLM."""
        named = await name_groups(result.groups, model=model, temperature=temperature)
        return result.model_copy(update={"groups": named})

    def _build_result(
        self,
        unique_texts: list[str],
        counts: Counter[str],
        embeddings: np.ndarray,
    ) -> Result:
        """Build Result from unique texts, counts, and embeddings."""
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        # Find groups
        component_indices = find_groups(normalized, self.threshold)

        # Handle overflow components (split at higher threshold)
        final_components: list[list[int]] = []
        for comp in component_indices:
            if len(comp) > self.max_component_size:
                sub_embeddings = normalized[comp]
                higher_threshold = self.threshold + self.overflow_threshold_bump
                sub_groups = find_groups(sub_embeddings, higher_threshold)
                for sg in sub_groups:
                    final_components.append([comp[i] for i in sg])
                # Items not in any sub-group become singletons (handled below)
                sub_grouped = {i for sg in sub_groups for i in sg}
                for i, idx in enumerate(comp):
                    if i not in sub_grouped:
                        final_components.append([idx])
            else:
                final_components.append(comp)

        # Track which indices are grouped
        grouped_indices: set[int] = set()
        for comp in final_components:
            if len(comp) > 1:
                grouped_indices.update(comp)

        # Build Group objects
        groups: list[Group] = []
        for group_id, comp in enumerate(final_components):
            if len(comp) < 2:
                continue
            members = [Member(text=unique_texts[i], count=counts[unique_texts[i]]) for i in comp]
            members.sort(key=lambda m: m.count, reverse=True)
            groups.append(
                Group(
                    group_id=group_id,
                    members=members,
                    size=len(members),
                    total_occurrences=sum(m.count for m in members),
                )
            )

        groups.sort(key=lambda g: g.total_occurrences, reverse=True)

        # Build singletons
        singletons = [
            Member(text=unique_texts[i], count=counts[unique_texts[i]])
            for i in range(len(unique_texts))
            if i not in grouped_indices
        ]

        return Result(
            groups=groups,
            singletons=singletons,
            threshold=self.threshold,
            model=self._model_name,
            n_input=len(unique_texts),
            n_grouped=len(grouped_indices),
            n_singletons=len(singletons),
        )
