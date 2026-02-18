from __future__ import annotations

import numpy as np

from semdedup.similarity import build_similarity_graph


def find_groups(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Find connected components in thresholded similarity graph.

    Embeddings must be L2-normalized. Returns lists of indices for
    multi-member components only (singletons dropped).
    """
    from scipy.sparse.csgraph import connected_components  # type: ignore[reportUnknownVariableType]

    graph = build_similarity_graph(embeddings, threshold)
    n_components: int
    labels: np.ndarray
    n_components, labels = connected_components(graph, directed=False)  # type: ignore[reportUnknownMemberType]

    components: list[list[int]] = [[] for _ in range(n_components)]  # type: ignore[reportUnknownArgumentType]
    for idx, label in enumerate(labels):  # type: ignore[reportUnknownArgumentType]
        components[int(label)].append(idx)

    return [comp for comp in components if len(comp) > 1]
