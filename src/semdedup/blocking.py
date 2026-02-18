from __future__ import annotations

import numpy as np

from semdedup.components import find_groups


def find_groups_blocked(
    embeddings: np.ndarray,
    threshold: float,
    n_blocks: int = 100,
) -> list[list[int]]:
    """Blocked pairwise comparison for large corpora (>50K items).

    Uses K-means to partition into coarse blocks, then runs pairwise
    similarity within each block.

    NOTE: Cross-block merge is not yet implemented. Items near block
    boundaries may not be grouped with their true duplicates in other blocks.
    """
    from sklearn.cluster import MiniBatchKMeans  # type: ignore[reportUnknownVariableType]

    kmeans = MiniBatchKMeans(n_clusters=n_blocks, random_state=42)
    block_labels: np.ndarray = kmeans.fit_predict(embeddings)  # type: ignore[reportUnknownMemberType]

    all_groups: list[list[int]] = []
    for block_id in range(n_blocks):
        block_indices = np.where(block_labels == block_id)[0]  # type: ignore[reportUnknownArgumentType]
        if len(block_indices) < 2:
            continue
        block_embeddings = embeddings[block_indices]
        local_groups = find_groups(block_embeddings, threshold)
        for group in local_groups:
            all_groups.append([int(block_indices[i]) for i in group])

    # TODO: cross-block edge detection
    return all_groups
