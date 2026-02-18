from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def build_similarity_graph(embeddings: np.ndarray, threshold: float) -> csr_matrix:
    """Build sparse adjacency matrix from L2-normalized embeddings.

    Embeddings must be L2-normalized so dot product == cosine similarity.
    """
    sim_matrix: np.ndarray = embeddings @ embeddings.T

    sim_matrix[sim_matrix < threshold] = 0.0
    np.fill_diagonal(sim_matrix, 0.0)

    return csr_matrix(sim_matrix)
