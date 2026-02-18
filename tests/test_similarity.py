from __future__ import annotations

import numpy as np

from semdedup.components import find_groups
from semdedup.similarity import build_similarity_graph


def test_build_similarity_graph_identical_vectors():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = build_similarity_graph(embeddings, threshold=0.9)
    # Vectors 0 and 1 are identical (sim=1.0), vector 2 is orthogonal (sim=0.0)
    assert graph[0, 1] == 1.0
    assert graph[1, 0] == 1.0
    assert graph[0, 2] == 0.0
    assert graph[2, 0] == 0.0


def test_build_similarity_graph_no_self_loops():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
    graph = build_similarity_graph(embeddings, threshold=0.5)
    assert graph[0, 0] == 0.0
    assert graph[1, 1] == 0.0


def test_find_groups_basic():
    # Three identical vectors and one orthogonal
    embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float64)
    groups = find_groups(embeddings, threshold=0.9)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_find_groups_no_duplicates():
    # All orthogonal — no groups
    embeddings = np.eye(4, dtype=np.float64)
    groups = find_groups(embeddings, threshold=0.5)
    assert len(groups) == 0


def test_find_groups_threshold_sensitivity():
    # Two vectors with sim ~0.87
    v1 = np.array([1.0, 0.5])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array([1.0, 0.6])
    v2 = v2 / np.linalg.norm(v2)
    embeddings = np.array([v1, v2])

    sim = float(np.dot(v1, v2))
    # At threshold below sim → grouped
    groups_low = find_groups(embeddings, threshold=sim - 0.01)
    assert len(groups_low) == 1
    # At threshold above sim → not grouped
    groups_high = find_groups(embeddings, threshold=sim + 0.01)
    assert len(groups_high) == 0
