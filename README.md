# semdedup

Semantic deduplication of text labels. Given messy free-text values that should be a controlled vocabulary, semdedup embeds them, finds near-duplicates via connected components on a cosine similarity graph, and optionally names the resulting groups.

One parameter (similarity threshold). Deterministic output. No parameter sweeps.

## Install

```bash
pip install semdedup
```

## Quick Start

### Python API

```python
from semdedup import SemDedup

sd = SemDedup(threshold=0.85)
result = await sd.dedupe([
    "depression severity",
    "depressive symptoms",
    "depressive symptom severity",
    "depression / depressive symptoms",
    "sleep quality",
])

# result.groups = [Group(members=["depression severity", "depressive symptoms", ...], size=4)]
# result.singletons = [Member(text="sleep quality")]

# Flat mapping for pipeline integration
mapping = result.to_mapping()
# {"depression severity": "depression severity", "depressive symptoms": "depression severity", ...}
```

### CLI

```bash
# Deduplicate a text file (one label per line)
semdedup dedupe labels.txt

# JSON input with path extraction
semdedup dedupe data.json --path ".[].label"

# CSV input
semdedup dedupe data.csv --column category

# Tune threshold
semdedup dedupe labels.txt --threshold 0.90

# Output as mapping (for pipeline integration)
semdedup dedupe labels.txt --output-format mapping > mapping.json

# Name groups with LLM
semdedup dedupe labels.txt --name-groups

# Sweep thresholds
semdedup sweep labels.txt --thresholds 0.80,0.85,0.90,0.95

# Quick stats
semdedup stats labels.txt
```

## How It Works

```
1. Embed all strings → vectors (via OpenAI text-embedding-3-small)
2. L2-normalize vectors
3. Compute pairwise cosine similarity (dot product on normalized vectors)
4. Build graph: edge between items with similarity >= threshold
5. Find connected components → each component is a group
6. Singletons are items that matched nothing
```

This is **entity resolution**, not clustering. Connected components at a cosine similarity threshold. One parameter, deterministic, no parameter sweeps.

**Known tradeoff — transitive chaining:** If A~B and B~C but A≠C, all three end up in the same group. Mitigated by `max_component_size` which re-runs at a higher threshold to split oversized groups.

## Configuration

All settings via environment variables (prefix `SEMDEDUP_`):

| Variable | Default | Description |
|---|---|---|
| `SEMDEDUP_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold |
| `SEMDEDUP_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `SEMDEDUP_EMBEDDING_BATCH_SIZE` | `512` | Texts per API call |
| `SEMDEDUP_CACHE_DIR` | `~/.cache/semdedup` | Embedding cache directory |
| `SEMDEDUP_CACHE_ENABLED` | `true` | Enable embedding cache |
| `SEMDEDUP_MAX_COMPONENT_SIZE` | `100` | Split groups larger than this |

## Embedding Providers

```python
# Default: OpenAI
from semdedup import SemDedup, OpenAIEmbedder
sd = SemDedup(embedder=OpenAIEmbedder(model="text-embedding-3-large"))

# Precomputed embeddings (no API calls)
from semdedup import SemDedup, PrecomputedEmbedder
import numpy as np
embeddings = np.load("my_embeddings.npy")
sd = SemDedup(embedder=PrecomputedEmbedder(embeddings))
result = sd.dedupe_precomputed(texts, embeddings)

# LiteLLM (any provider)
# pip install semdedup[litellm]
from semdedup.embedders.litellm import LiteLLMEmbedder
sd = SemDedup(embedder=LiteLLMEmbedder(model="text-embedding-3-small"))

# Local models (no API)
# pip install semdedup[local]
from semdedup.embedders.sentence import SentenceTransformerEmbedder
sd = SemDedup(embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2"))
```

## License

MIT
