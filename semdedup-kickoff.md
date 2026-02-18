# semdedup — Project Kickoff Document

> Working name. Final name TBD. Check PyPI availability before committing.
> Created: 2026-02-17

---

## What Is This

A standalone Python package for **semantic deduplication of text labels**. Given a corpus of messy free-text values that should be a controlled vocabulary, semdedup embeds them, finds near-duplicates via connected components on a cosine similarity graph, and optionally names the resulting groups.

The core insight: this is an **entity resolution** problem, not a clustering problem. The algorithm is SemDeDup (Abbas et al. 2023) simplified: embed, threshold, connected components. One parameter (similarity threshold), deterministic output, no parameter sweeps.

---

## Origin Story

Built out of a real problem in an NLP pipeline. A free-text field on extracted clinical paper outcomes had 2,313 unique values across 3,281 fills — 70% unique, 83% singletons. The LLM generating them was inconsistent:

```
depression severity          (16x)
depressive symptoms          (13x)
depressive symptom severity  (11x)
depression / depressive symptoms (8x)
```

These are all the same concept. A POC using connected components at cosine threshold 0.85 produced 268 groups: 60% TIGHT (true synonyms), 37% LOOSE (related, minor chaining), 3% JUNK (transitive chaining). The approach works.

The problem is universal. Any corpus of human- or LLM-generated text labels benefits from this: product categories, support ticket tags, survey responses, medical codes, job titles, ingredient names, error messages.

---

## Core Algorithm

### Connected Components at Cosine Similarity Threshold

```
Input:  list of text strings
Output: list of groups (each group = near-duplicate strings)

1. Embed all strings → vectors (via OpenAI, or any embedding provider)
2. L2-normalize vectors
3. Compute pairwise cosine similarity (dot product on normalized vectors)
4. Build graph: edge between items with sim >= threshold (e.g. 0.85)
5. Find connected components → each component is a group
6. Singletons are items that matched nothing
```

**Why connected components, not clustering:**
- Directly optimizes the objective: "items within cosine distance X always co-locate"
- One parameter (threshold), not k or resolution
- Deterministic — same input always produces same output
- No parameter sweep needed
- Guaranteed: if A~B and B~C, then A, B, C co-locate (by definition)

**Known failure mode — transitive chaining:**
If A~B (0.86) and B~C (0.86) but A~C (0.70), all three end up in the same component even though A and C aren't similar. This creates monster groups for broad terms ("inflammatory cytokine" chains TNF-α → IL-6 → CRP → inflammation → ...).

**Mitigation:** Max component size cap. If a component exceeds N items, re-run at a higher threshold (e.g. 0.90) to split it. Or use balanced graph partitioning (minimize cut edges = don't split high-similarity pairs).

### SemDeDup Blocking (for scale)

At >50K items, pairwise similarity is O(N^2) and the similarity matrix doesn't fit in memory. SemDeDup (Meta/NVIDIA) adds a blocking step:

```
1. K-means into coarse blocks (just for compute reduction)
2. Pairwise cosine similarity WITHIN each block
3. Threshold → connected components within blocks
4. Cross-block dedup: check block centroids against each other
```

The K-means clustering doesn't need to be good — it just needs to not split true duplicates. Use a large K (oversegment) so blocks are small enough for pairwise comparison.

For <50K items (most use cases), skip blocking entirely and go straight to pairwise.

### Optional: Stop Word Stripping

For domain-specific corpora, surface lexical features can dominate embeddings. Example from biomedical data:

- "serum cortisol concentration" and "plasma CoQ10 concentration" cluster together because "concentration" dominates
- "NF-κB expression" and "BDNF expression" cluster together because "expression" dominates

Fix: user-supplied stop word list, stripped before embedding but preserved in output mapping.

```
Input:  "serum cortisol concentration"
Embed:  "cortisol"  (stripped "serum", "concentration")
Output: group contains "serum cortisol concentration" (original preserved)
```

### Optional: LLM Naming

After grouping, optionally send each group's members to an LLM to generate a canonical name:

```
Cluster members:
  16x  depression severity
  13x  depressive symptoms
  11x  depressive symptom severity
   8x  depression / depressive symptoms

→ LLM output: "depressive symptoms"
```

This is the most expensive step and is optional. Users who just want dedup don't need naming.

---

## Input Formats

The tool should accept text from multiple sources:

| Format | Example | Extraction |
|--------|---------|------------|
| **JSON** | `[{"label": "foo"}, ...]` | jq-style path: `.[].label` |
| **JSONL** | `{"label": "foo"}\n{"label": "bar"}` | jq-style path: `.label` |
| **CSV/TSV** | Column name | `--column label` |
| **Plain text** | One item per line | No extraction needed |
| **XML** | XPath expression | `--xpath //item/label` |
| **Stdin pipe** | Stream of lines | `cat labels.txt \| semdedup` |
| **Python API** | `list[str]` | Direct |

Priority: JSON/JSONL and plain text first. CSV second. XML if someone asks.

---

## Output Formats

| Format | Description |
|--------|------------|
| **JSON** (default) | `{"groups": [{"canonical": "foo", "members": ["foo", "Foo", "FOO"], "count": 3}], "singletons": ["bar", "baz"]}` |
| **JSONL** | One group per line |
| **CSV** | `original,canonical,group_id` — flat mapping |
| **Mapping JSON** | `{"foo": "foo", "Foo": "foo", "FOO": "foo"}` — direct lookup table |

The **mapping** output is the most useful for downstream consumption: feed it back into your pipeline to normalize values.

---

## CLI Design

```bash
# Basic: deduplicate a text file
semdedup dedupe labels.txt

# JSON input with path extraction
semdedup dedupe data.json --path ".[].label"

# CSV input
semdedup dedupe data.csv --column "category"

# Tune threshold
semdedup dedupe labels.txt --threshold 0.90

# Different embedding model
semdedup dedupe labels.txt --model text-embedding-3-large --dimensions 512

# Output as mapping (for pipeline integration)
semdedup dedupe labels.txt --output-format mapping > mapping.json

# With LLM naming
semdedup dedupe labels.txt --name-groups --naming-model gpt-4o-mini

# With stop words
semdedup dedupe labels.txt --stop-words "concentration,level,activity,expression"

# Sweep thresholds to find the right one
semdedup sweep labels.txt --thresholds 0.80,0.85,0.90,0.95

# Inspect a specific group
semdedup inspect groups.json --group 5

# Stats overview
semdedup stats labels.txt
```

### Subcommands

| Command | Description |
|---------|-------------|
| `dedupe` | Core: embed → threshold → connected components → output groups |
| `sweep` | Run at multiple thresholds, show group count / singleton count / max group size |
| `stats` | Quick stats: unique count, estimated duplicate rate (via sampling), cardinality |
| `inspect` | Interactive exploration of a group file |
| `cache` | Manage embedding cache (`cache clear`, `cache stats`, `cache export`) |

---

## Python API

```python
from semdedup import SemDedup, OpenAIEmbedder

# Minimal
sd = SemDedup()
result = await sd.dedupe(["foo", "Foo", "FOO", "bar", "baz"])
# result.groups = [Group(canonical="foo", members=["foo", "Foo", "FOO"])]
# result.singletons = ["bar", "baz"]

# With config
sd = SemDedup(
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    threshold=0.85,
    max_component_size=50,  # split large components at higher threshold
    stop_words=["concentration", "level"],
)
result = await sd.dedupe(texts)

# Get flat mapping
mapping = result.to_mapping()
# {"foo": "foo", "Foo": "foo", "FOO": "foo", "bar": "bar", "baz": "baz"}

# Name groups with LLM
named = await sd.name_groups(result, model="gpt-4o-mini")

# From file
result = await sd.dedupe_file("data.json", path=".[].label")

# Bring your own embeddings (skip API calls entirely)
import numpy as np
embeddings = np.load("my_embeddings.npy")
texts = open("labels.txt").read().splitlines()
result = sd.dedupe_precomputed(texts, embeddings)
```

### Embedding Provider Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...
```

Built-in providers:
- `OpenAIEmbedder` — default, uses `openai` package
- `PrecomputedEmbedder` — wraps a numpy array (no API calls)

Optional extras:
- `pip install semdedup[litellm]` → `LiteLLMEmbedder` (any provider litellm supports)
- `pip install semdedup[sentence-transformers]` → `SentenceTransformerEmbedder` (local models, no API)

---

## Architecture

```
src/semdedup/
├── __init__.py          # Public API: SemDedup, Group, Result, EmbeddingProvider
├── py.typed             # PEP 561 marker
├── core.py              # SemDedup class — orchestrates embed → similarity → components
├── similarity.py        # Pairwise cosine similarity (numpy/scipy)
├── components.py        # Connected components (scipy.sparse.csgraph)
├── blocking.py          # K-means blocking for scale (>50K items)
├── embedders/
│   ├── __init__.py
│   ├── protocol.py      # EmbeddingProvider protocol
│   ├── openai.py        # OpenAIEmbedder (default)
│   ├── precomputed.py   # PrecomputedEmbedder
│   ├── litellm.py       # LiteLLMEmbedder (optional extra)
│   └── sentence.py      # SentenceTransformerEmbedder (optional extra)
├── naming.py            # LLM-based group naming (optional)
├── cache.py             # Embedding cache (diskcache, keyed by sha256)
├── io/
│   ├── __init__.py
│   ├── readers.py       # JSON/JSONL/CSV/text/XML readers
│   └── writers.py       # JSON/JSONL/CSV/mapping writers
├── models.py            # Pydantic models: Group, Result, SweepResult
├── config.py            # pydantic-settings: SemDedupConfig
├── cli.py               # Typer CLI app
└── _stop_words.py       # Stop word stripping utilities
```

### Key Design Decisions

1. **scipy.sparse.csgraph for connected components** — no graph library dependency. The similarity matrix IS already a sparse matrix after thresholding. `connected_components()` is one function call.

2. **Async-first** — embedding calls are I/O bound. `asyncio.gather()` for parallel batched embedding. The CLI wraps async with `asyncio.run()`.

3. **Embedding cache** — `diskcache` keyed by `sha256(text + model_name)`. Never re-embed the same text twice. Biggest cost saver. Cache location configurable, defaults to `~/.cache/semdedup/`.

4. **No clustering algorithms** — this is entity resolution, not clustering. Connected components at threshold. No k-means (except for blocking at scale), no agglomerative, no DBSCAN, no Leiden.

5. **Precomputed embeddings path** — users who already have embeddings skip all API calls. Pass numpy array + text list, get groups back. This also enables non-OpenAI embeddings without any provider integration.

6. **Stop words stripped before embedding, preserved in output** — the mapping always uses original text as keys.

---

## Data Models

```python
from pydantic import BaseModel

class Member(BaseModel):
    """One text item in a group."""
    text: str           # Original text
    count: int = 1      # Frequency in input (if duplicates exist)

class Group(BaseModel):
    """A set of near-duplicate text items."""
    group_id: int
    members: list[Member]           # Sorted by count desc
    canonical: str | None = None    # LLM-named canonical form (None if not named)
    size: int                       # len(members)
    total_occurrences: int          # sum of member counts

class Result(BaseModel):
    """Output of a dedup run."""
    groups: list[Group]             # Multi-member components, sorted by total_occurrences desc
    singletons: list[Member]        # Items that matched nothing
    threshold: float                # The threshold used
    model: str                      # Embedding model used
    n_input: int                    # Total unique input items
    n_grouped: int                  # Items that ended up in a group
    n_singletons: int               # Items that matched nothing

    def to_mapping(self) -> dict[str, str]:
        """Flat original → canonical mapping. Uses most frequent member as canonical if not LLM-named."""
        mapping = {}
        for group in self.groups:
            canonical = group.canonical or group.members[0].text
            for member in group.members:
                mapping[member.text] = canonical
        for singleton in self.singletons:
            mapping[singleton.text] = singleton.text
        return mapping

class SweepResult(BaseModel):
    """Output of a threshold sweep."""
    thresholds: list[float]
    results: list[ThresholdStats]

class ThresholdStats(BaseModel):
    threshold: float
    n_groups: int
    n_singletons: int
    max_group_size: int
    median_group_size: float
    pct_grouped: float              # % of items in a group
```

---

## Configuration

```python
from pydantic_settings import BaseSettings

class SemDedupConfig(BaseSettings):
    # Embedding
    openai_api_key: str = ""                    # Falls back to OPENAI_API_KEY env var
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None     # None = model default
    embedding_batch_size: int = 512             # Texts per API call

    # Dedup
    similarity_threshold: float = 0.85
    max_component_size: int = 100               # Re-cluster at higher threshold if exceeded
    overflow_threshold_bump: float = 0.05       # How much to raise threshold for overflow components

    # Cache
    cache_enabled: bool = True
    cache_dir: str = "~/.cache/semdedup"

    # Naming (optional)
    naming_model: str = "gpt-4o-mini"
    naming_temperature: float = 0.0

    model_config = {"env_prefix": "SEMDEDUP_"}
```

---

## Embedding Cache Design

```python
import hashlib
import diskcache

class EmbeddingCache:
    """Content-addressed embedding cache. Never re-embed the same text."""

    def __init__(self, cache_dir: str = "~/.cache/semdedup"):
        self._cache = diskcache.Cache(os.path.expanduser(cache_dir))

    def key(self, text: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        return self._cache.get(self.key(text, model))

    def put(self, text: str, model: str, embedding: list[float]) -> None:
        self._cache[self.key(text, model)] = embedding

    def get_batch(self, texts: list[str], model: str) -> tuple[list[str], dict[str, list[float]]]:
        """Split texts into cached (have embeddings) and uncached (need API call)."""
        cached = {}
        uncached = []
        for text in texts:
            emb = self.get(text, model)
            if emb is not None:
                cached[text] = emb
            else:
                uncached.append(text)
        return uncached, cached
```

The cache is the single biggest cost optimization. On re-runs with overlapping corpora, embedding API calls drop to near zero.

---

## Similarity Matrix Construction

For small corpora (<50K items):
```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def build_similarity_graph(embeddings: np.ndarray, threshold: float) -> csr_matrix:
    """Build sparse adjacency matrix from L2-normalized embeddings."""
    # embeddings already L2-normalized, so dot product = cosine similarity
    sim_matrix = embeddings @ embeddings.T

    # Zero out below threshold (make sparse)
    sim_matrix[sim_matrix < threshold] = 0.0
    np.fill_diagonal(sim_matrix, 0.0)  # No self-loops

    return csr_matrix(sim_matrix)

def find_groups(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Find connected components in thresholded similarity graph."""
    graph = build_similarity_graph(embeddings, threshold)
    n_components, labels = connected_components(graph, directed=False)

    components: list[list[int]] = [[] for _ in range(n_components)]
    for idx, label in enumerate(labels):
        components[label].append(idx)

    return [comp for comp in components if len(comp) > 1]  # Drop singletons
```

For large corpora (>50K items), the dense `embeddings @ embeddings.T` matrix doesn't fit in memory. Use blocking:

```python
from sklearn.cluster import MiniBatchKMeans

def find_groups_blocked(embeddings: np.ndarray, threshold: float, n_blocks: int = 100) -> list[list[int]]:
    """Blocked pairwise comparison for large corpora."""
    # Step 1: Coarse blocking with K-means (doesn't need to be good)
    kmeans = MiniBatchKMeans(n_clusters=n_blocks, random_state=42)
    block_labels = kmeans.fit_predict(embeddings)

    # Step 2: Pairwise within each block
    all_groups = []
    for block_id in range(n_blocks):
        block_indices = np.where(block_labels == block_id)[0]
        if len(block_indices) < 2:
            continue
        block_embeddings = embeddings[block_indices]
        local_groups = find_groups(block_embeddings, threshold)
        # Map back to global indices
        for group in local_groups:
            all_groups.append([block_indices[i] for i in group])

    # Step 3: Cross-block merge (check if any items near block boundaries should merge)
    # TODO: implement cross-block edge detection
    return all_groups
```

---

## Tooling Stack

| Decision | Choice | Rationale |
|---|---|---|
| Package manager | **uv** | 2026 consensus. Rust-fast. Replaces Poetry/pip/venv/pyenv in one tool |
| Build backend | **uv_build** | Zero config, integrated with uv |
| Python minimum | **3.12** | `tomllib` in stdlib, PEP 695, 3.10 EOL Oct 2026 |
| Testing | **pytest + pytest-asyncio + pytest-cov** | Only choice. Add hypothesis for property-based graph invariant tests |
| Linter/formatter | **ruff** | Replaces black+flake8+isort. 100-200x faster |
| Type checker | **pyright strict** | 3-5x faster than mypy. Checks unannotated code. Watch ty (Astral, beta) |
| CLI | **typer + rich** | Click underneath, type-hint-driven, progress bars |
| Similarity math | **numpy + scipy** | L2-norm + dot product + sparse csgraph connected_components |
| Graph | **scipy.sparse.csgraph** | No extra dep. Similarity matrix IS the graph |
| Embedding provider | **Protocol + OpenAI default** | litellm/sentence-transformers as optional extras |
| Config | **pydantic-settings** | Type-safe, .env support, already using pydantic |
| Embedding cache | **diskcache** | Content-addressed, sha256 keyed, thread-safe |
| Docs | **mkdocs-material** | Watch Zensical (successor, same team) |
| Publishing | **PyPI OIDC Trusted Publishers** | No API tokens. GitHub Actions OIDC |
| CI | **GitHub Actions** | astral-sh/setup-uv@v7, matrix 3.12+3.13 |
| Pre-commit | **ruff + pyright** | Catch issues before CI |
| Changelog | **git-cliff** | Auto-generate from conventional commits |
| Versioning | **python-semantic-release** | Auto-bump from commit messages |
| License | **MIT** | Simple, permissive, universal |
| Layout | **src/semdedup/** | Avoids accidental imports from working directory |

---

## pyproject.toml (Complete)

```toml
[project]
name = "semdedup"  # TBD — check PyPI availability
version = "0.1.0"
description = "Semantic deduplication: embed text, find near-duplicates, name groups"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
authors = [
    { name = "Tom Neyland" },
]
keywords = ["deduplication", "embeddings", "nlp", "clustering", "entity-resolution"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Text Processing",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]

dependencies = [
    "openai>=1.0",
    "numpy>=2.0",
    "scipy>=1.14",
    "pydantic>=2.0",
    "pydantic-settings>=2.5",
    "typer>=0.12",
    "rich>=13.0",
    "diskcache>=5.6",
]

[project.optional-dependencies]
litellm = ["litellm>=1.0"]
local = ["sentence-transformers>=3.0"]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.25",
    "pytest-cov>=6.0",
    "pytest-xdist>=3.5",
    "hypothesis>=6.100",
    "ruff>=0.15",
    "pyright>=1.1",
]
docs = [
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.26",
]

[project.scripts]
semdedup = "semdedup.cli:app"

[project.urls]
Homepage = "https://github.com/TomNeyland/semdedup"
Documentation = "https://tomneyland.github.io/semdedup"
Repository = "https://github.com/TomNeyland/semdedup"
Issues = "https://github.com/TomNeyland/semdedup/issues"

[build-system]
requires = ["uv_build"]
build-backend = "uv_build"

# --- Tool Config ---

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ANN"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["ANN"]  # Don't require annotations in tests

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "--cov=semdedup --cov-report=term-missing"

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:"]
```

---

## CI/CD

### `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v7
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
          cache-dependency-glob: "uv.lock"
      - run: uv sync --all-extras --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright
      - run: uv run pytest
```

### `.github/workflows/release.yml`

```yaml
name: Release
on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v7
      - run: uv build
      - run: uv publish
```

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## Testing Strategy

### Unit Tests

```
tests/
├── test_core.py           # SemDedup orchestration
├── test_similarity.py     # Pairwise similarity math
├── test_components.py     # Connected components correctness
├── test_blocking.py       # K-means blocking for scale
├── test_cache.py          # Embedding cache hit/miss/eviction
├── test_models.py         # Pydantic model validation
├── test_io/
│   ├── test_readers.py    # JSON/CSV/text/JSONL readers
│   └── test_writers.py    # Output format writers
├── test_stop_words.py     # Stop word stripping
├── test_naming.py         # LLM naming (mocked)
└── test_cli.py            # CLI integration tests (typer CliRunner)
```

### Property-Based Tests (Hypothesis)

Critical invariants to test:

```python
from hypothesis import given
import hypothesis.strategies as st

@given(st.lists(st.text(min_size=1), min_size=2, max_size=100))
def test_component_count_never_exceeds_input(texts):
    """Number of components <= number of unique texts."""
    ...

@given(st.floats(min_value=0.0, max_value=1.0))
def test_higher_threshold_fewer_groups(threshold):
    """Raising threshold should never increase group count."""
    ...

def test_deterministic():
    """Same input always produces same output."""
    ...

def test_singleton_coverage():
    """Every input text appears in exactly one group or singleton."""
    ...

def test_mapping_is_complete():
    """to_mapping() has an entry for every input text."""
    ...
```

### Integration Tests

- Real OpenAI API calls (gated behind `SEMDEDUP_INTEGRATION_TESTS=1` env var)
- Round-trip: text file → CLI dedupe → mapping → verify mapping is valid
- Cache: first run calls API, second run hits cache (zero API calls)

---

## POC Results

These validate the approach and inform threshold selection:

### POC Threshold Sweep Results

| Threshold | Groups | Singletons | Max group | Items grouped |
|---|---|---|---|---|
| 0.80 | 302 | 1,142 | 85 | 51% |
| 0.85 | 268 | 1,496 | 54 | 35% |
| 0.90 | 151 | 1,956 | 11 | 15% |
| 0.95 | 47 | 2,210 | 5 | 4% |

0.85 is a good default — captures most true duplicates while keeping groups tight. The monster group at 54 members (inflammatory cytokines) is the transitive chaining problem, fixable with max component size cap.

---

## Use Cases

| Domain | Messy Input | After Dedup |
|--------|------------|-------------|
| E-commerce | "iPhone 15 Pro", "Apple iPhone 15Pro", "iphone 15 pro max" | Product normalization |
| Support tickets | "can't login", "login broken", "unable to sign in", "authentication failed" | Ticket categorization |
| Surveys | "work-life balance", "work life balance", "balancing work and life" | Response coding |
| Medical | "Type 2 diabetes mellitus", "T2DM", "diabetes type II", "adult-onset diabetes" | Terminology normalization |
| Job postings | "Senior Software Engineer", "Sr. SWE", "Senior Dev", "Lead Developer" | Title normalization |
| Error logs | "Connection refused", "ECONNREFUSED", "connection reset by peer" | Error dedup |
| Ingredients | "Vitamin B12", "cyanocobalamin", "methylcobalamin", "B-12" | Ingredient matching |

---

## Roadmap

### v0.1.0 — MVP

- [ ] Core: embed → threshold → connected components → groups
- [ ] OpenAI embedder (async, batched)
- [ ] Embedding cache (diskcache)
- [ ] JSON/JSONL/text input
- [ ] JSON/mapping output
- [ ] CLI: `dedupe`, `sweep`, `stats`
- [ ] Stop word stripping
- [ ] Max component size cap with threshold escalation
- [ ] Precomputed embeddings path
- [ ] Tests with >90% coverage
- [ ] Docs site
- [ ] PyPI publish

### v0.2.0 — Provider Ecosystem

- [ ] `EmbeddingProvider` protocol finalized
- [ ] litellm optional extra
- [ ] sentence-transformers optional extra (local models)
- [ ] CSV input/output
- [ ] JSONL streaming for large files

### v0.3.0 — Scale

- [ ] K-means blocking for >50K items
- [ ] Cross-block edge detection
- [ ] Memory-efficient sparse similarity (no dense NxN)
- [ ] Benchmarks (pytest-benchmark, CI regression detection)

### v1.0.0 — Stable

- [ ] Stable public API
- [ ] LLM naming as first-class feature
- [ ] XML input (if requested)
- [ ] Plugin system for custom embedders/namers

---

## Open Questions

1. **Package name** — `semdedup`? `textdedup`? `vocabforge`? `termfold`? Check PyPI availability.
2. **Naming LLM** — which model for group naming? gpt-4o-mini is cheap and good enough. Should it be configurable or hardcoded?
3. **Threshold default** — 0.85 worked well for biomedical text. Is it universal? Need testing across domains.
4. **Embedding model default** — `text-embedding-3-small` (cheap, fast) vs `text-embedding-3-large` (better quality). Small is probably fine for dedup.
5. **Transitive chaining mitigation** — max component size cap vs balanced graph partitioning vs just using a higher threshold. Cap is simplest.
6. **Should we support count-weighted dedup?** — if "depression severity" appears 16x and "depressive symptoms" appears 13x, the canonical name should be weighted toward the more frequent variant. Currently the most frequent member is used as canonical (unless LLM-named).

---

## References

- Abbas et al. (2023). "SemDeDup: Data-efficient learning at web-scale through semantic deduplication." arXiv:2303.09540.
- [Connected components — SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html)
- [text-embedding-3 models — OpenAI](https://platform.openai.com/docs/guides/embeddings)
- [uv — Astral](https://docs.astral.sh/uv/)
- [ruff — Astral](https://docs.astral.sh/ruff/)
- [Typer — FastAPI team](https://typer.tiangolo.com/)
- [diskcache — Grant Jenks](https://grantjenks.com/docs/diskcache/)
