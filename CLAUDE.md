# semdedup

Semantic deduplication of text labels via connected components on a cosine similarity graph. One parameter (threshold), deterministic output. Entity resolution, not clustering.

**Origin:** @semdedup-kickoff.md has the full spec, algorithm details, and design decisions.

## This Repo Is 100% AI-Built

No humans write code here. Every session is issue-driven:
1. gh-prime.py auto-loads open issues on session start
2. Pick an issue (or create one), implement it, test it
3. Commit, push, close the issue — in that order
4. Use `/gh-issues` skill for epics, sub-issues, and dependencies

## Tech Stack

Python 3.12+ | uv | numpy/scipy | openai | pydantic | typer/rich | diskcache

## Commands

```bash
uv sync --all-extras          # Install everything
uv run pytest                 # Run tests
uv run pytest tests/test_X.py # Single test file
uv run ruff check .           # Lint
uv run ruff format .          # Format
uv run pyright                # Type check
# All checks:
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

## Architecture

```
src/semdedup/
  __init__.py          Public API: SemDedup, Group, Member, Result, EmbeddingProvider
  py.typed             PEP 561 marker
  core.py              SemDedup class — orchestrates embed → similarity → components
  similarity.py        build_similarity_graph() — pairwise cosine (numpy dot product on L2-normed vectors)
  components.py        find_groups() — connected components via scipy.sparse.csgraph
  blocking.py          find_groups_blocked() — K-means blocking for >50K items
  models.py            Pydantic: Member, Group, Result, SweepResult, ThresholdStats
  config.py            SemDedupConfig (pydantic-settings, env prefix SEMDEDUP_)
  cache.py             EmbeddingCache — diskcache keyed by sha256(model:text)
  naming.py            LLM-based group naming (optional, async)
  _stop_words.py       Stop word stripping (before embed, preserved in output)
  cli.py               Typer CLI: dedupe, sweep, stats, inspect, cache
  embedders/
    protocol.py        EmbeddingProvider Protocol
    openai.py          OpenAIEmbedder (default, async, batched, cached)
    precomputed.py     PrecomputedEmbedder (wraps numpy array, no API)
    litellm.py         LiteLLMEmbedder (optional extra: semdedup[litellm])
    sentence.py        SentenceTransformerEmbedder (optional extra: semdedup[local])
  io/
    readers.py         read_json, read_jsonl, read_csv, read_text
    writers.py         write_json, write_jsonl, write_csv, write_mapping
```

## Key Decisions

- **scipy.sparse.csgraph** for connected components — similarity matrix IS the graph
- **Async-first** — embedding calls are I/O bound, batched with asyncio
- **diskcache** for embedding cache — never re-embed the same text twice
- **EmbeddingProvider Protocol** — pluggable backends, bring your own embeddings
- **No clustering algorithms** — connected components at threshold, period

## Code Rules

- pyright strict — every function fully annotated
- `from __future__ import annotations` in every module
- ruff: `select = ["E", "F", "I", "UP", "B", "SIM", "ANN"]`
- Line length 100, pathlib.Path over os.path

**NO DEFENSIVE CODING:**
- NO `.get()` with defaults — use `data["key"]`, let it KeyError
- NO fallbacks — if something is missing, crash
- NO exception swallowing — errors propagate
- NO input normalization — bad data crashes, doesn't silently mutate
- FAIL FAST, FAIL LOUD

## Testing

- pytest + pytest-asyncio (asyncio_mode = "auto")
- hypothesis for property-based graph invariant tests
- Mock OpenAI in unit tests — never call real APIs
- Integration tests gated: `SEMDEDUP_INTEGRATION_TESTS=1`
- File naming mirrors source: `core.py` → `test_core.py`

## Git Workflow

- Conventional commits: `feat(scope):`, `fix(scope):`, `refactor:`, `test:`, `docs:`, `ci:`, `chore:`
- Labels: type (`epic feature task bug refactor`) + priority (`P0 P1 P2 P3 P4`)
- `/gh-issues` for sub-issues and blocked-by relationships
- Co-Authored-By: Claude <noreply@anthropic.com>
