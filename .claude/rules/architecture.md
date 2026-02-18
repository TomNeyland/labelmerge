---
paths:
  - "src/semdedup/**/*.py"
---

# Module Boundaries

Each module has a single responsibility. Do not leak abstractions across boundaries.

- **similarity.py** — Builds the sparse similarity matrix. Only numpy/scipy. No business logic.
- **components.py** — Extracts connected components from a similarity graph. Uses similarity.py's output. Returns index lists.
- **core.py** — Orchestrator. Imports from components, embedders, cache, naming. The only module that knows about the full pipeline.
- **models.py** — Pydantic data models only. No business logic, no I/O, no imports from other semdedup modules.
- **config.py** — pydantic-settings config only. No logic.
- **cache.py** — Embedding cache. No knowledge of what's being cached or why.
- **naming.py** — LLM naming. Takes groups, returns named groups. No knowledge of how groups were formed.
- **embedders/** — Each embedder is self-contained. Implements the EmbeddingProvider protocol. No cross-embedder imports.
- **io/** — File I/O. Readers produce `list[str]`, writers consume `Result`. No embedding/similarity logic.
- **cli.py** — Thin CLI layer. Constructs objects, calls async methods via asyncio.run(), formats output. No algorithm logic.
- **blocking.py** — Scale optimization. Uses components.py for per-block dedup. Only needed for >50K items.
