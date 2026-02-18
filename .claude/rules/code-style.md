---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Code Style Rules

- All functions must have complete type annotations (pyright strict)
- Use `from __future__ import annotations` in every module
- Import order: stdlib -> third-party -> local (ruff handles this)
- Line length: 100 characters
- Use pathlib.Path over os.path
- Async functions for any I/O (embedding calls, file reads, API calls)
- Pydantic models for all data structures crossing module boundaries
- NO defensive coding: no fallbacks, no `.get()` with defaults, no input normalization
- Let exceptions propagate -- crash on bad data
