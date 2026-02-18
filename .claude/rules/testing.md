---
paths:
  - "tests/**/*.py"
---

# Testing Rules

- Use pytest fixtures, not setUp/tearDown
- Mock external services (OpenAI, diskcache) -- never call real APIs in unit tests
- Use hypothesis for property-based tests on graph invariants
- Test file names mirror source: src/labelmerge/core.py -> tests/test_core.py
- No annotations required in test files (ruff ANN exemption)
- Use pytest.raises for expected exceptions, not try/except
- asyncio_mode = "auto" -- async test functions just work
- Integration tests behind LABELMERGE_INTEGRATION_TESTS=1 env var gate
