# LabelMerge CLI Design Principles

`labelmerge` should feel like a sharp UNIX tool: fast, composable, deterministic, and pleasant to use.

## Three Rules

1. Structured data to `stdout`, human context to `stderr`
- JSON/mapping/enum outputs belong on `stdout`
- Progress, summaries, and trust-building explanations belong on `stderr`
- Commands should remain pipe-friendly

2. Progress that respects time
- Single updating line to `stderr`
- Show meaningful progress (`Scanning`, `Embedding`, counts, percent)
- Stay silent for fast operations (under ~1s)
- Honor `--quiet` / `-q`

3. The output tells a story
- Summaries should explain how messy the data was and what changed
- Prefer compression/reduction framing over raw counts alone
- Show representative merges/normalizations to build trust

## Output Format Principles

- Default JSON output is compact (single-line). Use `--pretty` for readability.
- Mapping output should be key-sorted for deterministic diffs.
- Enum output should be sorted and deterministic.

## Naming Principles (`--name-groups`)

- Lowercase canonicals
- Concise phrasing
- Stable outputs (temperature `0`)
- Domain-neutral wording

## Composability Checklist

- `cmd | jq .` works
- `cmd 2>/dev/null` keeps structured output valid
- `cmd --quiet` suppresses `stderr`
- `cmd -o out.json` works when supported
- `cmd -` / `--stdin` works where applicable
- Deterministic output for identical inputs
