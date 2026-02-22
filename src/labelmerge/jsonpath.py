from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class PathToken:
    """A parsed token in the simplified jq-style path syntax."""

    kind: str  # "field" | "all"
    value: str | None = None


@dataclass
class JsonPathApplyStats:
    """Stats collected while applying a mapping to JSON values at a path."""

    values_checked: int = 0
    changed: int = 0
    already_canonical: int = 0
    unmapped: int = 0
    changes: Counter[tuple[str, str]] = field(default_factory=Counter)
    unmapped_values: Counter[str] = field(default_factory=Counter)


def parse_json_path(path: str) -> list[PathToken]:
    """Parse a simplified jq-style path (e.g. `.items[].label`)."""
    raw = path.strip()
    if not raw:
        raise ValueError("JSON path cannot be empty")
    if raw.startswith("."):
        raw = raw[1:]

    tokens: list[PathToken] = []
    field_buf: list[str] = []
    i = 0

    def flush_field() -> None:
        if field_buf:
            tokens.append(PathToken(kind="field", value="".join(field_buf)))
            field_buf.clear()

    while i < len(raw):
        ch = raw[i]
        if ch == ".":
            flush_field()
            i += 1
            continue
        if ch == "[":
            if raw[i : i + 2] != "[]":
                raise ValueError(f"Unsupported JSON path syntax near: {raw[i:]!r}")
            flush_field()
            tokens.append(PathToken(kind="all"))
            i += 2
            continue
        field_buf.append(ch)
        i += 1

    flush_field()
    if not tokens:
        raise ValueError(f"Unsupported JSON path: {path!r}")
    return tokens


def extract_json_path_strings(data: object, path: str) -> list[str]:
    """Extract terminal values from JSON and coerce to strings.

    Missing keys and null values are skipped.
    """
    tokens = parse_json_path(path)
    current: list[object] = [data]

    for token in tokens:
        next_items: list[object] = []
        if token.kind == "field":
            field_name = token.value
            assert field_name is not None
            for item in current:
                if item is None or not isinstance(item, dict):
                    continue
                if field_name not in item:
                    continue
                value = item[field_name]
                if value is None:
                    continue
                next_items.append(value)
        elif token.kind == "all":
            for item in current:
                if item is None or not isinstance(item, list):
                    continue
                for elem in item:
                    if elem is None:
                        continue
                    next_items.append(elem)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown token kind: {token.kind}")
        current = next_items

    result: list[str] = []
    for item in current:
        if item is None:
            continue
        result.append(item if isinstance(item, str) else str(item))
    return result


def apply_mapping_to_json_path(data: object, path: str, mapping: Mapping[str, str]) -> JsonPathApplyStats:
    """Mutate JSON in place by applying a mapping to terminal string values."""
    tokens = parse_json_path(path)
    stats = JsonPathApplyStats()
    _transform_node(data, tokens, 0, mapping, stats)
    return stats


def _transform_node(
    node: object,
    tokens: list[PathToken],
    token_index: int,
    mapping: Mapping[str, str],
    stats: JsonPathApplyStats,
) -> tuple[object, bool]:
    if token_index >= len(tokens):
        if node is None or not isinstance(node, str):
            return node, False
        stats.values_checked += 1
        if node in mapping:
            canonical = mapping[node]
            if canonical == node:
                stats.already_canonical += 1
                return node, False
            stats.changed += 1
            stats.changes[(node, canonical)] += 1
            return canonical, True
        stats.unmapped += 1
        stats.unmapped_values[node] += 1
        return node, False

    token = tokens[token_index]

    if token.kind == "field":
        field_name = token.value
        assert field_name is not None
        if node is None or not isinstance(node, dict):
            return node, False
        if field_name not in node:
            return node, False
        child = node[field_name]
        if child is None:
            return node, False
        new_child, changed = _transform_node(child, tokens, token_index + 1, mapping, stats)
        if changed:
            node[field_name] = new_child
        return node, changed

    if token.kind == "all":
        if node is None or not isinstance(node, list):
            return node, False
        any_changed = False
        for i, child in enumerate(node):
            if child is None:
                continue
            new_child, changed = _transform_node(child, tokens, token_index + 1, mapping, stats)
            if changed:
                node[i] = new_child
                any_changed = True
        return node, any_changed

    raise ValueError(f"Unknown token kind: {token.kind}")  # pragma: no cover
