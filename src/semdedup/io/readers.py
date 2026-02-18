from __future__ import annotations

import csv
import json
from pathlib import Path


def read_json(path: str | Path, json_path: str) -> list[str]:
    """Read texts from a JSON file using a jq-style path.

    Supports paths like '.[].label' to extract from arrays of objects.
    """
    with open(path) as f:
        data = json.load(f)

    return _extract_json_path(data, json_path)


def read_jsonl(path: str | Path, json_path: str) -> list[str]:
    """Read texts from a JSONL file using a jq-style path per line.

    Each line is a JSON object; json_path extracts the text field.
    """
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            extracted = _extract_json_path(obj, json_path)
            texts.extend(extracted)
    return texts


def read_text(path: str | Path) -> list[str]:
    """Read texts from a plain text file, one item per line.

    Empty lines are skipped.
    """
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def read_csv(path: str | Path, column: str) -> list[str]:
    """Read texts from a CSV file by column name."""
    texts: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row[column])
    return texts


def _extract_json_path(data: object, path: str) -> list[str]:
    """Extract values from JSON data using a simplified jq-style path.

    Supports:
      - '.[].field' — iterate array, extract field
      - '.field' — extract single field from object
      - '.field.subfield' — nested extraction
    """
    parts = path.lstrip(".").split(".")
    current: list[object] = [data]

    for part in parts:
        next_items: list[object] = []
        for item in current:
            if part == "[]":
                assert isinstance(item, list)
                next_items.extend(item)  # type: ignore[reportUnknownArgumentType]
            else:
                assert isinstance(item, dict)
                next_items.append(item[part])  # type: ignore[reportUnknownArgumentType]
        current = next_items

    return [str(r) for r in current]
