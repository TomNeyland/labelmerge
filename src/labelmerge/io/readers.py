from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Callable, TextIO

from labelmerge.jsonpath import extract_json_path_strings


def read_json(path: str | Path, json_path: str) -> list[str]:
    """Read texts from a JSON file using a jq-style path.

    Supports paths like '.[].label' to extract from arrays of objects.
    """
    with open(path) as f:
        data = json.load(f)

    return extract_json_path_strings(data, json_path)


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
            extracted = extract_json_path_strings(obj, json_path)
            texts.extend(extracted)
    return texts


def read_text(path: str | Path) -> list[str]:
    """Read texts from a plain text file, one item per line.

    Empty lines are skipped.
    """
    with open(path) as f:
        return read_text_stream(f)


def read_csv(path: str | Path, column: str) -> list[str]:
    """Read texts from a CSV file by column name."""
    texts: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row[column])
    return texts


def read_text_stream(stream: TextIO) -> list[str]:
    """Read texts from a text stream, one item per line, skipping blanks."""
    return [line.strip() for line in stream if line.strip()]


def read_json_array_stream(stream: TextIO) -> list[str]:
    """Read a JSON array from a stream and coerce values to strings."""
    data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("JSON stdin input must be an array")
    result: list[str] = []
    for item in data:
        if item is None:
            continue
        result.append(item if isinstance(item, str) else str(item))
    return result


def detect_input_format(path: str | Path, input_format: str | None = None) -> str:
    """Infer input format from path suffix unless an explicit format is given."""
    if input_format is not None and input_format != "auto":
        return input_format

    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    return "text"


def read_values(
    path: str | Path,
    *,
    path_expr: str | None = None,
    column: str | None = None,
    input_format: str | None = None,
) -> list[str]:
    """Read values from a single file with optional explicit format."""
    fmt = detect_input_format(path, input_format)
    if fmt == "json":
        if path_expr is None:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON input without --path must be a JSON array")
            return [item if isinstance(item, str) else str(item) for item in data if item is not None]
        return read_json(path, path_expr)
    if fmt == "jsonl":
        if path_expr is None:
            raise ValueError("JSONL input requires --path")
        return read_jsonl(path, path_expr)
    if fmt == "csv":
        if column is None:
            raise ValueError("CSV input requires --column")
        return read_csv(path, column)
    if fmt == "text":
        return read_text(path)
    raise ValueError(f"Unsupported input format: {fmt}")


def has_glob_chars(value: str) -> bool:
    """Return True when the string contains glob wildcard syntax."""
    return any(ch in value for ch in ("*", "?", "["))


def expand_input_patterns(inputs: list[str]) -> list[Path]:
    """Expand glob patterns and return a deterministic list of files."""
    seen: set[Path] = set()
    result: list[Path] = []

    for item in inputs:
        if item == "-":
            raise ValueError("'-' (stdin) cannot be combined in file pattern expansion")

        matches: list[str]
        if has_glob_chars(item):
            matches = sorted(glob.glob(item, recursive=True))
            if not matches:
                raise FileNotFoundError(f"No files matched pattern: {item}")
        else:
            matches = [item]

        for match in matches:
            path = Path(match)
            if path in seen:
                continue
            seen.add(path)
            result.append(path)

    return result


def read_many_values(
    paths: list[Path],
    *,
    path_expr: str | None = None,
    column: str | None = None,
    input_format: str | None = None,
    on_file_read: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Read and concatenate values from multiple files."""
    texts: list[str] = []
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        texts.extend(read_values(path, path_expr=path_expr, column=column, input_format=input_format))
        if on_file_read is not None:
            on_file_read(idx, total)
    return texts
