from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from labelmerge.models import Result


def dump_json(
    data: object,
    fp: TextIO,
    *,
    pretty: bool = False,
    sort_keys: bool = False,
) -> None:
    """Write JSON to a stream with compact output by default."""
    if pretty:
        json.dump(data, fp, indent=2, sort_keys=sort_keys)
    else:
        json.dump(data, fp, separators=(",", ":"), sort_keys=sort_keys)
    fp.write("\n")


def dumps_json(data: object, *, pretty: bool = False, sort_keys: bool = False) -> str:
    """Serialize JSON to a string with compact output by default."""
    if pretty:
        return json.dumps(data, indent=2, sort_keys=sort_keys)
    return json.dumps(data, separators=(",", ":"), sort_keys=sort_keys)


def write_json(result: Result, path: str | Path, *, pretty: bool = True) -> None:
    """Write result as JSON."""
    with open(path, "w") as f:
        dump_json(result.model_dump(), f, pretty=pretty)


def write_jsonl(result: Result, path: str | Path) -> None:
    """Write result as JSONL, one group per line."""
    with open(path, "w") as f:
        write_jsonl_stream(result, f)


def write_jsonl_stream(result: Result, fp: TextIO) -> None:
    """Write result as JSONL to a stream, one group/singleton per line."""
    for group in result.groups:
        fp.write(json.dumps(group.model_dump()) + "\n")
    for singleton in result.singletons:
        fp.write(
            json.dumps(
                {
                    "group_id": None,
                    "members": [singleton.model_dump()],
                    "canonical": singleton.text,
                    "size": 1,
                    "total_occurrences": singleton.count,
                }
            )
            + "\n"
        )


def write_csv(result: Result, path: str | Path) -> None:
    """Write result as CSV with columns: original, canonical, group_id."""
    with open(path, "w", newline="") as f:
        write_csv_stream(result, f)


def write_csv_stream(result: Result, fp: TextIO) -> None:
    """Write result as CSV to a stream with columns: original, canonical, group_id."""
    mapping = result.to_mapping()
    group_lookup: dict[str, int | None] = {}
    for group in result.groups:
        for member in group.members:
            group_lookup[member.text] = group.group_id
    for singleton in result.singletons:
        group_lookup[singleton.text] = None

    writer = csv.writer(fp)
    writer.writerow(["original", "canonical", "group_id"])
    for original, canonical in mapping.items():
        writer.writerow([original, canonical, group_lookup[original]])


def write_mapping(result: Result, path: str | Path) -> None:
    """Write result as a flat original -> canonical JSON mapping."""
    mapping = result.to_mapping()
    with open(path, "w") as f:
        dump_json(mapping, f, pretty=True, sort_keys=True)


def write_enum(result: Result, path: str | Path, *, pretty: bool = False) -> None:
    """Write sorted canonical values as a JSON array."""
    with open(path, "w") as f:
        dump_json(result.canonical_values(), f, pretty=pretty)


def write_jsonschema(result: Result, path: str | Path, *, pretty: bool = False) -> None:
    """Write a JSON Schema fragment for the canonical value enum."""
    with open(path, "w") as f:
        dump_json(result.to_jsonschema(), f, pretty=pretty)


def write_stdout_json(
    data: object,
    *,
    pretty: bool = False,
    sort_keys: bool = False,
) -> None:
    """Write structured JSON to stdout."""
    dump_json(data, sys.stdout, pretty=pretty, sort_keys=sort_keys)
