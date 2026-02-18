from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semdedup.models import Result


def write_json(result: Result, path: str | Path) -> None:
    """Write result as formatted JSON."""
    with open(path, "w") as f:
        json.dump(result.model_dump(), f, indent=2)


def write_jsonl(result: Result, path: str | Path) -> None:
    """Write result as JSONL, one group per line."""
    with open(path, "w") as f:
        for group in result.groups:
            f.write(json.dumps(group.model_dump()) + "\n")
        for singleton in result.singletons:
            f.write(
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
    mapping = result.to_mapping()
    group_lookup: dict[str, int | None] = {}
    for group in result.groups:
        for member in group.members:
            group_lookup[member.text] = group.group_id
    for singleton in result.singletons:
        group_lookup[singleton.text] = None

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "canonical", "group_id"])
        for original, canonical in mapping.items():
            writer.writerow([original, canonical, group_lookup[original]])


def write_mapping(result: Result, path: str | Path) -> None:
    """Write result as a flat original -> canonical JSON mapping."""
    mapping = result.to_mapping()
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
