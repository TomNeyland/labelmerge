from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labelmerge.config import LabelMergeConfig
from labelmerge.core import LabelMerge
from labelmerge.io.writers import write_csv, write_json, write_jsonl, write_mapping
from labelmerge.models import Result

app = typer.Typer(name="labelmerge", help="Semantic deduplication of text labels.")
cache_app = typer.Typer(name="cache", help="Manage embedding cache.")
app.add_typer(cache_app)

console = Console()


@app.command()
def dedupe(
    input_file: Path,
    threshold: float = typer.Option(0.85, help="Cosine similarity threshold."),
    path: str | None = typer.Option(None, help="jq-style path for JSON/JSONL."),
    column: str | None = typer.Option(None, help="Column name for CSV."),
    model: str = typer.Option("text-embedding-3-small", help="Embedding model."),
    dimensions: int | None = typer.Option(None, help="Embedding dimensions."),
    output_format: str = typer.Option("json", help="Output format: json, jsonl, csv, mapping."),
    output: Path | None = typer.Option(None, "-o", help="Output file. Defaults to stdout."),
    stop_words: str | None = typer.Option(None, help="Comma-separated stop words."),
    name_groups: bool = typer.Option(False, "--name-groups", help="Name groups with LLM."),
    naming_model: str = typer.Option("gpt-4o-mini", help="Model for group naming."),
    max_component_size: int = typer.Option(100, help="Max component size before splitting."),
) -> None:
    """Deduplicate text labels from a file."""
    from labelmerge.cache import EmbeddingCache
    from labelmerge.embedders.openai import OpenAIEmbedder

    config = LabelMergeConfig()
    cache = EmbeddingCache(config.cache_dir) if config.cache_enabled else None
    embedder = OpenAIEmbedder(
        model=model,
        dimensions=dimensions,
        batch_size=config.embedding_batch_size,
        cache=cache,
    )

    sw_list = [s.strip() for s in stop_words.split(",")] if stop_words else []

    sd = LabelMerge(
        embedder=embedder,
        threshold=threshold,
        max_component_size=max_component_size,
        stop_words=sw_list,
        cache_enabled=False,  # cache already managed by embedder
    )

    result = asyncio.run(sd.dedupe_file(input_file, path_expr=path, column=column))

    if name_groups:
        result = asyncio.run(sd.name_groups(result, model=naming_model))

    _write_output(result, output_format, output)


@app.command()
def sweep(
    input_file: Path,
    thresholds: str = typer.Option("0.80,0.85,0.90,0.95", help="Comma-separated thresholds."),
    path: str | None = typer.Option(None, help="jq-style path for JSON/JSONL."),
    column: str | None = typer.Option(None, help="Column name for CSV."),
    model: str = typer.Option("text-embedding-3-small", help="Embedding model."),
) -> None:
    """Run dedup at multiple thresholds to find the right one."""
    from labelmerge.cache import EmbeddingCache
    from labelmerge.embedders.openai import OpenAIEmbedder

    config = LabelMergeConfig()
    cache = EmbeddingCache(config.cache_dir) if config.cache_enabled else None
    embedder = OpenAIEmbedder(model=model, cache=cache)

    threshold_list = [float(t.strip()) for t in thresholds.split(",")]

    table = Table(title="Threshold Sweep")
    table.add_column("Threshold", justify="right")
    table.add_column("Groups", justify="right")
    table.add_column("Singletons", justify="right")
    table.add_column("Max Group", justify="right")
    table.add_column("% Grouped", justify="right")

    for t in threshold_list:
        sd = LabelMerge(embedder=embedder, threshold=t, cache_enabled=False)
        result = asyncio.run(sd.dedupe_file(input_file, path_expr=path, column=column))
        max_size = max((g.size for g in result.groups), default=0)
        pct = (result.n_grouped / result.n_input * 100) if result.n_input > 0 else 0.0
        table.add_row(
            f"{t:.2f}",
            str(len(result.groups)),
            str(result.n_singletons),
            str(max_size),
            f"{pct:.1f}%",
        )

    console.print(table)


@app.command()
def stats(
    input_file: Path,
    path: str | None = typer.Option(None, help="jq-style path for JSON/JSONL."),
    column: str | None = typer.Option(None, help="Column name for CSV."),
) -> None:
    """Quick stats about input data."""
    from semdedup.io.readers import read_csv, read_json, read_jsonl, read_text

    suffix = input_file.suffix.lower()
    if suffix == ".json" and path is not None:
        texts = read_json(input_file, path)
    elif suffix == ".jsonl" and path is not None:
        texts = read_jsonl(input_file, path)
    elif suffix == ".csv" and column is not None:
        texts = read_csv(input_file, column)
    else:
        texts = read_text(input_file)

    from collections import Counter

    counts = Counter(texts)
    total = len(texts)
    unique = len(counts)
    singletons = sum(1 for c in counts.values() if c == 1)

    console.print(f"Total items:    {total}")
    console.print(f"Unique items:   {unique}")
    console.print(f"Exact dupes:    {total - unique}")
    console.print(f"Singletons:     {singletons} ({singletons / unique * 100:.1f}%)")
    console.print(f"Unique rate:    {unique / total * 100:.1f}%")


@app.command()
def inspect(
    groups_file: Path,
    group: int = typer.Option(0, help="Group ID to inspect."),
) -> None:
    """Inspect a specific group from a results file."""
    with open(groups_file) as f:
        data = json.load(f)

    groups = data["groups"]
    for g in groups:
        if g["group_id"] == group:
            console.print(f"[bold]Group {g['group_id']}[/bold]")
            if g["canonical"]:
                console.print(f"Canonical: {g['canonical']}")
            console.print(f"Size: {g['size']} members, {g['total_occurrences']} occurrences")
            console.print()
            for m in g["members"]:
                console.print(f"  {m['count']:>4}x  {m['text']}")
            return

    console.print(f"[red]Group {group} not found.[/red]")
    raise SystemExit(1)


@cache_app.command("clear")
def cache_clear() -> None:
    """Clear the embedding cache."""
    from labelmerge.cache import EmbeddingCache

    config = LabelMergeConfig()
    cache = EmbeddingCache(config.cache_dir)
    cache.clear()
    console.print("Cache cleared.")


@cache_app.command("stats")
def cache_stats() -> None:
    """Show embedding cache statistics."""
    from labelmerge.cache import EmbeddingCache

    config = LabelMergeConfig()
    cache = EmbeddingCache(config.cache_dir)
    s = cache.stats()
    console.print(f"Cached embeddings: {s['size']}")
    console.print(f"Disk usage:        {s['volume']} bytes")


def _write_output(result: Result, fmt: str, output: Path | None) -> None:
    """Write result to file or stdout."""
    if output is not None:
        writers = {
            "json": write_json,
            "jsonl": write_jsonl,
            "csv": write_csv,
            "mapping": write_mapping,
        }
        writers[fmt](result, output)
    else:
        if fmt == "mapping":
            json.dump(result.to_mapping(), sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            json.dump(result.model_dump(), sys.stdout, indent=2)
            sys.stdout.write("\n")
