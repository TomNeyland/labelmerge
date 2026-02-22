from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from labelmerge.config import LabelMergeConfig
from labelmerge.core import LabelMerge
from labelmerge.io.readers import (
    expand_input_patterns,
    read_json_array_stream,
    read_many_values,
    read_text_stream,
)
from labelmerge.io.writers import (
    dump_json,
    dumps_json,
    write_csv,
    write_csv_stream,
    write_json,
    write_jsonl,
    write_jsonl_stream,
    write_stdout_json,
)
from labelmerge.jsonpath import JsonPathApplyStats, apply_mapping_to_json_path
from labelmerge.models import DiffExact, DiffMatch, DiffResult, DiffUnmatched, Result

app = typer.Typer(name="labelmerge", help="Semantic deduplication of text labels.")
cache_app = typer.Typer(name="cache", help="Manage embedding cache.")
app.add_typer(cache_app)

console = Console(stderr=True)

_RESULT_OUTPUT_FORMATS = {"json", "jsonl", "csv", "mapping", "enum", "jsonschema"}


@dataclass
class _DelayedProgress:
    prefix: str
    quiet: bool = False
    show_after_seconds: float = 1.0
    show_percent: bool = False

    _started_at: float | None = None
    _shown: bool = False
    _last_line: str = ""

    def update(self, current: int, total: int) -> None:
        if self.quiet:
            return
        now = time.monotonic()
        if self._started_at is None:
            self._started_at = now

        suffix = f"{current}/{total}"
        if self.show_percent and total > 0:
            pct = int((current / total) * 100)
            suffix += f" ({pct}%)"
        self._last_line = f"{self.prefix}... {suffix}"

        if not self._shown and (now - self._started_at) < self.show_after_seconds:
            return

        self._shown = True
        sys.stderr.write(f"\r{self._last_line}")
        sys.stderr.flush()

    def finish(self) -> None:
        if self.quiet or not self._shown:
            return
        sys.stderr.write(f"\r{self._last_line}\n")
        sys.stderr.flush()


@dataclass
class _ReadInputsResult:
    texts: list[str]
    files: list[Path]
    source_kind: str  # "stdin" | "files"


@dataclass
class _ApplyPreparedSource:
    label: str
    path: Path | None
    output_text: str
    changed: bool
    stats: JsonPathApplyStats


@dataclass
class _EmbedRunStats:
    total: int = 0
    cached: int = 0
    duration_s: float | None = None

    @property
    def cache_hit_pct(self) -> float | None:
        if self.total <= 0:
            return None
        return (self.cached / self.total) * 100.0


@dataclass
class _DedupeRunOutcome:
    result: Result
    embed: _EmbedRunStats


@dataclass
class _DiffRunOutcome:
    result: DiffResult
    embed: _EmbedRunStats | None


def _stderr(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        typer.echo(message, err=True)


def _resolve_input_sources(inputs: list[str] | None, stdin_flag: bool) -> tuple[str, list[Path]]:
    normalized_inputs = list(inputs or [])

    if stdin_flag and normalized_inputs:
        raise typer.BadParameter("Use either positional input(s) or --stdin, not both")

    if stdin_flag:
        return "stdin", []

    if normalized_inputs == ["-"]:
        return "stdin", []

    if any(item == "-" for item in normalized_inputs):
        raise typer.BadParameter("'-' (stdin) cannot be combined with file inputs")

    if not normalized_inputs:
        raise typer.BadParameter("Provide an input path/glob or use --stdin")

    try:
        paths = expand_input_patterns(normalized_inputs)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return "files", paths


def _read_inputs(
    *,
    inputs: list[str] | None,
    stdin_flag: bool,
    input_format: str,
    path_expr: str | None,
    column: str | None,
    quiet: bool,
) -> _ReadInputsResult:
    source_kind, paths = _resolve_input_sources(inputs, stdin_flag)
    if source_kind == "stdin":
        if input_format == "json":
            texts = read_json_array_stream(sys.stdin)
        elif input_format in {"auto", "text"}:
            texts = read_text_stream(sys.stdin)
        else:
            raise typer.BadParameter("stdin supports --input-format text (default) or json")
        return _ReadInputsResult(texts=texts, files=[], source_kind="stdin")

    progress = _DelayedProgress(prefix=f"Scanning {len(paths)} files", quiet=quiet)
    try:
        texts = read_many_values(
            paths,
            path_expr=path_expr,
            column=column,
            input_format=input_format,
            on_file_read=progress.update if len(paths) > 1 else None,
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        progress.finish()

    return _ReadInputsResult(texts=texts, files=paths, source_kind="files")


def _build_labelmerge(
    *,
    threshold: float,
    model: str,
    dimensions: int | None,
    stop_words: str | None,
    max_component_size: int,
    progress: _DelayedProgress | None,
) -> tuple[LabelMerge, object]:
    from labelmerge.cache import EmbeddingCache
    from labelmerge.embedders.openai import OpenAIEmbedder

    config = LabelMergeConfig()
    cache = EmbeddingCache(config.cache_dir) if config.cache_enabled else None
    embedder = OpenAIEmbedder(
        model=model,
        dimensions=dimensions,
        batch_size=config.embedding_batch_size,
        cache=cache,
        on_progress=progress.update if progress is not None else None,
    )

    stop_word_list = [s.strip() for s in stop_words.split(",") if s.strip()] if stop_words else []
    lm = LabelMerge(
        embedder=embedder,
        threshold=threshold,
        max_component_size=max_component_size,
        stop_words=stop_word_list,
        cache_enabled=False,
    )
    return lm, embedder


def _write_result_output(result: Result, fmt: str, output: Path | None, *, pretty: bool) -> None:
    if fmt not in _RESULT_OUTPUT_FORMATS:
        raise typer.BadParameter(
            f"Unsupported output format: {fmt}. Choose from: {', '.join(sorted(_RESULT_OUTPUT_FORMATS))}"
        )

    if output is not None:
        if fmt == "json":
            write_json(result, output, pretty=pretty)
            return
        if fmt == "jsonl":
            write_jsonl(result, output)
            return
        if fmt == "csv":
            write_csv(result, output)
            return
        with open(output, "w") as f:
            if fmt == "mapping":
                dump_json(result.to_mapping(), f, pretty=pretty, sort_keys=True)
            elif fmt == "enum":
                dump_json(result.canonical_values(), f, pretty=pretty)
            elif fmt == "jsonschema":
                dump_json(result.to_jsonschema(), f, pretty=pretty)
        return

    if fmt == "json":
        write_stdout_json(result.model_dump(), pretty=pretty)
    elif fmt == "mapping":
        write_stdout_json(result.to_mapping(), pretty=pretty, sort_keys=True)
    elif fmt == "enum":
        write_stdout_json(result.canonical_values(), pretty=pretty)
    elif fmt == "jsonschema":
        write_stdout_json(result.to_jsonschema(), pretty=pretty)
    elif fmt == "jsonl":
        write_jsonl_stream(result, sys.stdout)
    elif fmt == "csv":
        write_csv_stream(result, sys.stdout)


def _coerce_embed_run_stats(obj: object | None) -> _EmbedRunStats:
    if obj is None:
        return _EmbedRunStats()
    raw = getattr(obj, "last_embed_stats", None)
    if not isinstance(raw, dict):
        return _EmbedRunStats()
    total = raw.get("total")
    cached = raw.get("cached")
    duration_s = raw.get("duration_s")
    return _EmbedRunStats(
        total=int(total) if isinstance(total, (int, float)) else 0,
        cached=int(cached) if isinstance(cached, (int, float)) else 0,
        duration_s=float(duration_s) if isinstance(duration_s, (int, float)) else None,
    )


def _format_embed_done_line(*, unique_values: int, embed: _EmbedRunStats) -> str:
    pieces = [f"Embedding {unique_values} unique values... done"]
    meta: list[str] = []
    if embed.duration_s is not None:
        meta.append(f"{embed.duration_s:.1f}s")
    cache_hit_pct = embed.cache_hit_pct
    if cache_hit_pct is not None:
        meta.append(f"cache: {cache_hit_pct:.0f}% hit")
    if meta:
        pieces.append(f"({', '.join(meta)})")
    return " ".join(pieces)


def _print_dedupe_summary(
    *,
    result: Result,
    embed: _EmbedRunStats | None,
    total_values: int,
    unique_values: int,
    source_label: str,
    quiet: bool,
    mode_label: str,
) -> None:
    if quiet:
        return

    merge_groups = [group for group in result.groups if group.size > 1]
    canonical_count = len(result.canonical_values())
    reduction_pct = 0.0
    if unique_values > 0:
        reduction_pct = ((unique_values - canonical_count) / unique_values) * 100

    _stderr(f"{mode_label}: {source_label}", quiet=quiet)
    _stderr(f"{total_values} values → {unique_values} unique", quiet=quiet)
    if embed is not None and unique_values > 0:
        _stderr(_format_embed_done_line(unique_values=unique_values, embed=embed), quiet=quiet)
    _stderr(
        (
            f"After dedup: {unique_values} → {canonical_count} canonical "
            f"({reduction_pct:.1f}% reduction)"
        ),
        quiet=quiet,
    )

    if merge_groups:
        largest = max(merge_groups, key=lambda g: (g.size, g.total_occurrences))
        largest_canonical = largest.canonical or largest.members[0].text
        _stderr(
            f'Largest merge: {largest.size} variants → "{largest_canonical}"',
            quiet=quiet,
        )
        _stderr("Top merges:", quiet=quiet)
        for group in merge_groups[:3]:
            canonical = group.canonical or group.members[0].text
            variants = [m.text for m in group.members if m.text != canonical][:4]
            if not variants:
                continue
            _stderr(f'  "{canonical}" ← {", ".join(variants)}', quiet=quiet)


def _print_corpus_summary(
    *,
    result: Result,
    embed: _EmbedRunStats,
    read_result: _ReadInputsResult,
    threshold: float,
    quiet: bool,
) -> None:
    if quiet:
        return

    counts = Counter(read_result.texts)
    total_values = len(read_result.texts)
    unique_values = len(counts)
    exact_singletons = sum(1 for c in counts.values() if c == 1)
    merge_groups = [group for group in result.groups if group.size > 1]
    canonical_count = len(result.canonical_values())
    merged_values = max(unique_values - canonical_count, 0)
    reduction_pct = ((merged_values / unique_values) * 100.0) if unique_values else 0.0

    _stderr(
        f"Extracted {total_values} values ({unique_values} unique, {exact_singletons} singletons)",
        quiet=quiet,
    )
    if unique_values > 0:
        _stderr(_format_embed_done_line(unique_values=unique_values, embed=embed), quiet=quiet)
    _stderr("", quiet=quiet)
    _stderr(f"Dedup at threshold {threshold:.2f}:", quiet=quiet)
    _stderr(
        f"  {unique_values} unique → {canonical_count} canonical ({reduction_pct:.1f}% reduction)",
        quiet=quiet,
    )
    _stderr(
        f"  {merged_values} values merged into {len(merge_groups)} groups",
        quiet=quiet,
    )

    if merge_groups:
        largest = max(merge_groups, key=lambda g: (g.size, g.total_occurrences))
        largest_canonical = largest.canonical or largest.members[0].text
        _stderr(
            (
                f'  Largest: "{largest_canonical}" ← {largest.size} variants, '
                f"{largest.total_occurrences} occurrences"
            ),
            quiet=quiet,
        )
        _stderr("", quiet=quiet)
        _stderr("Top merges:", quiet=quiet)
        for group in sorted(
            merge_groups, key=lambda g: (g.total_occurrences, g.size, -(g.group_id)), reverse=True
        )[:5]:
            canonical = group.canonical or group.members[0].text
            variants = [m.text for m in group.members if m.text != canonical][:4]
            if not variants:
                continue
            _stderr(
                (
                    f'  "{canonical}" ← {", ".join(variants)}'
                    f"  ({group.total_occurrences} occurrences, {group.size} variants)"
                ),
                quiet=quiet,
            )


def _print_diff_summary(
    *,
    diff_result: DiffResult,
    embed: _EmbedRunStats | None,
    vocab_count: int,
    vocab_file: Path,
    read_result: _ReadInputsResult,
    quiet: bool,
) -> None:
    if quiet:
        return

    _stderr(f"Vocabulary: {vocab_count} canonical values (from {vocab_file})", quiet=quiet)
    _stderr(
        f"New data: {len(set(read_result.texts))} values (from {_command_source_label(read_result)})",
        quiet=quiet,
    )
    if embed is not None and embed.total > 0:
        # `embed.total` here includes vocab + non-exact candidates.
        meta: list[str] = []
        if embed.duration_s is not None:
            meta.append(f"{embed.duration_s:.1f}s")
        cache_hit_pct = embed.cache_hit_pct
        if cache_hit_pct is not None:
            meta.append(f"cache: {cache_hit_pct:.0f}% hit")
        suffix = f" ({', '.join(meta)})" if meta else ""
        _stderr(f"Embedding {embed.total} values... done{suffix}", quiet=quiet)

    _stderr("", quiet=quiet)
    _stderr(f"  {len(diff_result.matched)} matched existing canonicals (auto-mapped)", quiet=quiet)
    _stderr(f"  {len(diff_result.unmatched)} below threshold (need review)", quiet=quiet)
    _stderr(f"  {len(diff_result.exact)} exact matches (already in vocabulary)", quiet=quiet)

    if diff_result.matched:
        _stderr("", quiet=quiet)
        _stderr("Top auto-matches:", quiet=quiet)
        for item in sorted(diff_result.matched, key=lambda m: (-m.similarity, m.value))[:5]:
            _stderr(
                f'  "{item.value}" → "{item.canonical}" ({item.similarity:.2f})',
                quiet=quiet,
            )

    if diff_result.unmatched:
        _stderr("", quiet=quiet)
        _stderr("Top review items:", quiet=quiet)
        for item in sorted(diff_result.unmatched, key=lambda m: (-m.similarity, m.value))[:5]:
            if item.best_match is None:
                _stderr(f'  "{item.value}" → (no match) ({item.similarity:.2f})', quiet=quiet)
            else:
                _stderr(
                    f'  "{item.value}" → "{item.best_match}" ({item.similarity:.2f})',
                    quiet=quiet,
                )


async def _run_dedupe_async(
    *,
    texts: list[str],
    threshold: float,
    model: str,
    dimensions: int | None,
    stop_words: str | None,
    max_component_size: int,
    name_groups: bool,
    naming_model: str,
    review_profile: str,
    review_rules: Path | None,
    review_policy: str | None,
    quiet: bool,
) -> _DedupeRunOutcome:
    unique_count = len(set(texts))
    embed_progress = _DelayedProgress(
        prefix=f"Embedding {unique_count} unique values",
        quiet=quiet,
        show_percent=True,
    )
    lm, embedder = _build_labelmerge(
        threshold=threshold,
        model=model,
        dimensions=dimensions,
        stop_words=stop_words,
        max_component_size=max_component_size,
        progress=embed_progress if unique_count > 0 else None,
    )

    try:
        result = await lm.dedupe(texts)
    finally:
        embed_progress.finish()

    if name_groups:
        try:
            result = await lm.name_groups(
                result,
                model=naming_model,
                review_profile=review_profile,
                review_rules_path=review_rules,
                review_policy=review_policy,
            )
        except (ImportError, AttributeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    return _DedupeRunOutcome(result=result, embed=_coerce_embed_run_stats(embedder))


def _run_dedupe_with_metrics(
    *,
    texts: list[str],
    threshold: float,
    model: str,
    dimensions: int | None,
    stop_words: str | None,
    max_component_size: int,
    name_groups: bool,
    naming_model: str,
    review_profile: str,
    review_rules: Path | None,
    review_policy: str | None,
    quiet: bool,
) -> _DedupeRunOutcome:
    return asyncio.run(
        _run_dedupe_async(
            texts=texts,
            threshold=threshold,
            model=model,
            dimensions=dimensions,
            stop_words=stop_words,
            max_component_size=max_component_size,
            name_groups=name_groups,
            naming_model=naming_model,
            review_profile=review_profile,
            review_rules=review_rules,
            review_policy=review_policy,
            quiet=quiet,
        )
    )


def _run_dedupe(
    *,
    texts: list[str],
    threshold: float,
    model: str,
    dimensions: int | None,
    stop_words: str | None,
    max_component_size: int,
    name_groups: bool,
    naming_model: str,
    review_profile: str,
    review_rules: Path | None,
    review_policy: str | None,
    quiet: bool,
) -> Result:
    return _run_dedupe_with_metrics(
        texts=texts,
        threshold=threshold,
        model=model,
        dimensions=dimensions,
        stop_words=stop_words,
        max_component_size=max_component_size,
        name_groups=name_groups,
        naming_model=naming_model,
        review_profile=review_profile,
        review_rules=review_rules,
        review_policy=review_policy,
        quiet=quiet,
    ).result


def _write_json_data(
    data: object,
    *,
    output: Path | None,
    pretty: bool,
    sort_keys: bool = False,
) -> None:
    if output is None:
        write_stdout_json(data, pretty=pretty, sort_keys=sort_keys)
        return
    with open(output, "w") as f:
        dump_json(data, f, pretty=pretty, sort_keys=sort_keys)


def _load_vocab_canonicals(vocab_file: Path) -> list[str]:
    with open(vocab_file) as f:
        data: Any = json.load(f)

    if isinstance(data, list):
        return sorted({str(v) for v in data if v is not None})

    if isinstance(data, dict):
        enum_value = data.get("enum")
        if isinstance(enum_value, list):
            return sorted({str(v) for v in enum_value if v is not None})

        if "groups" in data and "singletons" in data:
            result = Result.model_validate(data)
            return result.canonical_values()

        values = [str(v) for v in data.values() if v is not None]
        return sorted(set(values))

    raise typer.BadParameter("Vocabulary file must be a JSON array, mapping object, or labelmerge result")


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


async def _run_diff_async(
    *,
    vocab: list[str],
    new_values: list[str],
    threshold: float,
    model: str,
    dimensions: int | None,
    quiet: bool,
) -> _DiffRunOutcome:
    unique_new_values = sorted(set(new_values))
    vocab_set = set(vocab)

    exact = [DiffExact(value=v, canonical=v) for v in unique_new_values if v in vocab_set]
    candidates = [v for v in unique_new_values if v not in vocab_set]

    if not candidates:
        return _DiffRunOutcome(result=DiffResult(matched=[], unmatched=[], exact=exact), embed=None)
    if not vocab:
        unmatched = [
            DiffUnmatched(value=v, best_match=None, similarity=0.0) for v in sorted(candidates)
        ]
        return _DiffRunOutcome(result=DiffResult(matched=[], unmatched=unmatched, exact=exact), embed=None)

    from labelmerge.cache import EmbeddingCache
    from labelmerge.embedders.openai import OpenAIEmbedder

    config = LabelMergeConfig()
    progress = _DelayedProgress(
        prefix=f"Embedding {len(vocab) + len(candidates)} values",
        quiet=quiet,
        show_percent=True,
    )
    cache = EmbeddingCache(config.cache_dir) if config.cache_enabled else None
    embedder = OpenAIEmbedder(
        model=model,
        dimensions=dimensions,
        batch_size=config.embedding_batch_size,
        cache=cache,
        on_progress=progress.update,
    )

    texts_to_embed = vocab + candidates
    try:
        raw = await embedder.embed(texts_to_embed)
    finally:
        progress.finish()

    emb = _normalize_rows(np.array(raw, dtype=np.float64))
    vocab_emb = emb[: len(vocab)]
    cand_emb = emb[len(vocab) :]

    sim_matrix = cand_emb @ vocab_emb.T
    best_idx = np.argmax(sim_matrix, axis=1)
    best_scores = sim_matrix[np.arange(len(candidates)), best_idx]

    matched: list[DiffMatch] = []
    unmatched: list[DiffUnmatched] = []
    for i, value in enumerate(candidates):
        score = float(best_scores[i])
        canonical = vocab[int(best_idx[i])]
        if score >= threshold:
            matched.append(DiffMatch(value=value, canonical=canonical, similarity=score))
        else:
            unmatched.append(DiffUnmatched(value=value, best_match=canonical, similarity=score))

    matched.sort(key=lambda item: item.value)
    unmatched.sort(key=lambda item: item.value)
    exact.sort(key=lambda item: item.value)
    return _DiffRunOutcome(
        result=DiffResult(matched=matched, unmatched=unmatched, exact=exact),
        embed=_coerce_embed_run_stats(embedder),
    )


def _run_diff_with_metrics(
    *,
    vocab: list[str],
    new_values: list[str],
    threshold: float,
    model: str,
    dimensions: int | None,
    quiet: bool,
) -> _DiffRunOutcome:
    return asyncio.run(
        _run_diff_async(
            vocab=vocab,
            new_values=new_values,
            threshold=threshold,
            model=model,
            dimensions=dimensions,
            quiet=quiet,
        )
    )


def _run_diff(
    *,
    vocab: list[str],
    new_values: list[str],
    threshold: float,
    model: str,
    dimensions: int | None,
    quiet: bool,
) -> DiffResult:
    return _run_diff_with_metrics(
        vocab=vocab,
        new_values=new_values,
        threshold=threshold,
        model=model,
        dimensions=dimensions,
        quiet=quiet,
    ).result


def _merge_apply_stats(dest: JsonPathApplyStats, src: JsonPathApplyStats) -> None:
    dest.values_checked += src.values_checked
    dest.changed += src.changed
    dest.already_canonical += src.already_canonical
    dest.unmapped += src.unmapped
    dest.changes.update(src.changes)
    dest.unmapped_values.update(src.unmapped_values)


def _apply_text_lines(lines: list[str], mapping: dict[str, str]) -> tuple[list[str], JsonPathApplyStats]:
    stats = JsonPathApplyStats()
    output: list[str] = []
    for line in lines:
        if line == "":
            output.append(line)
            continue
        stats.values_checked += 1
        if line in mapping:
            canonical = mapping[line]
            if canonical == line:
                stats.already_canonical += 1
                output.append(line)
            else:
                stats.changed += 1
                stats.changes[(line, canonical)] += 1
                output.append(canonical)
        else:
            stats.unmapped += 1
            stats.unmapped_values[line] += 1
            output.append(line)
    return output, stats


def _prepare_apply_text_source(path: Path | None, mapping: dict[str, str]) -> _ApplyPreparedSource:
    if path is None:
        raw = sys.stdin.read()
        label = "<stdin>"
    else:
        raw = path.read_text()
        label = str(path)

    line_chunks = raw.splitlines(keepends=True)
    if not line_chunks and raw:
        line_chunks = [raw]

    stripped_lines: list[str] = []
    endings: list[str] = []
    for chunk in line_chunks:
        line = chunk.rstrip("\r\n")
        stripped_lines.append(line)
        endings.append(chunk[len(line) :])

    if not line_chunks and raw == "":
        stripped_lines = []
        endings = []

    new_lines, stats = _apply_text_lines(stripped_lines, mapping)
    output_text = "".join(f"{line}{ending}" for line, ending in zip(new_lines, endings, strict=True))

    if not line_chunks and raw == "":
        output_text = ""

    return _ApplyPreparedSource(
        label=label,
        path=path,
        output_text=output_text,
        changed=output_text != raw,
        stats=stats,
    )


def _prepare_apply_json_source(path: Path, path_expr: str, mapping: dict[str, str]) -> _ApplyPreparedSource:
    suffix = path.suffix.lower()
    raw = path.read_text()

    if suffix == ".json":
        data = json.loads(raw)
        stats = apply_mapping_to_json_path(data, path_expr, mapping)
        output_text = raw if stats.changed == 0 else f"{dumps_json(data, pretty=True)}\n"
        return _ApplyPreparedSource(
            label=str(path),
            path=path,
            output_text=output_text,
            changed=stats.changed > 0,
            stats=stats,
        )

    if suffix == ".jsonl":
        aggregate = JsonPathApplyStats()
        out_lines: list[str] = []
        for line in raw.splitlines():
            if not line.strip():
                out_lines.append(line)
                continue
            obj = json.loads(line)
            line_stats = apply_mapping_to_json_path(obj, path_expr, mapping)
            _merge_apply_stats(aggregate, line_stats)
            out_lines.append(dumps_json(obj, pretty=False))
        output_text = "\n".join(out_lines)
        if raw.endswith("\n"):
            output_text += "\n"
        return _ApplyPreparedSource(
            label=str(path),
            path=path,
            output_text=output_text,
            changed=aggregate.changed > 0,
            stats=aggregate,
        )

    raise typer.BadParameter("--path apply mode currently supports .json and .jsonl files")


def _load_mapping_file(mapping_file: Path) -> dict[str, str]:
    with open(mapping_file) as f:
        data: Any = json.load(f)
    if not isinstance(data, dict):
        raise typer.BadParameter("Mapping file must be a JSON object of raw -> canonical")
    mapping: dict[str, str] = {}
    for key, value in data.items():
        if value is None:
            continue
        mapping[str(key)] = str(value)
    return mapping


def _print_apply_summary(
    *,
    prepared: list[_ApplyPreparedSource],
    strict: bool,
    dry_run: bool,
    quiet: bool,
) -> tuple[dict[str, object], int]:
    aggregate = JsonPathApplyStats()
    change_files: dict[tuple[str, str], set[str]] = defaultdict(set)

    for item in prepared:
        _merge_apply_stats(aggregate, item.stats)
        for pair in item.stats.changes:
            change_files[pair].add(item.label)

    changes_sorted = sorted(
        aggregate.changes.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )
    changes_payload: list[dict[str, object]] = []
    for (from_value, to_value), occurrences in changes_sorted:
        changes_payload.append(
            {
                "from": from_value,
                "to": to_value,
                "occurrences": occurrences,
                "files": len(change_files[(from_value, to_value)]),
            }
        )

    summary: dict[str, object] = {
        "files_scanned": len(prepared),
        "values_checked": aggregate.values_checked,
        "would_change": aggregate.changed,
        "already_canonical": aggregate.already_canonical,
        "unmapped": aggregate.unmapped,
        "changes": changes_payload,
    }

    if not quiet:
        _stderr(
            f"{len(prepared)} file{'s' if len(prepared) != 1 else ''} scanned, "
            f"{aggregate.values_checked} values checked",
            quiet=quiet,
        )
        if dry_run:
            _stderr("", quiet=quiet)
            _stderr("Changes:", quiet=quiet)
            _stderr(
                f"  {aggregate.changed} values would be normalized "
                f"(across {sum(1 for p in prepared if p.stats.changed > 0)} files)",
                quiet=quiet,
            )
        else:
            _stderr("", quiet=quiet)
            _stderr("Changes:", quiet=quiet)
            _stderr(
                f"  {aggregate.changed} values normalized "
                f"(across {sum(1 for p in prepared if p.stats.changed > 0)} files)",
                quiet=quiet,
            )
        _stderr(
            f"  {aggregate.already_canonical} values already canonical (no change)",
            quiet=quiet,
        )
        _stderr(f"  {aggregate.unmapped} values unmapped (not in mapping)", quiet=quiet)

        if changes_payload:
            _stderr("", quiet=quiet)
            _stderr("Top normalizations:", quiet=quiet)
            for change in changes_payload[:5]:
                _stderr(
                    (
                        f'  "{change["from"]}" → "{change["to"]}"  '
                        f'({change["occurrences"]} occurrences in {change["files"]} files)'
                    ),
                    quiet=quiet,
                )

        if strict and aggregate.unmapped > 0:
            _stderr("", quiet=quiet)
            _stderr("Unmapped values (strict mode):", quiet=quiet)
            for value, count in aggregate.unmapped_values.most_common(20):
                _stderr(f'  {count}x  "{value}"', quiet=quiet)
            remaining = len(aggregate.unmapped_values) - 20
            if remaining > 0:
                _stderr(f"  ... and {remaining} more", quiet=quiet)

    return summary, aggregate.unmapped


def _command_source_label(read_result: _ReadInputsResult) -> str:
    if read_result.source_kind == "stdin":
        return "stdin"
    if len(read_result.files) == 1:
        return str(read_result.files[0])
    return f"{len(read_result.files)} files"


@app.command()
def dedupe(
    inputs: list[str] = typer.Argument(None, help="Input file(s), glob(s), or '-' for stdin."),
    stdin: bool = typer.Option(False, "--stdin", help="Read input values from stdin."),
    input_format: str = typer.Option(
        "auto",
        help="Input format: auto, text, json, jsonl, csv. For stdin, use text or json.",
    ),
    threshold: float = typer.Option(0.85, help="Cosine similarity threshold."),
    path: str | None = typer.Option(None, help="jq-style path for JSON/JSONL."),
    column: str | None = typer.Option(None, help="Column name for CSV."),
    model: str = typer.Option("text-embedding-3-small", help="Embedding model."),
    dimensions: int | None = typer.Option(None, help="Embedding dimensions."),
    output_format: str = typer.Option(
        "json",
        help="Output format: json, jsonl, csv, mapping, enum, jsonschema.",
    ),
    output: Path | None = typer.Option(None, "-o", help="Output file. Defaults to stdout."),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress stderr output."),
    stop_words: str | None = typer.Option(None, help="Comma-separated stop words."),
    name_groups: bool = typer.Option(False, "--name-groups", help="Name groups with LLM."),
    naming_model: str = typer.Option("gpt-4o-mini", help="Model for group naming."),
    review_profile: str = typer.Option(
        "generic",
        help="Review heuristic profile for --name-groups (default: generic).",
    ),
    review_rules: Path | None = typer.Option(
        None,
        help="Optional JSON file to customize generic split-review heuristics.",
    ),
    review_policy: str | None = typer.Option(
        None,
        help="Advanced: Python review policy plugin in 'module.path:attr' form.",
    ),
    max_component_size: int = typer.Option(100, help="Max component size before splitting."),
) -> None:
    """Deduplicate text labels from files, globs, or stdin."""
    read_result = _read_inputs(
        inputs=inputs,
        stdin_flag=stdin,
        input_format=input_format,
        path_expr=path,
        column=column,
        quiet=quiet,
    )

    dedupe_outcome = _run_dedupe_with_metrics(
        texts=read_result.texts,
        threshold=threshold,
        model=model,
        dimensions=dimensions,
        stop_words=stop_words,
        max_component_size=max_component_size,
        name_groups=name_groups,
        naming_model=naming_model,
        review_profile=review_profile,
        review_rules=review_rules,
        review_policy=review_policy,
        quiet=quiet,
    )
    result = dedupe_outcome.result

    total_values = len(read_result.texts)
    unique_values = len(set(read_result.texts))
    _print_dedupe_summary(
        result=result,
        embed=dedupe_outcome.embed,
        total_values=total_values,
        unique_values=unique_values,
        source_label=_command_source_label(read_result),
        quiet=quiet,
        mode_label="Dedupe",
    )
    _write_result_output(result, output_format, output, pretty=pretty)


@app.command()
def corpus(
    inputs: list[str] = typer.Argument(..., help="JSON/JSONL file(s) or glob(s) to scan."),
    path: str = typer.Option(..., help="jq-style path for the target field."),
    input_format: str = typer.Option(
        "auto", help="Input format override: auto, json, jsonl, csv, text."
    ),
    threshold: float = typer.Option(0.85, help="Cosine similarity threshold (default is usually good)."),
    model: str = typer.Option("text-embedding-3-small", help="Embedding model."),
    dimensions: int | None = typer.Option(None, help="Embedding dimensions."),
    output_format: str = typer.Option(
        "mapping",
        help="Output format: mapping (default), enum, jsonschema, json, jsonl, csv.",
    ),
    output: Path | None = typer.Option(None, "-o", help="Output file. Defaults to stdout."),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress stderr output."),
    stop_words: str | None = typer.Option(None, help="Comma-separated stop words."),
    name_groups: bool = typer.Option(False, "--name-groups", help="Name groups with LLM."),
    naming_model: str = typer.Option("gpt-4o-mini", help="Model for group naming."),
    review_profile: str = typer.Option(
        "generic",
        help="Review heuristic profile for --name-groups (default: generic).",
    ),
    review_rules: Path | None = typer.Option(
        None,
        help="Optional JSON file to customize generic split-review heuristics.",
    ),
    review_policy: str | None = typer.Option(
        None,
        help="Advanced: Python review policy plugin in 'module.path:attr' form.",
    ),
    max_component_size: int = typer.Option(100, help="Max component size before splitting."),
) -> None:
    """Extract a field across a corpus and output a dedup mapping."""
    read_result = _read_inputs(
        inputs=inputs,
        stdin_flag=False,
        input_format=input_format,
        path_expr=path,
        column=None,
        quiet=quiet,
    )

    dedupe_outcome = _run_dedupe_with_metrics(
        texts=read_result.texts,
        threshold=threshold,
        model=model,
        dimensions=dimensions,
        stop_words=stop_words,
        max_component_size=max_component_size,
        name_groups=name_groups,
        naming_model=naming_model,
        review_profile=review_profile,
        review_rules=review_rules,
        review_policy=review_policy,
        quiet=quiet,
    )
    result = dedupe_outcome.result
    _print_corpus_summary(
        result=result,
        embed=dedupe_outcome.embed,
        read_result=read_result,
        threshold=threshold,
        quiet=quiet,
    )
    _write_result_output(result, output_format, output, pretty=pretty)


@app.command()
def diff(
    vocab_file: Path,
    inputs: list[str] = typer.Argument(None, help="New data file(s), glob(s), or '-' for stdin."),
    stdin: bool = typer.Option(False, "--stdin", help="Read new values from stdin."),
    input_format: str = typer.Option(
        "auto",
        help="Input format for new data: auto, text, json, jsonl, csv (stdin: text/json).",
    ),
    path: str | None = typer.Option(None, help="jq-style path for JSON/JSONL new data."),
    column: str | None = typer.Option(None, help="Column name for CSV new data."),
    threshold: float = typer.Option(0.85, help="Cosine similarity threshold."),
    model: str = typer.Option("text-embedding-3-small", help="Embedding model."),
    dimensions: int | None = typer.Option(None, help="Embedding dimensions."),
    output_format: str = typer.Option("json", help="Output format: json or mapping."),
    output: Path | None = typer.Option(None, "-o", help="Output file. Defaults to stdout."),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress stderr output."),
) -> None:
    """Compare new values against an existing vocabulary."""
    vocab = _load_vocab_canonicals(vocab_file)
    read_result = _read_inputs(
        inputs=inputs,
        stdin_flag=stdin,
        input_format=input_format,
        path_expr=path,
        column=column,
        quiet=quiet,
    )

    diff_outcome = _run_diff_with_metrics(
        vocab=vocab,
        new_values=read_result.texts,
        threshold=threshold,
        model=model,
        dimensions=dimensions,
        quiet=quiet,
    )
    diff_result = diff_outcome.result
    _print_diff_summary(
        diff_result=diff_result,
        embed=diff_outcome.embed,
        vocab_count=len(vocab),
        vocab_file=vocab_file,
        read_result=read_result,
        quiet=quiet,
    )

    if output_format == "json":
        _write_json_data(diff_result.model_dump(), output=output, pretty=pretty)
    elif output_format == "mapping":
        _write_json_data(diff_result.to_mapping(), output=output, pretty=pretty, sort_keys=True)
    else:
        raise typer.BadParameter("diff --output-format must be 'json' or 'mapping'")


@app.command()
def apply(
    mapping_file: Path,
    inputs: list[str] = typer.Argument(None, help="Data file(s), glob(s), or '-' for stdin."),
    stdin: bool = typer.Option(False, "--stdin", help="Read data from stdin (text mode)."),
    path: str | None = typer.Option(None, help="jq-style path for JSON/JSONL field normalization."),
    in_place: bool = typer.Option(False, "--in-place", help="Write changes back to source files."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show structured summary without writing."),
    strict: bool = typer.Option(False, "--strict", help="Fail if unmapped values are encountered."),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print dry-run JSON output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress stderr output."),
) -> None:
    """Normalize values using an existing mapping."""
    if dry_run and in_place:
        raise typer.BadParameter("Use either --dry-run or --in-place, not both")

    mapping = _load_mapping_file(mapping_file)
    source_kind, paths = _resolve_input_sources(inputs, stdin)

    if path is not None and source_kind == "stdin":
        raise typer.BadParameter("--path apply mode does not support stdin yet")

    if path is not None and (not in_place and not dry_run) and len(paths) > 1:
        raise typer.BadParameter(
            "For multiple structured files, use --in-place or --dry-run (stdout is ambiguous)"
        )

    prepared: list[_ApplyPreparedSource] = []
    scan_progress = _DelayedProgress(prefix=f"Scanning {len(paths)} files", quiet=quiet)

    if source_kind == "stdin":
        prepared.append(_prepare_apply_text_source(None, mapping))
    else:
        try:
            for idx, file_path in enumerate(paths, start=1):
                if path is None:
                    prepared.append(_prepare_apply_text_source(file_path, mapping))
                else:
                    prepared.append(_prepare_apply_json_source(file_path, path, mapping))
                if len(paths) > 1:
                    scan_progress.update(idx, len(paths))
        finally:
            scan_progress.finish()

    summary, unmapped_count = _print_apply_summary(
        prepared=prepared,
        strict=strict,
        dry_run=dry_run,
        quiet=quiet,
    )

    if dry_run:
        _write_json_data(summary, output=None, pretty=pretty)
        if strict and unmapped_count > 0:
            raise typer.Exit(1)
        return

    if strict and unmapped_count > 0:
        raise typer.Exit(1)

    if in_place:
        if source_kind == "stdin":
            raise typer.BadParameter("--in-place cannot be used with stdin")
        for item in prepared:
            if item.path is None or not item.changed:
                continue
            item.path.write_text(item.output_text)
        return

    if len(prepared) != 1:
        raise typer.BadParameter("Stdout apply mode requires a single input source")
    sys.stdout.write(prepared[0].output_text)


@app.command()
def sweep(
    inputs: list[str] = typer.Argument(None, help="Input file(s), glob(s), or '-' for stdin."),
    stdin: bool = typer.Option(False, "--stdin", help="Read input values from stdin."),
    input_format: str = typer.Option(
        "auto",
        help="Input format: auto, text, json, jsonl, csv (stdin: text/json).",
    ),
    thresholds: str = typer.Option("0.80,0.85,0.90,0.95", help="Comma-separated thresholds."),
    path: str | None = typer.Option(None, help="jq-style path for JSON/JSONL."),
    column: str | None = typer.Option(None, help="Column name for CSV."),
    model: str = typer.Option("text-embedding-3-small", help="Embedding model."),
    dimensions: int | None = typer.Option(None, help="Embedding dimensions."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress scan progress to stderr."),
) -> None:
    """Run dedup at multiple thresholds to find the right one."""
    from labelmerge.cache import EmbeddingCache
    from labelmerge.embedders.openai import OpenAIEmbedder

    read_result = _read_inputs(
        inputs=inputs,
        stdin_flag=stdin,
        input_format=input_format,
        path_expr=path,
        column=column,
        quiet=quiet,
    )

    config = LabelMergeConfig()
    cache = EmbeddingCache(config.cache_dir) if config.cache_enabled else None
    embedder = OpenAIEmbedder(model=model, dimensions=dimensions, cache=cache)

    threshold_list = [float(t.strip()) for t in thresholds.split(",")]

    table = Table(title="Threshold Sweep")
    table.add_column("Threshold", justify="right")
    table.add_column("Groups", justify="right")
    table.add_column("Singletons", justify="right")
    table.add_column("Max Group", justify="right")
    table.add_column("% Grouped", justify="right")

    for t in threshold_list:
        sd = LabelMerge(embedder=embedder, threshold=t, cache_enabled=False)
        result = asyncio.run(sd.dedupe(read_result.texts))
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
    inputs: list[str] = typer.Argument(None, help="Input file(s), glob(s), or '-' for stdin."),
    stdin: bool = typer.Option(False, "--stdin", help="Read values from stdin."),
    input_format: str = typer.Option(
        "auto", help="Input format: auto, text, json, jsonl, csv (stdin: text/json)."
    ),
    path: str | None = typer.Option(None, help="jq-style path for JSON/JSONL."),
    column: str | None = typer.Option(None, help="Column name for CSV."),
) -> None:
    """Quick stats about input data."""
    read_result = _read_inputs(
        inputs=inputs,
        stdin_flag=stdin,
        input_format=input_format,
        path_expr=path,
        column=column,
        quiet=False,
    )

    counts = Counter(read_result.texts)
    total = len(read_result.texts)
    unique = len(counts)
    singletons = sum(1 for c in counts.values() if c == 1)

    console.print(f"Total items:    {total}")
    console.print(f"Unique items:   {unique}")
    console.print(f"Exact dupes:    {total - unique}")
    if unique > 0:
        console.print(f"Singletons:     {singletons} ({singletons / unique * 100:.1f}%)")
    else:
        console.print("Singletons:     0 (0.0%)")
    if total > 0:
        console.print(f"Unique rate:    {unique / total * 100:.1f}%")
    else:
        console.print("Unique rate:    0.0%")


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


if __name__ == "__main__":  # pragma: no cover
    app()
