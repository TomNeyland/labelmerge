from __future__ import annotations

from io import StringIO

from labelmerge.io.readers import expand_input_patterns, read_json_array_stream, read_values


def test_read_json_array_stream_coerces_and_skips_null() -> None:
    stream = StringIO('["PHQ-9", null, 42]')
    assert read_json_array_stream(stream) == ["PHQ-9", "42"]


def test_read_values_json_array_without_path(tmp_path) -> None:
    path = tmp_path / "values.json"
    path.write_text('["a", "b", null]')

    assert read_values(path) == ["a", "b"]


def test_expand_input_patterns_sorts_and_deduplicates_matches(tmp_path) -> None:
    (tmp_path / "b.json").write_text("[]")
    (tmp_path / "a.json").write_text("[]")

    pattern = str(tmp_path / "*.json")
    matches = expand_input_patterns([pattern, str(tmp_path / "a.json")])

    assert [p.name for p in matches] == ["a.json", "b.json"]
