from __future__ import annotations

from labelmerge.jsonpath import apply_mapping_to_json_path, extract_json_path_strings, parse_json_path


def test_parse_json_path_handles_items_array_field() -> None:
    tokens = parse_json_path(".items[].label")
    assert [t.kind for t in tokens] == ["field", "all", "field"]
    assert [t.value for t in tokens] == ["items", None, "label"]


def test_extract_json_path_strings_skips_null_and_missing_values() -> None:
    data = {
        "items": [
            {"label": "PHQ-9"},
            {"label": None},
            {"other": "ignored"},
            {"label": "BDI-II"},
        ]
    }

    assert extract_json_path_strings(data, ".items[].label") == ["PHQ-9", "BDI-II"]


def test_apply_mapping_to_json_path_updates_values_and_tracks_stats() -> None:
    data = {
        "items": [
            {"label": "Total Cholesterol"},
            {"label": "total cholesterol"},
            {"label": "retinal thickness"},
            {"label": None},
        ]
    }
    mapping = {
        "Total Cholesterol": "total cholesterol",
        "total cholesterol": "total cholesterol",
    }

    stats = apply_mapping_to_json_path(data, ".items[].label", mapping)

    assert data["items"][0]["label"] == "total cholesterol"
    assert data["items"][1]["label"] == "total cholesterol"
    assert data["items"][2]["label"] == "retinal thickness"
    assert stats.values_checked == 3
    assert stats.changed == 1
    assert stats.already_canonical == 1
    assert stats.unmapped == 1
    assert stats.changes[("Total Cholesterol", "total cholesterol")] == 1
    assert stats.unmapped_values["retinal thickness"] == 1
