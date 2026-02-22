from __future__ import annotations

import json

from typer.testing import CliRunner

import labelmerge.cli as cli_mod
from labelmerge.models import Group, Member, Result


def _runner() -> CliRunner:
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:  # pragma: no cover - older Click versions
        return CliRunner()


def _fake_result() -> Result:
    return Result(
        groups=[
            Group(
                group_id=0,
                members=[
                    Member(text="Total Cholesterol", count=2),
                    Member(text="total cholesterol", count=5),
                ],
                canonical="total cholesterol",
                size=2,
                total_occurrences=7,
            )
        ],
        singletons=[Member(text="sleep quality", count=1)],
        threshold=0.85,
        model="test",
        n_input=3,
        n_grouped=2,
        n_singletons=1,
    )


def test_dedupe_stdin_enum_output_with_stubbed_engine(monkeypatch) -> None:
    def fake_run_dedupe(**kwargs):
        assert kwargs["texts"] == ["Total Cholesterol", "sleep quality"]
        return cli_mod._DedupeRunOutcome(result=_fake_result(), embed=cli_mod._EmbedRunStats(total=2, cached=0))

    monkeypatch.setattr(cli_mod, "_run_dedupe_with_metrics", fake_run_dedupe)

    result = _runner().invoke(
        cli_mod.app,
        ["dedupe", "--stdin", "--output-format", "enum", "--quiet"],
        input="Total Cholesterol\nsleep quality\n",
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == ["sleep quality", "total cholesterol"]
    if hasattr(result, "stderr"):
        assert result.stderr == ""


def test_diff_exact_only_avoids_embedding_and_outputs_structured_json(tmp_path) -> None:
    vocab_file = tmp_path / "vocab.json"
    data_file = tmp_path / "new.txt"
    vocab_file.write_text('["sleep quality", "total cholesterol"]')
    data_file.write_text("total cholesterol\nsleep quality\ntotal cholesterol\n")

    result = _runner().invoke(cli_mod.app, ["diff", str(vocab_file), str(data_file), "--quiet"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["matched"] == []
    assert payload["unmatched"] == []
    assert payload["exact"] == [
        {"value": "sleep quality", "canonical": "sleep quality"},
        {"value": "total cholesterol", "canonical": "total cholesterol"},
    ]


def test_apply_text_stdout_normalizes_and_keeps_summary_on_stderr(tmp_path) -> None:
    mapping_file = tmp_path / "mapping.json"
    data_file = tmp_path / "data.txt"
    mapping_file.write_text(json.dumps({"Total Cholesterol": "total cholesterol"}))
    data_file.write_text("Total Cholesterol\nretinal thickness\n")

    result = _runner().invoke(cli_mod.app, ["apply", str(mapping_file), str(data_file)])

    assert result.exit_code == 0
    assert result.stdout == "total cholesterol\nretinal thickness\n"
    stderr = result.stderr if hasattr(result, "stderr") else result.output
    assert "values checked" in stderr
    assert "Top normalizations" in stderr


def test_apply_json_dry_run_reports_changes_without_writing(tmp_path) -> None:
    mapping_file = tmp_path / "mapping.json"
    data_file = tmp_path / "data.json"
    mapping_file.write_text(
        json.dumps(
            {
                "Total Cholesterol": "total cholesterol",
                "total cholesterol": "total cholesterol",
            }
        )
    )
    original = {
        "items": [
            {"label": "Total Cholesterol"},
            {"label": "total cholesterol"},
            {"label": "retinal thickness"},
            {"label": None},
        ]
    }
    data_file.write_text(json.dumps(original))

    result = _runner().invoke(
        cli_mod.app,
        ["apply", str(mapping_file), str(data_file), "--path", ".items[].label", "--dry-run", "--quiet"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["files_scanned"] == 1
    assert payload["values_checked"] == 3
    assert payload["would_change"] == 1
    assert payload["already_canonical"] == 1
    assert payload["unmapped"] == 1
    assert payload["changes"][0]["from"] == "Total Cholesterol"
    assert json.loads(data_file.read_text()) == original
