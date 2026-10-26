"""CLI-Verhalten ohne externe Dienste: Fehlerpfade, fetch-Protokoll."""

import json
from pathlib import Path

from typer.testing import CliRunner

from hrqde_c.cli import app

runner = CliRunner()


def test_evaluate_without_gold_file_gives_hint(tmp_path: Path):
    result = runner.invoke(app, ["evaluate", "-g", str(tmp_path / "gold.json"),
                                 "-i", "data/input/example_posting.json"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "Gold-Datei" in result.output and "gold-import" in result.output


def test_calibrate_without_reference_uris_gives_hint(tmp_path: Path):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([{
        "raw_id": "ad-1", "partition": "kalibrier", "tokens": ["Java"], "bio_labels": ["B-K"],
        "spans": [{"text": "Java", "span_type": "knowledge", "token_start": 0, "token_end": 1}],
        "annotator_ids": ["a1"],
    }]), encoding="utf-8")
    result = runner.invoke(app, ["calibrate", "-g", str(gold), "-o", str(tmp_path / "c.csv")])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "esco_uri" in result.output and "gold-esco-candidates" in result.output


def test_run_without_input_gives_hint(tmp_path: Path):
    result = runner.invoke(app, ["run", "-i", str(tmp_path / "nix.json"), "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "Eingabedatei" in result.output and "fetch" in result.output


def test_load_neo4j_without_directory_gives_hint(tmp_path: Path):
    result = runner.invoke(app, ["load-neo4j", "-i", str(tmp_path / "nix")])
    assert result.exit_code == 1
    assert "Verzeichnis" in result.output and "hrqde-c run" in result.output


def test_fetch_writes_fetch_log(tmp_path: Path, monkeypatch):
    from hrqde_c.acquisition import jobsuche

    monkeypatch.setattr(jobsuche, "fetch_with_stats",
                        lambda was, wo=None, count=25: ([], {"was": was, "wo": wo,
                                                            "count": count, "hits": 0,
                                                            "fetched": 0, "skipped": 0,
                                                            "errors": 0}))
    out = tmp_path / "fetched.json"
    result = runner.invoke(app, ["fetch", "--was", "Pflege", "-n", "1", "-o", str(out)])
    assert result.exit_code == 0, result.output
    log = json.loads((tmp_path / "fetched.fetch.json").read_text(encoding="utf-8"))
    assert log["was"] == "Pflege" and log["output"] == str(out)
    assert "Abrufprotokoll" in result.output
