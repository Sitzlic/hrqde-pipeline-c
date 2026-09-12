from __future__ import annotations

import logging
from pathlib import Path

import typer

from hrqde_c import pipeline
from hrqde_c.acquisition import jobsuche

app = typer.Typer(add_completion=False, help="HR-QDE Säule C - Pipeline-Lauf")


@app.command()
def run(
    input: Path = typer.Option(
        Path("data/input/example_posting.json"),
        "--input",
        "-i",
        help="JSON-Datei mit strukturierten Stellenanzeigen.",
    ),
    output: Path = typer.Option(
        Path("data/output"),
        "--output",
        "-o",
        help="Verzeichnis für die TTL-Lieferung.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    paths = pipeline.run(input, output)
    typer.echo(f"\nFertig. {len(paths)} TTL-Datei(en) in {output}:")
    for p in paths:
        typer.echo(f"  - {p}")


@app.command()
def fetch(
    was: str = typer.Option(..., "--was", help="Suchbegriff, z.B. Berufsbezeichnung."),
    wo: str = typer.Option(None, "--wo", help="Ort (optional)."),
    count: int = typer.Option(25, "--count", "-n", help="Anzahl Anzeigen mit Volltext."),
    output: Path = typer.Option(
        Path("data/input/fetched.json"),
        "--output",
        "-o",
        help="Zieldatei, direkt als Pipeline-Input verwendbar.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stellenanzeigen über die BA-Jobsuche-API einsammeln (braucht deutsche IP)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    raws = jobsuche.fetch(was, wo=wo, count=count)
    jobsuche.save(raws, output)
    typer.echo(f"{len(raws)} Anzeigen -> {output}")


@app.command("gold-sample")
def gold_sample(
    input: Path = typer.Option(..., "--input", "-i", help="JSON mit RawAdvertisements."),
    size: int = typer.Option(60, "--size", "-n", help="Stichprobengröße."),
    calib_ratio: float = typer.Option(0.5, "--calib-ratio", help="Anteil Kalibrierteil."),
    seed: int = typer.Option(42, "--seed", help="Seed für die Zufallszuordnung."),
    out_dir: Path = typer.Option(Path("data/gold"), "--out-dir", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stichprobe ziehen, Kalibrier-/Test-Split festhalten, Label-Studio-Tasks
    mit Vor-Annotationen aus beiden Extraktoren schreiben."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    from hrqde_c.gold import label_studio, sampling

    raws = pipeline.acquire(input)
    partition = sampling.draw(raws, size=size, calib_ratio=calib_ratio, seed=seed)
    sampling.write_manifest(out_dir / "sample_manifest.json", partition, seed, calib_ratio)
    tasks = label_studio.build_tasks(raws, partition)
    label_studio.write_tasks(tasks, out_dir / "labelstudio_tasks.json")
    label_studio.write_config(out_dir / "labelstudio_config.xml")
    typer.echo(f"{len(tasks)} Tasks mit Vor-Annotationen -> {out_dir}")


@app.command("gold-import")
def gold_import(
    export: Path = typer.Option(..., "--export", "-e", help="Label-Studio-Export (JSON)."),
    manifest: Path = typer.Option(
        Path("data/gold/sample_manifest.json"), "--manifest", "-m"
    ),
    output: Path = typer.Option(Path("data/gold/gold_advertisements.json"), "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Label-Studio-Export als BIO-Gold-Standard übernehmen, Kappa berechnen."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    import json

    from hrqde_c.gold import label_studio, sampling

    payload = json.loads(export.read_text(encoding="utf-8"))
    partition = sampling.load_manifest(manifest)
    gold = label_studio.import_gold(payload, partition)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([g.model_dump(exclude_none=True) for g in gold],
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    doubled = [g for g in gold if g.inter_annotator_agreement is not None]
    parts = {"kalibrier": 0, "test": 0}
    for g in gold:
        parts[g.partition] += 1
    typer.echo(
        f"{len(gold)} GoldAdvertisements -> {output} "
        f"({parts['kalibrier']} Kalibrier, {parts['test']} Test)"
    )
    if doubled:
        mean = sum(g.inter_annotator_agreement for g in doubled) / len(doubled)
        typer.echo(f"Doppelannotation: {len(doubled)} Anzeigen, mittleres Kappa {mean:.3f}")


@app.command()
def evaluate(
    gold: Path = typer.Option(
        Path("data/gold/gold_advertisements.json"), "--gold", "-g"
    ),
    input: Path = typer.Option(..., "--input", "-i", help="JSON mit den Rohanzeigen."),
    out_dir: Path = typer.Option(Path("data/gold"), "--out-dir", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Extraktionsvarianten gegen den Testteil auswerten (Span-F1) und
    den Vergleichsbericht als CSV exportieren."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    import json

    from hrqde_c.gold import evaluator, reporter
    from hrqde_c.models import GoldAdvertisement

    gold_ads = [GoldAdvertisement(**g) for g in json.loads(gold.read_text(encoding="utf-8"))]
    raws_by_id = {r.id: r for r in pipeline.acquire(input)}
    runs = [
        evaluator.evaluate_variant(variant, gold_ads, raws_by_id, str(gold))
        for variant in evaluator.VARIANTS
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_runs.json").write_text(
        json.dumps(
            [{"run": run.model_dump(), "result": result.model_dump()} for run, result in runs],
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    reporter.write_report(runs, out_dir / "evaluation_report.csv")
    for run, result in sorted(runs, key=lambda r: r[1].f1, reverse=True):
        typer.echo(
            f"{run.variant:16s} P={result.precision:.3f} R={result.recall:.3f} "
            f"F1={result.f1:.3f} (tp={result.true_positives} fp={result.false_positives} "
            f"fn={result.false_negatives})"
        )
    typer.echo(f"Bericht -> {out_dir / 'evaluation_report.csv'}")


@app.command()
def calibrate(
    gold: Path = typer.Option(
        Path("data/gold/gold_advertisements.json"), "--gold", "-g"
    ),
    output: Path = typer.Option(Path("data/gold/calibration.csv"), "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Schwellenwert-Tabelle auf dem Kalibrierteil erzeugen (Tuning-Grundlage
    für den Mapping-Threshold)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    import csv
    import json

    from hrqde_c.gold import calibration
    from hrqde_c.models import GoldAdvertisement

    gold_ads = [GoldAdvertisement(**g) for g in json.loads(gold.read_text(encoding="utf-8"))]
    rows = calibration.sweep(gold_ads)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda r: r["f1"])
    typer.echo(f"{len(rows)} Schwellenwerte -> {output}")
    typer.echo(f"Bestes F1 {best['f1']:.3f} bei Threshold {best['threshold']}")


@app.command("load-neo4j")
def load_neo4j(
    input_dir: Path = typer.Option(Path("data/output"), "--input", "-i",
                                   help="Verzeichnis mit TTL-Dateien."),
    uri: str = typer.Option("bolt://localhost:7687", "--uri"),
    user: str = typer.Option("neo4j", "--user"),
    password: str = typer.Option("hrqde-dev", "--password"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """TTL-Lieferung über neosemantics in Neo4j ablegen (docker compose up -d)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    from hrqde_c.graph import neo4j_import

    files, triples = neo4j_import.load_directory(input_dir, uri, user, password)
    typer.echo(f"{files} TTL-Dateien, {triples} Tripel -> {uri}")
