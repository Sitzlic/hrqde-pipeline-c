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
