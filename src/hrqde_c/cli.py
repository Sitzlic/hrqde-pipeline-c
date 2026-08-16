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
