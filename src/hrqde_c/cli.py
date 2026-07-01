from __future__ import annotations

import logging
from pathlib import Path

import typer

from hrqde_c import pipeline

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
