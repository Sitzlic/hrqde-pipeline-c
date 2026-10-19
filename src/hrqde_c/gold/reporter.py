"""Vergleichsbericht ueber die Auswertungs-Laeufe (Drei-Wege-Vergleich)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from hrqde_c.models import EvaluationResult, EvaluationRun

log = logging.getLogger(__name__)


# Varianten, die auf Span-Ebene nicht gegen den deutschen Gold Standard messbar sind
# (UCon3-UCas5, Ausnahme): sie stehen im Bericht als eigene Zeile mit Vermerk
UNMEASURED_VARIANTS: dict[str, str] = {
    "adj-noun-en-uebersetzt": "nicht gemessen: Spans liegen auf dem uebersetzten Text",
}


def write_report(
    runs: list[tuple[EvaluationRun, EvaluationResult]],
    path: Path,
    unmeasured: dict[str, str] | None = None,
) -> None:
    unmeasured = UNMEASURED_VARIANTS if unmeasured is None else unmeasured
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "variant", "precision", "recall", "f1",
            "true_positives", "false_positives", "false_negatives",
            "advertisements", "run_id", "run_timestamp", "note",
        ])
        for run, result in sorted(runs, key=lambda r: r[1].f1, reverse=True):
            writer.writerow([
                run.variant, result.precision, result.recall, result.f1,
                result.true_positives, result.false_positives, result.false_negatives,
                run.advertisement_count, run.id, run.run_timestamp, "",
            ])
        for variant, note in unmeasured.items():
            writer.writerow([variant, "", "", "", "", "", "", "", "", "", note])
    log.info("reporter: %d Laeufe, %d nicht gemessen -> %s", len(runs), len(unmeasured), path)
