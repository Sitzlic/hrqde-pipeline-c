"""Vergleichsbericht ueber die Auswertungs-Laeufe (Drei-Wege-Vergleich)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from hrqde_c.models import EvaluationResult, EvaluationRun

log = logging.getLogger(__name__)


def write_report(
    runs: list[tuple[EvaluationRun, EvaluationResult]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "variant", "precision", "recall", "f1",
            "true_positives", "false_positives", "false_negatives",
            "advertisements", "run_id", "run_timestamp",
        ])
        for run, result in sorted(runs, key=lambda r: r[1].f1, reverse=True):
            writer.writerow([
                run.variant, result.precision, result.recall, result.f1,
                result.true_positives, result.false_positives, result.false_negatives,
                run.advertisement_count, run.id, run.run_timestamp,
            ])
    log.info("reporter: %d Laeufe -> %s", len(runs), path)
