"""Schwellenwert-Bestimmung auf dem Kalibrierteil (SA-C.3.05).

Fuer jede GoldSpan mit Referenz (esco_uri gesetzt) wird der Mapper befragt.
Referenz kann eine ESCO-URI sein oder der Marker "kein Konzept": dann ist
jede akzeptierte Zuordnung ein Fehlgriff und drueckt die Precision. Pro
Schwellenwert entsteht eine Zeile Precision/Recall/F1, die Tabelle ist die
Tuning-Grundlage fuer AF-C.2.03.
"""

from __future__ import annotations

import logging
from typing import Callable

from hrqde_c.gold.reference import NO_CONCEPT
from hrqde_c.models import GoldAdvertisement

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [round(0.30 + i * 0.05, 2) for i in range(13)]  # 0.30 .. 0.90


def _default_match_fn(text: str) -> tuple[str, float]:
    from hrqde_c.mapping import esco

    match = esco.get_index().match(text, top_k=1)[0]
    return match.concept.uri, match.score


def sweep(
    gold_ads: list[GoldAdvertisement],
    thresholds: list[float] | None = None,
    match_fn: Callable[[str], tuple[str, float]] | None = None,
) -> list[dict]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    match = match_fn or _default_match_fn

    references = [
        (span.text, span.esco_uri)
        for gold in gold_ads if gold.partition == "kalibrier"
        for span in gold.spans if span.esco_uri
    ]
    if not references:
        raise ValueError(
            "Kalibrierteil enthaelt keine GoldSpans mit esco_uri, "
            "erst Referenz-URIs annotieren"
        )
    with_concept = sum(1 for _, uri in references if uri != NO_CONCEPT)
    log.info(
        "calibration: %d Referenz-Spans im Kalibrierteil (%d ohne Konzept)",
        len(references), len(references) - with_concept,
    )

    scored = [(match(text), uri) for text, uri in references]
    rows = []
    for threshold in thresholds:
        accepted = [(pred, uri) for (pred, uri) in scored if pred[1] >= threshold]
        correct = sum(
            1 for (pred_uri, _), uri in accepted if uri != NO_CONCEPT and pred_uri == uri
        )
        precision = correct / len(accepted) if accepted else 0.0
        recall = correct / with_concept if with_concept else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accepted": len(accepted),
            "references": len(references),
            "no_concept": len(references) - with_concept,
        })
    return rows


def mark_best(rows: list[dict]) -> dict:
    """Spalte best setzen: "true" nur in der Zeile mit dem hoechsten F1
    (bei Gleichstand die niedrigste Schwelle). -> beste Zeile"""
    best = max(rows, key=lambda r: r["f1"])
    for row in rows:
        row["best"] = "true" if row is best else "false"
    return best
