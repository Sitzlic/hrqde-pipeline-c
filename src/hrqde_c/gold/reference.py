"""Referenz-URIs fuer die Gold-Spans des Kalibrierteils (esco_uri, SA-C.3.05).

Zwei Schritte. write_candidates schickt jeden Gold-Span des Kalibrierteils
durch den ESCO-Mapper und schreibt die Top-k-Vorschlaege als CSV. Der
Annotator traegt pro Span in der Spalte decision entweder den Rang des
richtigen Vorschlags (1..k), eine andere ESCO-URI oder "-" fuer "kein
passendes Konzept" ein. apply_decisions liest das zurueck in die Gold-Datei.

Die Zuordnung passiert erst nach der Zusammenfuehrung der Doppelannotation,
damit die Referenz auf der konsolidierten Span-Menge liegt.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable

from hrqde_c.models import GoldAdvertisement

log = logging.getLogger(__name__)

# Marker fuer "kein passendes ESCO-Konzept", zaehlt in der Kalibrierung als
# Referenz, bei der jede akzeptierte Zuordnung ein Fehlgriff ist
NO_CONCEPT = "-"

FIELDS = [
    "raw_id", "token_start", "token_end", "span_text", "span_type",
    "rank", "concept_uri", "matched_label", "score", "decision",
]

Candidate = tuple[str, str, float]  # (uri, label, score)


def _default_candidates(text: str, top_k: int) -> list[Candidate]:
    from hrqde_c.mapping import esco

    return [
        (m.concept.uri, m.label, m.score)
        for m in esco.get_index().match(text, top_k=top_k)
    ]


def write_candidates(
    gold_ads: list[GoldAdvertisement],
    path: Path,
    top_k: int = 3,
    candidates_fn: Callable[[str, int], list[Candidate]] | None = None,
) -> int:
    """Vorschlagsliste fuer den Kalibrierteil. -> Anzahl Spans"""
    fetch = candidates_fn or _default_candidates
    path.parent.mkdir(parents=True, exist_ok=True)
    spans = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for gold in gold_ads:
            if gold.partition != "kalibrier":
                continue
            for span in gold.spans:
                spans += 1
                for rank, (uri, label, score) in enumerate(fetch(span.text, top_k), start=1):
                    writer.writerow({
                        "raw_id": gold.raw_id,
                        "token_start": span.token_start,
                        "token_end": span.token_end,
                        "span_text": span.text,
                        "span_type": span.span_type.value,
                        "rank": rank,
                        "concept_uri": uri,
                        "matched_label": label,
                        "score": f"{score:.4f}",
                        # nur in der ersten Zeile eines Spans auszufuellen
                        "decision": "",
                    })
    log.info("reference: %d Kalibrier-Spans mit Top-%d-Vorschlaegen -> %s", spans, top_k, path)
    return spans


def _resolve(decision: str, candidates: dict[int, str]) -> str | None:
    decision = decision.strip()
    if not decision:
        return None
    if decision == NO_CONCEPT:
        return NO_CONCEPT
    if decision.isdigit():
        uri = candidates.get(int(decision))
        if uri is None:
            raise ValueError(f"Rang {decision} hat keinen Vorschlag")
        return uri
    return decision  # freie URI


def apply_decisions(gold_ads: list[GoldAdvertisement], csv_path: Path) -> int:
    """Entscheidungen aus der CSV in die esco_uri der Gold-Spans uebernehmen.
    -> Anzahl gesetzter Referenzen"""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))

    # Kandidaten und Entscheidung pro Span einsammeln
    by_span: dict[tuple[str, int, int], dict] = {}
    for row in rows:
        key = (row["raw_id"], int(row["token_start"]), int(row["token_end"]))
        entry = by_span.setdefault(key, {"candidates": {}, "decision": ""})
        entry["candidates"][int(row["rank"])] = row["concept_uri"]
        if row.get("decision", "").strip():
            entry["decision"] = row["decision"]

    spans_by_key = {
        (gold.raw_id, span.token_start, span.token_end): span
        for gold in gold_ads
        for span in gold.spans
    }
    applied = 0
    for key, entry in by_span.items():
        span = spans_by_key.get(key)
        if span is None:
            log.warning("reference: Span %s nicht im Gold Standard, uebersprungen", key)
            continue
        uri = _resolve(entry["decision"], entry["candidates"])
        if uri is None:
            continue
        span.esco_uri = uri
        applied += 1
    log.info("reference: %d Referenzen gesetzt (%d Spans in der CSV)", applied, len(by_span))
    return applied
