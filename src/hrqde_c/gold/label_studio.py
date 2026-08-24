"""Export nach und Import aus Label Studio (SA-C.3.01).

Der Pre-Annotator vereinigt die Span-Kandidaten aller Extraktionsvarianten
und schreibt sie als Predictions in die Tasks. Annotatoren bestaetigen,
passen an oder loeschen im Tool, der Export kommt als JSON zurueck.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hrqde_c.gold import bio
from hrqde_c.gold.adjudicator import cohens_kappa
from hrqde_c.models import GoldAdvertisement, RawAdvertisement, SpanCandidate, SpanType
from hrqde_c.pipeline import _load_nlp

log = logging.getLogger(__name__)

CONFIG_XML = """<View>
  <Header value="$title"/>
  <Labels name="label" toName="text">
    <Label value="Skill" background="#5b9bd5"/>
    <Label value="Knowledge" background="#70ad47"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
"""


def union_spans(spans: list[SpanCandidate]) -> list[SpanCandidate]:
    """Variantenneutrale Vereinigung: exakte Doppel (Lage + Typ) fallen raus."""
    seen: set[tuple[int, int, SpanType]] = set()
    out: list[SpanCandidate] = []
    for span in sorted(spans, key=lambda s: (s.char_start, s.char_end)):
        key = (span.char_start, span.char_end, span.span_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


def _prediction(spans: list[SpanCandidate]) -> dict:
    return {
        "model_version": "pre-annotator",
        "result": [
            {
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": s.char_start,
                    "end": s.char_end,
                    "text": s.text,
                    "labels": [bio.LABEL_BY_TYPE[s.span_type]],
                },
            }
            for s in spans
        ],
    }


def build_tasks(raws: list[RawAdvertisement], partition: dict[str, str]) -> list[dict]:
    # Import hier, damit das Basis-Package ohne ml-Extras nutzbar bleibt
    from hrqde_c.extractors import adj_noun, escoxlm_r

    nlp = _load_nlp()
    tasks = []
    for raw in raws:
        if raw.id not in partition:
            continue
        spans = union_spans(adj_noun.extract(raw, nlp(raw.raw_text)) + escoxlm_r.extract(raw))
        tasks.append({
            "data": {
                "raw_id": raw.id,
                "title": raw.title,
                "partition": partition[raw.id],
                "text": raw.raw_text,
            },
            "predictions": [_prediction(spans)],
        })
        log.info("label-studio: Task %s mit %d Vor-Annotationen", raw.id, len(spans))
    return tasks


def write_tasks(tasks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("label-studio: %d Tasks -> %s", len(tasks), path)


def write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_XML, encoding="utf-8")


def _annotator_id(annotation: dict) -> str:
    completed_by = annotation.get("completed_by")
    if isinstance(completed_by, dict):
        return str(completed_by.get("email") or completed_by.get("id") or "unbekannt")
    return str(completed_by or "unbekannt")


def _regions(annotation: dict) -> list[tuple[int, int, SpanType]]:
    regions = []
    for item in annotation.get("result") or []:
        if item.get("type") != "labels":
            continue
        value = item["value"]
        labels = value.get("labels") or []
        if not labels or labels[0] not in bio.TYPE_BY_LABEL:
            continue
        regions.append((int(value["start"]), int(value["end"]), bio.TYPE_BY_LABEL[labels[0]]))
    return regions


def import_gold(
    export_payload: list[dict], partition: dict[str, str]
) -> list[GoldAdvertisement]:
    """LS-Export -> GoldAdvertisements. Erste Annotation ist massgeblich,
    bei Doppelannotation wird Cohen's Kappa auf Token-Ebene berechnet."""
    gold: list[GoldAdvertisement] = []
    for task in export_payload:
        data = task.get("data") or {}
        raw_id, text = data.get("raw_id"), data.get("text")
        if not raw_id or not text:
            log.warning("import: Task ohne raw_id/text, uebersprungen")
            continue
        if raw_id not in partition:
            log.warning("import: %s nicht im Manifest, uebersprungen", raw_id)
            continue
        annotations = task.get("annotations") or []
        if not annotations:
            log.warning("import: %s ohne Annotation, uebersprungen", raw_id)
            continue

        tokens, labels, spans = bio.to_bio(text, _regions(annotations[0]))
        kappa = None
        if len(annotations) >= 2:
            _, labels_b, _ = bio.to_bio(text, _regions(annotations[1]))
            kappa = cohens_kappa(labels, labels_b)
        gold.append(GoldAdvertisement(
            raw_id=raw_id,
            partition=partition[raw_id],
            tokens=tokens,
            bio_labels=labels,
            spans=spans,
            annotator_ids=[_annotator_id(a) for a in annotations],
            inter_annotator_agreement=kappa,
        ))
    log.info("import: %d GoldAdvertisements", len(gold))
    return gold
