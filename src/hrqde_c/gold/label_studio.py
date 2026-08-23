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
from hrqde_c.models import RawAdvertisement, SpanCandidate, SpanType
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
