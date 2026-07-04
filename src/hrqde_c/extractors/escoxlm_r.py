from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from hrqde_c.models import RawAdvertisement, SpanCandidate, SpanType

if TYPE_CHECKING:
    from transformers.pipelines import Pipeline

log = logging.getLogger(__name__)

SKILL_MODEL = "jjzha/escoxlmr_skill_extraction"
KNOWLEDGE_MODEL = "jjzha/escoxlmr_knowledge_extraction"

TRAILING_PUNCT = ".,;:!?)"


@lru_cache(maxsize=2)
def _load(model_id: str) -> "Pipeline":
    # lazy import, Basis-Package soll ohne die ml-Extras installierbar sein
    from transformers import pipeline

    log.info("escoxlm_r: lade %s", model_id)
    return pipeline(model=model_id, aggregation_strategy="first")


def _strip_trailing_punct(hit: dict) -> dict:
    word = hit["word"]
    end = hit["end"]
    while word and word[-1] in TRAILING_PUNCT:
        word = word[:-1]
        end -= 1
    return {**hit, "word": word, "end": end}


def _merge_adjacent(hits: list[dict]) -> list[dict]:
    # wie aggregate_span im offiziellen Demo-Space: Luecke <= 1 Zeichen wird gemerged
    if not hits:
        return []
    sorted_hits = sorted(hits, key=lambda h: h["start"])
    merged: list[dict] = [dict(sorted_hits[0])]
    for h in sorted_hits[1:]:
        prev = merged[-1]
        if h["start"] - prev["end"] <= 1:
            prev["word"] = f"{prev['word']} {h['word']}"
            prev["end"] = h["end"]
            prev["score"] = (prev["score"] + h["score"]) / 2
        else:
            merged.append(dict(h))
    return merged


def _postprocess(hits: list[dict]) -> list[dict]:
    stripped = [_strip_trailing_punct(h) for h in hits]
    return _merge_adjacent([h for h in stripped if h["word"]])


def extract(raw: RawAdvertisement) -> list[SpanCandidate]:
    text = raw.raw_text
    labelled: list[tuple[SpanType, dict]] = []
    for hit in _postprocess(_load(SKILL_MODEL)(text)):
        labelled.append((SpanType.SKILL, hit))
    for hit in _postprocess(_load(KNOWLEDGE_MODEL)(text)):
        labelled.append((SpanType.KNOWLEDGE, hit))

    spans: list[SpanCandidate] = []
    for idx, (kind, hit) in enumerate(labelled, start=1):
        spans.append(
            SpanCandidate(
                id=f"{raw.id}-esco-{idx}",
                processed_raw_id=raw.id,
                text=hit["word"],
                span_type=kind,
                char_start=int(hit["start"]),
                char_end=int(hit["end"]),
                extractor="escoxlm_r",
                score=float(hit["score"]),
            )
        )
    return spans
