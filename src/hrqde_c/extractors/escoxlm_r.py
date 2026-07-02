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


@lru_cache(maxsize=2)
def _load(model_id: str) -> "Pipeline":
    # lazy import, Basis-Package soll ohne die ml-Extras installierbar sein
    from transformers import pipeline

    log.info("escoxlm_r: lade %s", model_id)
    return pipeline(model=model_id, aggregation_strategy="first")


def extract(raw: RawAdvertisement) -> list[SpanCandidate]:
    text = raw.raw_text
    hits: list[tuple[str, dict]] = []
    for hit in _load(SKILL_MODEL)(text):
        hits.append(("skill", hit))
    for hit in _load(KNOWLEDGE_MODEL)(text):
        hits.append(("knowledge", hit))

    spans: list[SpanCandidate] = []
    for idx, (kind, hit) in enumerate(hits, start=1):
        spans.append(
            SpanCandidate(
                id=f"{raw.id}-esco-{idx}",
                processed_raw_id=raw.id,
                text=hit["word"],
                span_type=SpanType.SKILL if kind == "skill" else SpanType.KNOWLEDGE,
                char_start=int(hit["start"]),
                char_end=int(hit["end"]),
                extractor="escoxlm_r",
                score=float(hit["score"]),
            )
        )
    return spans
