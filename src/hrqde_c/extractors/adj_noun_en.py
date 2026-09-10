"""ADJ+NOUN nach Uebersetzung ins Englische (dritte Extraktionsvariante).

Uebersetzung satzweise mit Helsinki-NLP/opus-mt-de-en (lokal, kein
API-Key noetig), Extraktion mit derselben Attribut-Regel wie adj_noun,
nur auf dem englischen spaCy-Modell (dort heisst die Relation amod).

Achtung fuers Benchmarking: die char-Offsets der Spans beziehen sich auf
den UEBERSETZTEN Text, nicht auf das deutsche Original. Ein Span-F1
gegen den deutschen Gold Standard ist damit nicht direkt moeglich
(dafuer braeuchte es ein Wort-Alignment) - die Variante ist deshalb
nicht in der Evaluator-Registry, sondern dient dem Mapping-Pfad und
qualitativen Vergleichen.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from hrqde_c.models import RawAdvertisement, SpanCandidate, SpanType

if TYPE_CHECKING:
    from spacy.language import Language
    from transformers.pipelines import Pipeline

log = logging.getLogger(__name__)

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-de-en"
EN_SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _load_translator() -> "Pipeline":
    from transformers import pipeline

    log.info("adj_noun_en: lade %s", TRANSLATION_MODEL)
    return pipeline("translation", model=TRANSLATION_MODEL)


@lru_cache(maxsize=1)
def _load_en_nlp() -> "Language":
    import spacy

    try:
        return spacy.load(EN_SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy-Modell '{EN_SPACY_MODEL}' nicht installiert. "
            f"Installieren mit: python -m spacy download {EN_SPACY_MODEL}"
        ) from exc


def translate(text: str) -> str:
    """Satzweise Uebersetzung, damit lange Anzeigen nicht am Sequenzlimit reissen."""
    from hrqde_c.pipeline import _load_nlp

    sentences = [s.text.strip() for s in _load_nlp()(text).sents if s.text.strip()]
    if not sentences:
        return ""
    translator = _load_translator()
    results = translator(sentences, max_length=512)
    return " ".join(r["translation_text"] for r in results)


def _candidate_spans(doc):
    # gleiche Regel wie adj_noun, englisches Schema: amod statt nk
    for token in doc:
        if token.pos_ != "ADJ":
            continue
        if token.dep_ != "amod":
            continue
        head = token.head
        if head.pos_ not in ("NOUN", "PROPN"):
            continue
        if token.i > head.i:
            continue
        yield doc[token.i : head.i + 1]


def extract(raw: RawAdvertisement) -> list[SpanCandidate]:
    """Spans auf dem uebersetzten Text, Offsets beziehen sich auf translate()."""
    translated = translate(raw.raw_text)
    doc = _load_en_nlp()(translated)
    spans: list[SpanCandidate] = []
    for idx, span in enumerate(_candidate_spans(doc), start=1):
        spans.append(
            SpanCandidate(
                id=f"{raw.id}-adjnen-{idx}",
                processed_raw_id=raw.id,
                text=span.text,
                span_type=SpanType.SKILL,
                char_start=span.start_char,
                char_end=span.end_char,
                extractor="adj_noun_en",
            )
        )
    log.info("adj_noun_en: %s -> %d Spans (auf Uebersetzung)", raw.id, len(spans))
    return spans
