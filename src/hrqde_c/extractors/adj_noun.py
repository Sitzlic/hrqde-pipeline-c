from __future__ import annotations

import logging

from spacy.tokens import Doc

from hrqde_c.models import RawAdvertisement, SpanCandidate, SpanType

log = logging.getLogger(__name__)


def _candidate_spans(doc: Doc):
    # SQW-Regel: Adjektiv-Attribut + Nomen ("fundierte Java-Kenntnisse").
    # Achtung, deutsche spaCy-Modelle taggen nach TIGER, die Relation heisst
    # dort nk statt amod. nk deckt auch Artikel ab, daher der ADJ-Check.
    for token in doc:
        if token.pos_ != "ADJ":
            continue
        if token.dep_ not in ("nk", "amod"):
            continue
        head = token.head
        if head.pos_ not in ("NOUN", "PROPN"):
            continue
        if token.i > head.i:
            # nachgestellte Attribute sind fast immer Parser-Fehler
            continue
        yield doc[token.i : head.i + 1]


def extract(raw: RawAdvertisement, doc: Doc) -> list[SpanCandidate]:
    # TODO Uebersetzungsstufe fehlt noch (SA-C.1.03, Komposita-Problem),
    # bis dahin deutsches Original
    spans: list[SpanCandidate] = []
    for idx, span in enumerate(_candidate_spans(doc), start=1):
        spans.append(
            SpanCandidate(
                id=f"{raw.id}-adjn-{idx}",
                processed_raw_id=raw.id,
                text=span.text,
                # Regel kann skill/knowledge nicht unterscheiden, pauschal SKILL
                span_type=SpanType.SKILL,
                char_start=span.start_char,
                char_end=span.end_char,
                extractor="adj_noun",
            )
        )
    return spans
