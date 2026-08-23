"""Zeichen-Spans in Token-BIO-Labels umrechnen (SA-C.3.02).

Tokenbestand wie in process_nlp: spaCy-Tokens ohne Whitespace, damit die
Token-Positionen der GoldSpans zum uebrigen Pipeline-Bestand passen.
"""

from __future__ import annotations

import logging

from hrqde_c.models import GoldSpan, SpanType
from hrqde_c.pipeline import _load_nlp

log = logging.getLogger(__name__)

LABEL_BY_TYPE = {SpanType.SKILL: "Skill", SpanType.KNOWLEDGE: "Knowledge"}
TYPE_BY_LABEL = {v: k for k, v in LABEL_BY_TYPE.items()}


def to_bio(
    text: str, regions: list[tuple[int, int, SpanType]]
) -> tuple[list[str], list[str], list[GoldSpan]]:
    """(tokens, bio_labels, gold_spans) fuer eine Anzeige.

    Ueberlappt eine Region eine bereits belegte, wird sie uebersprungen.
    """
    doc = _load_nlp()(text)
    toks = [tok for tok in doc if not tok.is_space]
    tokens = [tok.text for tok in toks]
    labels = ["O"] * len(tokens)
    spans: list[GoldSpan] = []

    for start, end, span_type in sorted(regions, key=lambda r: (r[0], r[1])):
        covered = [
            i for i, tok in enumerate(toks)
            if tok.idx < end and tok.idx + len(tok.text) > start
        ]
        if not covered:
            log.warning("bio: Region %d-%d trifft keinen Token, uebersprungen", start, end)
            continue
        if any(labels[i] != "O" for i in covered):
            log.warning("bio: Region %d-%d ueberlappt eine andere, uebersprungen", start, end)
            continue
        name = LABEL_BY_TYPE[span_type]
        labels[covered[0]] = f"B-{name}"
        for i in covered[1:]:
            labels[i] = f"I-{name}"
        spans.append(GoldSpan(
            text=text[start:end],
            span_type=span_type,
            token_start=covered[0],
            token_end=covered[-1] + 1,
        ))
    return tokens, labels, spans
