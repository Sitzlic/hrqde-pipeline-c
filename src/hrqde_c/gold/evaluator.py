"""Span-F1 der Extraktionsvarianten gegen den Testteil (SA-C.3.04).

Span-Uebereinstimmung ist der exakte Match auf Token-Spanne und Span-Typ.
Die Varianten-Bezeichner folgen der Konzeption, adj-noun-en-uebersetzt ist
im Prototyp nicht realisiert (deklarierte Abweichung von SA-C.1.03).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable

from hrqde_c.models import (
    EvaluationResult,
    EvaluationRun,
    GoldAdvertisement,
    RawAdvertisement,
    SpanCandidate,
    SpanType,
)
from hrqde_c.pipeline import _load_nlp

log = logging.getLogger(__name__)

TokenSpan = tuple[int, int, SpanType]


def _extract_adj_noun(raw: RawAdvertisement) -> list[SpanCandidate]:
    from hrqde_c.extractors import adj_noun

    return adj_noun.extract(raw, _load_nlp()(raw.raw_text))


def _extract_escoxlm_r(raw: RawAdvertisement) -> list[SpanCandidate]:
    from hrqde_c.extractors import escoxlm_r

    return escoxlm_r.extract(raw)


VARIANTS: dict[str, Callable[[RawAdvertisement], list[SpanCandidate]]] = {
    "adj-noun-de": _extract_adj_noun,
    "escoxlm-r-de": _extract_escoxlm_r,
}


def token_spans(text: str, candidates: list[SpanCandidate]) -> set[TokenSpan]:
    """Zeichen-Spans der Extraktoren auf Token-Spannen umrechnen,
    dieselbe Abdeckungsregel wie beim Gold-Import."""
    toks = [tok for tok in _load_nlp()(text) if not tok.is_space]
    spans: set[TokenSpan] = set()
    for cand in candidates:
        covered = [
            i for i, tok in enumerate(toks)
            if tok.idx < cand.char_end and tok.idx + len(tok.text) > cand.char_start
        ]
        if covered:
            spans.add((covered[0], covered[-1] + 1, cand.span_type))
    return spans


def _gold_spans(gold: GoldAdvertisement) -> set[TokenSpan]:
    return {(s.token_start, s.token_end, s.span_type) for s in gold.spans}


def evaluate_variant(
    variant: str,
    gold_ads: list[GoldAdvertisement],
    raws_by_id: dict[str, RawAdvertisement],
    gold_dataset: str,
    extract_fn: Callable[[RawAdvertisement], list[SpanCandidate]] | None = None,
) -> tuple[EvaluationRun, EvaluationResult]:
    extract = extract_fn or VARIANTS[variant]
    test_ads = [g for g in gold_ads if g.partition == "test"]
    tp = fp = fn = 0
    for gold in test_ads:
        raw = raws_by_id.get(gold.raw_id)
        if raw is None:
            log.warning("evaluator: kein Rohtext zu %s, uebersprungen", gold.raw_id)
            continue
        predicted = token_spans(raw.raw_text, extract(raw))
        expected = _gold_spans(gold)
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    run = EvaluationRun(
        id=f"run-{uuid.uuid4().hex[:8]}",
        variant=variant,
        gold_dataset=gold_dataset,
        advertisement_count=len(test_ads),
        run_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    result = EvaluationResult(
        run_id=run.id,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )
    log.info(
        "evaluator: %s -> P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)",
        variant, precision, recall, f1, tp, fp, fn,
    )
    return run, result
