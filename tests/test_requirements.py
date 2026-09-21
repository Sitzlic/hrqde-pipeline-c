from hrqde_c import pipeline
from hrqde_c.models import (
    MappingDecision,
    RawAdvertisement,
    RequirementKind,
    SpanCandidate,
    SpanMapping,
    SpanType,
)


def _raw() -> RawAdvertisement:
    return RawAdvertisement(
        id="r1", source_uri="https://example.org/r1", crawl_date="2026-09-22",
        platform="test", employer="Firma", posting_date="2026-09-01",
        title="Zoologischer Garteninspektor", raw_text="x",
    )


def _span(sid: str, kind: RequirementKind) -> SpanCandidate:
    return SpanCandidate(
        id=sid, processed_raw_id="r1", text=sid, span_type=SpanType.SKILL,
        char_start=0, char_end=1, extractor="adj_noun", requirement_kind=kind,
    )


def _mapping(sid: str, score: float) -> SpanMapping:
    return SpanMapping(
        id=f"m-{sid}", span_id=sid, concept_uri="esco:1", score=score,
        decision=MappingDecision.ACCEPTED,
    )


def test_must_beats_nice_to_have_regardless_of_order_and_score():
    must = _span("a", RequirementKind.MUST)
    nice = _span("b", RequirementKind.NICE_TO_HAVE)
    for spans in ([must, nice], [nice, must]):
        reqs, _ = pipeline.build_requirements(
            _raw(), spans, [_mapping("a", 0.6), _mapping("b", 0.9)]
        )
        assert len(reqs) == 1
        assert reqs[0].requirement_kind == RequirementKind.MUST
        assert reqs[0].provenance_confidence == 0.6


def test_highest_score_wins_within_same_kind():
    spans = [_span("a", RequirementKind.MUST), _span("b", RequirementKind.MUST)]
    reqs, _ = pipeline.build_requirements(_raw(), spans, [_mapping("a", 0.6), _mapping("b", 0.9)])
    assert len(reqs) == 1
    assert reqs[0].provenance_confidence == 0.9
