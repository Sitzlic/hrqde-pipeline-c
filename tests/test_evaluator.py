import csv

from hrqde_c.gold import evaluator, reporter
from hrqde_c.models import (
    GoldAdvertisement,
    GoldSpan,
    RawAdvertisement,
    SpanCandidate,
    SpanType,
)

TEXT = "Wir suchen Erfahrung mit Java und agilen Methoden."


def _raw() -> RawAdvertisement:
    return RawAdvertisement(
        id="ad-1", source_uri="https://example.org/1", crawl_date="2026-09-06",
        platform="test", employer="Firma", posting_date="2026-09-01",
        title="Titel", raw_text=TEXT,
    )


def _candidate(start: int, end: int, span_type: SpanType) -> SpanCandidate:
    return SpanCandidate(
        id=f"c-{start}", processed_raw_id="ad-1", text=TEXT[start:end],
        span_type=span_type, char_start=start, char_end=end, extractor="adj_noun",
    )


def _gold(partition: str = "test", esco_uri: str | None = None) -> GoldAdvertisement:
    # "Java" ist Token 4 (Wir suchen Erfahrung mit Java ...)
    return GoldAdvertisement(
        raw_id="ad-1", partition=partition,
        tokens=TEXT.replace(".", " .").split(),
        bio_labels=[], spans=[GoldSpan(
            text="Java", span_type=SpanType.KNOWLEDGE,
            token_start=4, token_end=5, esco_uri=esco_uri,
        )],
        annotator_ids=["a1"],
    )


def test_evaluate_variant_counts_exact_matches():
    java = TEXT.index("Java")
    agil = TEXT.index("agilen")

    def fake_extract(raw):
        return [
            _candidate(java, java + 4, SpanType.KNOWLEDGE),      # Treffer
            _candidate(agil, agil + len("agilen"), SpanType.SKILL),  # falsch positiv
        ]

    run, result = evaluator.evaluate_variant(
        "adj-noun-de", [_gold()], {"ad-1": _raw()}, "gold.json", extract_fn=fake_extract
    )
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert run.variant == "adj-noun-de"
    assert run.advertisement_count == 1


def test_evaluate_variant_type_mismatch_is_no_match():
    java = TEXT.index("Java")

    def fake_extract(raw):
        return [_candidate(java, java + 4, SpanType.SKILL)]  # Typ falsch

    _, result = evaluator.evaluate_variant(
        "adj-noun-de", [_gold()], {"ad-1": _raw()}, "gold.json", extract_fn=fake_extract
    )
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 1, 1)


def test_evaluator_ignores_kalibrier_partition():
    _, result = evaluator.evaluate_variant(
        "adj-noun-de", [_gold(partition="kalibrier")], {"ad-1": _raw()},
        "gold.json", extract_fn=lambda raw: [],
    )
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 0, 0)


def test_report_csv_sorted_by_f1(tmp_path):
    def fake_extract_hit(raw):
        java = TEXT.index("Java")
        return [_candidate(java, java + 4, SpanType.KNOWLEDGE)]

    good = evaluator.evaluate_variant(
        "escoxlm-r-de", [_gold()], {"ad-1": _raw()}, "g", extract_fn=fake_extract_hit)
    bad = evaluator.evaluate_variant(
        "adj-noun-de", [_gold()], {"ad-1": _raw()}, "g", extract_fn=lambda raw: [])
    path = tmp_path / "report.csv"
    reporter.write_report([bad, good], path)
    rows = list(csv.DictReader(path.open()))
    assert [r["variant"] for r in rows] == ["escoxlm-r-de", "adj-noun-de"]
    assert rows[0]["f1"] == "1.0"