import json

from hrqde_c.gold import bio, label_studio, sampling
from hrqde_c.models import RawAdvertisement, SpanCandidate, SpanType

TEXT = "Wir suchen Erfahrung mit Java und agilen Methoden."


def _raws(n: int) -> list[RawAdvertisement]:
    return [
        RawAdvertisement(
            id=f"ad-{i}", source_uri=f"https://example.org/{i}", crawl_date="2026-09-06",
            platform="test", employer="Firma", posting_date="2026-09-01",
            title="Titel", raw_text=TEXT,
        )
        for i in range(n)
    ]


def test_to_bio_single_and_multi_token():
    java = TEXT.index("Java")
    agil = TEXT.index("agilen")
    regions = [
        (java, java + len("Java"), SpanType.KNOWLEDGE),
        (agil, agil + len("agilen Methoden"), SpanType.SKILL),
    ]
    tokens, labels, spans = bio.to_bio(TEXT, regions)
    assert labels[tokens.index("Java")] == "B-Knowledge"
    assert labels[tokens.index("agilen")] == "B-Skill"
    assert labels[tokens.index("Methoden")] == "I-Skill"
    assert labels.count("O") == len(tokens) - 3
    assert [s.text for s in spans] == ["Java", "agilen Methoden"]
    assert spans[1].token_end - spans[1].token_start == 2


def test_to_bio_skips_overlapping_region():
    java = TEXT.index("Java")
    regions = [
        (java, java + 4, SpanType.KNOWLEDGE),
        (java, java + 4, SpanType.SKILL),  # gleiche Stelle, anderer Typ
    ]
    _, labels, spans = bio.to_bio(TEXT, regions)
    assert len(spans) == 1
    assert labels.count("B-Knowledge") == 1


def test_sampling_is_deterministic_and_split():
    raws = _raws(20)
    p1 = sampling.draw(raws, size=10, calib_ratio=0.5, seed=42)
    p2 = sampling.draw(raws, size=10, calib_ratio=0.5, seed=42)
    assert p1 == p2
    assert len(p1) == 10
    assert sum(1 for v in p1.values() if v == "kalibrier") == 5


def test_manifest_roundtrip(tmp_path):
    partition = {"ad-1": "kalibrier", "ad-2": "test"}
    path = tmp_path / "manifest.json"
    sampling.write_manifest(path, partition, seed=42, calib_ratio=0.5)
    assert sampling.load_manifest(path) == partition
    assert json.loads(path.read_text())["seed"] == 42


def test_union_spans_removes_exact_duplicates():
    java = TEXT.index("Java")
    def span(extractor, start, end, span_type):
        return SpanCandidate(
            id=f"{extractor}-{start}", processed_raw_id="ad-1", text=TEXT[start:end],
            span_type=span_type, char_start=start, char_end=end, extractor=extractor,
        )
    spans = [
        span("adj_noun", java, java + 4, SpanType.KNOWLEDGE),
        span("escoxlm_r", java, java + 4, SpanType.KNOWLEDGE),  # exaktes Doppel
        span("escoxlm_r", java, java + 4, SpanType.SKILL),      # anderer Typ, bleibt
    ]
    assert len(label_studio.union_spans(spans)) == 2