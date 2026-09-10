from hrqde_c.extractors import adj_noun_en
from hrqde_c.models import RawAdvertisement


def _raw():
    return RawAdvertisement(
        id="ad-en-1", source_uri="https://example.org/1", crawl_date="2026-09-07",
        platform="test", employer="Firma", posting_date="2026-09-01",
        title="Titel", raw_text="Wir erwarten fundierte Java-Kenntnisse.",
    )


def test_extract_on_translated_text(monkeypatch):
    # Uebersetzung gemockt, der Test prueft die englische amod-Regel offline
    monkeypatch.setattr(
        adj_noun_en, "translate",
        lambda text: "We require strong analytical skills and modern tools.",
    )
    spans = adj_noun_en.extract(_raw())
    texts = [s.text for s in spans]
    assert "strong analytical skills" in texts
    assert "modern tools" in texts
    for s in spans:
        assert s.extractor == "adj_noun_en"
        # Offsets zeigen in den uebersetzten Text
        assert "We require strong analytical skills and modern tools."[
            s.char_start:s.char_end] == s.text
