import json
from pathlib import Path

from hrqde_c import pipeline

SHAPES = Path("data/shapes/hrqde-shapes-all.ttl")


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    input_path = Path("data/input/example_posting.json")
    outputs = pipeline.run(input_path, tmp_path / "out")
    assert len(outputs) == 1

    ttl = outputs[0].read_text(encoding="utf-8")
    assert "hrqde:JobPosting" in ttl
    assert "hrqde:Employer" in ttl
    assert "hrqde:hasRequirement" in ttl
    assert "hrqde:QualificationRequirement" in ttl

    meta_path = tmp_path / "out" / "run_metadata.json"
    assert meta_path.exists()

    # Demand-Bestand des Laufs (UCon1-UCas4): Drafts und Kandidaten-Sicht
    drafts_path = tmp_path / "out" / "drafts.json"
    candidates_path = tmp_path / "out" / "span_candidates.csv"
    assert drafts_path.exists()
    assert candidates_path.exists()
    drafts = json.loads(drafts_path.read_text(encoding="utf-8"))
    assert len(drafts) == 1
    assert drafts[0]["span_candidates"]
    header = candidates_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("raw_id,span_id,span_text,span_type,extractor")


def test_process_nlp_token_annotations() -> None:
    from hrqde_c.models import RawAdvertisement

    raw = RawAdvertisement(
        id="t1", source_uri="https://example.org/1", crawl_date="2026-01-01",
        platform="test", employer="Firma", posting_date="2026-01-01",
        title="Test", raw_text="Wir suchen eine erfahrene Pflegekraft. Sie arbeiten im Team.",
    )
    doc = pipeline._load_nlp()(raw.raw_text)
    processed = pipeline.process_nlp(raw, doc)
    assert processed.tokens
    assert all(tok.dep for tok in processed.tokens)
    assert {tok.sentence for tok in processed.tokens} == {0, 1}


def test_ttl_conforms_to_hrqde_shapes(tmp_path: Path) -> None:
    from pyshacl import validate

    input_path = Path("data/input/example_posting.json")
    outputs = pipeline.run(input_path, tmp_path / "out")
    conforms, _, report = validate(str(outputs[0]), shacl_graph=str(SHAPES))
    assert conforms, report


def test_ttl_escapes_quotes_in_labels(tmp_path):
    import json

    from hrqde_c import pipeline

    posting = {
        "id": "quote-1", "source_uri": "https://example.org/1", "crawl_date": "2026-09-13",
        "platform": "test", "employer": 'Infineon AG "Standort Innsbruck"',
        "posting_date": "2026-09-01", "title": 'Entwickler "Embedded"',
        "raw_text": "Wir suchen Erfahrung mit Java und Python.",
    }
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps([posting]), encoding="utf-8")
    outputs = pipeline.run(inp, tmp_path / "out")

    # muss als valides Turtle parsebar sein
    import rdflib
    g = rdflib.Graph()
    g.parse(str(outputs[0]), format="turtle")
    assert any('Standort Innsbruck' in str(o) for o in g.objects())