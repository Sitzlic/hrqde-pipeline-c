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


def test_ttl_conforms_to_hrqde_shapes(tmp_path: Path) -> None:
    from pyshacl import validate

    input_path = Path("data/input/example_posting.json")
    outputs = pipeline.run(input_path, tmp_path / "out")
    conforms, _, report = validate(str(outputs[0]), shacl_graph=str(SHAPES))
    assert conforms, report
