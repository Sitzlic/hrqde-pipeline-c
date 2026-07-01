from pathlib import Path

from hrqde_c import pipeline


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    input_path = Path("data/input/example_posting.json")
    outputs = pipeline.run(input_path, tmp_path / "out")
    assert len(outputs) == 1
    ttl = outputs[0].read_text(encoding="utf-8")
    assert "hrqde:JobPosting" in ttl
    assert "hrqde:QualificationRequirement" in ttl
