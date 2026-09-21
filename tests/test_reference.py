import csv

from hrqde_c.gold import reference
from hrqde_c.models import GoldAdvertisement, GoldSpan, SpanType


def _gold(raw_id: str, partition: str = "kalibrier") -> GoldAdvertisement:
    return GoldAdvertisement(
        raw_id=raw_id, partition=partition,
        tokens=["Wir", "suchen", "Java", "und", "Teamgeist"],
        bio_labels=["O", "O", "B-Knowledge", "O", "B-Skill"],
        spans=[
            GoldSpan(text="Java", span_type=SpanType.KNOWLEDGE, token_start=2, token_end=3),
            GoldSpan(text="Teamgeist", span_type=SpanType.SKILL, token_start=4, token_end=5),
        ],
        annotator_ids=["a"],
    )


def _fake_candidates(text: str, top_k: int):
    return [(f"uri:{text.lower()}-{i}", f"{text} {i}", 0.9 - i * 0.1) for i in range(1, top_k + 1)]


def test_candidates_only_for_kalibrier_partition(tmp_path):
    path = tmp_path / "cand.csv"
    n = reference.write_candidates(
        [_gold("k1"), _gold("t1", partition="test")], path, top_k=2,
        candidates_fn=_fake_candidates,
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert n == 2
    assert len(rows) == 4  # zwei Spans x Top-2
    assert {r["raw_id"] for r in rows} == {"k1"}


def test_apply_decisions_rank_uri_and_no_concept(tmp_path):
    gold = _gold("k1")
    path = tmp_path / "cand.csv"
    reference.write_candidates([gold], path, top_k=2, candidates_fn=_fake_candidates)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["decision"] = "2"                # Java -> Rang 2
    rows[2]["decision"] = reference.NO_CONCEPT  # Teamgeist -> kein Konzept
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=reference.FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    applied = reference.apply_decisions([gold], path)
    assert applied == 2
    assert gold.spans[0].esco_uri == "uri:java-2"
    assert gold.spans[1].esco_uri == reference.NO_CONCEPT


def test_apply_decisions_skips_empty(tmp_path):
    gold = _gold("k1")
    path = tmp_path / "cand.csv"
    reference.write_candidates([gold], path, top_k=1, candidates_fn=_fake_candidates)
    assert reference.apply_decisions([gold], path) == 0
    assert gold.spans[0].esco_uri is None
