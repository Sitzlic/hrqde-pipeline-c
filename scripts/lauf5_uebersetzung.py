"""Lauf 5: Übersetzungsvariante auf dem Testteil, Kandidatenliste mit Mapping-Ergebnis."""
import csv, json, sys, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
from hrqde_c import pipeline
from hrqde_c.extractors import adj_noun, adj_noun_en
from hrqde_c.mapping import esco

manifest = json.loads(Path("data/gold/sample_manifest.json").read_text("utf-8"))
test = {it["raw_id"]: it["stratum"] for it in manifest["items"] if it["partition"] == "test"}
raws = [r for r in pipeline.acquire(Path("data/input/ba_korpus.json")) if r.id in test]
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
index = esco.get_index(); nlp = pipeline._load_nlp()
rows, summary = [], []
for raw in raws:
    de = adj_noun.extract(raw, nlp(raw.raw_text))
    en_text = adj_noun_en.translate(raw.raw_text)
    en = adj_noun_en.extract(raw)
    for s in en:
        m = index.match(s.text, top_k=1)[0]
        rows.append({"raw_id": raw.id, "stratum": test[raw.id], "span_text_en": s.text, "span_type": s.span_type.value,
                     "concept_uri": m.concept.uri, "matched_label": m.label, "score": f"{m.score:.4f}"})
    summary.append({"raw_id": raw.id, "stratum": test[raw.id], "chars_de": len(raw.raw_text), "chars_en": len(en_text),
                    "spans_adj_noun_de": len(de), "spans_adj_noun_en": len(en)})
with out.open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with out.with_name(out.stem + "_summary.csv").open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
print(len(raws), "Anzeigen,", len(rows), "EN-Spans ->", out)
