"""Zeigt, was jeder Extraktor findet und wo sie sich ueberlappen.

Kein Gold-Standard-Vergleich, das kommt erst mit RC-C.3.2.

Aufruf: python scripts/compare_extractors.py [--input PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hrqde_c import pipeline
from hrqde_c.extractors import adj_noun, escoxlm_r


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/input/postings_sample.json"))
    args = parser.parse_args()

    raws = pipeline.acquire(args.input)
    nlp = pipeline._load_nlp()

    total = {"adj": 0, "esco": 0, "overlap": 0}

    for raw in raws:
        adj_spans = adj_noun.extract(raw, nlp(raw.raw_text))
        esco_spans = escoxlm_r.extract(raw)

        overlap_adj = set()
        overlap_esco = set()
        for i, a in enumerate(adj_spans):
            for j, e in enumerate(esco_spans):
                if _overlaps(a.char_start, a.char_end, e.char_start, e.char_end):
                    overlap_adj.add(i)
                    overlap_esco.add(j)

        only_adj = [a for i, a in enumerate(adj_spans) if i not in overlap_adj]
        only_esco = [e for j, e in enumerate(esco_spans) if j not in overlap_esco]

        total["adj"] += len(adj_spans)
        total["esco"] += len(esco_spans)
        total["overlap"] += len(overlap_adj)

        print(f"\n{'=' * 72}")
        print(f"{raw.id}  -  {raw.title}")
        print(f"{'=' * 72}")
        print(f"  ADJ+NOUN: {len(adj_spans):2d} Spans   "
              f"ESCOXLM-R: {len(esco_spans):2d} Spans   "
              f"überlappend: {len(overlap_adj)}")

        print(f"\n  Nur ADJ+NOUN ({len(only_adj)}):")
        for s in only_adj:
            print(f"    - {s.text}")

        print(f"\n  Nur ESCOXLM-R ({len(only_esco)}):")
        for s in only_esco:
            print(f"    - [{s.span_type.value}] {s.text}")

    print(f"\n{'=' * 72}")
    print("Summe über alle Anzeigen")
    print(f"{'=' * 72}")
    print(f"  ADJ+NOUN gesamt:   {total['adj']}")
    print(f"  ESCOXLM-R gesamt:  {total['esco']}")
    print(f"  überlappend:       {total['overlap']}")


if __name__ == "__main__":
    main()
